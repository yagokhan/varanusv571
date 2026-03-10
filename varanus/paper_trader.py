"""
varanus/paper_trader.py — Varanus Tier 2 Paper Trading Engine

Lifecycle per 4h candle close:
  1. check_exits()    — for each open paper trade, fetch latest bar, check TP/SL/time
  2. _check_and_halt() — evaluate circuit breaker (daily -5%, drawdown -15%)
  3. scan()           — for each asset: build features → model.predict() on latest bar
  4. For each signal: open paper trade, send Telegram entry alert (with position size)
  5. save_state()     — persist state to config/paper_state.json

No real orders are placed. All trades tracked in config/paper_state.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

import ccxt
import numpy as np
import pandas as pd

from varanus.universe import TIER2_UNIVERSE, HIGH_VOL_SUBTIER
from varanus.pa_features import build_features, compute_atr
from varanus.tbm_labeler import label_trades, calculate_barriers, TBM_CONFIG, build_dual_labels
from varanus.model import VaranusModel, VaranusDualModel, get_leverage, MODEL_CONFIG
from varanus.risk import (
    RISK_CONFIG,
    get_position_size,
    would_breach_leverage,
    compute_portfolio_leverage,
)
from varanus.alerts import (send_alert, send_exit_alert, send_halt_alert,
                            send_no_signal_alert, send_heartbeat_alert)

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
CACHE_DIR  = _HERE / "data" / "cache"
STATE_FILE = _HERE / "config" / "paper_state.json"
TRADES_CSV = _HERE / "results"  / "paper_trades.csv"
ENV_FILE   = _HERE / "config"   / "telegram.env"

# ── Live data lookback ─────────────────────────────────────────────────────────
LIVE_BARS_4H = 400   # ~67 days — enough for EMA55, ATR100, rolling percentiles
LIVE_BARS_1D = 150   # 150 calendar days for HTF daily bias


# ══════════════════════════════════════════════════════════════════════════════
# Barrier helper  (mirrors backtest._check_barriers with flash-wick guard)
# ══════════════════════════════════════════════════════════════════════════════

def _check_barriers(bar: pd.Series, trade: dict) -> Optional[dict]:
    """
    Check TP / SL / time barrier for one OHLCV bar against an open paper trade.
    Returns {'type': 'tp'|'sl'|'time', 'price': float} or None.
    """
    d   = trade["direction"]
    ts  = bar.name

    # Time barrier (checked first)
    max_hold = pd.Timestamp(trade["max_hold_ts"]).tz_localize("UTC") \
        if pd.Timestamp(trade["max_hold_ts"]).tzinfo is None \
        else pd.Timestamp(trade["max_hold_ts"])
    if ts >= max_hold:
        return {"type": "time", "price": float(bar["close"])}

    tp = trade["take_profit"]
    sl = trade["stop_loss"]

    # TP — wick touch is sufficient (we want the gain)
    if d == 1  and bar["high"] >= tp:
        return {"type": "tp", "price": tp}
    if d == -1 and bar["low"]  <= tp:
        return {"type": "tp", "price": tp}

    # SL — flash-wick guard: body close required
    if d == 1  and bar["close"] < sl:
        return {"type": "sl", "price": sl}
    if d == -1 and bar["close"] > sl:
        return {"type": "sl", "price": sl}

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PaperTrader
# ══════════════════════════════════════════════════════════════════════════════

class PaperTrader:
    """
    Varanus Tier 2 Paper Trading Engine.

    Usage
    -----
    trader = PaperTrader(initial_capital=5000.0)
    trader.train()          # once at startup: trains XGBoost on historical cache
    trader.run_cycle()      # call every 4h: exits → health → scan → alerts
    """

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        initial_capital: float = 5_000.0,
        dry_run: bool = False,
        state_file: Path = STATE_FILE,
    ):
        self.dry_run    = dry_run
        self.state_file = Path(state_file)
        self.model: Optional[VaranusDualModel] = None

        # Best Optuna params — v5.7
        params_path = _HERE / "config" / "best_params_v57.json"
        if not params_path.exists():
            params_path = _HERE / "config" / "best_params_v52.json"
        if not params_path.exists():
            params_path = _HERE / "config" / "best_params_v51.json"
        if not params_path.exists():
            params_path = _HERE / "config" / "best_params.json"
        with open(params_path) as f:
            self.params = json.load(f)

        # Telegram credentials
        self._bot_token, self._chat_id = self._load_credentials()

        # Persistent state
        self.state = self._load_state(initial_capital)

        # Exchange disabled — network blocked on this machine.
        # Paper trading uses local parquet cache (updated daily via fetch_historical_data.py)
        self.exchange = None

        logger.info(
            "PaperTrader ready | capital=$%.2f | open_trades=%d | dry_run=%s",
            self.state["equity"],
            len(self.state["open_trades"]),
            dry_run,
        )

    # ── Credentials ───────────────────────────────────────────────────────────

    def _load_credentials(self) -> tuple[str, str]:
        env: dict[str, str] = {}
        if ENV_FILE.exists():
            with open(ENV_FILE) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        env[k.strip()] = v.strip()
        token   = env.get("VARANUS_BOT_TOKEN", os.environ.get("VARANUS_BOT_TOKEN", ""))
        chat_id = env.get("VARANUS_CHAT_ID",   os.environ.get("VARANUS_CHAT_ID",   ""))
        return token, chat_id

    # ── State persistence ─────────────────────────────────────────────────────

    def _load_state(self, initial_capital: float) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        default = {
            "initial_capital":    initial_capital,
            "equity":             initial_capital,
            "peak_equity":        initial_capital,
            "daily_start_equity": initial_capital,
            "daily_start_date":   datetime.now(timezone.utc).date().isoformat(),
            "open_trades":        {},   # {asset: trade_dict}
            "closed_trades":      [],
            "halted":             False,
        }
        self._write_state(default)
        return default

    def _write_state(self, state: Optional[dict] = None) -> None:
        if self.dry_run:
            return
        target = state if state is not None else self.state
        os.makedirs(self.state_file.parent, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(target, f, indent=2, default=str)

    # ── Model training ────────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Load full historical data from Chameleon parquet cache, build features,
        generate TBM labels, and train the XGBoost model.

        Uses the last 10 % of rows as validation set for early stopping.
        Call once at startup (~60 s for 15 assets).
        """
        logger.info("Loading historical data from cache ...")
        data_4h, data_1d = self._load_cache()

        X_all: list[pd.DataFrame]  = []
        y_all: list[pd.Series]     = []
        y_short_all: list[pd.Series] = []

        for asset in data_4h:
            if asset not in data_1d:
                continue
            try:
                X = build_features(data_4h[asset], data_1d[asset], asset, self.params)
                if X.empty:
                    continue
                # Dual labels: long model trained on build_dual_labels, short on v5.1-style mss_signal
                y = build_dual_labels(data_4h[asset], X, {**self.params, '_asset': asset})
                y = y.reindex(X.index).fillna(0).astype(int)
                y_short = label_trades(
                    data_4h[asset].loc[X.index],
                    X["mss_signal"],
                    TBM_CONFIG, asset, self.params,
                )
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                X_all.append(X)
                y_all.append(y)
                y_short_all.append(y_short)
                logger.debug("  %-6s  rows=%-5d  long=%.1f%%  short=%.1f%%",
                             asset, len(X), (y == 1).mean() * 100, (y_short == -1).mean() * 100)
            except Exception as exc:
                logger.warning("Feature build failed for %s: %s", asset, exc)

        if not X_all:
            raise RuntimeError("No training data built — check cache path.")

        # Concatenate in order — do NOT sort_index (multi-asset = duplicate timestamps)
        X_train      = pd.concat(X_all,       ignore_index=True)
        y_train      = pd.concat(y_all,       ignore_index=True).fillna(0).astype(int)
        y_short_train = pd.concat(y_short_all, ignore_index=True).fillna(0).astype(int)

        # Last 10 % → validation for early stopping
        n_val  = max(50, len(X_train) // 10)
        X_val,      y_val      = X_train.iloc[-n_val:],       y_train.iloc[-n_val:]
        X_tr,       y_tr       = X_train.iloc[:-n_val],        y_train.iloc[:-n_val]
        y_short_val = y_short_train.iloc[-n_val:]
        y_short_tr  = y_short_train.iloc[:-n_val]

        self.model = VaranusDualModel(MODEL_CONFIG)
        self.model.fit(X_tr, y_tr, X_val, y_val, y_short_tr, y_short_val)

        logger.info(
            "Model trained | train=%d  val=%d  assets=%d",
            len(X_tr), len(X_val), len(data_4h),
        )

    # ── Cache loading ─────────────────────────────────────────────────────────

    def _load_cache(self) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        data_4h: dict[str, pd.DataFrame] = {}
        data_1d: dict[str, pd.DataFrame] = {}
        for asset in TIER2_UNIVERSE:
            try:
                df4 = self._read_parquet(asset, "4h")
                df1 = self._read_parquet(asset, "1d")
                if df4 is not None and df1 is not None:
                    data_4h[asset] = df4
                    data_1d[asset] = df1
            except Exception as exc:
                logger.warning("Cache load failed for %s: %s", asset, exc)
        logger.info("Loaded %d / %d assets from cache", len(data_4h), len(TIER2_UNIVERSE))
        return data_4h, data_1d

    def _read_parquet(self, asset: str, tf: str) -> Optional[pd.DataFrame]:
        file_sym = "ASTER" if asset == "ASTR" else asset
        if tf == "4h":
            path = CACHE_DIR / f"{file_sym}_USDT.parquet"
        else:
            path = CACHE_DIR / f"{file_sym}_USDT_1h.parquet"
            if not path.exists():
                path = CACHE_DIR / f"{file_sym}_USDT.parquet"

        if not path.exists():
            return None

        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        if tf == "1d":
            df = df.resample("1D").agg(
                {"open": "first", "high": "max",
                 "low": "min",    "close": "last", "volume": "sum"}
            ).dropna()

        return df

    # ── Live data fetching ────────────────────────────────────────────────────

    _BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

    # Max bars to keep in cache (4h bars) — ~3 years, enough for training + live
    _CACHE_MAX_4H = 6500
    _CACHE_MAX_1H = 26000  # 6500 × 4

    def _fetch_binance_1h(self, asset: str, since: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """
        Fetch closed 1h bars from Binance public REST API.

        If `since` is provided, fetches only bars newer than that timestamp
        (gap-fill mode). Otherwise fetches the latest 48 bars.
        Drops the still-forming bar (last entry).
        """
        symbol = f"{asset}USDT"
        params: dict = {"symbol": symbol, "interval": "1h"}

        if since is not None:
            # startTime = 1ms after last cached bar to avoid re-fetching it
            params["startTime"] = int(since.timestamp() * 1000) + 1
            params["limit"] = 1000   # max Binance allows
        else:
            params["limit"] = 49    # 48 closed + 1 forming

        try:
            r = requests.get(self._BINANCE_KLINES, params=params, timeout=15)
            if r.status_code != 200:
                logger.warning("Binance klines %s HTTP %d", symbol, r.status_code)
                return None
            raw = r.json()
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore",
            ])
            df["timestamp"] = pd.to_datetime(df["open_time"].astype(float), unit="ms", utc=True)
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)
            df = df.sort_index()
            # Drop the last (still-forming) bar
            df = df.iloc[:-1]
            return df if not df.empty else None
        except Exception as exc:
            logger.warning("Binance fetch failed %s: %s", asset, exc)
            return None

    @staticmethod
    def _safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
        """Write parquet atomically via a temp file to avoid corruption on crash."""
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)   # atomic on Linux

    def _refresh_cache(self) -> None:
        """
        Fetch only the gap (bars newer than last cached bar) from Binance public
        API for all assets and merge into the local parquet cache.

        - Fetches only missing bars (not a fixed 48 every time)
        - Writes via temp file → atomic rename (crash-safe)
        - Trims cache to _CACHE_MAX_4H / _CACHE_MAX_1H bars to prevent unbounded growth
        """
        logger.info("Refreshing cache from Binance public API ...")
        updated = 0

        for asset in TIER2_UNIVERSE:
            file_sym = "ASTER" if asset == "ASTR" else asset
            path_1h  = CACHE_DIR / f"{file_sym}_USDT_1h.parquet"
            path_4h  = CACHE_DIR / f"{file_sym}_USDT.parquet"

            # Determine last cached 1h bar to fetch only the gap
            since: Optional[pd.Timestamp] = None
            old_1h: Optional[pd.DataFrame] = None
            if path_1h.exists():
                try:
                    old_1h = pd.read_parquet(path_1h)
                    old_1h.index = pd.to_datetime(old_1h.index, utc=True)
                    since = old_1h.index[-1]
                except Exception:
                    old_1h = None

            df_new = self._fetch_binance_1h(asset, since=since)
            if df_new is None or df_new.empty:
                logger.debug("Cache refresh: no new bars for %s", asset)
                continue

            # ── Merge & trim 1h ───────────────────────────────────────────────
            if old_1h is not None:
                df_1h = pd.concat([old_1h, df_new])
                df_1h = df_1h[~df_1h.index.duplicated(keep="last")].sort_index()
            else:
                df_1h = df_new
            if len(df_1h) > self._CACHE_MAX_1H:
                df_1h = df_1h.iloc[-self._CACHE_MAX_1H:]
            self._safe_write_parquet(df_1h, path_1h)

            # ── Resample new bars to 4h, merge & trim ─────────────────────────
            df_new_4h = df_new.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna(subset=["close"])

            old_4h: Optional[pd.DataFrame] = None
            if path_4h.exists():
                try:
                    old_4h = pd.read_parquet(path_4h)
                    old_4h.index = pd.to_datetime(old_4h.index, utc=True)
                except Exception:
                    old_4h = None

            if old_4h is not None:
                df_4h = pd.concat([old_4h, df_new_4h])
                df_4h = df_4h[~df_4h.index.duplicated(keep="last")].sort_index()
            else:
                df_4h = df_new_4h
            if len(df_4h) > self._CACHE_MAX_4H:
                df_4h = df_4h.iloc[-self._CACHE_MAX_4H:]
            self._safe_write_parquet(df_4h, path_4h)

            updated += 1
            time.sleep(0.05)   # be polite to the public API

        logger.info("Cache refresh done — %d/%d assets updated.", updated, len(TIER2_UNIVERSE))

    def _fetch_live(self, asset: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch recent OHLCV bars from Binance (public API)."""
        symbol  = f"{asset}/USDT"
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            if not ohlcv:
                return None
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            # Drop the last (still-forming) bar
            now = pd.Timestamp.now(tz="UTC")
            df = df[df.index < now - pd.Timedelta(minutes=30)]
            return df if not df.empty else None
        except Exception as exc:
            logger.warning("Live fetch failed %s %s: %s", asset, tf, exc)
            return None

    def _get_live_data(self, asset: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Use local parquet cache as live data source."""
        df_4h = self._read_parquet(asset, "4h")
        df_1d = self._read_parquet(asset, "1d")
        # Trim to last LIVE_BARS_4H / LIVE_BARS_1D bars to match live behaviour
        if df_4h is not None and len(df_4h) > LIVE_BARS_4H:
            df_4h = df_4h.iloc[-LIVE_BARS_4H:]
        if df_1d is not None and len(df_1d) > LIVE_BARS_1D:
            df_1d = df_1d.iloc[-LIVE_BARS_1D:]
        return df_4h, df_1d

    # ── Signal scanning ───────────────────────────────────────────────────────

    def scan(self) -> list[dict]:
        """
        Scan all Tier 2 assets for entry signals on the current 4h close.

        Steps per asset:
          1. Fetch live 4h + 1d bars
          2. Build 19-feature vector
          3. model.predict() → direction {-1, 0, +1}
          4. Confidence gate (≥ best_params threshold)
          5. TBM barrier calculation (ATR-based TP/SL)
          6. Portfolio checks (leverage, concurrent positions)
          7. Open paper trade + send Telegram entry alert

        Returns list of opened trade dicts.
        """
        if self.model is None:
            raise RuntimeError("Call trainer.train() before scan().")
        if self.state.get("halted"):
            logger.info("Scan skipped — circuit breaker active.")
            return []

        open_trades       = self.state["open_trades"]
        conf_thresh_short = self.params.get("conf_thresh_short", self.params.get("confidence_thresh", 0.786))
        conf_thresh_long  = self.params.get("conf_thresh_long", 0.7728)
        candidates: list[dict] = []

        # ── 1. Collect candidates ──────────────────────────────────────────────
        for asset in TIER2_UNIVERSE:
            if asset in open_trades:
                continue   # Already in a position

            df_4h, df_1d = self._get_live_data(asset)
            if df_4h is None or df_1d is None or len(df_4h) < 120:
                continue

            try:
                X = build_features(df_4h, df_1d, asset, self.params)
                if X.empty:
                    continue

                probs     = self.model.predict_proba(X)
                preds     = self.model.predict(X)
                direction = int(preds[-1])

                if direction == 0:
                    continue

                # Direction-specific confidence gate
                if direction == -1:
                    confidence  = float(probs[-1, 0])   # p_short
                    conf_thresh = conf_thresh_short
                else:
                    confidence  = float(probs[-1, 2])   # p_long
                    conf_thresh = conf_thresh_long

                if confidence < conf_thresh:
                    continue

                latest_bar  = df_4h.iloc[-1]
                entry_price = float(latest_bar["close"])
                atr_val     = float(compute_atr(df_4h, 14).iloc[-1])

                if np.isnan(atr_val) or atr_val <= 0:
                    continue

                # TBM barriers — direction-specific (Short: v5.1 frozen, Long: v5.2 Long Runner)
                tbm_cfg = TBM_CONFIG.copy()
                if direction == -1:
                    # SHORT HUNTER LOCK-DOWN: frozen Trial #183 params
                    tbm_cfg["take_profit_atr"] = self.params.get("tp_atr_mult",  5.768)
                    tbm_cfg["stop_loss_atr"]   = self.params.get("sl_atr_mult",  0.709)
                else:
                    # Long Runner: optimized v5.2 Deep Search params
                    tbm_cfg["take_profit_atr"] = self.params.get("tp_mult_long", 2.616)
                    tbm_cfg["stop_loss_atr"]   = self.params.get("sl_mult_long", 1.000)
                tbm_cfg["max_holding_candles"] = self.params.get("max_holding", 30)

                barriers = calculate_barriers(entry_price, atr_val, direction, tbm_cfg, asset)
                if not barriers["min_rr_satisfied"]:
                    continue

                feat = X.iloc[-1]
                candidates.append({
                    "asset":       asset,
                    "direction":   direction,
                    "confidence":  confidence,
                    "entry_price": entry_price,
                    "atr":         atr_val,
                    "barriers":    barriers,
                    "entry_ts":    latest_bar.name,
                    "feat":        feat,
                })

            except Exception as exc:
                logger.warning("Scan error %s: %s", asset, exc)

        # ── 2. Sort by confidence, apply portfolio constraints ─────────────────
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        opened: list[dict] = []
        capital = self.state["equity"]

        for cand in candidates:
            if len(open_trades) >= RISK_CONFIG["max_concurrent_positions"]:
                break

            asset = cand["asset"]
            lev     = get_leverage(cand["confidence"])
            pos_usd = 100.0   # Fixed paper trade size: $100 per position

            mock_sig = {"confidence": cand["confidence"], "asset": asset}
            if would_breach_leverage(open_trades, capital, mock_sig, RISK_CONFIG):
                logger.debug("Skipped %s — portfolio leverage would breach cap", asset)
                continue

            # Compute portfolio leverage after adding this trade
            port_lev = compute_portfolio_leverage(
                {**open_trades, asset: {"position_usd": pos_usd}}, capital
            )

            entry_ts    = cand["entry_ts"]
            max_hold_ts = entry_ts + pd.Timedelta(hours=4 * self.params.get("max_holding", 30))

            trade = {
                "asset":        asset,
                "direction":    cand["direction"],
                "confidence":   round(cand["confidence"], 4),
                "leverage":     lev,
                "entry_price":  round(cand["entry_price"], 6),
                "take_profit":  round(cand["barriers"]["take_profit"], 6),
                "stop_loss":    round(cand["barriers"]["stop_loss"], 6),
                "rr_ratio":     cand["barriers"]["rr_ratio"],
                "atr_14":       round(cand["atr"], 6),
                "position_usd": round(pos_usd, 2),
                "entry_ts":     entry_ts.isoformat(),
                "max_hold_ts":  max_hold_ts.isoformat(),
                "trail_active": False,
                "trail_peak":   None,
                "trail_stop":   None,
            }

            # ── 3. Send Telegram entry alert with position size ────────────────
            feat      = cand["feat"]
            dir_label = "LONG ↑" if cand["direction"] == 1 else "SHORT ↓"
            mss_val   = float(feat.get("mss_signal", 0))
            htf_val   = float(feat.get("htf_bias",   0))
            mss_label = "↑ Bullish" if mss_val ==  1 else ("↓ Bearish" if mss_val == -1 else "→ Neutral")
            htf_label = "↑ Bullish" if htf_val ==  1 else ("↓ Bearish" if htf_val == -1 else "→ Neutral")

            alert_dict = {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "asset":         asset,
                "direction":     dir_label,
                "confidence":    cand["confidence"],
                "leverage":      lev,
                "entry_price":   round(cand["entry_price"], 6),
                "take_profit":   round(cand["barriers"]["take_profit"], 6),
                "stop_loss":     round(cand["barriers"]["stop_loss"], 6),
                "rr_ratio":      cand["barriers"]["rr_ratio"],
                "atr_14":        round(cand["atr"], 6),
                "mss":           mss_label,
                "fvg_valid":     "✓",
                "sweep_confirmed": "✓",
                "rvol":          round(float(feat.get("relative_volume", 1.0)), 2),
                "rsi":           round(float(feat.get("rsi_14", 50.0)), 1),
                "htf_bias":      htf_label,
                "position_usd":  round(pos_usd, 2),
                "port_lev":      round(port_lev, 2),
            }

            send_alert(alert_dict, self._bot_token, self._chat_id, dry_run=self.dry_run)

            # ── 4. Record paper trade ──────────────────────────────────────────
            open_trades[asset] = trade
            opened.append(trade)

            logger.info(
                "OPENED  %-6s  %s  entry=%.6f  TP=%.6f  SL=%.6f  "
                "size=$%.0f  lev=%.0fx  conf=%.1f%%  RR=%.1fx",
                asset, dir_label,
                cand["entry_price"],
                cand["barriers"]["take_profit"],
                cand["barriers"]["stop_loss"],
                pos_usd, lev,
                cand["confidence"] * 100,
                cand["barriers"]["rr_ratio"],
            )

        self.state["open_trades"] = open_trades
        self._write_state()
        return opened

    # ── Exit checking ─────────────────────────────────────────────────────────

    def check_exits(self) -> list[dict]:
        """
        For each open paper trade, fetch the latest 4h bars and check TP / SL / time.
        Closes trades, updates equity, and sends Telegram exit alerts.

        Returns list of closed trade dicts.
        """
        open_trades = self.state["open_trades"]
        closed:      list[dict] = []

        for asset, trade in list(open_trades.items()):
            df = self._read_parquet(asset, "4h")
            if df is not None and len(df) > 10:
                df = df.iloc[-10:]
            if df is None or df.empty:
                continue

            entry_ts      = pd.Timestamp(trade["entry_ts"]).tz_localize("UTC") \
                if pd.Timestamp(trade["entry_ts"]).tzinfo is None \
                else pd.Timestamp(trade["entry_ts"])
            bars_to_check = df[df.index > entry_ts]

            TRAIL_TRIGGER_PCT  = 0.01147
            TRAIL_DISTANCE_PCT = 0.01147

            outcome = None
            exit_ts = None
            for ts, bar in bars_to_check.iterrows():
                d  = trade["direction"]
                ep = trade["entry_price"]

                # ── Trailing stop logic (v5.7.1) ──────────────────────────
                if d == 1:  # LONG
                    if (bar["high"] - ep) / ep >= TRAIL_TRIGGER_PCT:
                        trade["trail_active"] = True
                    if trade.get("trail_active"):
                        peak = trade.get("trail_peak")
                        if peak is None or bar["high"] > peak:
                            trade["trail_peak"] = bar["high"]
                            trade["trail_stop"] = bar["high"] * (1 - TRAIL_DISTANCE_PCT)
                        if bar["close"] < trade["trail_stop"]:
                            outcome = {"type": "trailing_sl_hit", "price": trade["trail_stop"]}
                            exit_ts = ts
                            break
                elif d == -1:  # SHORT
                    if (ep - bar["low"]) / ep >= TRAIL_TRIGGER_PCT:
                        trade["trail_active"] = True
                    if trade.get("trail_active"):
                        peak = trade.get("trail_peak")
                        if peak is None or bar["low"] < peak:
                            trade["trail_peak"] = bar["low"]
                            trade["trail_stop"] = bar["low"] * (1 + TRAIL_DISTANCE_PCT)
                        if bar["close"] > trade["trail_stop"]:
                            outcome = {"type": "trailing_sl_hit", "price": trade["trail_stop"]}
                            exit_ts = ts
                            break

                if outcome:
                    break
                outcome = _check_barriers(bar, trade)
                if outcome:
                    exit_ts = ts
                    break

            if outcome is None:
                continue   # Trade still open

            exit_price = outcome["price"]

            # Net PnL (mirrors backtest._calculate_pnl)
            direction = trade["direction"]
            raw_ret   = direction * (exit_price - trade["entry_price"]) / trade["entry_price"]
            taker_exits = ("sl", "signal_decay", "mss_invalidation", "trailing_sl_hit")
            fee       = 0.0005 if outcome["type"] in taker_exits else 0.0002
            net_ret   = raw_ret - fee - 0.0008          # slippage
            pnl_usd   = trade["position_usd"] * net_ret

            self.state["equity"]      = round(self.state["equity"] + pnl_usd, 2)
            self.state["peak_equity"] = max(
                self.state["peak_equity"], self.state["equity"]
            )

            closed_trade = {
                **trade,
                "exit_ts":    exit_ts.isoformat(),
                "exit_price": round(exit_price, 6),
                "outcome":    outcome["type"],
                "pnl_usd":    round(pnl_usd, 2),
            }
            self.state["closed_trades"].append(closed_trade)
            del open_trades[asset]
            closed.append(closed_trade)

            send_exit_alert(
                closed_trade, self._bot_token, self._chat_id, dry_run=self.dry_run
            )

            sign = "+" if pnl_usd >= 0 else ""
            logger.info(
                "CLOSED  %-6s  %s  exit=%.6f  PnL=%s$%.2f  equity=$%.2f",
                asset,
                outcome["type"].upper(),
                exit_price,
                sign, abs(pnl_usd),
                self.state["equity"],
            )

        self.state["open_trades"] = open_trades
        self._write_state()
        self._append_csv(closed)
        return closed

    # ── Circuit breaker ───────────────────────────────────────────────────────

    def get_health(self) -> dict:
        """
        Compute portfolio health (daily loss + peak drawdown).
        Resets daily baseline at UTC midnight.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.get("daily_start_date") != today:
            self.state["daily_start_date"]   = today
            self.state["daily_start_equity"] = self.state["equity"]
            self._write_state()

        cur       = self.state["equity"]
        day_start = self.state.get("daily_start_equity", cur)
        peak      = self.state.get("peak_equity", cur)

        daily_pct = (cur - day_start) / day_start * 100 if day_start > 0 else 0.0
        dd_pct    = (cur - peak)      / peak       * 100 if peak > 0       else 0.0

        return {
            "current_equity": round(cur, 2),
            "daily_loss_pct": round(daily_pct, 2),
            "drawdown_pct":   round(dd_pct, 2),
            "halt_signals":   (daily_pct <= -5.0) or (dd_pct <= -15.0),
        }

    def _check_and_halt(self) -> bool:
        health = self.get_health()
        if health["halt_signals"] and not self.state.get("halted"):
            self.state["halted"] = True
            self._write_state()
            send_halt_alert(
                health, self._bot_token, self._chat_id, dry_run=self.dry_run
            )
            logger.warning(
                "CIRCUIT BREAKER TRIPPED | daily=%.1f%%  dd=%.1f%%  equity=$%.2f",
                health["daily_loss_pct"],
                health["drawdown_pct"],
                health["current_equity"],
            )
        return bool(self.state.get("halted", False))

    def reset_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self.state["halted"] = False
        self._write_state()
        logger.info("Circuit breaker reset.")

    # ── Full cycle ────────────────────────────────────────────────────────────

    def run_cycle(self) -> dict:
        """
        Full paper trading cycle — call every 4h at candle close.

        Order:
          1. check_exits()      — close any TP / SL / time-hit trades + alert
          2. _check_and_halt()  — evaluate circuit breaker
          3. scan()             — find new signals + open trades + alert

        Returns summary dict.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        logger.info("═══ Paper cycle  %s  equity=$%.2f ═══",
                    now_str, self.state["equity"])

        self._refresh_cache()
        closed = self.check_exits()
        halted = self._check_and_halt()

        if halted:
            logger.info("Scan skipped (halted).")
            return {"closed": closed, "opened": [], "halted": True}

        opened = self.scan()

        h = self.get_health()
        logger.info(
            "Cycle done | open=%d  closed=%d  new=%d  equity=$%.2f  "
            "daily=%.1f%%  dd=%.1f%%",
            len(self.state["open_trades"]),
            len(closed), len(opened),
            h["current_equity"],
            h["daily_loss_pct"], h["drawdown_pct"],
        )

        if not opened:
            send_no_signal_alert(
                now_str, h["current_equity"], h["daily_loss_pct"],
                self._bot_token, self._chat_id, dry_run=self.dry_run,
            )

        return {"closed": closed, "opened": opened, "halted": False}

    # ── CSV logging ───────────────────────────────────────────────────────────

    def _append_csv(self, closed: list[dict]) -> None:
        if not closed or self.dry_run:
            return
        os.makedirs(TRADES_CSV.parent, exist_ok=True)
        df     = pd.DataFrame(closed)
        header = not TRADES_CSV.exists()
        df.to_csv(TRADES_CSV, mode="a", index=False, header=header)

    # ── Telegram listener (heartbeat) ────────────────────────────────────────

    def _mins_to_next_cycle(self) -> int:
        """Return minutes until the next scheduled 4h cycle (at xx:05 UTC)."""
        from datetime import timedelta
        now         = datetime.now(timezone.utc)
        cycle_hours = [0, 4, 8, 12, 16, 20]
        candidates  = []
        for delta_days in (0, 1):
            base = now.replace(hour=0, minute=0, second=0, microsecond=0) \
                   + timedelta(days=delta_days)
            for h in cycle_hours:
                t = base + timedelta(hours=h, minutes=5)
                if t > now:
                    candidates.append(t)
        if not candidates:
            return 0
        return int((min(candidates) - now).total_seconds() / 60)

    def start_listener(self) -> None:
        """
        Start a background thread that polls Telegram for incoming messages.
        When the authorised user sends 'heartbeat', reply with current status.
        """
        thread = threading.Thread(target=self._poll_loop, daemon=True, name="tg-listener")
        thread.start()
        logger.info("Telegram listener started (heartbeat command active).")

    def _poll_loop(self) -> None:
        offset = 0
        url    = f"https://api.telegram.org/bot{self._bot_token}/getUpdates"
        while True:
            try:
                resp = requests.get(
                    url,
                    params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                    timeout=40,
                )
                data = resp.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg    = update.get("message", {})
                    text   = msg.get("text", "").strip().lower()
                    from_id = str(msg.get("chat", {}).get("id", ""))
                    if from_id != str(self._chat_id):
                        continue   # Ignore messages from unknown chats
                    if text in ("heartbeat", "/status", "status", "/start"):
                        logger.info("Status request received: '%s'", text)
                        health = self.get_health()
                        send_heartbeat_alert(
                            self.state, health,
                            self._bot_token, self._chat_id,
                            next_cycle_mins=self._mins_to_next_cycle(),
                        )
            except Exception as exc:
                logger.debug("Telegram poll error: %s", exc)
                time.sleep(5)

    # ── Status summary ────────────────────────────────────────────────────────

    def status(self) -> None:
        """Print a human-readable status summary."""
        h       = self.get_health()
        closed  = self.state.get("closed_trades", [])
        initial = self.state["initial_capital"]

        print("\n╔══════════ VARANUS PAPER TRADING STATUS ══════════╗")
        print(f"  Equity:        ${h['current_equity']:,.2f}  "
              f"(initial: ${initial:,.2f})")
        total_pnl = h["current_equity"] - initial
        sign = "+" if total_pnl >= 0 else ""
        print(f"  Total PnL:     {sign}${total_pnl:,.2f}  "
              f"({sign}{total_pnl / initial * 100:.1f}%)")
        print(f"  Daily P&L:     {h['daily_loss_pct']:+.1f}%")
        print(f"  Peak Drawdown: {h['drawdown_pct']:+.1f}%")
        print(f"  Halted:        {'YES 🚨' if self.state.get('halted') else 'No'}")
        print(f"  Open trades:   {len(self.state['open_trades'])}")
        print(f"  Closed trades: {len(closed)}")

        if self.state["open_trades"]:
            print("\n  ── Open Positions ──")
            for asset, t in self.state["open_trades"].items():
                d_label = "LONG ↑" if t["direction"] == 1 else "SHORT ↓"
                print(f"    {asset:<6} {d_label}  "
                      f"entry={t['entry_price']:.6f}  "
                      f"TP={t['take_profit']:.6f}  "
                      f"SL={t['stop_loss']:.6f}  "
                      f"size=${t['position_usd']:.0f}  "
                      f"lev={t['leverage']:.0f}x")

        if closed:
            wins  = sum(1 for t in closed if t.get("pnl_usd", 0) > 0)
            total = len(closed)
            tot_pnl = sum(t.get("pnl_usd", 0) for t in closed)
            print(f"\n  ── Closed Trade Summary ──")
            print(f"    Win rate:  {wins}/{total} ({wins/total*100:.0f}%)")
            print(f"    Total PnL: ${tot_pnl:+,.2f}")

        print("╚══════════════════════════════════════════════════╝\n")
