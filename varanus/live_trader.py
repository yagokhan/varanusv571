"""
varanus/live_trader.py — Varanus Tier 2 Live Trading Engine (Binance USDT-M Futures)

Mirrors PaperTrader exactly for signal generation, exit logic, and
walk-forward maintenance. Replaces simulated execution with real
Binance Futures orders via BinanceFuturesClient.

Lifecycle per 4h candle close:
  1. _verify_all_positions() — reconcile Binance vs local state
  2. check_exits()           — evaluate exits, close positions on exchange
  3. _check_and_halt()       — circuit breaker (tighter limits than paper)
  4. maintenance_check()     — walk-forward maintenance monitor
  5. scan()                  — generate signals, place real market orders
  6. save_state()            — persist to config/live_state.json
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

import numpy as np
import pandas as pd

from varanus.universe import TIER2_UNIVERSE, HIGH_VOL_SUBTIER
from varanus.pa_features import build_features, compute_atr, detect_mss
from varanus.tbm_labeler import label_trades, calculate_barriers, TBM_CONFIG, build_dual_labels
from varanus.model import VaranusDualModel, get_leverage, MODEL_CONFIG
from varanus.risk import (
    RISK_CONFIG,
    get_position_size,
    would_breach_leverage,
    compute_portfolio_leverage,
)
from varanus.alerts import (send_alert, send_exit_alert, send_halt_alert,
                            send_no_signal_alert, send_heartbeat_alert,
                            send_maintenance_alert)
from varanus.binance_client import BinanceFuturesClient

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
CACHE_DIR   = _HERE / "data" / "cache"
STATE_FILE  = _HERE / "config" / "live_state.json"
TRADES_CSV  = _HERE / "results" / "live_trades.csv"
ENV_FILE    = _HERE / "config" / "telegram.env"
BINANCE_ENV = _HERE / "config" / "binance.env"

# ── Live data lookback ─────────────────────────────────────────────────────────
LIVE_BARS_4H = 400
LIVE_BARS_1D = 150

# ── Live-specific risk overrides (more conservative than paper) ────────────────
LIVE_RISK_CONFIG = {
    **RISK_CONFIG,
    "max_concurrent_positions": 3,
    "max_portfolio_leverage":   2.0,
    "daily_loss_limit_pct":     0.03,      # 3% daily loss → halt
    "portfolio_stop_pct":       0.10,      # 10% drawdown → halt
}

# Safety constants
MAX_DAILY_TRADES          = 6        # circuit break if >6 trades in 24h
EMERGENCY_EQUITY_DROP_PCT = 0.08     # emergency close all if -8% from session start


# ══════════════════════════════════════════════════════════════════════════════
# Barrier helper (identical to paper_trader._check_barriers)
# ══════════════════════════════════════════════════════════════════════════════

def _check_barriers(bar: pd.Series, trade: dict) -> Optional[dict]:
    d  = trade["direction"]
    ts = bar.name

    max_hold = pd.Timestamp(trade["max_hold_ts"]).tz_localize("UTC") \
        if pd.Timestamp(trade["max_hold_ts"]).tzinfo is None \
        else pd.Timestamp(trade["max_hold_ts"])
    if ts >= max_hold:
        return {"type": "time", "price": float(bar["close"])}

    tp = trade["take_profit"]
    sl = trade["stop_loss"]

    if d == 1  and bar["high"] >= tp:
        return {"type": "tp", "price": tp}
    if d == -1 and bar["low"]  <= tp:
        return {"type": "tp", "price": tp}

    if d == 1  and bar["close"] < sl:
        return {"type": "sl", "price": sl}
    if d == -1 and bar["close"] > sl:
        return {"type": "sl", "price": sl}

    return None


# ══════════════════════════════════════════════════════════════════════════════
# LiveTrader
# ══════════════════════════════════════════════════════════════════════════════

class LiveTrader:
    """
    Varanus Tier 2 Live Trading Engine — Binance USDT-M Futures.

    Usage
    -----
    trader = LiveTrader(initial_capital=1000.0, testnet=True)
    trader.train()
    trader.run_cycle()
    """

    # Hunter active management constants — identical to PaperTrader
    _TRAIL_TRIGGER_PCT     = 0.01147
    _TRAIL_DISTANCE_PCT    = 0.01147
    _SIGNAL_DECAY_THRESH   = 0.35
    _BREAKEVEN_TRIGGER_PCT = 0.75
    _BREAKEVEN_BUFFER_ATR  = 0.05

    # Walk-forward maintenance constants — identical to PaperTrader
    _RETRAIN_CYCLE_DAYS     = 120
    _ATH_STAGNATION_DAYS    = 30
    _FOLD_DEPTH             = 8
    _BASELINE_TRAINING_DATE = "2026-02-28"

    _CACHE_MAX_4H = 6500
    _CACHE_MAX_1H = 26000

    # ── Init ────────────────────────────────────────────────────────────────

    def __init__(
        self,
        initial_capital: float = 1_000.0,
        dry_run: bool = False,
        testnet: bool = False,
        state_file: Path = STATE_FILE,
    ):
        self.dry_run    = dry_run
        self.testnet    = testnet
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
        self._bot_token, self._chat_id = self._load_telegram_credentials()

        # Binance credentials
        api_key, api_secret = self._load_binance_credentials()
        self.client = BinanceFuturesClient(api_key, api_secret, testnet=testnet)

        # Verify connectivity
        if not self.client.ping():
            raise RuntimeError("Cannot reach Binance API — check network and credentials.")

        # Fetch real balance for verification
        bal = self.client.get_balance()
        logger.info("Binance balance: total=$%.2f  free=$%.2f  used=$%.2f",
                     bal["total"], bal["free"], bal["used"])

        # Persistent state
        self.state = self._load_state(initial_capital)

        # Safety state
        self._kill_switch       = False
        self._daily_trade_count = 0
        self._session_start_equity: Optional[float] = None

        # Exchange ref for _fetch_live (ccxt instance)
        self.exchange = self.client.exchange

        logger.info(
            "LiveTrader ready | capital=$%.2f | open_trades=%d | "
            "dry_run=%s | testnet=%s",
            self.state["equity"],
            len(self.state["open_trades"]),
            dry_run, testnet,
        )

    # ── Credentials ────────────────────────────────────────────────────────

    def _load_telegram_credentials(self) -> tuple[str, str]:
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

    def _load_binance_credentials(self) -> tuple[str, str]:
        env: dict[str, str] = {}
        if BINANCE_ENV.exists():
            with open(BINANCE_ENV) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        env[k.strip()] = v.strip()
        api_key    = env.get("BINANCE_API_KEY",    os.environ.get("BINANCE_API_KEY", ""))
        api_secret = env.get("BINANCE_API_SECRET", os.environ.get("BINANCE_API_SECRET", ""))
        if not api_key or not api_secret:
            raise RuntimeError(
                "Binance credentials not found. Set BINANCE_API_KEY and "
                "BINANCE_API_SECRET in varanus/config/binance.env or env vars."
            )
        return api_key, api_secret

    # ── State persistence ──────────────────────────────────────────────────

    def _load_state(self, initial_capital: float) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        default = {
            "initial_capital":     initial_capital,
            "equity":              initial_capital,
            "peak_equity":         initial_capital,
            "daily_start_equity":  initial_capital,
            "daily_start_date":    datetime.now(timezone.utc).date().isoformat(),
            "open_trades":         {},
            "closed_trades":       [],
            "halted":              False,
            "last_training_date":  self._BASELINE_TRAINING_DATE,
            "peak_equity_date":    datetime.now(timezone.utc).date().isoformat(),
            "maintenance_required": False,
            "performance_drift":   False,
            "fold_registry": [
                {"fold": 1, "start": "2023-01-01", "end": "2023-05-13"},
                {"fold": 2, "start": "2023-05-14", "end": "2023-09-23"},
                {"fold": 3, "start": "2023-09-24", "end": "2024-02-04"},
                {"fold": 4, "start": "2024-02-05", "end": "2024-06-17"},
                {"fold": 5, "start": "2024-06-18", "end": "2024-10-28"},
                {"fold": 6, "start": "2024-10-29", "end": "2025-03-11"},
                {"fold": 7, "start": "2025-03-12", "end": "2025-07-22"},
                {"fold": 8, "start": "2025-07-23", "end": "2025-10-31"},
            ],
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

    # ── Model training (identical to PaperTrader) ──────────────────────────

    def train(self) -> None:
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
            except Exception as exc:
                logger.warning("Feature build failed for %s: %s", asset, exc)

        if not X_all:
            raise RuntimeError("No training data built — check cache path.")

        X_train       = pd.concat(X_all,       ignore_index=True)
        y_train       = pd.concat(y_all,       ignore_index=True).fillna(0).astype(int)
        y_short_train = pd.concat(y_short_all, ignore_index=True).fillna(0).astype(int)

        n_val       = max(50, len(X_train) // 10)
        X_val       = X_train.iloc[-n_val:]
        X_tr        = X_train.iloc[:-n_val]
        y_val       = y_train.iloc[-n_val:]
        y_tr        = y_train.iloc[:-n_val]
        y_short_val = y_short_train.iloc[-n_val:]
        y_short_tr  = y_short_train.iloc[:-n_val]

        self.model = VaranusDualModel(MODEL_CONFIG)
        self.model.fit(X_tr, y_tr, X_val, y_val, y_short_tr, y_short_val)

        self.state["last_training_date"] = datetime.now(timezone.utc).date().isoformat()
        self.state["maintenance_required"] = False
        self._write_state()

        logger.info(
            "Model trained | train=%d  val=%d  assets=%d",
            len(X_tr), len(X_val), len(data_4h),
        )

    # ── Cache loading (identical to PaperTrader) ───────────────────────────

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
                 "low": "min", "close": "last", "volume": "sum"}
            ).dropna()

        return df

    # ── Live data fetching (identical to PaperTrader) ──────────────────────

    _BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
    _BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price"

    def fetch_spot_prices(self, assets: list[str]) -> dict[str, float]:
        prices = {}
        for asset in assets:
            sym = f"{asset}USDT"
            try:
                r = requests.get(self._BINANCE_TICKER, params={"symbol": sym}, timeout=5)
                if r.status_code == 200:
                    prices[asset] = float(r.json()["price"])
            except Exception:
                pass
        return prices

    def _fetch_binance_1h(self, asset: str, since: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        symbol = f"{asset}USDT"
        params: dict = {"symbol": symbol, "interval": "1h"}
        if since is not None:
            params["startTime"] = int(since.timestamp() * 1000) + 1
            params["limit"] = 1000
        else:
            params["limit"] = 49

        try:
            r = requests.get(self._BINANCE_KLINES, params=params, timeout=15)
            if r.status_code != 200:
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
            df = df.iloc[:-1]
            return df if not df.empty else None
        except Exception as exc:
            logger.warning("Binance fetch failed %s: %s", asset, exc)
            return None

    @staticmethod
    def _safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(path)

    def _refresh_cache(self) -> None:
        logger.info("Refreshing cache from Binance public API ...")
        updated = 0

        for asset in TIER2_UNIVERSE:
            file_sym = "ASTER" if asset == "ASTR" else asset
            path_1h  = CACHE_DIR / f"{file_sym}_USDT_1h.parquet"
            path_4h  = CACHE_DIR / f"{file_sym}_USDT.parquet"

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
                continue

            if old_1h is not None:
                df_1h = pd.concat([old_1h, df_new])
                df_1h = df_1h[~df_1h.index.duplicated(keep="last")].sort_index()
            else:
                df_1h = df_new
            if len(df_1h) > self._CACHE_MAX_1H:
                df_1h = df_1h.iloc[-self._CACHE_MAX_1H:]
            self._safe_write_parquet(df_1h, path_1h)

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
            time.sleep(0.05)

        self.state["last_cache_refresh"] = datetime.now(timezone.utc).isoformat()
        self.state["last_cache_ok"] = updated >= len(TIER2_UNIVERSE) - 1
        logger.info("Cache refresh done — %d/%d assets updated.", updated, len(TIER2_UNIVERSE))

    def _get_live_data(self, asset: str) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        df_4h = self._read_parquet(asset, "4h")
        df_1d = self._read_parquet(asset, "1d")
        return df_4h, df_1d

    # ══════════════════════════════════════════════════════════════════════
    # Position verification — LIVE-SPECIFIC
    # ══════════════════════════════════════════════════════════════════════

    def _verify_position_closed(self, asset: str) -> bool:
        """Confirm no residual position on Binance after a close order."""
        pos = self.client.get_position(asset)
        if pos is None:
            return True
        # Still open — attempt one more market close
        contracts = abs(float(pos.get("contracts", 0)))
        side_str = pos.get("side", "")
        close_side = "sell" if side_str == "long" else "buy"
        logger.critical(
            "RESIDUAL POSITION detected for %s (%.8f contracts) — forcing close",
            asset, contracts,
        )
        try:
            self.client.place_market_order(asset, close_side, contracts, reduce_only=True)
        except Exception as exc:
            logger.critical("FAILED to close residual %s: %s", asset, exc)
            return False
        return True

    def _verify_all_positions(self) -> None:
        """
        Reconcile Binance positions against local state.
        Detects orphaned positions and ghost trades.
        """
        open_trades = self.state["open_trades"]

        # Check each tracked trade still exists on Binance
        for asset in list(open_trades.keys()):
            pos = self.client.get_position(asset)
            if pos is None:
                # Position gone — server-side SL/liquidation triggered
                trade = open_trades.pop(asset)
                logger.critical(
                    "POSITION GONE on Binance for %s — likely server-side SL or liquidation",
                    asset,
                )
                # Record as emergency close with approximate PnL
                trade["exit_ts"]    = datetime.now(timezone.utc).isoformat()
                trade["exit_price"] = trade.get("stop_loss", trade["entry_price"])
                trade["outcome"]    = "server_side_close"
                d = trade["direction"]
                raw_ret = d * (trade["exit_price"] - trade["entry_price"]) / trade["entry_price"]
                trade["pnl_usd"] = round(trade["position_usd"] * (raw_ret - 0.0013), 2)
                self.state["equity"] = round(self.state["equity"] + trade["pnl_usd"], 2)
                self.state["closed_trades"].append(trade)
                self._send_live_alert(
                    f"⚠️ *LIVE — SERVER-SIDE CLOSE* | {asset}\n"
                    f"Position closed externally (SL/liquidation).\n"
                    f"Approx PnL: ${trade['pnl_usd']:+.2f}"
                )

        self._write_state()

    def _emergency_close_all(self, reason: str) -> None:
        """Close ALL open positions immediately and activate kill switch."""
        logger.critical("EMERGENCY CLOSE ALL — reason: %s", reason)
        self._kill_switch = True

        for asset, trade in list(self.state["open_trades"].items()):
            try:
                self.client.cancel_all_orders(asset)
                close_side = "sell" if trade["direction"] == 1 else "buy"
                qty = trade.get("fill_qty")
                if qty is None:
                    price = self.client.get_ticker(asset)
                    qty = self.client.compute_quantity(asset, trade["position_usd"], price)
                order = self.client.place_market_order(asset, close_side, qty, reduce_only=True)

                exit_price = float(order.get("average", trade["entry_price"]))
                d = trade["direction"]
                raw_ret = d * (exit_price - trade["entry_price"]) / trade["entry_price"]
                pnl_usd = round(trade["position_usd"] * (raw_ret - 0.0013), 2)

                trade["exit_ts"]    = datetime.now(timezone.utc).isoformat()
                trade["exit_price"] = exit_price
                trade["outcome"]    = f"emergency_close_{reason[:20]}"
                trade["pnl_usd"]    = pnl_usd
                self.state["equity"] = round(self.state["equity"] + pnl_usd, 2)
                self.state["closed_trades"].append(trade)
            except Exception as exc:
                logger.critical("Emergency close FAILED for %s: %s", asset, exc)

        self.state["open_trades"] = {}
        self.state["halted"] = True
        self._write_state()

        self._send_live_alert(
            f"🚨 *LIVE — EMERGENCY CLOSE ALL*\n"
            f"Reason: {reason}\n"
            f"All positions closed. Kill switch ACTIVE.\n"
            f"Equity: ${self.state['equity']:,.2f}"
        )

    def _send_live_alert(self, msg: str) -> None:
        """Send a raw message to Telegram (live-specific alerts)."""
        if self.dry_run:
            print(f"[dry-run] {msg}")
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._bot_token}/sendMessage",
                json={"chat_id": self._chat_id, "text": msg, "parse_mode": "Markdown"},
                timeout=5,
            )
        except Exception as exc:
            logger.warning("Telegram alert failed: %s", exc)

    # ══════════════════════════════════════════════════════════════════════
    # Signal scanning — LIVE (real order placement)
    # ══════════════════════════════════════════════════════════════════════

    def scan(self) -> list[dict]:
        if self.model is None:
            raise RuntimeError("Call train() before scan().")
        if self.state.get("halted"):
            logger.info("Scan skipped — circuit breaker active.")
            return []
        if self._kill_switch:
            logger.critical("Scan skipped — KILL SWITCH active.")
            return []

        open_trades       = self.state["open_trades"]
        conf_thresh_short = self.params.get("conf_thresh_short", self.params.get("confidence_thresh", 0.786))
        conf_thresh_long  = self.params.get("conf_thresh_long", 0.7728)
        p_short_max       = self.params.get("p_short_max_for_long", 1.0)
        candidates: list[dict] = []

        # ── 1. Collect candidates (identical to PaperTrader) ─────────────────
        for asset in TIER2_UNIVERSE:
            if asset in open_trades:
                continue

            df_4h, df_1d = self._get_live_data(asset)
            if df_4h is None or df_1d is None or len(df_4h) < 120:
                continue

            try:
                X = build_features(df_4h, df_1d, asset, self.params)
                if X.empty:
                    continue

                probs = self.model.predict_proba(X)
                p_long_val  = float(probs[-1, 2])
                p_short_val = float(probs[-1, 0])

                direction  = 0
                confidence = 0.0

                if p_long_val >= conf_thresh_long and p_short_val <= p_short_max:
                    direction  = 1
                    confidence = p_long_val
                if p_short_val >= conf_thresh_short and p_short_val >= p_long_val:
                    direction  = -1
                    confidence = p_short_val

                if direction == 0:
                    continue

                latest_bar  = df_4h.iloc[-1]
                entry_price = float(latest_bar["close"])
                atr_val     = float(compute_atr(df_4h, 14).iloc[-1])

                if np.isnan(atr_val) or atr_val <= 0:
                    continue

                tbm_cfg = TBM_CONFIG.copy()
                if direction == -1:
                    tbm_cfg["take_profit_atr"] = self.params.get("tp_atr_mult",  5.768)
                    tbm_cfg["stop_loss_atr"]   = self.params.get("sl_atr_mult",  0.709)
                else:
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

        # ── 2. Sort by confidence, apply portfolio constraints ───────────────
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        opened: list[dict] = []
        capital = self.state["equity"]

        for cand in candidates:
            if len(open_trades) >= LIVE_RISK_CONFIG["max_concurrent_positions"]:
                break
            if self._daily_trade_count >= MAX_DAILY_TRADES:
                logger.warning("Daily trade limit reached (%d) — stopping scan", MAX_DAILY_TRADES)
                break

            asset = cand["asset"]
            lev   = get_leverage(cand["confidence"])
            # Position size: 10% of equity as margin, notional = margin for now
            # Using fixed $100 notional like paper trader for initial phase
            pos_usd = 100.0

            mock_sig = {"confidence": cand["confidence"], "asset": asset}
            if would_breach_leverage(open_trades, capital, mock_sig, LIVE_RISK_CONFIG):
                logger.debug("Skipped %s — portfolio leverage would breach cap", asset)
                continue

            port_lev = compute_portfolio_leverage(
                {**open_trades, asset: {"position_usd": pos_usd}}, capital
            )

            entry_ts    = cand["entry_ts"]
            max_hold_ts = entry_ts + pd.Timedelta(hours=4 * self.params.get("max_holding", 30))

            # ── 3. LIVE: Place real order on Binance ─────────────────────────
            try:
                # Set isolated margin and leverage
                if not self.client.set_margin_mode(asset, "isolated"):
                    logger.error("Failed to set isolated margin for %s — skipping", asset)
                    continue
                if not self.client.set_leverage(asset, int(lev)):
                    logger.error("Failed to set leverage for %s — skipping", asset)
                    continue

                # Get current price and compute quantity
                current_price = self.client.get_ticker(asset)
                qty = self.client.compute_quantity(asset, pos_usd, current_price)

                # Place market entry order
                side = "buy" if cand["direction"] == 1 else "sell"
                order = self.client.place_market_order(asset, side, qty)

                if order.get("status") not in ("closed", "filled"):
                    logger.error("Order not filled for %s — status: %s", asset, order.get("status"))
                    continue

                actual_fill_price = float(order.get("average", current_price))
                actual_qty        = float(order.get("filled", qty))

                # Place server-side STOP_MARKET as safety net
                sl_side = "sell" if cand["direction"] == 1 else "buy"
                sl_order = None
                try:
                    sl_order = self.client.place_stop_market(
                        asset, sl_side, actual_qty, cand["barriers"]["stop_loss"]
                    )
                except Exception as exc:
                    logger.warning("Server-side SL failed for %s: %s (continuing)", asset, exc)

            except Exception as exc:
                logger.error("ORDER FAILED for %s: %s", asset, exc)
                continue

            # ── 4. Record trade with actual fill data ────────────────────────
            trade = {
                "asset":               asset,
                "direction":           cand["direction"],
                "confidence":          round(cand["confidence"], 4),
                "entry_confidence":    round(cand["confidence"], 4),
                "leverage":            lev,
                "entry_price":         round(actual_fill_price, 6),
                "take_profit":         round(cand["barriers"]["take_profit"], 6),
                "stop_loss":           round(cand["barriers"]["stop_loss"], 6),
                "rr_ratio":            cand["barriers"]["rr_ratio"],
                "atr_14":              round(cand["atr"], 6),
                "position_usd":        round(pos_usd, 2),
                "entry_ts":            entry_ts.isoformat(),
                "max_hold_ts":         max_hold_ts.isoformat(),
                "trail_active":        False,
                "trail_peak":          None,
                "trail_stop":          None,
                "breakeven_activated": False,
                # Live-specific fields
                "fill_price":          actual_fill_price,
                "fill_qty":            actual_qty,
                "order_id":            order.get("id"),
                "sl_order_id":         sl_order.get("id") if sl_order else None,
            }

            # ── 5. Send Telegram entry alert ─────────────────────────────────
            feat      = cand["feat"]
            dir_label = "LONG ↑" if cand["direction"] == 1 else "SHORT ↓"
            mss_val   = float(feat.get("mss_signal", 0))
            htf_val   = float(feat.get("htf_bias",   0))
            mss_label = "↑ Bullish" if mss_val == 1 else ("↓ Bearish" if mss_val == -1 else "→ Neutral")
            htf_label = "↑ Bullish" if htf_val == 1 else ("↓ Bearish" if htf_val == -1 else "→ Neutral")

            alert_dict = {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "asset":         asset,
                "direction":     f"🔴 LIVE {dir_label}",
                "confidence":    cand["confidence"],
                "leverage":      lev,
                "entry_price":   round(actual_fill_price, 6),
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

            open_trades[asset] = trade
            opened.append(trade)
            self._daily_trade_count += 1

            logger.info(
                "LIVE OPENED  %-6s  %s  fill=%.6f  TP=%.6f  SL=%.6f  "
                "size=$%.0f  lev=%.0fx  conf=%.1f%%  qty=%.8f  order=%s",
                asset, dir_label, actual_fill_price,
                cand["barriers"]["take_profit"],
                cand["barriers"]["stop_loss"],
                pos_usd, lev, cand["confidence"] * 100,
                actual_qty, order.get("id"),
            )

        self.state["open_trades"] = open_trades
        self._write_state()
        return opened

    # ══════════════════════════════════════════════════════════════════════
    # Exit checking — LIVE (real order closure)
    # ══════════════════════════════════════════════════════════════════════

    def check_exits(self) -> list[dict]:
        open_trades = self.state["open_trades"]
        closed: list[dict] = []
        mss_lookback = self.params.get("mss_lookback", 40)

        for asset, trade in list(open_trades.items()):
            df    = self._read_parquet(asset, "4h")
            df_1d = self._read_parquet(asset, "1d")
            if df is None or df.empty:
                continue

            entry_ts = pd.Timestamp(trade["entry_ts"])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            bars_to_check = df[df.index > entry_ts]
            if bars_to_check.empty:
                open_trades[asset] = trade
                continue

            # Precompute (identical to PaperTrader)
            mss_series = detect_mss(df, mss_lookback)

            proba_dir:   Optional[pd.Series] = None
            proba_long:  Optional[pd.Series] = None
            proba_short: Optional[pd.Series] = None
            if self.model is not None and df_1d is not None:
                try:
                    X = build_features(df, df_1d, asset, self.params)
                    if not X.empty:
                        probs       = self.model.predict_proba(X)
                        preds       = self.model.predict(X)
                        proba_dir   = pd.Series(preds,       index=X.index, dtype=int)
                        proba_long  = pd.Series(probs[:, 2], index=X.index)
                        proba_short = pd.Series(probs[:, 0], index=X.index)
                except Exception as exc:
                    logger.debug("Signal decay data build failed %s: %s", asset, exc)

            atr_series = compute_atr(df, 14)

            # Reset trailing state for bar replay
            trade["trail_active"] = False
            trade["trail_peak"]   = None
            trade["trail_stop"]   = None

            outcome = None
            exit_ts = None

            for ts, bar in bars_to_check.iterrows():
                d  = trade["direction"]
                ep = trade["entry_price"]

                # 1. Trailing Stop
                if d == 1:
                    if (bar["high"] - ep) / ep >= self._TRAIL_TRIGGER_PCT:
                        trade["trail_active"] = True
                    if trade.get("trail_active"):
                        peak = trade.get("trail_peak")
                        if peak is None or bar["high"] > peak:
                            trade["trail_peak"] = bar["high"]
                            trade["trail_stop"] = bar["high"] * (1 - self._TRAIL_DISTANCE_PCT)
                        if bar["close"] < trade["trail_stop"]:
                            outcome = {"type": "trailing_sl_hit", "price": trade["trail_stop"]}
                            exit_ts = ts
                            break
                elif d == -1:
                    if (ep - bar["low"]) / ep >= self._TRAIL_TRIGGER_PCT:
                        trade["trail_active"] = True
                    if trade.get("trail_active"):
                        peak = trade.get("trail_peak")
                        if peak is None or bar["low"] < peak:
                            trade["trail_peak"] = bar["low"]
                            trade["trail_stop"] = bar["low"] * (1 + self._TRAIL_DISTANCE_PCT)
                        if bar["close"] > trade["trail_stop"]:
                            outcome = {"type": "trailing_sl_hit", "price": trade["trail_stop"]}
                            exit_ts = ts
                            break

                # 2. MSS Invalidation
                current_mss = int(mss_series.get(ts, 0))
                if d == 1 and current_mss == -1:
                    outcome = {"type": "mss_invalidation", "price": float(bar["open"])}
                    exit_ts = ts
                    break
                if d == -1 and current_mss == 1:
                    outcome = {"type": "mss_invalidation", "price": float(bar["open"])}
                    exit_ts = ts
                    break

                # 3. Signal Decay
                if proba_dir is not None and ts in proba_dir.index:
                    if int(proba_dir[ts]) == d:
                        current_conf = (
                            float(proba_long[ts])  if d == 1 else
                            float(proba_short[ts])
                        )
                        entry_conf = trade.get("entry_confidence", trade["confidence"])
                        if entry_conf - current_conf >= self._SIGNAL_DECAY_THRESH:
                            outcome = {"type": "signal_decay", "price": float(bar["open"])}
                            exit_ts = ts
                            break

                # 4. Dynamic Breakeven
                current_atr = float(atr_series.get(ts, 0) or 0)
                if current_atr > 0 and not trade.get("breakeven_activated"):
                    buf = self._BREAKEVEN_BUFFER_ATR * current_atr
                    if d == 1:
                        target_dist = trade["take_profit"] - ep
                        if bar["high"] >= ep + self._BREAKEVEN_TRIGGER_PCT * target_dist:
                            new_sl = ep + buf
                            if new_sl > trade["stop_loss"]:
                                trade["stop_loss"]           = new_sl
                                trade["breakeven_activated"] = True
                                logger.info("BREAKEVEN  %-6s  new_sl=%.6f", asset, new_sl)
                                # Update server-side SL
                                self._update_server_sl(asset, trade)
                    elif d == -1:
                        target_dist = ep - trade["take_profit"]
                        if bar["low"] <= ep - self._BREAKEVEN_TRIGGER_PCT * target_dist:
                            new_sl = ep - buf
                            if new_sl < trade["stop_loss"]:
                                trade["stop_loss"]           = new_sl
                                trade["breakeven_activated"] = True
                                logger.info("BREAKEVEN  %-6s  new_sl=%.6f", asset, new_sl)
                                self._update_server_sl(asset, trade)

                # 5. Standard Barriers
                outcome = _check_barriers(bar, trade)
                if outcome:
                    exit_ts = ts
                    break

            if outcome is None:
                open_trades[asset] = trade
                continue

            # ── LIVE: Close position on Binance ──────────────────────────────
            actual_exit_price = outcome["price"]
            try:
                self.client.cancel_all_orders(asset)
                close_side = "sell" if trade["direction"] == 1 else "buy"
                qty = trade.get("fill_qty")
                if qty is None:
                    qty = self.client.compute_quantity(
                        asset, trade["position_usd"], actual_exit_price
                    )
                order = self.client.place_market_order(asset, close_side, qty, reduce_only=True)
                actual_exit_price = float(order.get("average", actual_exit_price))

                # Verify closed
                self._verify_position_closed(asset)

            except Exception as exc:
                logger.critical("EXIT ORDER FAILED for %s: %s", asset, exc)
                # Keep trade open — will retry next cycle
                open_trades[asset] = trade
                continue

            # Net PnL — using actual fill prices (no simulated fees)
            direction = trade["direction"]
            raw_ret   = direction * (actual_exit_price - trade["entry_price"]) / trade["entry_price"]
            # Real fees are already deducted by the exchange; approximate for equity tracking
            taker_exits = ("sl", "signal_decay", "mss_invalidation", "trailing_sl_hit")
            fee       = 0.0005 if outcome["type"] in taker_exits else 0.0002
            net_ret   = raw_ret - fee - 0.0003   # smaller slippage estimate for live (actual fill used)
            pnl_usd   = trade["position_usd"] * net_ret

            self.state["equity"]      = round(self.state["equity"] + pnl_usd, 2)
            self.state["peak_equity"] = max(self.state["peak_equity"], self.state["equity"])

            closed_trade = {
                **trade,
                "exit_ts":           exit_ts.isoformat(),
                "exit_price":        round(actual_exit_price, 6),
                "outcome":           outcome["type"],
                "pnl_usd":           round(pnl_usd, 2),
                "exit_fill_price":   actual_exit_price,
            }
            self.state["closed_trades"].append(closed_trade)
            del open_trades[asset]
            closed.append(closed_trade)

            send_exit_alert(
                closed_trade, self._bot_token, self._chat_id, dry_run=self.dry_run
            )

            sign = "+" if pnl_usd >= 0 else ""
            pnl_pct = pnl_usd / trade.get("position_usd", 1) * 100
            logger.info(
                "LIVE CLOSED  %-6s  %s  exit=%.6f  PnL=%s$%.2f (%s%.2f%%)  equity=$%.2f",
                asset, outcome["type"].upper(), actual_exit_price,
                sign, abs(pnl_usd), sign, abs(pnl_pct), self.state["equity"],
            )

        self.state["open_trades"] = open_trades
        self._write_state()
        self._append_csv(closed)
        return closed

    def _update_server_sl(self, asset: str, trade: dict) -> None:
        """Update the server-side STOP_MARKET after breakeven/trail adjustment."""
        try:
            self.client.cancel_all_orders(asset)
            sl_side = "sell" if trade["direction"] == 1 else "buy"
            qty = trade.get("fill_qty", 0)
            if qty > 0:
                sl_order = self.client.place_stop_market(
                    asset, sl_side, qty, trade["stop_loss"]
                )
                trade["sl_order_id"] = sl_order.get("id")
        except Exception as exc:
            logger.warning("Failed to update server-side SL for %s: %s", asset, exc)

    # ── Circuit breaker (tighter limits for live) ──────────────────────────

    def get_health(self) -> dict:
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
            "halt_signals":   (daily_pct <= -3.0) or (dd_pct <= -10.0),  # tighter
        }

    def _check_and_halt(self) -> bool:
        health = self.get_health()

        # Emergency equity drop check
        if self._session_start_equity is not None:
            cur = self.state["equity"]
            drop = (cur - self._session_start_equity) / self._session_start_equity
            if drop <= -EMERGENCY_EQUITY_DROP_PCT:
                self._emergency_close_all(
                    f"equity_emergency_{drop*100:.1f}pct"
                )
                return True

        if health["halt_signals"] and not self.state.get("halted"):
            self.state["halted"] = True
            self._write_state()
            send_halt_alert(
                health, self._bot_token, self._chat_id, dry_run=self.dry_run
            )
            logger.warning(
                "CIRCUIT BREAKER TRIPPED | daily=%.1f%%  dd=%.1f%%  equity=$%.2f",
                health["daily_loss_pct"], health["drawdown_pct"],
                health["current_equity"],
            )
        return bool(self.state.get("halted", False))

    def reset_breaker(self) -> None:
        self.state["halted"] = False
        self._kill_switch    = False
        self._write_state()
        logger.info("Circuit breaker and kill switch reset.")

    # ── Walk-forward maintenance (identical to PaperTrader) ────────────────

    def _ensure_maintenance_state(self) -> None:
        defaults = {
            "last_training_date":   self._BASELINE_TRAINING_DATE,
            "peak_equity_date":     datetime.now(timezone.utc).date().isoformat(),
            "maintenance_required": False,
            "performance_drift":    False,
            "fold_registry": [
                {"fold": 1, "start": "2023-01-01", "end": "2023-05-13"},
                {"fold": 2, "start": "2023-05-14", "end": "2023-09-23"},
                {"fold": 3, "start": "2023-09-24", "end": "2024-02-04"},
                {"fold": 4, "start": "2024-02-05", "end": "2024-06-17"},
                {"fold": 5, "start": "2024-06-18", "end": "2024-10-28"},
                {"fold": 6, "start": "2024-10-29", "end": "2025-03-11"},
                {"fold": 7, "start": "2025-03-12", "end": "2025-07-22"},
                {"fold": 8, "start": "2025-07-23", "end": "2025-10-31"},
            ],
        }
        for key, val in defaults.items():
            if key not in self.state:
                self.state[key] = val

    def _backfill_missing_data(self) -> int:
        logger.info("Backfilling missing market data since last session ...")
        updated = 0
        for asset in TIER2_UNIVERSE:
            file_sym = "ASTER" if asset == "ASTR" else asset
            path_1h  = CACHE_DIR / f"{file_sym}_USDT_1h.parquet"
            since = None
            if path_1h.exists():
                try:
                    old = pd.read_parquet(path_1h)
                    old.index = pd.to_datetime(old.index, utc=True)
                    since = old.index[-1]
                    gap_hours = (pd.Timestamp.now(tz="UTC") - since).total_seconds() / 3600
                    if gap_hours < 2:
                        continue
                except Exception:
                    pass
            df_new = self._fetch_binance_1h(asset, since=since)
            if df_new is not None and not df_new.empty:
                if since is not None:
                    old = pd.read_parquet(path_1h)
                    old.index = pd.to_datetime(old.index, utc=True)
                    df_1h = pd.concat([old, df_new])
                    df_1h = df_1h[~df_1h.index.duplicated(keep="last")].sort_index()
                else:
                    df_1h = df_new
                if len(df_1h) > self._CACHE_MAX_1H:
                    df_1h = df_1h.iloc[-self._CACHE_MAX_1H:]
                self._safe_write_parquet(df_1h, path_1h)

                path_4h = CACHE_DIR / f"{file_sym}_USDT.parquet"
                df_new_4h = df_new.resample("4h").agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
                ).dropna(subset=["close"])
                if path_4h.exists():
                    old4 = pd.read_parquet(path_4h)
                    old4.index = pd.to_datetime(old4.index, utc=True)
                    df_4h = pd.concat([old4, df_new_4h])
                    df_4h = df_4h[~df_4h.index.duplicated(keep="last")].sort_index()
                else:
                    df_4h = df_new_4h
                if len(df_4h) > self._CACHE_MAX_4H:
                    df_4h = df_4h.iloc[-self._CACHE_MAX_4H:]
                self._safe_write_parquet(df_4h, path_4h)
                updated += 1
            time.sleep(0.05)
        logger.info("Backfill complete — %d/%d assets updated.", updated, len(TIER2_UNIVERSE))
        return updated

    def _rotate_fold_window(self) -> dict:
        from datetime import date, timedelta
        registry = self.state["fold_registry"]
        last_fold_end = date.fromisoformat(registry[-1]["end"])
        new_fold_start = last_fold_end + timedelta(days=1)
        new_fold_end = new_fold_start + timedelta(days=self._RETRAIN_CYCLE_DAYS - 1)

        dropped = registry.pop(0)
        for i, fold in enumerate(registry):
            fold["fold"] = i + 1
        new_fold = {
            "fold":  self._FOLD_DEPTH,
            "start": new_fold_start.isoformat(),
            "end":   new_fold_end.isoformat(),
        }
        registry.append(new_fold)
        self.state["fold_registry"] = registry
        return new_fold

    def maintenance_check(self) -> dict:
        from datetime import date
        today = datetime.now(timezone.utc).date()
        self._ensure_maintenance_state()

        current_equity = self.state["equity"]
        peak_equity    = self.state.get("peak_equity", current_equity)

        if current_equity >= peak_equity:
            self.state["peak_equity"]       = current_equity
            self.state["peak_equity_date"]  = today.isoformat()
            self.state["performance_drift"] = False

        last_train = date.fromisoformat(self.state["last_training_date"])
        days_since_training = (today - last_train).days
        maintenance_needed = days_since_training >= self._RETRAIN_CYCLE_DAYS

        new_fold = None
        if maintenance_needed and not self.state["maintenance_required"]:
            self.state["maintenance_required"] = True
            backfilled = self._backfill_missing_data()
            new_fold = self._rotate_fold_window()

            logger.critical(
                "MAINTENANCE_REQUIRED | %d days since last training. "
                "Backfilled %d assets. New Fold 8: %s → %s.",
                days_since_training, backfilled,
                new_fold["start"], new_fold["end"],
            )
            registry = self.state["fold_registry"]
            fold_lines = "\n".join(
                f"  Fold {f['fold']}: {f['start']} → {f['end']}"
                for f in registry
            )
            send_maintenance_alert(
                trigger="120-Day Cycle — Automated Backfill & Fold Rotation",
                details=(
                    f"Days since last training: {days_since_training}\n"
                    f"Backfilled assets: {backfilled}/{len(TIER2_UNIVERSE)}\n"
                    f"New Fold 8: {new_fold['start']} → {new_fold['end']}\n"
                    f"\nUpdated 8-Fold Window:\n{fold_lines}"
                ),
                bot_token=self._bot_token,
                chat_id=self._chat_id,
                dry_run=self.dry_run,
            )

        peak_date = date.fromisoformat(self.state["peak_equity_date"])
        days_since_ath = (today - peak_date).days
        drift_detected = days_since_ath >= self._ATH_STAGNATION_DAYS

        if drift_detected and not self.state["performance_drift"]:
            self.state["performance_drift"] = True
            logger.critical(
                "PERFORMANCE_DRIFT_ALERT | No new ATH for %d days",
                days_since_ath,
            )
            send_maintenance_alert(
                trigger="Performance Drift (30-Day Stagnation)",
                details=(
                    f"Days since last ATH: {days_since_ath}\n"
                    f"Current equity: ${current_equity:,.2f}\n"
                    f"Peak equity: ${peak_equity:,.2f}"
                ),
                bot_token=self._bot_token,
                chat_id=self._chat_id,
                dry_run=self.dry_run,
            )

        self._write_state()
        return {
            "maintenance_required": maintenance_needed,
            "performance_drift":    drift_detected,
            "days_since_training":  days_since_training,
            "days_since_ath":       days_since_ath,
            "fold_registry":        self.state["fold_registry"],
            "new_fold":             new_fold,
        }

    # ── Full cycle ─────────────────────────────────────────────────────────

    def run_cycle(self) -> dict:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        logger.info("═══ LIVE cycle  %s  equity=$%.2f ═══",
                     now_str, self.state["equity"])

        # Reset daily counters
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.get("daily_start_date") != today:
            self._daily_trade_count = 0
            self._session_start_equity = self.state["equity"]

        self._refresh_cache()
        self._verify_all_positions()
        closed  = self.check_exits()
        halted  = self._check_and_halt()
        integrity = self.maintenance_check()

        if halted:
            logger.info("Scan skipped (halted).")
            return {"closed": closed, "opened": [], "halted": True,
                    "integrity": integrity}

        opened = self.scan()

        h = self.get_health()
        all_closed = self.state.get("closed_trades", [])
        total_deployed = sum(t.get("position_usd", 0.0) for t in all_closed)
        total_pnl_closed = sum(t.get("pnl_usd", 0.0) for t in all_closed)
        roi_pct = total_pnl_closed / total_deployed * 100 if total_deployed else 0.0
        logger.info(
            "LIVE cycle done | open=%d  closed=%d  new=%d  equity=$%.2f  "
            "daily=%.1f%%  dd=%.1f%%  deployed=$%.2f  PnL=$%.2f (%.2f%%)",
            len(self.state["open_trades"]),
            len(closed), len(opened),
            h["current_equity"],
            h["daily_loss_pct"], h["drawdown_pct"],
            total_deployed, total_pnl_closed, roi_pct,
        )

        if not opened:
            send_no_signal_alert(
                now_str, h["current_equity"], h["daily_loss_pct"],
                self._bot_token, self._chat_id, dry_run=self.dry_run,
            )

        return {"closed": closed, "opened": opened, "halted": False,
                "integrity": integrity}

    # ── CSV logging ────────────────────────────────────────────────────────

    def _append_csv(self, closed: list[dict]) -> None:
        if not closed or self.dry_run:
            return
        os.makedirs(TRADES_CSV.parent, exist_ok=True)
        df     = pd.DataFrame(closed)
        header = not TRADES_CSV.exists()
        df.to_csv(TRADES_CSV, mode="a", index=False, header=header)

    # ── Telegram listener ─────────────────────────────────────────────────

    def _mins_to_next_cycle(self) -> int:
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
        thread = threading.Thread(target=self._poll_loop, daemon=True, name="tg-live-listener")
        thread.start()
        logger.info("Telegram listener started (live commands active).")

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
                        continue

                    if text in ("heartbeat", "/status", "status", "/start"):
                        logger.info("Status request received: '%s'", text)
                        health = self.get_health()
                        open_assets = list(self.state.get("open_trades", {}).keys())
                        live_prices = self.fetch_spot_prices(open_assets) if open_assets else {}
                        send_heartbeat_alert(
                            self.state, health,
                            self._bot_token, self._chat_id,
                            next_cycle_mins=self._mins_to_next_cycle(),
                            live_prices=live_prices,
                        )

                    elif text == "/kill":
                        logger.critical("KILL command received via Telegram!")
                        self._emergency_close_all("telegram_kill_command")

                    elif text == "/resume confirm":
                        logger.info("RESUME command received via Telegram.")
                        self.reset_breaker()
                        self._send_live_alert("✅ *LIVE — Kill switch deactivated.* Bot resumed.")

                    elif text == "/positions":
                        self._send_positions_report()

            except Exception as exc:
                logger.debug("Telegram poll error: %s", exc)
                time.sleep(5)

    def _send_positions_report(self) -> None:
        """Fetch real Binance positions and report via Telegram."""
        lines = ["📊 *LIVE Positions (Binance)*\n"]
        open_trades = self.state.get("open_trades", {})
        if not open_trades:
            lines.append("No open positions.")
        else:
            for asset, trade in open_trades.items():
                pos = self.client.get_position(asset)
                if pos:
                    upnl = float(pos.get("unrealizedPnl", 0))
                    liq  = pos.get("liquidationPrice", "N/A")
                    lines.append(
                        f"  {asset} | {pos.get('side','?').upper()} | "
                        f"uPnL: ${upnl:+.2f} | Liq: {liq}"
                    )
                else:
                    lines.append(f"  {asset} | ⚠ NOT FOUND on Binance")
        bal = self.client.get_balance()
        lines.append(f"\n💰 Balance: ${bal['total']:,.2f} (free: ${bal['free']:,.2f})")
        self._send_live_alert("\n".join(lines))

    # ── Status summary ─────────────────────────────────────────────────────

    def status(self) -> None:
        h       = self.get_health()
        closed  = self.state.get("closed_trades", [])
        initial = self.state["initial_capital"]

        print("\n╔══════════ VARANUS LIVE TRADING STATUS ══════════╗")
        print(f"  Mode:          {'TESTNET' if self.testnet else 'PRODUCTION'}")
        print(f"  Kill Switch:   {'ACTIVE 🚨' if self._kill_switch else 'Off'}")
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

        # Real Binance balance
        try:
            bal = self.client.get_balance()
            print(f"\n  ── Binance Account ──")
            print(f"  Total:         ${bal['total']:,.2f}")
            print(f"  Free:          ${bal['free']:,.2f}")
            print(f"  In Positions:  ${bal['used']:,.2f}")
        except Exception:
            print(f"  Binance:       (unable to fetch)")

        print("╚═════════════════════════════════════════════════╝\n")
