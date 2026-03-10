"""
varanus/tbm_labeler.py — Triple-Barrier Labeling (Step 3).
Varanus v5.2 Dual-Engine — includes build_dual_labels() for directional training.

ATR-based barriers adapt to each asset's current volatility regime,
solving the problem of fixed-percentage barriers that are trivial on one
asset and noise on another.

Label encoding:
     1  →  TP hit first   (win)
     0  →  Time barrier   (neutral / no-trade bars)
    -1  →  SL hit first   (loss)

Flash-Wick Guard:
    Mid-caps engineer wicks that sweep stop-losses before reversing.
    With the guard ON, a Stop-Loss requires a candle body close beyond
    the stop level (wick touches alone do not trigger).
    A severe wick (> 0.3×ATR beyond stop) is treated as a hit regardless.

Entry convention for labeling:
    Entry price = close[i] on the signal bar.
    Forward scan starts at bar i+1.
    This is consistent with training data generation; the live backtest
    engine (Step 5) applies slippage on the next bar's open.

Public API:
    calculate_barriers(entry, atr, direction, cfg, asset) -> dict
    label_trades(df, signals, cfg, asset, params) -> pd.Series
    label_trades_all_assets(data_dict, signals_dict, cfg, params) -> pd.DataFrame
    barrier_stats(labels, signals) -> dict
    build_dual_labels(df_4h, X, params) -> pd.Series   [v5.2]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from varanus.universe import is_high_vol

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

TBM_CONFIG: dict = {
    "atr_window":           14,
    "take_profit_atr":      3.5,    # v5.1: TP = entry ± (3.5 × ATR14) — floor of [3.5–6.0] search range
    "stop_loss_atr":        0.9,    # v5.1: SL = entry ∓ (0.9 × ATR14) — midpoint of [0.7–1.2] search range
    "max_holding_candles":  30,     # Time barrier: 30 × 4h = 5 days
    "min_rr_ratio":         2.5,    # v5.1: raised from 2.0 — wider TP range demands higher R:R floor
    "flash_wick_guard":     True,

    # High-Volatility Sub-Tier overrides: TAO, ASTR, KITE, ICP
    "high_vol_overrides": {
        "take_profit_atr":  4.5,    # v5.1: raised from 3.0 — extra runway for high-vol assets
        "stop_loss_atr":    1.2,    # v5.1: ceiling of search range — max tolerated for high-vol
    },
}

FLASH_WICK_GUARD: dict = {
    "enabled":                        True,
    "require_body_close_beyond_stop": True,   # Wick alone does NOT trigger SL
    "wick_tolerance_atr_ratio":       0.3,    # Wick may pierce up to 0.3×ATR past SL
    "confirmation_candles":           1,      # 1 close-beyond confirmation required
}


# ═══════════════════════════════════════════════════════════════════════════════
# Barrier calculation
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_barriers(
    entry:     float,
    atr:       float,
    direction: int,
    cfg:       dict = TBM_CONFIG,
    asset:     str  = "",
) -> dict:
    """
    Compute ATR-based TP/SL barriers for a single trade setup.

    Args:
        entry:     Entry price (close of signal bar).
        atr:       ATR(14) value at the entry bar.
        direction: +1 for long, -1 for short.
        cfg:       TBM_CONFIG or Optuna override dict.
        asset:     Base currency string — triggers high-vol overrides if needed.

    Returns:
        dict with keys:
          take_profit      float
          stop_loss        float
          rr_ratio         float  (rounded to 2dp)
          min_rr_satisfied bool
    """
    hv       = is_high_vol(asset)
    hv_cfg   = cfg.get("high_vol_overrides", {})

    tp_mul   = hv_cfg.get("take_profit_atr", cfg["take_profit_atr"]) if hv else cfg["take_profit_atr"]
    sl_mul   = hv_cfg.get("stop_loss_atr",   cfg["stop_loss_atr"])   if hv else cfg["stop_loss_atr"]

    take_profit = entry + direction * tp_mul * atr
    stop_loss   = entry - direction * sl_mul * atr

    tp_dist = abs(take_profit - entry)
    sl_dist = abs(entry - stop_loss)
    rr      = tp_dist / sl_dist if sl_dist > 0 else 0.0

    return {
        "take_profit":      take_profit,
        "stop_loss":        stop_loss,
        "rr_ratio":         round(rr, 2),
        "min_rr_satisfied": rr >= cfg.get("min_rr_ratio", TBM_CONFIG["min_rr_ratio"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ATR helper (local, avoids circular import with pa_features)
# ═══════════════════════════════════════════════════════════════════════════════

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple ATR: rolling mean of True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Core labeler
# ═══════════════════════════════════════════════════════════════════════════════

def label_trades(
    df:       pd.DataFrame,
    signals:  pd.Series,
    cfg:      dict = TBM_CONFIG,
    asset:    str  = "",
    params:   Optional[dict] = None,
) -> pd.Series:
    """
    Apply Triple-Barrier Labeling to generate {-1, 0, 1} labels.

    For each bar i where signals[i] != 0:
      - Compute ATR-based TP/SL barriers in the signal direction.
      - Skip if R:R < min_rr_ratio (label stays 0).
      - Scan forward bar-by-bar up to max_holding_candles:
            TP hit (wick touch) first  → label  1
            SL hit (body close or severe wick) first → label -1
            Neither before time limit  → label  0

    For bars where signals[i] == 0: label = 0 (no trade, neutral).

    Args:
        df:      OHLCV DataFrame (DatetimeIndex, columns: open/high/low/close/volume).
        signals: Direction series aligned to df.index; values {-1, 0, 1}.
                 Typically the mss_signal column from build_features().
        cfg:     TBM_CONFIG or override dict.
        asset:   Base currency — used for high-vol barrier overrides.
        params:  Optional Optuna param overrides:
                   tp_atr_mult   float  → overrides take_profit_atr
                   sl_atr_mult   float  → overrides stop_loss_atr
                   max_holding   int    → overrides max_holding_candles

    Returns:
        pd.Series of int8 {-1, 0, 1} aligned to df.index.
    """
    # ── Resolve effective config (apply Optuna overrides if provided) ─────────
    eff                        = cfg.copy()
    eff["high_vol_overrides"]  = cfg.get("high_vol_overrides", {}).copy()

    if params:
        if "tp_atr_mult" in params:
            eff["take_profit_atr"] = params["tp_atr_mult"]
            eff["high_vol_overrides"]["take_profit_atr"] = params.get(
                "tp_atr_mult_hv", params["tp_atr_mult"]
            )
        if "sl_atr_mult" in params:
            eff["stop_loss_atr"] = params["sl_atr_mult"]
            eff["high_vol_overrides"]["stop_loss_atr"] = params.get(
                "sl_atr_mult_hv", params["sl_atr_mult"]
            )
        if "max_holding" in params:
            eff["max_holding_candles"] = params["max_holding"]

    # ── Pre-compute ATR series ─────────────────────────────────────────────────
    atr_series = _atr(df, eff["atr_window"])

    max_hold   = eff["max_holding_candles"]
    flash_wick = eff.get("flash_wick_guard", True)
    wick_tol   = FLASH_WICK_GUARD["wick_tolerance_atr_ratio"]

    # ── Raw arrays for fast indexing ───────────────────────────────────────────
    high_arr   = df["high"].values
    low_arr    = df["low"].values
    close_arr  = df["close"].values
    atr_arr    = atr_series.values
    sig_arr    = signals.reindex(df.index).fillna(0).values.astype(int)
    n          = len(df)

    labels = np.zeros(n, dtype=np.int8)

    for i in range(n):
        direction = sig_arr[i]
        if direction == 0:
            continue

        atr_val = atr_arr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        entry_price = close_arr[i]
        barriers    = calculate_barriers(entry_price, atr_val, direction, eff, asset)

        if not barriers["min_rr_satisfied"]:
            continue   # Label stays 0

        tp       = barriers["take_profit"]
        sl       = barriers["stop_loss"]
        wick_ext = wick_tol * atr_val  # Maximum allowable SL wick penetration

        outcome = 0  # Default: time barrier

        for j in range(i + 1, min(i + max_hold + 1, n)):
            h = high_arr[j]
            l = low_arr[j]
            c = close_arr[j]

            # ── TP check (wick touch sufficient, checked before SL) ───────────
            if direction == 1 and h >= tp:
                outcome = 1
                break
            if direction == -1 and l <= tp:
                outcome = 1
                break

            # ── SL check ─────────────────────────────────────────────────────
            if flash_wick:
                # Body must close beyond stop, OR wick exceeds tolerance
                if direction == 1:
                    body_sl  = c < sl
                    wick_sl  = l < sl - wick_ext
                    sl_hit   = body_sl or wick_sl
                else:
                    body_sl  = c > sl
                    wick_sl  = h > sl + wick_ext
                    sl_hit   = body_sl or wick_sl
            else:
                # Standard: any wick touch triggers
                sl_hit = (direction == 1 and l <= sl) or (direction == -1 and h >= sl)

            if sl_hit:
                outcome = -1
                break

        labels[i] = outcome

    return pd.Series(labels, index=df.index, dtype=np.int8, name="label")


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-asset labeler
# ═══════════════════════════════════════════════════════════════════════════════

def label_trades_all_assets(
    data_dict:    dict[str, pd.DataFrame],
    signals_dict: dict[str, pd.Series],
    cfg:          dict = TBM_CONFIG,
    params:       Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply TBM labeling across all assets and return a combined DataFrame.

    Args:
        data_dict:    {asset: OHLCV DataFrame}
        signals_dict: {asset: direction Series aligned to asset's OHLCV index}
        cfg:          TBM_CONFIG or override dict.
        params:       Optional Optuna overrides forwarded to label_trades().

    Returns:
        DataFrame with columns [asset, label] indexed by timestamp.
        Only rows where label != 0 or signals != 0 are included.
    """
    records = []
    for asset, df in data_dict.items():
        sigs = signals_dict.get(asset, pd.Series(0, index=df.index))
        lbl  = label_trades(df, sigs, cfg=cfg, asset=asset, params=params)
        active = (sigs != 0) | (lbl != 0)
        if not active.any():
            continue
        tmp = lbl[active].reset_index()
        tmp.columns = ["timestamp", "label"]
        tmp["asset"] = asset
        records.append(tmp)

    if not records:
        return pd.DataFrame(columns=["timestamp", "asset", "label"])

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def build_dual_labels(
    df_4h:  pd.DataFrame,
    X:      pd.DataFrame,
    params: dict,
) -> pd.Series:
    """
    v5.2 Dual-Engine: generate direction-specific TBM labels for training.

    Long labels  — direction +1 with tp_mult_long / sl_mult_long.
                   Candidates: mss_signal==1  PLUS  bars where
                   bias_bypass_long==1 AND fvg_type==1 AND sweep_occurred==1.

    Short labels — direction -1 with tp_atr_mult / sl_atr_mult (frozen).

    Short labels take priority on any bar where both would fire.

    Args:
        df_4h:  OHLCV DataFrame for the asset (full slice, not just X.index).
        X:      Feature matrix from build_features(), indexed to df_4h subset.
        params: Full parameter dict including tp_mult_long, sl_mult_long,
                tp_atr_mult, sl_atr_mult, and optionally '_asset' str.

    Returns:
        pd.Series {-1, 0, +1} aligned to X.index.
    """
    asset = params.get('_asset', '')
    mss   = X['mss_signal']

    # ── Long signal candidates ────────────────────────────────────────────────
    long_sig = mss.clip(lower=0).copy()   # +1 where bullish MSS, else 0
    if 'bias_bypass_long' in X.columns:
        # Bypass candidates: oversold condition active + bullish FVG OR relaxed SSL sweep.
        # ssl_sweep_long uses a 20% relaxed min_sweep_pct — catches more dip-buy setups
        # that have a genuine liquidity sweep but may not have a full FVG formation.
        # SHORT HUNTER LOCK-DOWN: these candidates are only for direction == +1.
        has_ssl_sweep = X['ssl_sweep_long'] == 1 if 'ssl_sweep_long' in X.columns \
                        else pd.Series(False, index=X.index)
        bypass_candidates = (
            (X['bias_bypass_long'] == 1) &
            ((X['fvg_type'] == 1) | has_ssl_sweep) &
            (mss == 0)
        )
        long_sig = long_sig.copy()
        long_sig[bypass_candidates] = 1

    # v5.2 Deep Search: minor MSS long candidates.
    # A minor bullish MSS (8-bar lookback) with a quality filter (bullish FVG OR
    # SSL sweep present) qualifies as a long candidate even without a full MSS.
    # SHORT HUNTER LOCK-DOWN: minor_mss_long is long-side only.
    if 'minor_mss_long' in X.columns:
        has_ssl_sweep_m = X['ssl_sweep_long'] == 1 if 'ssl_sweep_long' in X.columns \
                          else pd.Series(False, index=X.index)
        minor_mss_candidates = (
            (X['minor_mss_long'] == 1) &
            ((X['fvg_type'] == 1) | has_ssl_sweep_m) &
            (mss == 0)
        )
        long_sig[minor_mss_candidates] = 1

    # ── Short signal candidates ───────────────────────────────────────────────
    short_sig = mss.where(mss == -1, 0)

    # ── Direction-specific barrier params ────────────────────────────────────
    long_params = {
        **params,
        'tp_atr_mult': params.get('tp_mult_long', params.get('tp_atr_mult', 3.5)),
        'sl_atr_mult': params.get('sl_mult_long', params.get('sl_atr_mult', 0.80)),
    }

    df_slice = df_4h.loc[X.index]
    y_long   = label_trades(df_slice, long_sig,  TBM_CONFIG, asset, long_params)
    y_short  = label_trades(df_slice, short_sig, TBM_CONFIG, asset, params)

    # ── Merge: short takes priority ───────────────────────────────────────────
    y = pd.Series(0, index=X.index, dtype=np.int8)
    y[y_long  ==  1] =  1
    y[y_short == -1] = -1
    return y


def barrier_stats(
    labels:  pd.Series,
    signals: Optional[pd.Series] = None,
) -> dict:
    """
    Summarise label distribution for a single asset.

    Args:
        labels:  Output of label_trades() — values {-1, 0, 1}.
        signals: Optional direction series; if provided, stats are
                 computed only on signal bars (where signals != 0).

    Returns:
        dict with keys:
          total_bars, signal_bars, win_count, loss_count, neutral_count,
          win_rate, loss_rate, neutral_rate, rr_implied
    """
    if signals is not None:
        active   = signals[signals != 0].index
        filtered = labels.reindex(active)
    else:
        filtered = labels[labels != 0]

    total   = len(labels)
    n_sig   = len(filtered)
    wins    = int((filtered == 1).sum())
    losses  = int((filtered == -1).sum())
    neutral = int((filtered == 0).sum())

    win_rate     = wins    / n_sig if n_sig > 0 else 0.0
    loss_rate    = losses  / n_sig if n_sig > 0 else 0.0
    neutral_rate = neutral / n_sig if n_sig > 0 else 0.0
    rr_implied   = win_rate / loss_rate if loss_rate > 0 else float("inf")

    return {
        "total_bars":    total,
        "signal_bars":   n_sig,
        "win_count":     wins,
        "loss_count":    losses,
        "neutral_count": neutral,
        "win_rate":      round(win_rate,     4),
        "loss_rate":     round(loss_rate,    4),
        "neutral_rate":  round(neutral_rate, 4),
        "rr_implied":    round(rr_implied,   3),
    }
