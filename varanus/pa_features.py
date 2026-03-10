"""
varanus/pa_features.py — Price Action Feature Engineering (Step 2).
Varanus v5.2 Dual-Engine.

Three feature groups:
  2.1  Market Structure Shift (MSS)        → mss_signal, htf_bias
  2.2  Fair Value Gap + Liquidity Sweep    → fvg_type, fvg_distance_atr,
                                             fvg_age_candles, sweep_occurred
  2.3  Chameleon Confirmation              → relative_volume, rsi_14, rsi_slope_3,
                                             ema21_55_alignment, atr_percentile_100

Market Character (per-bar, single-asset):
       volatility_rank, volume_rank, asset_tier_flag,
       hour_of_day, day_of_week

v5.2 Additions:
  2.4  Long Runner Bias Bypass             → bias_bypass_long
       Neutralises bearish HTF bias when the market is deeply oversold
       (RSI_1D < 45 OR price in bottom 25% of its 100-day range).

  2.5  SSL Sweep Long (v5.2 Volume Injection) → ssl_sweep_long
       Detects Sell-Side Liquidity (SSL) sweeps with a 20% relaxed threshold
       specifically for long entries. Does NOT affect short detection.

  2.6  Minor MSS Long (v5.2 Deep Search) → minor_mss_long
       Short-lookback (8-bar) bullish market structure break. Detects breaks
       of the most recent 8-bar high — 'minor' structural shifts that precede
       full MSS confirmation. Long-side only.

Master entry point:
  build_features(df_4h, df_1d, asset, params=None) -> pd.DataFrame

NOTE: detect_mss() uses a trailing rolling window, correct for both
      training/backtesting and live use.

SHORT HUNTER LOCK-DOWN: No parameters, logic, or thresholds for short trades
(direction == -1) may be modified. The v5.1 Short Hunter (Trial #183) is the
Gold Standard baseline. All changes in this file are strictly long-side only.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from varanus.universe import is_high_vol, HIGH_VOL_SUBTIER

logger = logging.getLogger(__name__)

# ── 2.1 MSS config ─────────────────────────────────────────────────────────────

MSS_CONFIG: dict = {
    "lookback_range":     (30, 50),  # Optuna search range
    "lookback_default":   40,        # Starting point / training default
    "swing_confirmation": 3,         # Candles each side to confirm swing H/L
    "body_filter":        0.6,       # Close >= 60% of candle body past swing
    "wick_tolerance":     0.005,     # Ignore sweeps < 0.5% beyond swing
    "htf_bias_required":  True,      # MSS must align with 1D trend direction
}

# ── v5.2 Bias Bypass config ────────────────────────────────────────────────────
# SHORT HUNTER LOCK-DOWN: These settings apply to Long Runner (direction==1) only.

BIAS_BYPASS_CONFIG: dict = {
    "rsi_1d_threshold":       50,    # Daily RSI below this → bypass bearish HTF filter
                                     # v5.2 Deep Search: relaxed from 45 → 50
                                     # Captures mid-range pullbacks, not just oversold zones.
    "price_range_bottom_pct": 0.40,  # Price in bottom 40% of 100-day H-L range → bypass
                                     # v5.2 Deep Search: relaxed from 25% → 40%
                                     # Allows dip-buying across the lower half of range.
    "range_lookback_4h":      600,   # 100 days × 6 4h bars per day
    "rsi_1d_min_periods":     10,    # Minimum 1D bars required to compute RSI
    "ssl_sweep_mult_long":    0.80,  # SSL sweep threshold multiplier for Long entries only
                                     # min_sweep_pct × 0.80 — 20% more sensitive to
                                     # Sell-Side Liquidity sweeps for dip-buying.
                                     # Does NOT affect bearish sweep detection.
    "minor_mss_lookback":     8,     # Short lookback for minor MSS detection (bars)
                                     # Break of most recent 8-bar high = minor bullish MSS
}

# ── 2.2 FVG config ─────────────────────────────────────────────────────────────

FVG_CONFIG: dict = {
    "min_gap_atr_ratio":    0.3,    # FVG >= 30% of ATR(14)
    "max_gap_age_candles":  20,     # Invalidate FVGs older than 20 candles
    "sweep_lookback":       15,     # Bars to look back for a swept swing point
    "min_sweep_pct":        0.004,  # Breach swing by >= 0.4%
    "sweep_close_reversal": True,   # Must close back inside range after sweep
    "fvg_partial_fill_pct": 0.5,    # Invalidate if 50%+ of gap is filled
    "require_sweep":        True,   # CORE TIER 2 FILTER. Never set to False.
}

# ── 2.3 Confirmation feature config ───────────────────────────────────────────

CONFIRMATION_FEATURES: dict = {
    # PRIMARY — both required for signal emission
    "relative_volume": {
        "window":    20,
        "threshold": 1.5,    # Current vol >= 1.5x 20-period avg
        "weight":    0.40,
    },
    "rsi_14": {
        "oversold":     35,   # Bullish entry zone (wider than Tier 1)
        "overbought":   65,   # Bearish entry zone
        "neutral_band": (45, 55),
        "weight":       0.35,
    },
    # SECONDARY — at least 1 required
    "atr_percentile": {
        "window":         100,
        "min_percentile": 40,  # ATR in top 60% — avoid dead markets
        "weight":         0.15,
    },
    "ema_alignment": {
        "fast":              21,
        "slow":              55,
        "require_alignment": True,
        "weight":            0.10,
    },
}

CONFIRMATION_SCORE_MIN: float = 0.70

# Full feature list produced by build_features() — matches Step 4 FEATURE_LIST
FEATURE_COLS: list[str] = [
    # PA features
    "mss_signal",
    "fvg_type",
    "fvg_distance_atr",
    "fvg_age_candles",
    "sweep_occurred",
    "htf_bias",
    # Chameleon confirmation
    "relative_volume",
    "rsi_14",
    "rsi_slope_3",
    "ema21_55_alignment",
    "atr_percentile_100",
    # Market character
    "volatility_rank",
    "volume_rank",
    "asset_tier_flag",
    "hour_of_day",
    "day_of_week",
    # v5.2 Long Runner Bias Bypass
    "bias_bypass_long",
    # v5.2 Volume Injection — SSL sweep for long entries (relaxed threshold, long-side only)
    "ssl_sweep_long",
    # v5.2 Deep Search — Minor MSS long (8-bar short-lookback bullish structure break)
    "minor_mss_long",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Average True Range (simple rolling mean of TR)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (EWM with alpha = 1/period)."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    For each bar, return the fraction of the preceding *window* values
    that are strictly below the current value.  Result is in [0, 1].
    """
    def _pct(w: np.ndarray) -> float:
        return float((w[:-1] < w[-1]).mean()) if len(w) > 1 else 0.0

    return series.rolling(window).apply(_pct, raw=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1  Market Structure Shift
# ═══════════════════════════════════════════════════════════════════════════════

def detect_mss(df: pd.DataFrame, lookback: int = 40) -> pd.Series:
    """
    Detect Market Structure Shift on a single-asset OHLCV DataFrame.

    Spec intent: "A valid MSS is the first close beyond the most recent
    significant swing point, confirming a change in market character."

    Implementation: trailing rolling max/min of the past *lookback* bars.
    For each bar t, the "swing high" is max(high[t-lookback : t-1]), and
    a bullish MSS fires on the first close that crosses above it.

    A wick_tolerance filter (0.5%) ensures micro-wicks are not treated as
    structural breaks.

    NOTE on the spec's center=True code: using center=True and then shift(1)
    creates an impossibility — swing_highs[t-1] includes highs up to t+19,
    so close[t] < high[t] ≤ swing_highs[t-1] always. Corrected here.

    Args:
        df:       DataFrame with columns [high, low, close], DatetimeIndex.
        lookback: Trailing window (Optuna range: 30–50, default 40).

    Returns:
        pd.Series of {1: Bullish MSS, -1: Bearish MSS, 0: No signal}.
    """
    tol         = MSS_CONFIG["wick_tolerance"]
    prior_high  = df["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
    prior_low   = df["low"].rolling(window=lookback, min_periods=lookback).min().shift(1)

    bullish_mss = (
        (df["close"] > prior_high * (1 + tol)) &
        (df["close"].shift(1) <= prior_high)
    )
    bearish_mss = (
        (df["close"] < prior_low * (1 - tol)) &
        (df["close"].shift(1) >= prior_low)
    )

    signal = pd.Series(0, index=df.index, dtype=np.int8)
    signal[bullish_mss] =  1
    signal[bearish_mss] = -1
    return signal


def detect_mss_live(df: pd.DataFrame, lookback: int = 40) -> pd.Series:
    """
    Alias for detect_mss() — same trailing-window implementation.
    Kept as a named variant so callers can be explicit about live use.
    """
    return detect_mss(df, lookback)


def detect_htf_bias(df_1d: pd.DataFrame, lookback: int = 40) -> pd.Series:
    """
    Compute MSS on the 1D timeframe.

    Returns a Series indexed by df_1d.index with values {-1, 0, 1}.
    The raw signal marks only the MSS bar; forward-filling to 4h is done
    by align_htf_to_4h().
    """
    raw = detect_mss_live(df_1d, lookback)
    # Forward-fill the last non-zero direction so every 1D bar has a bias
    filled = raw.copy()
    last_dir = 0
    for i, v in enumerate(raw.values):
        if v != 0:
            last_dir = v
        filled.iloc[i] = last_dir
    return filled


def align_htf_to_4h(htf_signal: pd.Series, df_4h: pd.DataFrame) -> pd.Series:
    """
    Forward-fill a 1D bias signal onto the 4h bar index.

    Args:
        htf_signal: Series of 1D MSS bias, indexed by daily DatetimeIndex.
        df_4h:      4h OHLCV DataFrame whose index drives the output.

    Returns:
        Series of {-1, 0, 1} aligned to df_4h.index.
    """
    # Union of both indices, sort, forward-fill, then select 4h rows
    combined_idx = htf_signal.index.union(df_4h.index).sort_values()
    aligned = htf_signal.reindex(combined_idx).ffill()
    return aligned.reindex(df_4h.index).fillna(0).astype(np.int8)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2  Fair Value Gap + Liquidity Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def is_liquidity_sweep(df: pd.DataFrame, idx: int, cfg: dict) -> bool:
    """
    Determine whether bar at position *idx* is a liquidity sweep.

    A sweep: price violates a prior swing High or Low by >= min_sweep_pct,
    then closes back inside the range on the same candle.

    Args:
        df:  Full OHLCV DataFrame (integer-positional indexing).
        idx: The bar to evaluate (0-based integer position).
        cfg: FVG_CONFIG dict.

    Returns:
        True if a qualifying sweep occurred at *idx*.
    """
    start    = max(0, idx - cfg["sweep_lookback"])
    window   = df.iloc[start: idx + 1]
    prior_high = window["high"].iloc[:-1].max()
    prior_low  = window["low"].iloc[:-1].min()
    current    = df.iloc[idx]

    # Bearish sweep: hunts buy-stops above prior high, then closes below
    if (
        current["high"] > prior_high * (1 + cfg["min_sweep_pct"]) and
        cfg["sweep_close_reversal"] and
        current["close"] < prior_high
    ):
        return True

    # Bullish sweep: hunts sell-stops below prior low, then closes above
    if (
        current["low"] < prior_low * (1 - cfg["min_sweep_pct"]) and
        cfg["sweep_close_reversal"] and
        current["close"] > prior_low
    ):
        return True

    return False


def detect_fvg(
    df: pd.DataFrame,
    atr: pd.Series,
    cfg: dict = FVG_CONFIG,
) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (3-candle imbalances) across the DataFrame.

    A bullish FVG exists when prev2.high < curr.low (gap above).
    A bearish FVG exists when prev2.low  > curr.high (gap below).
    Validity requires: gap >= min_gap_atr_ratio × ATR(14) AND a preceding
    liquidity sweep (when cfg['require_sweep'] is True).

    Args:
        df:  OHLCV DataFrame with positional integer or DatetimeIndex.
        atr: ATR(14) Series aligned to df.index.
        cfg: FVG_CONFIG (or overrides).

    Returns:
        DataFrame indexed by the integer position of the *impulse* candle
        (the 3rd bar of the 3-candle pattern), with columns:
            fvg_type    {1: bullish, -1: bearish}
            fvg_top     upper edge of the gap
            fvg_bottom  lower edge of the gap
            fvg_valid   True if size and sweep criteria are satisfied
    """
    fvgs = []

    # Work with a positional reset to simplify iloc arithmetic
    df_r   = df.reset_index(drop=True)
    atr_r  = atr.reset_index(drop=True)

    for i in range(2, len(df_r)):
        prev2 = df_r.iloc[i - 2]
        curr  = df_r.iloc[i]

        fvg_type = fvg_top = fvg_bottom = None

        if prev2["high"] < curr["low"]:           # Bullish FVG
            gap_size              = curr["low"] - prev2["high"]
            fvg_type, fvg_top, fvg_bottom = 1, curr["low"], prev2["high"]
        elif prev2["low"] > curr["high"]:          # Bearish FVG
            gap_size              = prev2["low"] - curr["high"]
            fvg_type, fvg_top, fvg_bottom = -1, prev2["low"], curr["high"]
        else:
            continue

        atr_val   = atr_r.iloc[i]
        atr_ratio = gap_size / atr_val if atr_val > 0 else 0.0
        sweep_ok  = (
            is_liquidity_sweep(df_r, i - 1, cfg)
            if cfg["require_sweep"] else True
        )
        valid = (atr_ratio >= cfg["min_gap_atr_ratio"]) and sweep_ok

        fvgs.append({
            "pos":        i,
            "fvg_type":   fvg_type,
            "fvg_top":    fvg_top,
            "fvg_bottom": fvg_bottom,
            "fvg_valid":  valid,
        })

    if not fvgs:
        return pd.DataFrame(
            columns=["fvg_type", "fvg_top", "fvg_bottom", "fvg_valid"]
        )

    fvg_df = pd.DataFrame(fvgs).set_index("pos")
    # Restore original index labels so callers can cross-reference df.index
    fvg_df.index = df.index[fvg_df.index]
    return fvg_df


def _compute_fvg_bar_features(
    df: pd.DataFrame,
    fvg_df: pd.DataFrame,
    atr: pd.Series,
    cfg: dict = FVG_CONFIG,
) -> pd.DataFrame:
    """
    Map FVG formations back to per-bar features.

    For each bar, finds the most recently formed *valid* FVG that is
    within max_gap_age_candles and has not been 50%+ filled.

    Returns:
        DataFrame with columns: fvg_type, fvg_distance_atr,
                                 fvg_age_candles, sweep_occurred
        (all zeros where no active valid FVG exists).
    """
    max_age     = cfg["max_gap_age_candles"]
    fill_thresh = cfg["fvg_partial_fill_pct"]

    n           = len(df)
    idx_array   = df.index
    close       = df["close"].values
    high        = df["high"].values
    low         = df["low"].values
    atr_vals    = atr.values

    fvg_types    = np.zeros(n, dtype=np.float32)
    fvg_dist_atr = np.zeros(n, dtype=np.float32)
    fvg_ages     = np.zeros(n, dtype=np.float32)
    sweep_flags  = np.zeros(n, dtype=np.float32)

    # Pre-build positional lookup for fvg_df
    # Map each fvg formation timestamp → integer position in df
    ts_to_pos = {ts: i for i, ts in enumerate(idx_array)}
    valid_fvgs = [
        (ts_to_pos[ts], row)
        for ts, row in fvg_df.iterrows()
        if row["fvg_valid"] and ts in ts_to_pos
    ]

    for j in range(n):
        best_dist   = np.inf
        best_type   = 0
        best_age    = 0
        best_sweep  = 0

        for form_pos, fvg in valid_fvgs:
            age = j - form_pos
            if age <= 0:
                continue      # FVG not yet formed at bar j
            if age > max_age:
                continue      # Too old

            fvg_top    = fvg["fvg_top"]
            fvg_bottom = fvg["fvg_bottom"]
            gap_size   = fvg_top - fvg_bottom

            if gap_size <= 0:
                continue

            # Check partial fill: look at bars between formation and j
            midpoint = fvg_bottom + fill_thresh * gap_size
            if fvg["fvg_type"] == 1:   # Bullish gap: filled if price drops in
                fill_check = low[form_pos + 1: j + 1]
                if len(fill_check) and fill_check.min() <= midpoint:
                    continue  # Gap 50%+ filled — invalid
            else:                       # Bearish gap: filled if price rises in
                fill_check = high[form_pos + 1: j + 1]
                if len(fill_check) and fill_check.max() >= midpoint:
                    continue

            # Distance: signed close-to-midpoint / ATR, positive on favourable side
            atr_j    = atr_vals[j]
            midpt    = (fvg_top + fvg_bottom) / 2.0
            signed   = (close[j] - midpt) * fvg["fvg_type"]
            dist_atr = abs(signed) / atr_j if atr_j > 0 else 0.0

            if dist_atr < best_dist:
                best_dist  = dist_atr
                best_type  = fvg["fvg_type"]
                best_age   = age
                best_sweep = 1  # All valid FVGs have sweep confirmed

        fvg_types[j]    = best_type
        fvg_dist_atr[j] = best_dist if best_dist < np.inf else 0.0
        fvg_ages[j]     = best_age
        sweep_flags[j]  = best_sweep

    return pd.DataFrame({
        "fvg_type":        fvg_types,
        "fvg_distance_atr": fvg_dist_atr,
        "fvg_age_candles": fvg_ages,
        "sweep_occurred":  sweep_flags,
    }, index=idx_array)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3  Chameleon Confirmation Features
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_confirmation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all confirmation and market-character features for a single asset.

    Args:
        df: OHLCV DataFrame (columns: open, high, low, close, volume).

    Returns:
        DataFrame with 9 new columns aligned to df.index:
          relative_volume, rsi_14, rsi_slope_3,
          ema21_55_alignment, atr_percentile_100,
          volatility_rank, volume_rank, hour_of_day, day_of_week
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    cfg_rv  = CONFIRMATION_FEATURES["relative_volume"]
    cfg_rsi = CONFIRMATION_FEATURES["rsi_14"]
    cfg_atr = CONFIRMATION_FEATURES["atr_percentile"]
    cfg_ema = CONFIRMATION_FEATURES["ema_alignment"]

    out = pd.DataFrame(index=df.index)

    # ── Relative volume ────────────────────────────────────────────────────────
    vol_ma = volume.rolling(cfg_rv["window"]).mean()
    out["relative_volume"] = volume / vol_ma.replace(0, np.nan)

    # ── RSI(14) and 3-bar slope ────────────────────────────────────────────────
    rsi = _rsi(close, 14)
    out["rsi_14"]      = rsi
    out["rsi_slope_3"] = rsi - rsi.shift(3)

    # ── EMA alignment: {1: bullish, -1: bearish, 0: mixed} ────────────────────
    ema_fast = close.ewm(span=cfg_ema["fast"], adjust=False).mean()
    ema_slow = close.ewm(span=cfg_ema["slow"], adjust=False).mean()
    bullish_align = (close > ema_fast) & (ema_fast > ema_slow)
    bearish_align = (close < ema_fast) & (ema_fast < ema_slow)
    alignment = pd.Series(0, index=df.index, dtype=np.int8)
    alignment[bullish_align] =  1
    alignment[bearish_align] = -1
    out["ema21_55_alignment"] = alignment

    # ── ATR percentile (ATR14 ranked within 100-bar window) ───────────────────
    atr14 = _atr(high, low, close, 14)
    out["atr_percentile_100"] = _rolling_percentile(atr14, cfg_atr["window"])

    # ── Volatility rank (ATR20 vs 100-bar window — market character) ──────────
    atr20 = _atr(high, low, close, 20)
    out["volatility_rank"] = _rolling_percentile(atr20, 100)

    # ── Volume rank (current volume vs 100-bar window) ────────────────────────
    out["volume_rank"] = _rolling_percentile(volume, 100)

    # ── Session features ──────────────────────────────────────────────────────
    # Normalize hour to [0, 1] so tree models split cleanly;
    # day_of_week kept as integer category (0=Monday … 6=Sunday).
    out["hour_of_day"]  = df.index.hour.astype(np.float32)
    out["day_of_week"]  = df.index.dayofweek.astype(np.float32)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2.5  SSL Sweep Long — Long-side only, relaxed threshold
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_ssl_sweep_long(
    df:           pd.DataFrame,
    sweep_lookback: int,
    min_sweep_pct: float,
    sweep_mult:   float = 0.80,
) -> pd.Series:
    """
    Per-bar feature: 1 if a bullish Sell-Side Liquidity (SSL) sweep occurred.

    A bullish SSL sweep: bar's low pierces a prior swing low by >= threshold,
    then closes ABOVE that low on the same bar (indicating reversal conviction).

    Uses a relaxed threshold = min_sweep_pct × sweep_mult (default 0.80, i.e.
    20% more sensitive than the standard FVG sweep filter).

    LONG-SIDE ONLY — only detects downward sweeps. Bearish sweep detection
    (for Short Hunter) is never modified by this function.

    Args:
        df:             OHLCV DataFrame (positional-indexed internally).
        sweep_lookback: Number of bars to look back for a prior swing low.
        min_sweep_pct:  Base sweep threshold from FVG_CONFIG (e.g. 0.00641).
        sweep_mult:     Relaxation multiplier for long entries (default 0.80).

    Returns:
        pd.Series of float32 {0.0, 1.0} aligned to df.index.
    """
    threshold = min_sweep_pct * sweep_mult
    n         = len(df)
    result    = np.zeros(n, dtype=np.float32)

    low_arr   = df["low"].values
    close_arr = df["close"].values

    for i in range(1, n):
        start     = max(0, i - sweep_lookback)
        prior_low = low_arr[start:i].min()
        cur_low   = low_arr[i]
        cur_close = close_arr[i]

        # Bullish SSL sweep: dips below prior low by threshold, closes above it
        if cur_low < prior_low * (1 - threshold) and cur_close > prior_low:
            result[i] = 1.0

    return pd.Series(result, index=df.index, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.6  Minor MSS Long — Long-side only, short lookback
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_minor_mss_long(
    df:      pd.DataFrame,
    lookback: int = 8,
) -> pd.Series:
    """
    Per-bar feature: 1 if a 'minor' bullish Market Structure Shift occurred.

    A minor bullish MSS is a close that breaks the rolling *lookback*-bar high
    (the most recent short-term swing high), where the previous bar did NOT
    already close above that level (first-break only).

    Uses a shorter lookback (default 8) than the main detect_mss() (30–50),
    enabling detection of smaller structural breaks that often precede the
    larger confirmed MSS — useful as an early-entry signal for longs.

    LONG-SIDE ONLY — only bullish (upward) breaks are flagged. Bearish minor
    MSS detection is never computed here (Short Hunter lock-down).

    Args:
        df:       OHLCV DataFrame with 'high' and 'close' columns.
        lookback: Short lookback window for swing high (default 8 bars).

    Returns:
        pd.Series of float32 {0.0, 1.0} aligned to df.index.
    """
    tol = MSS_CONFIG["wick_tolerance"]
    prior_high = df["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)

    bullish_minor_mss = (
        (df["close"] > prior_high * (1 + tol)) &
        (df["close"].shift(1) <= prior_high)
    )

    result = bullish_minor_mss.fillna(False).astype(np.float32)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Master builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    asset: str,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute all 19 PA + confirmation features for a single asset.

    Args:
        df_4h:  4h OHLCV DataFrame (DatetimeIndex, columns: open/high/low/close/volume).
        df_1d:  1d OHLCV DataFrame (DatetimeIndex) — used for HTF bias only.
        asset:  Base currency string (e.g. "LINK"), used for asset_tier_flag.
        params: Optional overrides; supports keys:
                  mss_lookback        (int)
                  fvg_min_atr_ratio   (float)
                  sweep_min_pct       (float)
                  fvg_max_age         (int)

    Returns:
        DataFrame aligned to df_4h.index with columns matching FEATURE_COLS.
        Rows that cannot be computed (warm-up period) are dropped (NaN rows).
    """
    # ── Apply any Optuna-supplied param overrides ─────────────────────────────
    mss_lookback = MSS_CONFIG["lookback_default"]
    fvg_cfg      = FVG_CONFIG.copy()

    if params:
        mss_lookback              = params.get("mss_lookback", mss_lookback)
        fvg_cfg["min_gap_atr_ratio"] = params.get("fvg_min_atr_ratio", fvg_cfg["min_gap_atr_ratio"])
        fvg_cfg["min_sweep_pct"]     = params.get("sweep_min_pct",      fvg_cfg["min_sweep_pct"])
        fvg_cfg["max_gap_age_candles"]= params.get("fvg_max_age",       fvg_cfg["max_gap_age_candles"])

    df = df_4h.copy()

    # ── 2.1 MSS signal ────────────────────────────────────────────────────────
    mss = detect_mss(df, lookback=mss_lookback)

    # ── HTF bias (1D MSS → forward-filled to 4h) ──────────────────────────────
    if df_1d is not None and len(df_1d) > 0:
        htf_raw  = detect_htf_bias(df_1d, lookback=mss_lookback)
        htf_bias = align_htf_to_4h(htf_raw, df)
    else:
        htf_bias = pd.Series(0, index=df.index, dtype=np.int8)
        logger.warning(f"[{asset}] No 1D data provided — HTF bias set to 0.")

    # ── v5.2 Long Runner Bias Bypass ──────────────────────────────────────────
    # Condition 1: Daily RSI < 35 (oversold on higher timeframe)
    bbc = BIAS_BYPASS_CONFIG
    rsi_1d_aligned = pd.Series(50.0, index=df.index)
    if df_1d is not None and len(df_1d) >= bbc["rsi_1d_min_periods"]:
        rsi_1d_raw = _rsi(df_1d["close"], 14)
        combined_idx = rsi_1d_raw.index.union(df.index).sort_values()
        rsi_1d_aligned = (
            rsi_1d_raw.reindex(combined_idx)
            .ffill()
            .reindex(df.index)
            .fillna(50.0)
        )

    # Condition 2: Price in bottom 40% of its 100-day (600 x 4h) H-L range
    lb = bbc["range_lookback_4h"]
    roll_high  = df["high"].rolling(lb, min_periods=lb // 4).max()
    roll_low   = df["low"].rolling(lb,  min_periods=lb // 4).min()
    range_span = (roll_high - roll_low).replace(0, np.nan)
    price_range_pct = ((df["close"] - roll_low) / range_span).fillna(0.5)

    # v5.5: rsi_1d_long_limit is injectable via Optuna params (search range [45, 65]).
    # Falls back to BIAS_BYPASS_CONFIG["rsi_1d_threshold"] = 50 when not provided.
    _rsi_1d_thr = (
        params.get("rsi_1d_long_limit", bbc["rsi_1d_threshold"])
        if params else bbc["rsi_1d_threshold"]
    )
    bypass_active = (
        (rsi_1d_aligned < _rsi_1d_thr) |
        (price_range_pct < bbc["price_range_bottom_pct"])
    )

    # NOTE: htf_bias is NOT modified here. Keeping it as the raw 1D MSS direction
    # ensures the short model sees the same feature distribution as v5.1.
    # bias_bypass_long is passed as a separate feature so the long runner can learn
    # to override bearish bias independently, without contaminating short training.

    # ── 2.5 SSL Sweep Long (Volume Injection — long-side only) ────────────────
    # Uses min_sweep_pct × ssl_sweep_mult_long (20% relaxed) to detect more
    # Sell-Side Liquidity sweeps for long bypass candidates.
    # SHORT HUNTER LOCK-DOWN: only bullish sweeps computed here.
    ssl_sweep_long = _compute_ssl_sweep_long(
        df,
        sweep_lookback = fvg_cfg["sweep_lookback"],
        min_sweep_pct  = fvg_cfg["min_sweep_pct"],
        sweep_mult     = bbc["ssl_sweep_mult_long"],
    )

    # ── 2.6 Minor MSS Long (Deep Search — long-side only) ─────────────────────
    # Short-lookback (8-bar) bullish structure break. Detects the first close
    # above the recent 8-bar high — minor MSS that can precede full confirmation.
    # SHORT HUNTER LOCK-DOWN: only bullish minor breaks computed here.
    # v5.5: minor_mss_lookback injectable via params (search range [5, 12]).
    minor_mss_long = _compute_minor_mss_long(
        df,
        lookback = params.get("minor_mss_lookback", bbc["minor_mss_lookback"]) if params else bbc["minor_mss_lookback"],
    )

    # ── 2.2 FVG + sweep features ──────────────────────────────────────────────
    atr14   = _atr(df["high"], df["low"], df["close"], 14)
    fvg_df  = detect_fvg(df, atr14, fvg_cfg)
    fvg_bar = _compute_fvg_bar_features(df, fvg_df, atr14, fvg_cfg)

    # ── 2.3 Confirmation features ─────────────────────────────────────────────
    confirm = _compute_confirmation_features(df)

    # ── Assemble output ───────────────────────────────────────────────────────
    out = df.copy()
    out["mss_signal"]       = mss
    out["htf_bias"]         = htf_bias          # True 1D MSS direction, unmodified
    out["bias_bypass_long"] = bypass_active.astype(np.float32)
    out["ssl_sweep_long"]   = ssl_sweep_long    # Long-only SSL sweep (relaxed threshold)
    out["minor_mss_long"]   = minor_mss_long   # Long-only minor MSS (8-bar lookback)
    out = out.join(fvg_bar)
    out = out.join(confirm)

    # Asset tier flag — constant per asset, useful tree-split signal
    out["asset_tier_flag"] = 1 if is_high_vol(asset) else 0

    # Keep only the 19 feature columns
    out = out[FEATURE_COLS]

    # Drop warm-up rows that could not be fully computed
    n_before = len(out)
    out = out.dropna()
    n_after  = len(out)
    if n_before - n_after > 0:
        logger.debug(
            f"[{asset}] Dropped {n_before - n_after} warm-up rows "
            f"({n_after} usable candles remain)."
        )

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: compute ATR for external callers (backtest, scanner)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Convenience wrapper — compute ATR for a given period."""
    return _atr(df["high"], df["low"], df["close"], period)
