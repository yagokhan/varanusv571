#!/usr/bin/env python3
"""
fetch_historical_data.py — Download 2 years of 4h OHLCV via Yahoo Finance (1h→4h resample).
Falls back to extending with existing cache data.

Saves to: varanus/data/cache/{SYMBOL}_USDT.parquet    (4h)
          varanus/data/cache/{SYMBOL}_USDT_1h.parquet  (1h)
"""

import sys, time, datetime
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import yfinance as yf
import pandas as pd
from pathlib import Path

from varanus.universe import TIER2_UNIVERSE

CACHE_DIR = Path(__file__).parent / "varanus" / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo Finance 1h data: max 730 days back
START_DATE = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=729)).strftime('%Y-%m-%d')

# Yahoo Finance ticker map
YAHOO_MAP = {
    "ADA":  "ADA-USD",  "AVAX": "AVAX-USD", "LINK": "LINK-USD",
    "DOT":  "DOT-USD",  "TRX":  "TRX-USD",  "NEAR": "NEAR-USD",
    "UNI":  "UNI-USD",  "SUI":  "SUI-USD",  "ARB":  "ARB-USD",
    "OP":   "OP-USD",   "POL":  "POL-USD",  "APT":  "APT-USD",
    "ATOM": "ATOM-USD", "FIL":  "FIL-USD",  "ICP":  "ICP-USD",
}

def fetch_and_resample(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker = YAHOO_MAP.get(asset)
    if not ticker:
        return pd.DataFrame(), pd.DataFrame()

    # Fetch 1h data (up to 730 days)
    raw = yf.download(ticker, start=START_DATE, interval="1h", progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Flatten MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.sort_index()

    # Resample to 4h
    df_4h = raw.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["close"])

    return df_4h, raw  # (4h, 1h)

def merge_with_existing(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge new data with existing parquet, new data takes precedence."""
    if not path.exists() or new_df.empty:
        return new_df
    try:
        old = pd.read_parquet(path)
        old.columns = [c.lower() for c in old.columns]
        if not isinstance(old.index, pd.DatetimeIndex):
            old = old.set_index("timestamp")
        old.index = pd.to_datetime(old.index, utc=True)
        # Combine: old for earlier dates, new for overlap onwards
        combined = pd.concat([old, new_df])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined
    except Exception as e:
        print(f"    Warning: could not merge with existing: {e}")
        return new_df

def run():
    print("=== Varanus v5.1 — Yahoo Finance Historical Data Fetch ===")
    print(f"    Source   : Yahoo Finance (1h → resample to 4h)")
    print(f"    From     : {START_DATE}  (~730 days)")
    print(f"    Assets   : {len(TIER2_UNIVERSE)}")
    print(f"    Cache    : {CACHE_DIR}\n")

    results = {}
    for asset in TIER2_UNIVERSE:
        file_sym = "ASTER" if asset == "ASTR" else asset
        path_4h  = CACHE_DIR / f"{file_sym}_USDT.parquet"
        path_1h  = CACHE_DIR / f"{file_sym}_USDT_1h.parquet"

        print(f"  [{asset}] Fetching...", end=" ", flush=True)
        df_4h, df_1h = fetch_and_resample(asset)

        if df_4h.empty:
            print("FAILED (no data)")
            results[asset] = 0
            continue

        # Merge with existing older cache data to extend history
        df_4h = merge_with_existing(df_4h, path_4h)
        df_1h = merge_with_existing(df_1h, path_1h)

        df_4h.to_parquet(path_4h)
        df_1h.to_parquet(path_1h)

        print(f"OK — {len(df_4h)} x 4h candles | {df_4h.index[0].date()} → {df_4h.index[-1].date()}")
        results[asset] = len(df_4h)
        time.sleep(0.3)

    print(f"\n=== Summary ===")
    ok = sum(1 for v in results.values() if v > 0)
    print(f"  Fetched : {ok}/{len(TIER2_UNIVERSE)} assets")
    avg = sum(results.values()) / max(ok, 1)
    print(f"  Avg 4h candles : {avg:.0f}  (~{avg*4/24:.0f} days)")

    # Check if enough for 8-fold WFV
    min_c = min(v for v in results.values() if v > 0) if ok else 0
    n_folds, t_r, v_r, s_r, gap = 8, 0.4, 0.3, 0.3, 24
    fw = int(min_c / (n_folds * s_r + t_r + v_r))
    train_len = int(fw * t_r) - gap
    print(f"\n  Min candles : {min_c}  →  est. train_len = {train_len}  (need ≥ 800 for 8-fold)")
    if train_len < 800:
        needed = int((800 + gap) / t_r * (n_folds * s_r + t_r + v_r))
        print(f"  ⚠  Insufficient for 8-fold. Need {needed} candles.")
        # Suggest adjusted n_folds
        for nf in [5, 4, 3]:
            fw2 = int(min_c / (nf * s_r + t_r + v_r))
            tl2 = int(fw2 * t_r) - gap
            if tl2 >= 500:
                print(f"  ✓  {nf}-fold WFV works: train_len={tl2}")
                break

if __name__ == "__main__":
    run()
