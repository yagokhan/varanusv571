#!/usr/bin/env python3
"""
fetch_all_data_v57.py
Download Jan 2023 – Oct 2025 (hard cutoff Nov 01 2025) 1h OHLCV
from data.binance.vision, resample to 4h, save to cache as parquet.

Run this once before optimization. Takes ~15–20 min on a normal connection.
"""

import io, sys, zipfile, time
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from varanus.universe import TIER2_UNIVERSE as ASSETS

CACHE_DIR = Path(__file__).parent / "varanus" / "data" / "cache"
CUTOFF    = pd.Timestamp("2025-11-01", tz="UTC")

# Full range: Jan 2023 – Oct 2025
MONTHS = (
    [(2023, m) for m in range(1, 13)] +
    [(2024, m) for m in range(1, 13)] +
    [(2025, m) for m in range(1, 11)]
)

# POL replaced MATIC on Binance ~Sep 2024 on Binance Vision
# Months before this cutover: use MATIC symbol; from this month: use POL
POL_CUTOVER = pd.Timestamp("2024-09-01", tz="UTC")

# Assets with non-Jan-2023 start dates
ASSET_START = {
    "ARB": pd.Timestamp("2023-03-01", tz="UTC"),
    "SUI": pd.Timestamp("2023-05-01", tz="UTC"),
}

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _binance_symbol(asset: str, year: int, month: int) -> str:
    """Return the actual Binance Vision symbol for a given month.
    POL replaced MATIC on Binance Vision from Sep 2024 onward.
    """
    if asset == "POL" and pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC") < POL_CUTOVER:
        return "MATIC"
    return asset


def download_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    bsymbol = _binance_symbol(symbol, year, month)
    fname = f"{bsymbol}USDT-1h-{year}-{month:02d}.zip"
    url   = f"{BASE_URL}/{bsymbol}USDT/1h/{fname}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=COLS)
    except Exception as e:
        print(f"      ✗ {symbol} {year}-{month:02d}: {e}")
        return pd.DataFrame()

    # Auto-detect microseconds vs milliseconds (2025 files use microseconds)
    sample = df["open_time"].iloc[0]
    if sample > 1e15:
        df["open_time"] = df["open_time"] / 1000

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[df.index < CUTOFF]
    return df


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    return df_1h.resample("4h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["close"])


def run():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Varanus v5.7 — Full Data Fetch (Jan 2023 – Oct 2025) ===")
    print(f"    Cutoff  : {CUTOFF.date()}")
    print(f"    Assets  : {len(ASSETS)}")
    print(f"    Months  : {len(MONTHS)} per asset (minus early-launch assets)")
    print()

    for asset in ASSETS:
        asset_start = ASSET_START.get(asset, pd.Timestamp("2023-01-01", tz="UTC"))
        months_for_asset = [
            (y, m) for y, m in MONTHS
            if pd.Timestamp(f"{y}-{m:02d}-01", tz="UTC") >= asset_start
        ]

        print(f"  [{asset}] {len(months_for_asset)} months ...", end=" ", flush=True)

        frames = []
        for year, month in months_for_asset:
            chunk = download_month(asset, year, month)
            if not chunk.empty:
                frames.append(chunk)
            time.sleep(0.10)

        if not frames:
            print("FAILED — no data downloaded")
            continue

        df_1h = pd.concat(frames).sort_index()
        df_1h = df_1h[~df_1h.index.duplicated(keep="last")]

        path_1h = CACHE_DIR / f"{asset}_USDT_1h.parquet"
        path_4h = CACHE_DIR / f"{asset}_USDT.parquet"

        df_1h.to_parquet(path_1h)

        df_4h = resample_to_4h(df_1h)
        df_4h.to_parquet(path_4h)

        print(f"4h: {len(df_4h)} bars | {df_4h.index[0].date()} → {df_4h.index[-1].date()}")

    print("\n=== Done ===")
    print("\nCache summary:")
    for f in sorted(CACHE_DIR.glob("*_USDT.parquet")):
        if "_1h" in f.name:
            continue
        df = pd.read_parquet(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        print(f"  {f.stem:<20} {len(df):>5} bars | {df.index[0].date()} → {df.index[-1].date()}")


if __name__ == "__main__":
    run()
