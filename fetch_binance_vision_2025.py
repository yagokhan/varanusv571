#!/usr/bin/env python3
"""
fetch_binance_vision_2025.py
Download Jan 2025 – Oct 2025 (cutoff Nov 01 2025) 1h OHLCV from data.binance.vision,
resample to 4h, and merge with existing parquet cache (Jan 2023–Dec 2024).

Fix: 2025 Binance monthly files use microseconds (16-digit open_time).
     We auto-detect and divide by 1000 before converting.
"""

import io, zipfile, time
import requests
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "varanus" / "data" / "cache"
CUTOFF     = pd.Timestamp("2025-11-01", tz="UTC")

ASSETS = [
    "ADA", "APT", "ARB", "ATOM", "AVAX",
    "DOT", "FIL", "ICP", "LINK", "NEAR",
    "OP",  "SUI", "TRX", "UNI",
]

# Months to download (2025 Jan–Oct)
MONTHS_2025 = [
    (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2025, 5),
    (2025, 6), (2025, 7), (2025, 8), (2025, 9), (2025, 10),
]

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def download_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    fname = f"{symbol}USDT-1h-{year}-{month:02d}.zip"
    url   = f"{BASE_URL}/{symbol}USDT/1h/{fname}"
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

    # Auto-detect microseconds vs milliseconds
    sample = df["open_time"].iloc[0]
    if sample > 1e15:          # microseconds
        df["open_time"] = df["open_time"] / 1000

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[df.index < CUTOFF]
    return df


def fetch_asset(asset: str) -> pd.DataFrame:
    frames = []
    for year, month in MONTHS_2025:
        chunk = download_month(asset, year, month)
        if not chunk.empty:
            frames.append(chunk)
        time.sleep(0.15)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    return df_1h.resample("4h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["close"])


def merge_and_save(new_df: pd.DataFrame, path: Path):
    if path.exists():
        old = pd.read_parquet(path)
        if not isinstance(old.index, pd.DatetimeIndex):
            old = old.set_index("timestamp")
        old.index = pd.to_datetime(old.index, utc=True)
        combined = pd.concat([old, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df
    combined.to_parquet(path)
    return combined


def run():
    print("=== Binance Vision 2025 Data Fetch (Jan–Oct) ===")
    print(f"    Cutoff : {CUTOFF.date()}")
    print(f"    Assets : {len(ASSETS)}\n")

    for asset in ASSETS:
        print(f"  [{asset}]", end=" ", flush=True)

        df_1h_new = fetch_asset(asset)
        if df_1h_new.empty:
            print("FAILED — no 2025 data")
            continue

        path_1h = CACHE_DIR / f"{asset}_USDT_1h.parquet"
        path_4h = CACHE_DIR / f"{asset}_USDT.parquet"

        # Save 1h
        df_1h_merged = merge_and_save(df_1h_new, path_1h)

        # Resample new 2025 chunk, then merge 4h
        df_4h_new    = resample_to_4h(df_1h_new)
        df_4h_merged = merge_and_save(df_4h_new, path_4h)

        print(f"4h: {len(df_4h_merged)} bars "
              f"| {df_4h_merged.index[0].date()} → {df_4h_merged.index[-1].date()}")

    print("\n=== Done ===")
    print("Verifying final cache:")
    for f in sorted(CACHE_DIR.glob("*_USDT.parquet")):
        if "_1h" in f.name:
            continue
        df = pd.read_parquet(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("timestamp")
        print(f"  {f.stem:<20} {len(df):>5} bars "
              f"| {df.index[0].date()} → {df.index[-1].date()}")


if __name__ == "__main__":
    run()
