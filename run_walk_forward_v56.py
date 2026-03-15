#!/usr/bin/env python3
"""
run_walk_forward_v52.py — Varanus v5.6 Dual-Engine Walk-Forward Validation
Validates optimized best_params_v56.json across 5-fold rolling OOS windows.
Gate: long_win_rate >= 40%, consistency >= 75%.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from varanus.universe import TIER2_UNIVERSE
from varanus.walk_forward import run_walk_forward, WFV_CONFIG_V51

_HERE = Path(__file__).parent
CACHE = str(_HERE / "varanus" / "data" / "cache")
PARAMS_FILE = _HERE / "varanus" / "config" / "best_params_v56.json"


def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    file_symbol = "ASTER" if symbol == "ASTR" else symbol
    if timeframe == "1d":
        try:
            df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT_1h.parquet")
        except FileNotFoundError:
            df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT.parquet")
    elif timeframe == "4h":
        df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT.parquet")
    else:
        df = pd.read_parquet(f"{CACHE}/{file_symbol}_USDT_{timeframe}.parquet")

    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    if timeframe == "1d":
        agg = {}
        for col, fn in [("open", "first"), ("high", "max"), ("low", "min"),
                        ("close", "last"), ("volume", "sum")]:
            if col in df.columns:
                agg[col] = fn
        df = df.resample("1D").agg(agg).dropna()

    return df


def run():
    print("=" * 60)
    print("  Varanus v5.6 Dual-Engine — Walk-Forward Validation")
    print("  Gate : long_win_rate >= 40%, consistency >= 75%")
    print("=" * 60)

    with open(PARAMS_FILE) as f:
        params = json.load(f)
    # Strip comment fields
    params = {k: v for k, v in params.items() if not k.startswith("_")}

    print(f"\n[+] Loaded params from {PARAMS_FILE.name}")
    for k, v in params.items():
        print(f"    {k}: {v}")

    print("\n[+] Loading universe data...")
    data_4h, data_1d = {}, {}
    for asset in TIER2_UNIVERSE:
        try:
            df_4h = load_data(asset, "4h")
            df_1d = load_data(asset, "1d")
            df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
            data_4h[asset] = df_4h
            data_1d[asset] = df_1d
            print(f"  {asset}: {len(df_4h)} x 4h candles")
        except Exception as e:
            print(f"  Skipping {asset}: {e}")

    print(f"\n[+] Loaded {len(data_4h)} assets.")
    print(f"\n[+] Running 5-fold walk-forward validation...\n")

    results_df, consistency, all_trades = run_walk_forward(
        data_4h, data_1d, params, cfg=WFV_CONFIG_V51
    )

    print("\n" + "=" * 60)
    print("  FINAL WALK-FORWARD RESULTS")
    print("=" * 60)

    if results_df.empty:
        print("  No fold results — insufficient data or signals.")
        return

    gate_pass = consistency >= WFV_CONFIG_V51["performance_gate"]["consistency_req"]
    print(f"\n  Consistency : {consistency:.0%}  ({'PASS' if gate_pass else 'FAIL'})")

    if not all_trades.empty:
        long_trades  = all_trades[all_trades['direction'] ==  1] if 'direction' in all_trades.columns else pd.DataFrame()
        short_trades = all_trades[all_trades['direction'] == -1] if 'direction' in all_trades.columns else pd.DataFrame()

        total_wr = (all_trades['pnl_usd'] > 0).mean() * 100 if not all_trades.empty else 0
        long_wr  = (long_trades['pnl_usd'] > 0).mean()  * 100 if not long_trades.empty  else 0
        short_wr = (short_trades['pnl_usd'] > 0).mean() * 100 if not short_trades.empty else 0

        print(f"\n  Total Trades : {len(all_trades)}  (L:{len(long_trades)} S:{len(short_trades)})")
        print(f"  Overall WR   : {total_wr:.1f}%")
        print(f"  Long WR      : {long_wr:.1f}%  ({'PASS' if long_wr >= 40 else 'FAIL'} — gate: >=40%)")
        print(f"  Short WR     : {short_wr:.1f}%")
        print(f"  Total PnL    : ${all_trades['pnl_usd'].sum():.2f}")

    print("\n  Per-Fold Summary:")
    cols = ['fold', 'total_return_pct', 'win_rate_pct', 'long_win_rate_pct',
            'short_win_rate_pct', 'max_drawdown_pct', 'sharpe_ratio',
            'hunter_efficiency', 'total_trades']
    print(results_df[[c for c in cols if c in results_df.columns]].to_string(index=False))

    # Save fold summary
    out_csv = _HERE / "varanus" / "config" / "wfv_v56_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[+] Results saved to {out_csv.name}")

    # Save individual trades
    if not all_trades.empty:
        out_trades = _HERE / "varanus" / "config" / "wfv_v56_trades.csv"
        all_trades.to_csv(out_trades, index=False)
        print(f"[+] Trades saved to {out_trades.name}")
    print("=" * 60)


if __name__ == "__main__":
    run()
