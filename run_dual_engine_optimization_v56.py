#!/usr/bin/env python3
"""
run_dual_engine_optimization_v56.py — Varanus v5.6 The Golden Ratio.

Fixes v5.5 Long WR slippage (39.6% → target >41%) while maintaining density.

Short Hunter FROZEN (Trial #183).
Searches 5 Long Runner parameters:
  conf_thresh_long     [0.55 – 0.70]   raised lower — filters noise entries
  tp_mult_long         [2.0  – 3.5]    tightened upper — drives 120-180 density band
  sl_mult_long         [0.70 – 1.20]   tightened upper — tighter R:R control
  rsi_1d_long_limit    [45  – 65]
  p_short_max_for_long [0.55 – 0.75]   below conf_thresh_short dead zone (0.786)

Objective: Density Score = (WR × Count × ln(Count+1)) / DD_Impact
Hard WR gate: 41% per fold (raised from 33%).
Density gate: 80 total long trades (targets 120–180 band).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from varanus.universe import TIER2_UNIVERSE
from varanus.optimizer import (
    DUAL_ENGINE_OPTUNA_CONFIG_V56,
    LONG_RUNNER_SEARCH_SPACE_V56,
    V4_FROZEN_PARAMS,
    run_v56_optimization,
)
from varanus.backtest import V52_SHORT_FROZEN_PARAMS

_HERE = Path(__file__).parent
CACHE = str(_HERE / "varanus" / "data" / "cache")


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
    print("  Varanus v5.6 — The Golden Ratio")
    print("  Objective : (WR × Count × ln(Count+1)) / DD_Impact")
    print("  WR Gate   : 41% per fold (hard)")
    print("  Density   : 80 min long trades (target 120–180)")
    print("  Short Hunt: FROZEN (Trial #183)")
    print("  conf_long : [0.55, 0.70]  (raised lower from 0.50)")
    print("  tp_mult   : [2.0,  3.5]   (tightened upper from 5.0)")
    print("  sl_mult   : [0.70, 1.20]  (tightened upper from 1.50)")
    print("  p_short   : [0.55, 0.75]  (below dead zone 0.786)")
    print("  Pruner    : MedianPruner  (no back-loaded fold killing)")
    print("=" * 60)

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

    print(f"\n[+] Frozen Short Hunter params (Trial #183):")
    for k, v in V52_SHORT_FROZEN_PARAMS.items():
        print(f"    {k}: {v}")

    print(f"\n[+] v5.6 Long Runner search space:")
    for k, v in LONG_RUNNER_SEARCH_SPACE_V56.items():
        print(f"    {k}: {v}")

    n_trials = DUAL_ENGINE_OPTUNA_CONFIG_V56["n_trials"]
    print(f"\n[+] Starting {n_trials}-trial v5.6 search...\n")

    study = run_v56_optimization(
        data_4h, data_1d,
        n_trials   = n_trials,
        study_name = "varanus_v56_golden_ratio",
    )

    print("\n" + "=" * 60)
    print(f"  Best Trial:             {study.best_trial.number}")
    print(f"  Best v5.6 Score:        {study.best_value:.4f}")

    best_params = study.best_params
    print(f"\n  Long Runner Params:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.6f}")
        else:
            print(f"    {k}: {v}")

    full_params = {
        **V4_FROZEN_PARAMS,
        "xgb_lr":        0.060884936946609944,
        "xgb_max_depth": 6,
        **V52_SHORT_FROZEN_PARAMS,
        **best_params,
    }

    print(f"\n  Full v5.6 Configuration:")
    print(json.dumps(full_params, indent=4))

    out_dir  = str(_HERE / "varanus" / "config")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "best_params_v56.json")

    out_data = {
        "_comment": "Varanus v5.6 The Golden Ratio — Short Hunter FROZEN (Trial #183), Long Runner v5.6",
        "_version": "5.6.0",
        **{k: v for k, v in full_params.items() if not k.startswith("_")},
        "_short_hunter_frozen": "Trial #183 — DO NOT MODIFY",
        "_long_runner": f"v5.6 Optimized Trial #{study.best_trial.number}",
    }

    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=4)

    print(f"\n[+] Saved to {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    run()
