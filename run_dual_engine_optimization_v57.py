#!/usr/bin/env python3
"""
run_dual_engine_optimization_v57.py — Varanus v5.7 Anti-Overfit Edition.

Fixes v5.6 selection bias (5 repeated OOS windows → inflated +890.6% return)
by switching to 8 explicit-date rolling folds across Jan 2023 – Oct 2025.

Short Hunter FROZEN (Trial #183).
Searches 5 Long Runner parameters (identical search space to v5.6):
  conf_thresh_long     [0.55 – 0.70]
  tp_mult_long         [2.0  – 3.5]
  sl_mult_long         [0.70 – 1.20]
  rsi_1d_long_limit    [45  – 65]
  p_short_max_for_long [0.55 – 0.75]

Gates (v5.7 tightened):
  - Per-fold:  < 5 long trades in any fold → trial discarded (-999)
  - Combined:  < 80 long trades total → trial discarded (-999)
  - WR:        fold long WR < 41% → fold score -10 (soft)
  - Profit:    any fold net PnL ≤ 0 → -100 subtracted from final score
  - Pruner:    MedianPruner (n_startup=10, n_warmup=3) — prune weak trials early

Objective: Density Score = (WR × Count × ln(Count+1)) / DD_Impact
Hard cutoff: Nov 01 2025 — blind test region never touched.

Output: varanus/config/best_params_v57.json
Log:    logs/dual_engine_opt_v57.log
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from varanus.universe import TIER2_UNIVERSE
from varanus.optimizer import (
    DUAL_ENGINE_OPTUNA_CONFIG_V57,
    LONG_RUNNER_SEARCH_SPACE_V57,
    V4_FROZEN_PARAMS,
    run_v57_optimization,
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
        # Handle both named index ('timestamp') and unnamed integer index
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Drop rows with missing OHLCV — prevents NaN propagation into feature builder
    ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df.dropna(subset=ohlcv)

    if timeframe == "1d":
        agg = {}
        for col, fn in [("open", "first"), ("high", "max"), ("low", "min"),
                        ("close", "last"), ("volume", "sum")]:
            if col in df.columns:
                agg[col] = fn
        df = df.resample("1D").agg(agg).dropna(subset=["close"])

    return df


def run():
    print("=" * 65)
    print("  Varanus v5.7 — Anti-Overfit Edition")
    print("  8-Fold Explicit-Date WFV (Jan 2023 – Oct 2025)")
    print("  Hard Cutoff : Nov 01 2025 (blind test region protected)")
    print("  Objective   : (WR × Count × ln(Count+1)) / DD_Impact")
    print("  Per-Fold    : < 5 long trades → trial -999 (hard)")
    print("  Combined    : < 80 long trades total → trial -999 (hard)")
    print("  WR Gate     : 41% per fold (soft -10 penalty)")
    print("  Profit Gate : any fold PnL ≤ 0 → -100 final score penalty")
    print("  Pruner      : MedianPruner (startup=10, warmup=3 folds)")
    print("  Short Hunt  : FROZEN (Trial #183)")
    print("  conf_long   : [0.55, 0.70]")
    print("  tp_mult     : [2.0,  3.5]")
    print("  sl_mult     : [0.70, 1.20]")
    print("  p_short     : [0.55, 0.75]")
    print("=" * 65)

    print("\n[+] Loading universe data ...")
    data_4h, data_1d = {}, {}
    for asset in TIER2_UNIVERSE:
        try:
            df_4h = load_data(asset, "4h")
            df_1d = load_data(asset, "1d")
            df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
            data_4h[asset] = df_4h
            data_1d[asset] = df_1d
            print(f"  {asset}: {len(df_4h)} x 4h bars "
                  f"| {df_4h.index[0].date()} → {df_4h.index[-1].date()}")
        except Exception as e:
            print(f"  Skipping {asset}: {e}")

    print(f"\n[+] Loaded {len(data_4h)} assets.")

    print(f"\n[+] Frozen Short Hunter params (Trial #183):")
    for k, v in V52_SHORT_FROZEN_PARAMS.items():
        print(f"    {k}: {v}")

    print(f"\n[+] v5.7 Long Runner search space:")
    for k, v in LONG_RUNNER_SEARCH_SPACE_V57.items():
        print(f"    {k}: {v}")

    n_trials = DUAL_ENGINE_OPTUNA_CONFIG_V57["n_trials"]
    print(f"\n[+] Starting {n_trials}-trial v5.7 search ...\n")

    study = run_v57_optimization(
        data_4h, data_1d,
        n_trials   = n_trials,
        study_name = "varanus_v57_antioverfits",
    )

    print("\n" + "=" * 65)
    print(f"  Best Trial  : {study.best_trial.number}")
    print(f"  Best Score  : {study.best_value:.4f}")

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

    print(f"\n  Full v5.7 Configuration:")
    print(json.dumps(full_params, indent=4))

    out_dir  = str(_HERE / "varanus" / "config")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "best_params_v57.json")

    out_data = {
        "_comment": "Varanus v5.7 Anti-Overfit — Short Hunter FROZEN (Trial #183), Long Runner v5.7",
        "_version": "5.7.0",
        **{k: v for k, v in full_params.items() if not k.startswith("_")},
        "_short_hunter_frozen": "Trial #183 — DO NOT MODIFY",
        "_long_runner": f"v5.7 Optimized Trial #{study.best_trial.number}",
    }

    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=4)

    print(f"\n[+] Saved to {out_file}")
    print("=" * 65)


if __name__ == "__main__":
    run()
