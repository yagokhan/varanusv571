#!/usr/bin/env python3
"""
run_dual_engine_optimization.py — Varanus v5.2 Long Runner Optuna search (Deep Search).

Keeps Short Hunter params frozen from Trial #183.
Searches only 2 Long Runner parameters:
  conf_thresh_long  [0.55  - 0.72]
  tp_mult_long      [2.5   - 4.5]

sl_mult_long = 1.0 FIXED (Hard Stop guardrail — not searched).

Objective: Density Score = (WR × Count) / DD_Impact
Density gate: trials with < 50 long trades total → -999.0 penalty.
(Gate calibrated to model ceiling of ~56; raised to 60 in v5.2.1 after training imbalance fix.)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from varanus.universe import TIER2_UNIVERSE
from varanus.optimizer import (
    DUAL_ENGINE_OPTUNA_CONFIG,
    LONG_RUNNER_SEARCH_SPACE,
    V4_FROZEN_PARAMS,
    run_dual_engine_optimization,
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
        for col, fn in [("open","first"),("high","max"),("low","min"),
                        ("close","last"),("volume","sum")]:
            if col in df.columns:
                agg[col] = fn
        df = df.resample("1D").agg(agg).dropna()

    return df


def run():
    print("=" * 60)
    print("  Varanus v5.2 Dual-Engine — Long Runner Optimization")
    print("  Objective : Density Score = (WR x Count) / DD_Impact")
    print("  Short Hunt: FROZEN (Trial #183)")
    print("  Search    : conf_thresh_long, tp_mult_long")
    print("  sl_mult_long = 1.0 FIXED (Hard Stop guardrail)")
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

    print(f"\n[+] Long Runner search space:")
    for k, v in LONG_RUNNER_SEARCH_SPACE.items():
        print(f"    {k}: {v}")

    n_trials = DUAL_ENGINE_OPTUNA_CONFIG["n_trials"]
    print(f"\n[+] Starting {n_trials}-trial Long Runner search...\n")

    try:
        study = run_dual_engine_optimization(
            data_4h, data_1d,
            n_trials   = n_trials,
            study_name = "varanus_v52_dual_engine",
        )
    except KeyboardInterrupt:
        print("\n[!] Interrupted — saving best known params...")
        import optuna
        study = optuna.load_study(
            study_name = "varanus_v52_dual_engine",
            storage    = None,
        )

    print("\n" + "=" * 60)
    print(f"  Best Trial:              {study.best_trial.number}")
    print(f"  Best Long Runner Score:  {study.best_value:.4f}")

    best_long_params = study.best_params
    print(f"\n  Long Runner Params:")
    for k, v in best_long_params.items():
        print(f"    {k}: {v:.6f}")

    # Assemble full v5.2 param config.
    # sl_mult_long is fixed at 1.0 (Hard Stop guardrail) — not in study.best_params
    # since it was not searched. Must be injected explicitly.
    full_params = {
        **V4_FROZEN_PARAMS,
        "xgb_lr":        0.060884936946609944,
        "xgb_max_depth": 6,
        **V52_SHORT_FROZEN_PARAMS,
        **best_long_params,
        "sl_mult_long":  1.0,   # Deep Search Hard Stop guardrail
    }

    print(f"\n  Full v5.2 Configuration:")
    print(json.dumps(full_params, indent=4))

    out_dir  = str(_HERE / "varanus" / "config")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "best_params_v52.json")

    # Preserve comment fields
    out_data = {
        "_comment": "Varanus v5.2 Dual-Engine — Short Hunter FROZEN (Trial #183), Long Runner optimized",
        "_version": "5.2.0",
        **{k: v for k, v in full_params.items()
           if not k.startswith("_")},
        "_short_hunter_frozen": "Trial #183 — DO NOT MODIFY",
        "_long_runner": f"Optimized Trial #{study.best_trial.number}",
    }

    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=4)

    print(f"\n[+] Saved to {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    run()
