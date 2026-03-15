import optuna
import json
import os
import sys
import pandas as pd

from varanus.universe import TIER2_UNIVERSE
from varanus.optimizer import HUNTER_OPTUNA_CONFIG, run_hunter_optimization, V4_FROZEN_PARAMS

_HERE  = __import__('pathlib').Path(__file__).parent
CACHE  = str(_HERE / "varanus" / "data" / "cache")

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
        agg_dict = {}
        if 'open' in df.columns: agg_dict['open'] = 'first'
        if 'high' in df.columns: agg_dict['high'] = 'max'
        if 'low' in df.columns: agg_dict['low'] = 'min'
        if 'close' in df.columns: agg_dict['close'] = 'last'
        if 'volume' in df.columns: agg_dict['volume'] = 'sum'
        df = df.resample('1D').agg(agg_dict).dropna()
        
    return df

def run():
    print("=== Varanus v5.1 Hunter Optimization ===")
    print("    Objective: Maximize Hunter Efficiency (Net Profit / |MaxDD|)")
    print("    Structure: 5-fold rolling WFV, 40/30/30 split, -12% DD penalty\n")

    # 1. Load Parquet Data
    print("[+] Loading Universe Data...")
    data_4h = {}
    data_1d = {}

    for asset in TIER2_UNIVERSE:
        try:
            df_4h = load_data(asset, "4h")
            df_1d = load_data(asset, "1d")
            df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
            data_4h[asset] = df_4h
            data_1d[asset] = df_1d
            print(f"  Loaded {asset}: {len(df_4h)} x 4h candles")
        except Exception as e:
            print(f"  Skipping {asset}: {e}")

    print(f"\n[+] Loaded {len(data_4h)} assets.")

    # 2. Run Hunter Optimization
    print(f"\n[+] Commencing {HUNTER_OPTUNA_CONFIG['n_trials']}-trial Hunter search...\n")
    try:
        study = run_hunter_optimization(
            data_4h, data_1d,
            n_trials   = HUNTER_OPTUNA_CONFIG['n_trials'],
            study_name = "varanus_v51_hunter",
        )
    except KeyboardInterrupt:
        print("\n[!] Interrupted — saving best known params...")
        import optuna
        study = optuna.load_study(
            study_name   = "varanus_v51_hunter",
            storage      = None,
        )

    # 3. Report
    print("\n=== Optimization Complete ===")
    print(f"Best Trial:            {study.best_trial.number}")
    print(f"Best Hunter Efficiency: {study.best_value:.3f}")

    # Merge searched params with frozen v4 params for a complete config
    best_params = {**V4_FROZEN_PARAMS, **study.best_params}
    print(f"\nFull Parameter Configuration:\n{json.dumps(best_params, indent=4)}")

    # 4. Save to config/best_params_v51.json (keep v4 file untouched)
    out_dir  = str(_HERE / "varanus" / "config")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "best_params_v51.json")

    with open(out_file, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\n[+] Saved to {out_file}")

if __name__ == "__main__":
    run()
