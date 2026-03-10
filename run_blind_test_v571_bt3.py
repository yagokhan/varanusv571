#!/usr/bin/env python3
"""
run_blind_test_v571_bt3.py — Varanus v5.7.1 Blind Test 3

Tests best_params_v57.json on data the optimizer NEVER saw:
  Mar 01 2026 – Mar 09 2026

Data source: Binance Vision DAILY files (monthly not yet published).

Output:
  varanus/config/varanus_v571_blind3_trades.csv
  varanus/config/varanus_v571_blind3_summary.csv
"""
import io, json, sys, time, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import requests

from varanus.universe import TIER2_UNIVERSE
from varanus.model import VaranusDualModel, MODEL_CONFIG
from varanus.pa_features import build_features, compute_atr
from varanus.tbm_labeler import label_trades, TBM_CONFIG, build_dual_labels
from varanus.backtest import run_backtest, compute_metrics

_HERE       = Path(__file__).parent
CACHE_DIR   = _HERE / "varanus" / "data" / "cache"
PARAMS_FILE = _HERE / "varanus" / "config" / "best_params_v57.json"
OUT_TRADES  = _HERE / "varanus" / "config" / "varanus_v571_blind3_trades.csv"
OUT_SUMMARY = _HERE / "varanus" / "config" / "varanus_v571_blind3_summary.csv"

TRAIN_CUTOFF = pd.Timestamp("2025-11-01", tz="UTC")
BLIND_START  = pd.Timestamp("2026-03-01", tz="UTC")
BLIND_END    = pd.Timestamp("2026-03-09 23:59:59", tz="UTC")

# Daily file base URL (used because March 2026 monthly not yet published)
DAILY_URL = "https://data.binance.vision/data/spot/daily/klines"
COLS = ["open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]

BLIND_DAYS = pd.date_range("2026-03-01", "2026-03-09", freq="D")


# ── Data helpers ──────────────────────────────────────────────────────────────

def _download_day(asset, date):
    """Download a single daily 1h kline file from Binance Vision."""
    sym   = asset
    dstr  = date.strftime("%Y-%m-%d")
    fname = f"{sym}USDT-1h-{dstr}.zip"
    url   = f"{DAILY_URL}/{sym}USDT/1h/{fname}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=COLS)
    except Exception as e:
        print(f"    x {asset} {dstr}: {e}")
        return pd.DataFrame()
    sample = df["open_time"].iloc[0]
    if sample > 1e15:
        df["open_time"] = df["open_time"] / 1000
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")[["open","high","low","close","volume"]].astype(float)
    return df


def fetch_blind_data():
    """Download Mar 01–09 2026 1h data from daily files and merge into cache."""
    print("[+] Fetching blind test 3 data (Mar 01–09 2026, daily files) ...")
    for asset in TIER2_UNIVERSE:
        frames = []
        for date in BLIND_DAYS:
            chunk = _download_day(asset, date)
            if not chunk.empty:
                frames.append(chunk)
            time.sleep(0.08)

        if not frames:
            print(f"  {asset}: no data found")
            continue

        df_new_1h = pd.concat(frames).sort_index()
        df_new_1h = df_new_1h[~df_new_1h.index.duplicated(keep="last")]
        df_new_1h = df_new_1h[(df_new_1h.index >= BLIND_START) & (df_new_1h.index <= BLIND_END)]

        # Merge with existing 1h cache
        path_1h = CACHE_DIR / f"{asset}_USDT_1h.parquet"
        if path_1h.exists():
            old = pd.read_parquet(path_1h)
            old.index = pd.to_datetime(old.index, utc=True)
            df_1h = pd.concat([old, df_new_1h])
            df_1h = df_1h[~df_1h.index.duplicated(keep="last")].sort_index()
        else:
            df_1h = df_new_1h
        df_1h.to_parquet(path_1h)

        # Resample to 4h and merge
        df_new_4h = df_new_1h.resample("4h").agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna(subset=["close"])
        path_4h = CACHE_DIR / f"{asset}_USDT.parquet"
        if path_4h.exists():
            old4 = pd.read_parquet(path_4h)
            old4.index = pd.to_datetime(old4.index, utc=True)
            df_4h = pd.concat([old4, df_new_4h])
            df_4h = df_4h[~df_4h.index.duplicated(keep="last")].sort_index()
        else:
            df_4h = df_new_4h
        df_4h.to_parquet(path_4h)

        blind_4h = df_4h[(df_4h.index >= BLIND_START) & (df_4h.index <= BLIND_END)]
        if not blind_4h.empty:
            print(f"  {asset}: {len(blind_4h)} blind 4h bars "
                  f"| {blind_4h.index[0].date()} -> {blind_4h.index[-1].date()}")
        else:
            print(f"  {asset}: data fetched but no 4h bars in window")

    print()


def load_data(symbol, timeframe):
    file_symbol = "ASTER" if symbol == "ASTR" else symbol
    if timeframe == "1d":
        try:
            df = pd.read_parquet(CACHE_DIR / f"{file_symbol}_USDT_1h.parquet")
        except FileNotFoundError:
            df = pd.read_parquet(CACHE_DIR / f"{file_symbol}_USDT.parquet")
    else:
        df = pd.read_parquet(CACHE_DIR / f"{file_symbol}_USDT.parquet")

    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    ohlcv = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df = df.dropna(subset=ohlcv)

    if timeframe == "1d":
        agg = {c: f for c, f in [("open","first"),("high","max"),("low","min"),
                                   ("close","last"),("volume","sum")] if c in df.columns}
        df = df.resample("1D").agg(agg).dropna(subset=["close"])
    return df


def run():
    print("=" * 65)
    print("  Varanus v5.7.1 — Blind Test 3")
    print(f"  Train : Jan 2023 – Oct 2025 (full optimization region)")
    print(f"  Test  : Mar 01 2026 – Mar 09 2026 (never seen by optimizer)")
    print("  Dynamic Trailing Stop: TRAIL_TRIGGER=1.147% TRAIL_DIST=1.147%")
    print("=" * 65)

    fetch_blind_data()

    with open(PARAMS_FILE) as f:
        params = json.load(f)
    params = {k: v for k, v in params.items() if not k.startswith("_")}
    print(f"[+] Params: {PARAMS_FILE.name}")

    print("[+] Loading all data ...")
    data_4h_all, data_1d_all = {}, {}
    for asset in TIER2_UNIVERSE:
        try:
            df_4h = load_data(asset, "4h")
            df_1d = load_data(asset, "1d")
            df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
            data_4h_all[asset] = df_4h
            data_1d_all[asset] = df_1d
        except Exception as e:
            print(f"  Skipping {asset}: {e}")

    train_4h = {a: df[df.index < TRAIN_CUTOFF] for a, df in data_4h_all.items()}
    train_1d = {a: df[df.index < TRAIN_CUTOFF] for a, df in data_1d_all.items()}
    blind_4h = {a: df[(df.index >= BLIND_START) & (df.index <= BLIND_END)]
                for a, df in data_4h_all.items()}
    blind_4h = {a: df for a, df in blind_4h.items() if not df.empty}
    blind_1d = {a: df[(df.index >= BLIND_START) & (df.index <= BLIND_END)]
                for a, df in data_1d_all.items()}
    blind_1d = {a: df for a, df in blind_1d.items() if not df.empty}

    print(f"  Train assets : {len([a for a, df in train_4h.items() if not df.empty])}")
    print(f"  Blind assets : {len(blind_4h)}")
    if blind_4h:
        sample = next(iter(blind_4h.values()))
        print(f"  Blind window : {sample.index[0].date()} -> {sample.index[-1].date()} ({len(sample)} bars/asset)")

    print("\n[+] Training final model on full optimization data (Jan 2023 – Oct 2025) ...")
    X_tr_list, y_tr_list, y_short_tr_list = [], [], []

    for asset in train_4h:
        if asset not in train_1d: continue
        df4 = train_4h[asset]
        df1 = train_1d[asset]
        if df4.empty or df1.empty: continue
        X = build_features(df4, df1, asset, params)
        if X.empty: continue
        X = X.dropna()
        if X.empty: continue
        y = build_dual_labels(df4, X, {**params, "_asset": asset})
        y = y.reindex(X.index).fillna(0).astype(int)
        y_short = label_trades(df4.loc[X.index], X["mss_signal"], TBM_CONFIG, asset, params)
        y_short = y_short.reindex(X.index).fillna(0).astype(int)
        X_tr_list.append(X); y_tr_list.append(y); y_short_tr_list.append(y_short)

    if not X_tr_list:
        print("ERROR: No training data built.")
        return

    model_cfg = {**MODEL_CONFIG}
    model_cfg["xgb_params"] = {
        **MODEL_CONFIG["xgb_params"],
        "max_depth":     params["xgb_max_depth"],
        "learning_rate": params["xgb_lr"],
        "n_estimators":  params["xgb_n_estimators"],
        "subsample":     params["xgb_subsample"],
    }
    model = VaranusDualModel(model_cfg)
    model.fit(
        pd.concat(X_tr_list), pd.concat(y_tr_list),
        None, None,
        pd.concat(y_short_tr_list), None,
    )
    print(f"  Model trained on {sum(len(x) for x in X_tr_list):,} samples from {len(X_tr_list)} assets.")

    print("\n[+] Generating signals on blind test window ...")
    signals = {}
    _conf_long   = params["conf_thresh_long"]
    _conf_short  = params.get("conf_thresh_short", 0.786)
    _p_short_max = params["p_short_max_for_long"]

    for asset in blind_4h:
        if asset not in data_4h_all or asset not in data_1d_all: continue
        # Use full history for feature lookback, but predict only on blind window rows
        df_4h_full = data_4h_all[asset]
        df_1d_full = data_1d_all[asset]
        X_all = build_features(df_4h_full, df_1d_full, asset, params)
        if X_all.empty: continue
        X_all = X_all.dropna()
        if X_all.empty: continue
        # Filter to blind window
        X_t = X_all[(X_all.index >= BLIND_START) & (X_all.index <= BLIND_END)]
        if X_t.empty: continue
        df_4h = blind_4h[asset]
        probs   = model.predict_proba(X_t)
        p_short = probs[:, 0]
        p_long  = probs[:, 2]
        direction = np.zeros(len(X_t), dtype=int)
        direction[(p_long  >= _conf_long)  & (p_short <= _p_short_max)] =  1
        direction[(p_short >= _conf_short) & (p_short >= p_long)]       = -1
        active = direction != 0
        if not active.any(): continue
        idx_active = X_t.index[active]
        sig_df = pd.DataFrame(index=idx_active)
        sig_df["confidence"]  = np.where(direction[active]==1, p_long[active], p_short[active])
        sig_df["direction"]   = direction[active]
        sig_df["entry_price"] = df_4h.loc[idx_active, "close"].values
        sig_df["atr"]         = compute_atr(df_4h.loc[X_t.index], 14).loc[idx_active].values
        signals[asset] = sig_df
        print(f"  {asset}: {(direction==1).sum()} long + {(direction==-1).sum()} short signals")

    if not signals:
        print("No signals generated in blind window.")
        return

    print("\n[+] Running blind test 3 backtest ...")
    equity, trades = run_backtest(blind_4h, signals, model, params)
    metrics = compute_metrics(equity, trades)

    long_trades  = trades[trades["direction"] ==  1]
    short_trades = trades[trades["direction"] == -1]
    initial_cap  = 5_000.0

    total_pnl  = trades["pnl_usd"].sum()
    total_ret  = total_pnl / initial_cap * 100
    overall_wr = (trades["pnl_usd"] > 0).mean() * 100
    long_wr    = (long_trades["pnl_usd"] > 0).mean() * 100 if not long_trades.empty else 0.0
    short_wr   = (short_trades["pnl_usd"] > 0).mean() * 100 if not short_trades.empty else 0.0

    wins = trades["pnl_usd"] > 0
    loss = trades["pnl_usd"] < 0
    pf   = abs(trades.loc[wins,"pnl_usd"].sum() / trades.loc[loss,"pnl_usd"].sum()) if loss.any() else float("inf")
    avg_win  = trades.loc[wins,"pnl_usd"].mean() if wins.any() else 0.0
    avg_loss = trades.loc[loss,"pnl_usd"].mean() if loss.any() else 0.0
    expectancy = overall_wr/100 * avg_win + (1-overall_wr/100) * avg_loss
    by_outcome = trades["outcome"].value_counts().to_dict()

    summary = {
        "period":             "Mar 01 2026 – Mar 09 2026",
        "total_trades":       len(trades),
        "long_trades":        len(long_trades),
        "short_trades":       len(short_trades),
        "total_return_pct":   round(total_ret, 2),
        "net_profit_usd":     round(total_pnl, 2),
        "win_rate_pct":       round(overall_wr, 2),
        "long_win_rate_pct":  round(long_wr, 2),
        "short_win_rate_pct": round(short_wr, 2),
        "profit_factor":      round(pf, 3),
        "expectancy_usd":     round(expectancy, 2),
        "avg_win_usd":        round(avg_win, 2),
        "avg_loss_usd":       round(avg_loss, 2),
        "max_drawdown_pct":   metrics["max_drawdown_pct"],
        "sharpe_ratio":       metrics["sharpe_ratio"],
        **{f"exits_{k}": v for k, v in by_outcome.items()},
    }

    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUT_TRADES, index=False)
    pd.DataFrame([summary]).to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 65)
    print("  BLIND TEST 3 RESULTS — v5.7.1")
    print(f"  Period : Mar 01 2026 – Mar 09 2026 (9 days)")
    print("=" * 65)
    print(f"\n  Total Trades  : {len(trades)}  (L:{len(long_trades)}  S:{len(short_trades)})")
    print(f"  Overall WR    : {overall_wr:.1f}%")
    print(f"  Long WR       : {long_wr:.1f}%  ({'PASS' if long_wr >= 41 else 'FAIL'} — gate >=41%)")
    print(f"  Short WR      : {short_wr:.1f}%")
    print(f"  Total Return  : {total_ret:.1f}%  (${total_pnl:,.2f} on ${initial_cap:,.0f})")
    print(f"  Profit Factor : {pf:.2f}")
    print(f"  Expectancy    : ${expectancy:.2f}/trade")
    print(f"  Avg Win       : ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print(f"  Max DrawDown  : {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe        : {metrics['sharpe_ratio']:.3f}")
    print(f"\n  Exit Breakdown:")
    for k, v in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"    {k:<24}: {v}")
    print(f"\n[+] Trades saved : {OUT_TRADES.name}")
    print(f"[+] Summary saved: {OUT_SUMMARY.name}")
    print("=" * 65)


if __name__ == "__main__":
    run()
