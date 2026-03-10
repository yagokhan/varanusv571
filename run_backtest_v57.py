#!/usr/bin/env python3
"""
run_backtest_v57.py — Varanus v5.7 Full Backtest
Runs 8-fold explicit-date WFV, collects all OOS trades, computes metrics.
Outputs:
  varanus/config/varanusv57_backtest_trades.csv   — full trade log
  varanus/config/varanusv57_backtest_summary.csv  — fold + aggregate metrics
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from varanus.universe import TIER2_UNIVERSE
from varanus.walk_forward import run_walk_forward_v57, WFV_CONFIG_V57
from varanus.backtest import compute_metrics

_HERE       = Path(__file__).parent
CACHE       = str(_HERE / "varanus" / "data" / "cache")
PARAMS_FILE = _HERE / "varanus" / "config" / "best_params_v57.json"
OUT_TRADES  = _HERE / "varanus" / "config" / "varanusv57_backtest_trades.csv"
OUT_SUMMARY = _HERE / "varanus" / "config" / "varanusv57_backtest_summary.csv"


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
    print("=" * 65)
    print("  Varanus v5.7 — Full Backtest (8-Fold Explicit-Date WFV)")
    print("  Short Hunter FROZEN (Trial #183) + Long Runner v5.7")
    print("  Hard Cutoff: Nov 01 2025")
    print("=" * 65)

    if not PARAMS_FILE.exists():
        print(f"\n[!] Params file not found: {PARAMS_FILE}")
        print("    Run run_dual_engine_optimization_v57.py first.")
        return

    with open(PARAMS_FILE) as f:
        params = json.load(f)
    params = {k: v for k, v in params.items() if not k.startswith("_")}

    print(f"\n[+] Params: {PARAMS_FILE.name}")
    for k, v in params.items():
        print(f"    {k}: {v}")

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
    print(f"\n[+] Running 8-fold v5.7 walk-forward backtest ...\n")

    results_df, consistency, all_trades = run_walk_forward_v57(
        data_4h, data_1d, params, cfg=WFV_CONFIG_V57
    )

    if results_df.empty or all_trades.empty:
        print("No results generated.")
        return

    # ── Aggregate metrics across all folds ──────────────────────────────────
    initial_capital = 5_000.0
    long_trades  = all_trades[all_trades["direction"] ==  1]
    short_trades = all_trades[all_trades["direction"] == -1]

    total_pnl   = all_trades["pnl_usd"].sum()
    total_ret   = total_pnl / initial_capital * 100
    overall_wr  = (all_trades["pnl_usd"] > 0).mean() * 100
    long_wr     = (long_trades["pnl_usd"] > 0).mean() * 100 if not long_trades.empty else 0.0
    short_wr    = (short_trades["pnl_usd"] > 0).mean() * 100 if not short_trades.empty else 0.0
    long_pnl    = long_trades["pnl_usd"].sum() if not long_trades.empty else 0.0
    short_pnl   = short_trades["pnl_usd"].sum() if not short_trades.empty else 0.0

    wins = all_trades["pnl_usd"] > 0
    loss = all_trades["pnl_usd"] < 0
    profit_factor = (
        abs(all_trades.loc[wins, "pnl_usd"].sum() / all_trades.loc[loss, "pnl_usd"].sum())
        if loss.any() else float("inf")
    )

    avg_win  = all_trades.loc[wins, "pnl_usd"].mean() if wins.any() else 0.0
    avg_loss = all_trades.loc[loss, "pnl_usd"].mean() if loss.any() else 0.0
    expectancy = overall_wr / 100 * avg_win + (1 - overall_wr / 100) * avg_loss

    by_outcome = all_trades["outcome"].value_counts().to_dict()

    aggregate = {
        "fold":                    "ALL",
        "total_trades":            len(all_trades),
        "long_trades":             len(long_trades),
        "short_trades":            len(short_trades),
        "total_return_pct":        round(total_ret, 2),
        "net_profit_usd":          round(total_pnl, 2),
        "win_rate_pct":            round(overall_wr, 2),
        "long_win_rate_pct":       round(long_wr, 2),
        "short_win_rate_pct":      round(short_wr, 2),
        "long_net_pnl_usd":        round(long_pnl, 2),
        "short_net_pnl_usd":       round(short_pnl, 2),
        "profit_factor":           round(profit_factor, 3),
        "avg_win_usd":             round(avg_win, 2),
        "avg_loss_usd":            round(avg_loss, 2),
        "expectancy_usd":          round(expectancy, 2),
        "tp_hits":                 by_outcome.get("tp", 0),
        "sl_hits":                 by_outcome.get("sl", 0),
        "time_exits":              by_outcome.get("time", 0),
        "signal_decay_exits":      by_outcome.get("signal_decay", 0),
        "mss_invalidation_exits":  by_outcome.get("mss_invalidation", 0),
        "consistency_pct":         round(consistency * 100, 1),
        "cagr_pct":            None,
        "max_drawdown_pct":    round(results_df["max_drawdown_pct"].min(), 2),
        "calmar_ratio":        None,
        "sharpe_ratio":        round(results_df["sharpe_ratio"].mean(), 3),
        "hunter_efficiency":   round(results_df["hunter_efficiency"].mean(), 3),
        "long_net_pnl_pct":    None,
        "short_net_pnl_pct":   None,
        "long_sharpe": round(results_df["long_sharpe"].mean(), 3) if "long_sharpe" in results_df else None,
    }

    summary_df = pd.concat(
        [results_df, pd.DataFrame([aggregate])], ignore_index=True
    )

    # ── Save outputs ────────────────────────────────────────────────────────
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    all_trades.to_csv(OUT_TRADES, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    # ── Print report ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BACKTEST RESULTS — v5.7 Anti-Overfit (8-Fold)")
    print("=" * 65)
    print(f"\n  Consistency   : {consistency:.0%}  ({'PASS' if consistency >= 0.75 else 'FAIL'})")
    print(f"  Total Trades  : {len(all_trades)}  (L:{len(long_trades)}  S:{len(short_trades)})")
    print(f"  Overall WR    : {overall_wr:.1f}%")
    print(f"  Long WR       : {long_wr:.1f}%  ({'PASS' if long_wr >= 41 else 'FAIL'} — gate >=41%)")
    print(f"  Short WR      : {short_wr:.1f}%")
    print(f"  Total PnL     : ${total_pnl:,.2f}  ({total_ret:.1f}% on ${initial_capital:,.0f})")
    print(f"  Long PnL      : ${long_pnl:,.2f}")
    print(f"  Short PnL     : ${short_pnl:,.2f}")
    print(f"  Profit Factor : {profit_factor:.2f}")
    print(f"  Expectancy    : ${expectancy:.2f}/trade")
    print(f"  Avg Win       : ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}")
    print(f"  Max DD (worst): {results_df['max_drawdown_pct'].min():.1f}%")
    print(f"  Avg Sharpe    : {results_df['sharpe_ratio'].mean():.3f}")

    print("\n  Exit Breakdown:")
    for k, v in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"    {k:<24}: {v}")

    print("\n  Per-Fold Summary:")
    fold_cols = ["fold", "total_trades", "long_trades", "short_trades",
                 "total_return_pct", "win_rate_pct", "long_win_rate_pct",
                 "short_win_rate_pct", "max_drawdown_pct", "sharpe_ratio",
                 "hunter_efficiency", "net_profit_usd"]
    print(results_df[[c for c in fold_cols if c in results_df.columns]].to_string(index=False))

    print(f"\n[+] Trade log saved  : {OUT_TRADES.name}")
    print(f"[+] Summary saved    : {OUT_SUMMARY.name}")
    print("=" * 65)


if __name__ == "__main__":
    run()
