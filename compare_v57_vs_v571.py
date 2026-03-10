#!/usr/bin/env python3
"""
compare_v57_vs_v571.py
Compare v5.7 vs v5.7.1 across backtest and both blind tests.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ── File paths ──────────────────────────────────────────────────────────────
V57_DIR  = Path("/home/gokhan/varanusv57-master/varanus/config")
V571_DIR = Path("/home/gokhan/varanus_v571/varanus/config")

FILES = {
    "backtest": {
        "v57":  V57_DIR  / "varanusv57_backtest_trades.csv",
        "v571": V571_DIR / "varanus_v571_backtest_trades.csv",
        "label": "Backtest (8-Fold WFV)",
    },
    "blind1": {
        "v57":  V57_DIR  / "varanusv57_blind_trades.csv",
        "v571": V571_DIR / "varanus_v571_blind1_trades.csv",
        "label": "Blind Test 1 (Nov 2025 – Jan 24 2026)",
    },
    "blind2": {
        "v57":  V57_DIR  / "varanusv57_blind2_trades.csv",
        "v571": V571_DIR / "varanus_v571_blind2_trades.csv",
        "label": "Blind Test 2 (Jan 24 – Mar 09 2026)",
    },
}

INITIAL_CAPITAL = 5_000.0


def max_drawdown(pnl_series: pd.Series) -> float:
    """Max drawdown from cumulative PnL series (fraction)."""
    equity = INITIAL_CAPITAL + pnl_series.cumsum()
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min() * 100


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compare_set(label: str, path_v57: Path, path_v571: Path):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    t57  = load(path_v57)
    t571 = load(path_v571)

    for name, df in [("v5.7", t57), ("v5.7.1", t571)]:
        wins = (df["pnl_usd"] > 0).sum()
        losses = (df["pnl_usd"] < 0).sum()
        total_pnl = df["pnl_usd"].sum()
        wr = (df["pnl_usd"] > 0).mean() * 100
        mdd = max_drawdown(df["pnl_usd"])
        trailing = (df["outcome"] == "trailing_sl_hit").sum() if "outcome" in df.columns else 0
        print(f"\n  [{name}]")
        print(f"    Total trades    : {len(df)}  (W:{wins}  L:{losses})")
        print(f"    Net profit      : ${total_pnl:,.2f}  ({total_pnl / INITIAL_CAPITAL * 100:.1f}%)")
        print(f"    Win rate        : {wr:.1f}%")
        print(f"    Max drawdown    : {mdd:.1f}%")
        print(f"    trailing_sl_hit : {trailing}")

    # Delta
    net57  = t57["pnl_usd"].sum()
    net571 = t571["pnl_usd"].sum()
    wr57   = (t57["pnl_usd"] > 0).mean() * 100
    wr571  = (t571["pnl_usd"] > 0).mean() * 100
    mdd57  = max_drawdown(t57["pnl_usd"])
    mdd571 = max_drawdown(t571["pnl_usd"])

    print(f"\n  Delta (v5.7.1 - v5.7):")
    print(f"    Net profit      : {net571 - net57:+,.2f} USD  ({(net571 - net57) / INITIAL_CAPITAL * 100:+.1f}%)")
    print(f"    Win rate        : {wr571 - wr57:+.1f}pp")
    print(f"    Max drawdown    : {mdd571 - mdd57:+.1f}pp")

    # ── Trailing stop improvement analysis ──────────────────────────────────
    print(f"\n  Trailing Stop Improvement Analysis:")

    trail571 = t571[t571["outcome"] == "trailing_sl_hit"].copy() if "outcome" in t571.columns else pd.DataFrame()
    print(f"    Trades exited by trailing_sl_hit in v5.7.1 : {len(trail571)}")

    if trail571.empty:
        print("    No trailing stop exits to analyze.")
        return

    # Match by (asset, entry_ts)
    t57_keyed = t57.set_index(["asset", "entry_ts"]) if "entry_ts" in t57.columns else None

    improvements = 0
    degradations = 0
    neutral      = 0

    for _, row in trail571.iterrows():
        if t57_keyed is None:
            break
        key = (row["asset"], row["entry_ts"])
        if key not in t57_keyed.index:
            continue
        v57_row = t57_keyed.loc[key]
        # Handle possible duplicate index (take first)
        if isinstance(v57_row, pd.DataFrame):
            v57_row = v57_row.iloc[0]

        v57_outcome = v57_row.get("outcome", "")
        v57_pnl     = float(v57_row["pnl_usd"])
        v571_pnl    = float(row["pnl_usd"])

        # Improvement: trailing stop locked in profit, but v5.7 was a loser (sl/time/signal_decay)
        if v571_pnl > 0 and v57_pnl <= 0:
            improvements += 1
        elif v571_pnl < v57_pnl:
            degradations += 1
        else:
            neutral += 1

    matched = improvements + degradations + neutral
    print(f"    Matched to v5.7 trades            : {matched}")
    print(f"    Improvements (v5.7.1 +ve, v5.7 -ve): {improvements}")
    print(f"    Degradations (v5.7.1 worse than v5.7): {degradations}")
    print(f"    Neutral / already winners         : {neutral}")

    # Extra: how much PnL is attributed to trailing exits
    trail_pnl = trail571["pnl_usd"].sum()
    trail_wr   = (trail571["pnl_usd"] > 0).mean() * 100
    print(f"\n    Trailing stop exit stats (v5.7.1):")
    print(f"      Total PnL from trail exits : ${trail_pnl:,.2f}")
    print(f"      Win rate of trail exits    : {trail_wr:.1f}%")
    print(f"      Avg PnL per trail exit     : ${trail_pnl / len(trail571):,.2f}")


def main():
    print("\n" + "=" * 70)
    print("  VARANUS v5.7 vs v5.7.1 — FULL COMPARISON")
    print("  Dynamic Trailing Stop: trigger=1.147%  distance=1.147%")
    print("=" * 70)

    for key, cfg in FILES.items():
        if not cfg["v57"].exists():
            print(f"\n[!] Missing v5.7 file: {cfg['v57']}")
            continue
        if not cfg["v571"].exists():
            print(f"\n[!] Missing v5.7.1 file: {cfg['v571']}")
            continue
        compare_set(cfg["label"], cfg["v57"], cfg["v571"])

    # ── Overall summary table ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Test':<40} {'Metric':<20} {'v5.7':>12} {'v5.7.1':>12} {'Delta':>12}")
    print("-" * 100)

    for key, cfg in FILES.items():
        if not cfg["v57"].exists() or not cfg["v571"].exists():
            continue
        t57  = load(cfg["v57"])
        t571 = load(cfg["v571"])
        label = cfg["label"][:38]

        metrics = [
            ("Net Profit USD",   t57["pnl_usd"].sum(),                 t571["pnl_usd"].sum()),
            ("Win Rate %",       (t57["pnl_usd"] > 0).mean() * 100,   (t571["pnl_usd"] > 0).mean() * 100),
            ("Max DD %",         max_drawdown(t57["pnl_usd"]),          max_drawdown(t571["pnl_usd"])),
            ("TrailSL Exits",    (t57["outcome"] == "trailing_sl_hit").sum() if "outcome" in t57.columns else 0,
                                 (t571["outcome"] == "trailing_sl_hit").sum() if "outcome" in t571.columns else 0),
        ]

        for i, (mname, v57val, v571val) in enumerate(metrics):
            row_label = label if i == 0 else ""
            delta = v571val - v57val
            print(f"  {row_label:<40} {mname:<20} {v57val:>12.2f} {v571val:>12.2f} {delta:>+12.2f}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
