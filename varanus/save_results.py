"""
save_results.py — Run full walk-forward with best_params.json and save
all plots + Excel artefacts to varanus/plots/ and varanus/results/.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from varanus.universe import TIER2_UNIVERSE
from varanus.walk_forward import run_walk_forward

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE      = "/home/yagokhan/chameleon/claude_code_project/data/cache"
PARAMS_FILE = "/home/yagokhan/varanus/config/best_params.json"
PLOTS_DIR  = "/home/yagokhan/varanus/plots"
RESULTS_DIR = "/home/yagokhan/varanus/results"

os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use("dark_background")
TEAL   = "#00ffcc"
RED    = "#ff4444"
ORANGE = "#ffaa00"
GREEN  = "#44ff88"
GREY   = "#888888"


# ── Data loader (mirrors run_optimization.py) ─────────────────────────────────
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
        for col, fn in [("open","first"),("high","max"),("low","min"),("close","last"),("volume","sum")]:
            if col in df.columns:
                agg[col] = fn
        df = df.resample("1D").agg(agg).dropna()

    return df


# ── Helper formatters ──────────────────────────────────────────────────────────
def _usd(x, _):
    if abs(x) >= 1_000:
        return f"${x/1_000:.1f}k"
    return f"${x:.0f}"

def _pct(x, _):
    return f"{x:.0f}%"


# ── Plot 1: Performance Dashboard (4-panel) ───────────────────────────────────
def plot_performance_dashboard(trades: pd.DataFrame, metrics: dict, params: dict):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    calmar = metrics.get("calmar_ratio", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    wr     = metrics.get("win_rate_pct", 0)
    mdd    = metrics.get("max_drawdown_pct", 0)
    cagr   = metrics.get("cagr_pct", 0)
    n      = metrics.get("total_trades", 0)

    fig.suptitle(
        f"Varanus v4.0 Tier 2 — Best Params Backtest\n"
        f"CAGR {cagr:.1f}%  |  Calmar {calmar:.2f}  |  Sharpe {sharpe:.2f}  |  "
        f"WR {wr:.1f}%  |  MaxDD {mdd:.1f}%  |  {n} trades",
        fontsize=16, color="white", y=0.98
    )

    # ── 1. Equity curve ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    initial_cap = 5_000.0
    tr = trades.sort_values("exit_ts").copy()
    tr["cumulative_pnl"] = tr["pnl_usd"].cumsum() + initial_cap
    ax1.plot(tr["exit_ts"], tr["cumulative_pnl"], color=TEAL, lw=2)
    ax1.fill_between(tr["exit_ts"], initial_cap, tr["cumulative_pnl"],
                     where=tr["cumulative_pnl"] >= initial_cap,
                     alpha=0.15, color=GREEN)
    ax1.fill_between(tr["exit_ts"], initial_cap, tr["cumulative_pnl"],
                     where=tr["cumulative_pnl"] < initial_cap,
                     alpha=0.20, color=RED)
    ax1.axhline(initial_cap, color=GREY, ls="--", lw=0.8, alpha=0.6)
    ax1.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax1.set_title("Equity Curve", color="white")
    ax1.set_xlabel("Date", color=GREY, fontsize=9)
    ax1.set_ylabel("Portfolio Value ($)", color=GREY, fontsize=9)
    ax1.tick_params(colors=GREY, labelsize=8)
    for sp in ax1.spines.values(): sp.set_color("#333")

    # ── 2. Drawdown ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    equity_s = tr.set_index("exit_ts")["cumulative_pnl"]
    roll_max = equity_s.cummax()
    drawdown = (equity_s - roll_max) / roll_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values, color=RED, lw=1)
    ax2.axhline(mdd, color=ORANGE, ls=":", lw=1, label=f"MaxDD {mdd:.1f}%")
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax2.set_title("Drawdown", color="white")
    ax2.set_xlabel("Date", color=GREY, fontsize=9)
    ax2.set_ylabel("Drawdown (%)", color=GREY, fontsize=9)
    ax2.legend(fontsize=8, facecolor="#1a1a1a", edgecolor=GREY)
    ax2.tick_params(colors=GREY, labelsize=8)
    for sp in ax2.spines.values(): sp.set_color("#333")

    # ── 3. PnL by asset ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    asset_pnl = trades.groupby("asset")["pnl_usd"].sum().sort_values()
    colors = [GREEN if v >= 0 else RED for v in asset_pnl.values]
    bars = ax3.barh(asset_pnl.index, asset_pnl.values, color=colors, edgecolor="#222", height=0.6)
    ax3.axvline(0, color=GREY, lw=0.8)
    for bar in bars:
        w = bar.get_width()
        ax3.text(w + (5 if w >= 0 else -5), bar.get_y() + bar.get_height()/2,
                 f"${w:,.0f}", va="center", ha="left" if w >= 0 else "right",
                 color="white", fontsize=7)
    ax3.xaxis.set_major_formatter(FuncFormatter(_usd))
    ax3.set_title("Total PnL by Asset", color="white")
    ax3.tick_params(colors=GREY, labelsize=8)
    for sp in ax3.spines.values(): sp.set_color("#333")

    # ── 4. Outcome distribution ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    outcome_counts = trades["outcome"].value_counts()
    pie_colors = {"tp": GREEN, "sl": RED, "time": ORANGE}
    wedge_colors = [pie_colors.get(k, GREY) for k in outcome_counts.index]
    wedges, texts, autotexts = ax4.pie(
        outcome_counts, labels=[o.upper() for o in outcome_counts.index],
        colors=wedge_colors, autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="#0d0d0d")
    )
    plt.setp(texts, color="white", size=11)
    plt.setp(autotexts, color="white", size=10, weight="bold")
    ax4.set_title("Exit Outcome Distribution", color="white")

    path = os.path.join(PLOTS_DIR, "01_performance_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: Fold Comparison ───────────────────────────────────────────────────
def plot_fold_comparison(results_df: pd.DataFrame):
    if results_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("Walk-Forward Fold Comparison", color="white", fontsize=14)

    metrics_to_plot = [
        ("calmar_ratio",     "Calmar Ratio",     TEAL),
        ("win_rate_pct",     "Win Rate (%)",      GREEN),
        ("max_drawdown_pct", "Max Drawdown (%)", RED),
    ]
    folds = results_df["fold"].astype(str)

    for ax, (col, title, color) in zip(axes, metrics_to_plot):
        ax.bar(folds, results_df[col], color=color, edgecolor="#222", width=0.5)
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel("Fold", color=GREY, fontsize=9)
        ax.tick_params(colors=GREY, labelsize=9)
        ax.set_facecolor("#111")
        for sp in ax.spines.values(): sp.set_color("#333")
        # Add value labels
        for i, v in enumerate(results_df[col]):
            ax.text(i, v + 0.01 * abs(results_df[col].max()), f"{v:.1f}",
                    ha="center", color="white", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_fold_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: Confidence vs PnL scatter ─────────────────────────────────────────
def plot_confidence_scatter(trades: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("Confidence Analysis", color="white", fontsize=14)

    outcome_colors = {"tp": GREEN, "sl": RED, "time": ORANGE}
    for outcome in trades["outcome"].unique():
        sub = trades[trades["outcome"] == outcome]
        ax1.scatter(sub["confidence"], sub["pnl_usd"],
                    color=outcome_colors.get(outcome, GREY),
                    alpha=0.5, s=40, label=outcome.upper(), edgecolors="none")

    ax1.axhline(0, color=GREY, ls="--", lw=0.8)
    ax1.set_title("Confidence vs PnL per Trade", color="white")
    ax1.set_xlabel("Model Confidence", color=GREY)
    ax1.set_ylabel("PnL ($)", color=GREY)
    ax1.legend(facecolor="#1a1a1a", edgecolor=GREY, labelcolor="white")
    ax1.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax1.tick_params(colors=GREY)
    ax1.set_facecolor("#111")
    for sp in ax1.spines.values(): sp.set_color("#333")

    # Confidence distribution by outcome
    for outcome in trades["outcome"].unique():
        sub = trades[trades["outcome"] == outcome]
        ax2.hist(sub["confidence"], bins=20, alpha=0.6,
                 color=outcome_colors.get(outcome, GREY),
                 label=outcome.upper(), edgecolor="none")

    ax2.set_title("Confidence Score Distribution", color="white")
    ax2.set_xlabel("Model Confidence", color=GREY)
    ax2.set_ylabel("Count", color=GREY)
    ax2.legend(facecolor="#1a1a1a", edgecolor=GREY, labelcolor="white")
    ax2.tick_params(colors=GREY)
    ax2.set_facecolor("#111")
    for sp in ax2.spines.values(): sp.set_color("#333")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_confidence_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4: Monthly PnL Heatmap ───────────────────────────────────────────────
def plot_monthly_heatmap(trades: pd.DataFrame):
    tr = trades.copy()
    tr["exit_ts"] = pd.to_datetime(tr["exit_ts"], utc=True)
    tr["year"]  = tr["exit_ts"].dt.year
    tr["month"] = tr["exit_ts"].dt.month

    monthly = tr.groupby(["year", "month"])["pnl_usd"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, max(3, len(monthly) * 1.5)))
    fig.patch.set_facecolor("#0d0d0d")

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    col_labels = [month_labels[m-1] for m in monthly.columns]

    sns.heatmap(monthly, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#222",
                xticklabels=col_labels,
                cbar_kws={"label": "PnL ($)", "shrink": 0.6},
                ax=ax)

    ax.set_title("Monthly PnL Heatmap ($)", color="white", fontsize=13)
    ax.set_xlabel("Month", color=GREY)
    ax.set_ylabel("Year", color=GREY)
    ax.tick_params(colors=GREY)

    path = os.path.join(PLOTS_DIR, "04_monthly_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 5: Asset confidence heatmap ─────────────────────────────────────────
def plot_asset_confidence_heatmap(trades: pd.DataFrame):
    tr = trades.copy()
    tr["conf_bin"] = pd.cut(
        tr["confidence"],
        bins=[0.78, 0.82, 0.86, 0.90, 1.01],
        labels=["0.78–0.82", "0.82–0.86", "0.86–0.90", "0.90+"]
    )

    pivot = tr.pivot_table(
        values="pnl_usd", index="asset", columns="conf_bin",
        aggfunc="sum", fill_value=0
    )

    fig, ax = plt.subplots(figsize=(12, max(5, len(pivot) * 0.55)))
    fig.patch.set_facecolor("#0d0d0d")

    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#222",
                cbar_kws={"label": "PnL ($)", "shrink": 0.7},
                ax=ax)

    ax.set_title("PnL Heatmap: Asset × Confidence Tier", color="white", fontsize=13)
    ax.set_xlabel("Confidence Tier", color=GREY)
    ax.set_ylabel("Asset", color=GREY)
    ax.tick_params(colors=GREY)

    path = os.path.join(PLOTS_DIR, "05_asset_confidence_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 6: Duration vs PnL scatter ──────────────────────────────────────────
def plot_duration_scatter(trades: pd.DataFrame):
    tr = trades.copy()
    tr["entry_ts"] = pd.to_datetime(tr["entry_ts"], utc=True)
    tr["exit_ts"]  = pd.to_datetime(tr["exit_ts"],  utc=True)
    tr["duration_h"] = (tr["exit_ts"] - tr["entry_ts"]).dt.total_seconds() / 3600

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0d0d0d")

    outcome_colors = {"tp": GREEN, "sl": RED, "time": ORANGE}
    for outcome in tr["outcome"].unique():
        sub = tr[tr["outcome"] == outcome]
        ax.scatter(sub["duration_h"], sub["pnl_usd"],
                   color=outcome_colors.get(outcome, GREY),
                   alpha=0.55, s=50, label=outcome.upper(), edgecolors="none")

    ax.axhline(0, color=GREY, ls="--", lw=0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(_usd))
    ax.set_title("Trade Duration vs PnL", color="white", fontsize=13)
    ax.set_xlabel("Duration (hours)", color=GREY)
    ax.set_ylabel("PnL ($)", color=GREY)
    ax.legend(facecolor="#1a1a1a", edgecolor=GREY, labelcolor="white")
    ax.tick_params(colors=GREY)
    ax.set_facecolor("#111")
    for sp in ax.spines.values(): sp.set_color("#333")

    path = os.path.join(PLOTS_DIR, "06_duration_vs_pnl.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Excel export ──────────────────────────────────────────────────────────────
def save_excel(trades: pd.DataFrame, results_df: pd.DataFrame,
               metrics: dict, params: dict):
    path = os.path.join(RESULTS_DIR, "varanus_backtest_results.xlsx")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # Sheet 1: Summary metrics
        summary_rows = [
            ("PERFORMANCE METRICS", ""),
            ("Total Return (%)",     metrics.get("total_return_pct", 0)),
            ("CAGR (%)",             metrics.get("cagr_pct", 0)),
            ("Max Drawdown (%)",     metrics.get("max_drawdown_pct", 0)),
            ("Calmar Ratio",         metrics.get("calmar_ratio", 0)),
            ("Sharpe Ratio",         metrics.get("sharpe_ratio", 0)),
            ("Win Rate (%)",         metrics.get("win_rate_pct", 0)),
            ("Profit Factor",        metrics.get("profit_factor", 0)),
            ("Total Trades",         metrics.get("total_trades", 0)),
            ("TP Hits",              metrics.get("tp_hits", 0)),
            ("SL Hits",              metrics.get("sl_hits", 0)),
            ("Time Exits",           metrics.get("time_exits", 0)),
            ("Avg Win ($)",          metrics.get("avg_win_usd", 0)),
            ("Avg Loss ($)",         metrics.get("avg_loss_usd", 0)),
            ("", ""),
            ("OPTIMAL PARAMETERS", ""),
        ]
        for k, v in params.items():
            summary_rows.append((k, round(v, 6) if isinstance(v, float) else v))

        summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Full trade log
        trades_out = trades.copy()
        for col in ["entry_ts", "exit_ts", "max_hold_bar"]:
            if col in trades_out.columns:
                trades_out[col] = pd.to_datetime(trades_out[col]).dt.tz_localize(None)
        trades_out.to_excel(writer, sheet_name="Trade Log", index=False)

        # Sheet 3: Per-asset stats
        asset_stats = trades.groupby("asset").agg(
            trades=("pnl_usd", "count"),
            total_pnl=("pnl_usd", "sum"),
            avg_pnl=("pnl_usd", "mean"),
            win_rate=("pnl_usd", lambda x: (x > 0).mean() * 100),
            avg_confidence=("confidence", "mean"),
            tp_hits=("outcome", lambda x: (x == "tp").sum()),
            sl_hits=("outcome", lambda x: (x == "sl").sum()),
            time_exits=("outcome", lambda x: (x == "time").sum()),
        ).round(3).reset_index()
        asset_stats.to_excel(writer, sheet_name="Asset Stats", index=False)

        # Sheet 4: Fold results
        if not results_df.empty:
            results_df.to_excel(writer, sheet_name="Fold Results", index=False)

        # Sheet 5: Monthly PnL
        trades_m = trades.copy()
        trades_m["exit_ts"] = pd.to_datetime(trades_m["exit_ts"], utc=True)
        trades_m["year"]  = trades_m["exit_ts"].dt.year
        trades_m["month"] = trades_m["exit_ts"].dt.strftime("%b")
        trades_m["month_num"] = trades_m["exit_ts"].dt.month
        monthly = (trades_m.groupby(["year", "month_num", "month"])["pnl_usd"]
                   .sum().reset_index()
                   .sort_values(["year", "month_num"])
                   .drop("month_num", axis=1))
        monthly.columns = ["Year", "Month", "PnL ($)"]
        monthly.to_excel(writer, sheet_name="Monthly PnL", index=False)

    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Varanus Results — Generating Plots & Excel ===\n")

    # 1. Load best params
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    print(f"[+] Loaded best params (Trial 161, Calmar = 4027.897)")

    # 2. Load market data
    print("\n[+] Loading universe data...")
    data_4h, data_1d = {}, {}
    for asset in TIER2_UNIVERSE:
        try:
            df_4h = load_data(asset, "4h")
            df_1d = load_data(asset, "1d")
            df_1d = df_1d[df_1d.index >= df_4h.index[0] - pd.Timedelta(days=100)]
            data_4h[asset] = df_4h
            data_1d[asset] = df_1d
        except Exception as e:
            print(f"  Skipping {asset}: {e}")
    print(f"  Loaded {len(data_4h)} assets")

    # 3. Run walk-forward with best params
    print("\n[+] Running walk-forward validation with best params...")
    results_df, consistency, all_trades = run_walk_forward(data_4h, data_1d, params)

    if all_trades.empty:
        print("[!] No trades generated. Aborting.")
        return

    print(f"\n  Total trades across all folds: {len(all_trades)}")
    print(f"  Consistency: {consistency:.0%}")

    # 4. Compute aggregate metrics across all trades
    from varanus.backtest import compute_metrics
    initial_cap = 5_000.0
    all_trades_sorted = all_trades.sort_values("exit_ts")
    # Build a simple equity series from cumulative PnL
    equity = (all_trades_sorted["pnl_usd"].cumsum() + initial_cap)
    equity.index = pd.to_datetime(all_trades_sorted["exit_ts"].values)
    metrics = compute_metrics(equity, all_trades_sorted)

    print(f"\n[+] Aggregate Metrics:")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    # 5. Save trade log CSV
    csv_path = os.path.join(RESULTS_DIR, "best_params_trade_log.csv")
    all_trades.to_csv(csv_path, index=False)
    print(f"\n[+] Saved trade log CSV: {csv_path}")

    # 6. Generate plots
    print("\n[+] Generating plots...")
    plot_performance_dashboard(all_trades, metrics, params)
    plot_fold_comparison(results_df)
    plot_confidence_scatter(all_trades)
    plot_monthly_heatmap(all_trades)
    plot_asset_confidence_heatmap(all_trades)
    plot_duration_scatter(all_trades)

    # 7. Save Excel
    print("\n[+] Saving Excel workbook...")
    save_excel(all_trades, results_df, metrics, params)

    print("\n=== Done ===")
    print(f"  Plots  → {PLOTS_DIR}/")
    print(f"  Excel  → {RESULTS_DIR}/varanus_backtest_results.xlsx")
    print(f"  CSV    → {RESULTS_DIR}/best_params_trade_log.csv")


if __name__ == "__main__":
    main()
