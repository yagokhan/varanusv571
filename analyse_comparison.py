#!/usr/bin/env python3
"""
Varanus v5.7 vs v5.7.1 Comparison Analysis Script
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── File Paths ──────────────────────────────────────────────────────────────
V57_BT_TRADES   = "/home/gokhan/varanusv57-master/varanus/config/varanusv57_backtest_trades.csv"
V57_BT_SUMMARY  = "/home/gokhan/varanusv57-master/varanus/config/varanusv57_backtest_summary.csv"
V57_B1_TRADES   = "/home/gokhan/varanusv57-master/varanus/config/varanusv57_blind_trades.csv"
V57_B1_SUMMARY  = "/home/gokhan/varanusv57-master/varanus/config/varanusv57_blind_summary.csv"
V57_B2_TRADES   = "/home/gokhan/varanusv57-master/varanus/config/varanusv57_blind2_trades.csv"
V57_B2_SUMMARY  = "/home/gokhan/varanusv57-master/varanus/config/varanusv57_blind2_summary.csv"

V571_BT_TRADES  = "/home/gokhan/varanus_v571/varanus/config/varanus_v571_backtest_trades.csv"
V571_BT_SUMMARY = "/home/gokhan/varanus_v571/varanus/config/varanus_v571_backtest_summary.csv"
V571_B1_TRADES  = "/home/gokhan/varanus_v571/varanus/config/varanus_v571_blind1_trades.csv"
V571_B1_SUMMARY = "/home/gokhan/varanus_v571/varanus/config/varanus_v571_blind1_summary.csv"
V571_B2_TRADES  = "/home/gokhan/varanus_v571/varanus/config/varanus_v571_blind2_trades.csv"
V571_B2_SUMMARY = "/home/gokhan/varanus_v571/varanus/config/varanus_v571_blind2_summary.csv"

# ─── Load Data ───────────────────────────────────────────────────────────────
def load_trades(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'entry_ts' in df.columns:
        df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
    return df

def load_summary(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

v57_bt  = load_trades(V57_BT_TRADES)
v57_b1  = load_trades(V57_B1_TRADES)
v57_b2  = load_trades(V57_B2_TRADES)
v571_bt = load_trades(V571_BT_TRADES)
v571_b1 = load_trades(V571_B1_TRADES)
v571_b2 = load_trades(V571_B2_TRADES)

v57_bt_sum  = load_summary(V57_BT_SUMMARY)
v57_b1_sum  = load_summary(V57_B1_SUMMARY)
v57_b2_sum  = load_summary(V57_B2_SUMMARY)
v571_bt_sum = load_summary(V571_BT_SUMMARY)
v571_b1_sum = load_summary(V571_B1_SUMMARY)
v571_b2_sum = load_summary(V571_B2_SUMMARY)

print("=== DATA LOADED ===")
print(f"v5.7  BT trades:  {len(v57_bt)}")
print(f"v5.7  B1 trades:  {len(v57_b1)}")
print(f"v5.7  B2 trades:  {len(v57_b2)}")
print(f"v5.7.1 BT trades: {len(v571_bt)}")
print(f"v5.7.1 B1 trades: {len(v571_b1)}")
print(f"v5.7.1 B2 trades: {len(v571_b2)}")

print("\nv5.7 BT columns:", list(v57_bt.columns))
print("v5.7.1 BT columns:", list(v571_bt.columns))
print("\nv5.7 B1 columns:", list(v57_b1.columns))
print("v5.7.1 B1 columns:", list(v571_b1.columns))

# ─── Compute Overall Metrics from Trades ─────────────────────────────────────
def compute_metrics(df, label=""):
    pnl = df['pnl_usd']
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    total = len(df)
    win_rate = len(wins) / total * 100 if total > 0 else 0
    net_profit = pnl.sum()
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    expectancy = pnl.mean() if total > 0 else 0
    # Sharpe approx
    std_pnl = pnl.std()
    sharpe = (pnl.mean() / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0

    # Max drawdown via cumulative pnl
    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd_usd = drawdown.min()
    # as % of running_max if possible
    if len(cumulative) > 0:
        peak = running_max.max()
        max_dd_pct = (drawdown / running_max.replace(0, np.nan)).min() * 100 if peak != 0 else 0
    else:
        max_dd_pct = 0

    return {
        'total_trades': total,
        'net_profit_usd': round(net_profit, 2),
        'win_rate_pct': round(win_rate, 2),
        'profit_factor': round(profit_factor, 3),
        'avg_win_usd': round(avg_win, 2),
        'avg_loss_usd': round(avg_loss, 2),
        'expectancy_usd': round(expectancy, 2),
        'sharpe_approx': round(sharpe, 3),
        'max_dd_usd': round(max_dd_usd, 2),
    }

def compute_exit_breakdown(df, label=""):
    total = len(df)
    counts = df['outcome'].value_counts()
    result = {}
    for exit_type in ['tp', 'sl', 'time', 'mss_invalidation', 'signal_decay', 'trailing_sl_hit']:
        c = counts.get(exit_type, 0)
        result[exit_type] = {'count': c, 'pct': round(c/total*100, 1) if total > 0 else 0}
    return result

def compute_directional(df, label=""):
    result = {}
    for direction, name in [(1.0, 'long'), (-1.0, 'short')]:
        sub = df[df['direction'] == direction]
        if len(sub) == 0:
            result[name] = {'count': 0, 'win_rate': 0, 'net_pnl': 0}
            continue
        wins = sub[sub['pnl_usd'] > 0]
        result[name] = {
            'count': len(sub),
            'win_rate': round(len(wins)/len(sub)*100, 2),
            'net_pnl': round(sub['pnl_usd'].sum(), 2)
        }
    return result

# ─── Trailing Stop Analysis ───────────────────────────────────────────────────
def trailing_stop_analysis(df, label=""):
    trail = df[df['outcome'] == 'trailing_sl_hit']
    if len(trail) == 0:
        print(f"  No trailing_sl_hit exits in {label}")
        return {}
    total_trail = len(trail)
    rescued = trail[trail['pnl_usd'] > 0]
    cut_short = trail[trail['pnl_usd'] < 0]
    win_rate = len(rescued) / total_trail * 100

    print(f"\n  --- Trailing Stop Analysis: {label} ---")
    print(f"  Total trailing_sl_hit exits: {total_trail}")
    print(f"  Win rate of trailing exits: {win_rate:.1f}%")
    print(f"  Average PnL of trailing exits: ${trail['pnl_usd'].mean():.2f}")
    print(f"  Total PnL from trailing exits: ${trail['pnl_usd'].sum():.2f}")
    print(f"  Rescued trades (trail exit, PnL > 0): {len(rescued)}")
    print(f"  Cut short (trail exit, PnL < 0): {len(cut_short)}")

    return {
        'count': total_trail,
        'win_rate': round(win_rate, 2),
        'avg_pnl': round(trail['pnl_usd'].mean(), 2),
        'total_pnl': round(trail['pnl_usd'].sum(), 2),
        'rescued': len(rescued),
        'cut_short': len(cut_short),
        'rescued_pnl': round(rescued['pnl_usd'].sum(), 2),
        'cut_short_pnl': round(cut_short['pnl_usd'].sum(), 2),
    }

# ─── Trade Matching Analysis ─────────────────────────────────────────────────
LOSS_EXITS = {'sl', 'time', 'signal_decay', 'mss_invalidation'}

def match_trades(df57, df571, label=""):
    print(f"\n  --- Trade Matching: {label} ---")

    # Normalize entry_ts
    df57 = df57.copy()
    df571 = df571.copy()
    df57['entry_ts_norm'] = pd.to_datetime(df57['entry_ts'], utc=True).dt.floor('min')
    df571['entry_ts_norm'] = pd.to_datetime(df571['entry_ts'], utc=True).dt.floor('min')
    df57['key'] = df57['asset'] + '|' + df57['entry_ts_norm'].astype(str)
    df571['key'] = df571['asset'] + '|' + df571['entry_ts_norm'].astype(str)

    keys57  = set(df57['key'])
    keys571 = set(df571['key'])
    common  = keys57 & keys571
    only57  = keys57 - keys571
    only571 = keys571 - keys57

    print(f"  Matched pairs (same asset+entry_ts): {len(common)}")
    print(f"  Only in v5.7 (unmatched): {len(only57)}")
    print(f"  Only in v5.7.1 (unmatched): {len(only571)}")

    df57m  = df57[df57['key'].isin(common)].set_index('key')
    df571m = df571[df571['key'].isin(common)].set_index('key')

    # For duplicates (same key), keep first
    df57m  = df57m[~df57m.index.duplicated(keep='first')]
    df571m = df571m[~df571m.index.duplicated(keep='first')]

    same_outcome = 0
    trailing_saved = 0
    trailing_cut_tp = 0
    trailing_cut_tp_pnl_diff = []
    other_diff = 0

    for key in common:
        if key not in df57m.index or key not in df571m.index:
            continue
        t57  = df57m.loc[key]
        t571 = df571m.loc[key]
        o57  = t57['outcome']
        o571 = t571['outcome']

        if o57 == o571 or (o57 not in ['trailing_sl_hit'] and o571 not in ['trailing_sl_hit'] and o57 == o571):
            same_outcome += 1
        elif o57 in LOSS_EXITS and o571 == 'trailing_sl_hit' and t571['pnl_usd'] > 0:
            trailing_saved += 1
        elif o57 == 'tp' and o571 == 'trailing_sl_hit':
            trailing_cut_tp += 1
            diff = t571['pnl_usd'] - t57['pnl_usd']
            trailing_cut_tp_pnl_diff.append(diff)
        else:
            same_outcome += 1  # other matched trades

    print(f"  Same outcome in both: {same_outcome}")
    print(f"  'Trailing saved trade' (v5.7=loss, v5.7.1=trailing+profit): {trailing_saved}")
    print(f"  'Trailing cut TP short' (v5.7=tp, v5.7.1=trailing): {trailing_cut_tp}")
    if trailing_cut_tp_pnl_diff:
        avg_diff = np.mean(trailing_cut_tp_pnl_diff)
        print(f"    Avg PnL difference for cut-TP trades: ${avg_diff:.2f} ({'loss' if avg_diff < 0 else 'gain'} vs v5.7)")

    return {
        'matched': len(common),
        'unmatched_57': len(only57),
        'unmatched_571': len(only571),
        'same_outcome': same_outcome,
        'trailing_saved': trailing_saved,
        'trailing_cut_tp': trailing_cut_tp,
        'avg_pnl_diff_cut_tp': round(np.mean(trailing_cut_tp_pnl_diff), 2) if trailing_cut_tp_pnl_diff else None,
    }

# ─── Run All Analysis ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BACKTEST RESULTS")
print("="*70)

bt57_metrics  = compute_metrics(v57_bt,  "v5.7 BT")
bt571_metrics = compute_metrics(v571_bt, "v5.7.1 BT")
print("\nv5.7 BT Metrics:")
for k, v in bt57_metrics.items():
    print(f"  {k}: {v}")
print("\nv5.7.1 BT Metrics:")
for k, v in bt571_metrics.items():
    print(f"  {k}: {v}")

# Read max_dd from summary (more accurate)
v57_bt_all  = v57_bt_sum[v57_bt_sum['fold'] == 'ALL'].iloc[0]
v571_bt_all = v571_bt_sum[v571_bt_sum['fold'] == 'ALL'].iloc[0]
print(f"\nv5.7  BT Summary (ALL fold): total_return_pct={v57_bt_all['total_return_pct']}, max_dd={v57_bt_all['max_drawdown_pct']}, sharpe={v57_bt_all['sharpe_ratio']}, win_rate={v57_bt_all['win_rate_pct']}, profit_factor={v57_bt_all['profit_factor']}, net_profit={v57_bt_all['net_profit_usd']}, expectancy={v57_bt_all['expectancy_usd']}")
print(f"v5.7.1 BT Summary (ALL fold): total_return_pct={v571_bt_all['total_return_pct']}, max_dd={v571_bt_all['max_drawdown_pct']}, sharpe={v571_bt_all['sharpe_ratio']}, win_rate={v571_bt_all['win_rate_pct']}, profit_factor={v571_bt_all['profit_factor']}, net_profit={v571_bt_all['net_profit_usd']}, expectancy={v571_bt_all['expectancy_usd']}")

print("\n-- Exit Breakdown BT --")
bt57_exits  = compute_exit_breakdown(v57_bt)
bt571_exits = compute_exit_breakdown(v571_bt)
print("v5.7:", bt57_exits)
print("v5.7.1:", bt571_exits)

print("\n-- Directional BT --")
bt57_dir  = compute_directional(v57_bt)
bt571_dir = compute_directional(v571_bt)
print("v5.7:", bt57_dir)
print("v5.7.1:", bt571_dir)

# Per-fold BT
print("\n-- BT Per-Fold (v5.7) --")
for _, row in v57_bt_sum.iterrows():
    print(f"  Fold {row['fold']}: return={row['total_return_pct']}%, trades={row['total_trades']}, wr={row['win_rate_pct']}%, pf={row['profit_factor']}, net=${row['net_profit_usd']}, sharpe={row['sharpe_ratio']}, dd={row['max_drawdown_pct']}%")

print("\n-- BT Per-Fold (v5.7.1) --")
for _, row in v571_bt_sum.iterrows():
    print(f"  Fold {row['fold']}: return={row['total_return_pct']}%, trades={row['total_trades']}, wr={row['win_rate_pct']}%, pf={row['profit_factor']}, net=${row['net_profit_usd']}, sharpe={row['sharpe_ratio']}, dd={row['max_drawdown_pct']}%")

print("\n" + "="*70)
print("BLIND TEST 1 RESULTS")
print("="*70)
b1_57_all  = v57_b1_sum.iloc[0]
b1_571_all = v571_b1_sum.iloc[0]
print(f"v5.7  B1 Summary: trades={b1_57_all['total_trades']}, return={b1_57_all['total_return_pct']}%, net=${b1_57_all['net_profit_usd']}, wr={b1_57_all['win_rate_pct']}%, pf={b1_57_all['profit_factor']}, exp=${b1_57_all['expectancy_usd']}, avg_win=${b1_57_all['avg_win_usd']}, avg_loss=${b1_57_all['avg_loss_usd']}, dd={b1_57_all['max_drawdown_pct']}%, sharpe={b1_57_all['sharpe_ratio']}")
print(f"v5.7.1 B1 Summary: trades={b1_571_all['total_trades']}, return={b1_571_all['total_return_pct']}%, net=${b1_571_all['net_profit_usd']}, wr={b1_571_all['win_rate_pct']}%, pf={b1_571_all['profit_factor']}, exp=${b1_571_all['expectancy_usd']}, avg_win=${b1_571_all['avg_win_usd']}, avg_loss=${b1_571_all['avg_loss_usd']}, dd={b1_571_all['max_drawdown_pct']}%, sharpe={b1_571_all['sharpe_ratio']}")

print("\n-- Exit Breakdown B1 --")
b1_57_exits  = compute_exit_breakdown(v57_b1)
b1_571_exits = compute_exit_breakdown(v571_b1)
print("v5.7:", b1_57_exits)
print("v5.7.1:", b1_571_exits)

print("\n-- Directional B1 --")
b1_57_dir  = compute_directional(v57_b1)
b1_571_dir = compute_directional(v571_b1)
print("v5.7:", b1_57_dir)
print("v5.7.1:", b1_571_dir)

# Additional metrics from trades for B1
b1_57_m  = compute_metrics(v57_b1)
b1_571_m = compute_metrics(v571_b1)
print("\nv5.7 B1 computed:", b1_57_m)
print("v5.7.1 B1 computed:", b1_571_m)

print("\n" + "="*70)
print("BLIND TEST 2 RESULTS")
print("="*70)
b2_57_all  = v57_b2_sum.iloc[0]
b2_571_all = v571_b2_sum.iloc[0]
print(f"v5.7  B2 Summary: trades={b2_57_all['total_trades']}, return={b2_57_all['total_return_pct']}%, net=${b2_57_all['net_profit_usd']}, wr={b2_57_all['win_rate_pct']}%, pf={b2_57_all['profit_factor']}, exp=${b2_57_all['expectancy_usd']}, avg_win=${b2_57_all['avg_win_usd']}, avg_loss=${b2_57_all['avg_loss_usd']}, dd={b2_57_all['max_drawdown_pct']}%, sharpe={b2_57_all['sharpe_ratio']}")
print(f"v5.7.1 B2 Summary: trades={b2_571_all['total_trades']}, return={b2_571_all['total_return_pct']}%, net=${b2_571_all['net_profit_usd']}, wr={b2_571_all['win_rate_pct']}%, pf={b2_571_all['profit_factor']}, exp=${b2_571_all['expectancy_usd']}, avg_win=${b2_571_all['avg_win_usd']}, avg_loss=${b2_571_all['avg_loss_usd']}, dd={b2_571_all['max_drawdown_pct']}%, sharpe={b2_571_all['sharpe_ratio']}")

print("\n-- Exit Breakdown B2 --")
b2_57_exits  = compute_exit_breakdown(v57_b2)
b2_571_exits = compute_exit_breakdown(v571_b2)
print("v5.7:", b2_57_exits)
print("v5.7.1:", b2_571_exits)

print("\n-- Directional B2 --")
b2_57_dir  = compute_directional(v57_b2)
b2_571_dir = compute_directional(v571_b2)
print("v5.7:", b2_57_dir)
print("v5.7.1:", b2_571_dir)

b2_57_m  = compute_metrics(v57_b2)
b2_571_m = compute_metrics(v571_b2)
print("\nv5.7 B2 computed:", b2_57_m)
print("v5.7.1 B2 computed:", b2_571_m)

print("\n" + "="*70)
print("TRAILING STOP ANALYSIS")
print("="*70)
trail_bt  = trailing_stop_analysis(v571_bt,  "BT")
trail_b1  = trailing_stop_analysis(v571_b1,  "B1")
trail_b2  = trailing_stop_analysis(v571_b2,  "B2")

print("\n" + "="*70)
print("TRADE MATCHING ANALYSIS")
print("="*70)
match_bt = match_trades(v57_bt, v571_bt, "Backtest")
match_b1 = match_trades(v57_b1, v571_b1, "Blind 1")
match_b2 = match_trades(v57_b2, v571_b2, "Blind 2")

# ─── Additional: v5.7 BT summary columns check ───────────────────────────────
print("\n" + "="*70)
print("SUMMARY COLUMNS CHECK")
print("="*70)
print("v5.7 BT summary columns:", list(v57_bt_sum.columns))
print("v5.7.1 BT summary columns:", list(v571_bt_sum.columns))
print("v5.7 B1 summary columns:", list(v57_b1_sum.columns))
print("v5.7.1 B1 summary columns:", list(v571_b1_sum.columns))

# ─── Combined Trailing Totals ─────────────────────────────────────────────────
print("\n" + "="*70)
print("COMBINED TRAILING STOP TOTALS (v5.7.1 all windows)")
print("="*70)
all_trail_count  = trail_bt.get('count',0) + trail_b1.get('count',0) + trail_b2.get('count',0)
all_trail_rescued = trail_bt.get('rescued',0) + trail_b1.get('rescued',0) + trail_b2.get('rescued',0)
all_trail_cut    = trail_bt.get('cut_short',0) + trail_b1.get('cut_short',0) + trail_b2.get('cut_short',0)
all_trail_pnl    = trail_bt.get('total_pnl',0) + trail_b1.get('total_pnl',0) + trail_b2.get('total_pnl',0)
print(f"Total trailing_sl_hit exits: {all_trail_count}")
print(f"Total rescued: {all_trail_rescued}")
print(f"Total cut short: {all_trail_cut}")
print(f"Overall win rate of trailing exits: {all_trail_rescued/all_trail_count*100:.1f}%" if all_trail_count > 0 else "N/A")
print(f"Total PnL from trailing exits: ${all_trail_pnl:.2f}")

# ─── Net profit comparison for all windows ────────────────────────────────────
print("\n" + "="*70)
print("QUICK REFERENCE TABLE")
print("="*70)
print(f"{'Metric':<30} {'v5.7 BT':>15} {'v5.7.1 BT':>15}")
print("-"*60)
metrics_to_show = [
    ('total_trades', v57_bt_all, v571_bt_all),
]
print(f"{'Total Trades':<30} {v57_bt_all['total_trades']:>15} {v571_bt_all['total_trades']:>15}")
print(f"{'Net Profit USD':<30} {v57_bt_all['net_profit_usd']:>15} {v571_bt_all['net_profit_usd']:>15}")
print(f"{'Total Return %':<30} {v57_bt_all['total_return_pct']:>15} {v571_bt_all['total_return_pct']:>15}")
print(f"{'Win Rate %':<30} {v57_bt_all['win_rate_pct']:>15} {v571_bt_all['win_rate_pct']:>15}")
print(f"{'Profit Factor':<30} {v57_bt_all['profit_factor']:>15} {v571_bt_all['profit_factor']:>15}")
print(f"{'Max Drawdown %':<30} {v57_bt_all['max_drawdown_pct']:>15} {v571_bt_all['max_drawdown_pct']:>15}")
print(f"{'Sharpe Ratio':<30} {v57_bt_all['sharpe_ratio']:>15} {v571_bt_all['sharpe_ratio']:>15}")
print(f"{'Expectancy USD':<30} {v57_bt_all['expectancy_usd']:>15} {v571_bt_all['expectancy_usd']:>15}")
print(f"{'Avg Win USD':<30} {v57_bt_all['avg_win_usd']:>15} {v571_bt_all['avg_win_usd']:>15}")
print(f"{'Avg Loss USD':<30} {v57_bt_all['avg_loss_usd']:>15} {v571_bt_all['avg_loss_usd']:>15}")

print(f"\n{'Metric':<30} {'v5.7 B1':>15} {'v5.7.1 B1':>15}")
print("-"*60)
print(f"{'Total Trades':<30} {b1_57_all['total_trades']:>15} {b1_571_all['total_trades']:>15}")
print(f"{'Net Profit USD':<30} {b1_57_all['net_profit_usd']:>15} {b1_571_all['net_profit_usd']:>15}")
print(f"{'Total Return %':<30} {b1_57_all['total_return_pct']:>15} {b1_571_all['total_return_pct']:>15}")
print(f"{'Win Rate %':<30} {b1_57_all['win_rate_pct']:>15} {b1_571_all['win_rate_pct']:>15}")
print(f"{'Profit Factor':<30} {b1_57_all['profit_factor']:>15} {b1_571_all['profit_factor']:>15}")
print(f"{'Max Drawdown %':<30} {b1_57_all['max_drawdown_pct']:>15} {b1_571_all['max_drawdown_pct']:>15}")
print(f"{'Sharpe Ratio':<30} {b1_57_all['sharpe_ratio']:>15} {b1_571_all['sharpe_ratio']:>15}")
print(f"{'Expectancy USD':<30} {b1_57_all['expectancy_usd']:>15} {b1_571_all['expectancy_usd']:>15}")
print(f"{'Avg Win USD':<30} {b1_57_all['avg_win_usd']:>15} {b1_571_all['avg_win_usd']:>15}")
print(f"{'Avg Loss USD':<30} {b1_57_all['avg_loss_usd']:>15} {b1_571_all['avg_loss_usd']:>15}")

print(f"\n{'Metric':<30} {'v5.7 B2':>15} {'v5.7.1 B2':>15}")
print("-"*60)
print(f"{'Total Trades':<30} {b2_57_all['total_trades']:>15} {b2_571_all['total_trades']:>15}")
print(f"{'Net Profit USD':<30} {b2_57_all['net_profit_usd']:>15} {b2_571_all['net_profit_usd']:>15}")
print(f"{'Total Return %':<30} {b2_57_all['total_return_pct']:>15} {b2_571_all['total_return_pct']:>15}")
print(f"{'Win Rate %':<30} {b2_57_all['win_rate_pct']:>15} {b2_571_all['win_rate_pct']:>15}")
print(f"{'Profit Factor':<30} {b2_57_all['profit_factor']:>15} {b2_571_all['profit_factor']:>15}")
print(f"{'Max Drawdown %':<30} {b2_57_all['max_drawdown_pct']:>15} {b2_571_all['max_drawdown_pct']:>15}")
print(f"{'Sharpe Ratio':<30} {b2_57_all['sharpe_ratio']:>15} {b2_571_all['sharpe_ratio']:>15}")
print(f"{'Expectancy USD':<30} {b2_57_all['expectancy_usd']:>15} {b2_571_all['expectancy_usd']:>15}")
print(f"{'Avg Win USD':<30} {b2_57_all['avg_win_usd']:>15} {b2_571_all['avg_win_usd']:>15}")
print(f"{'Avg Loss USD':<30} {b2_57_all['avg_loss_usd']:>15} {b2_571_all['avg_loss_usd']:>15}")

print("\n=== ANALYSIS COMPLETE ===")
