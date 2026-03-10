import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_performance_report():
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Load trade data
    try:
        df = pd.read_csv("step5_trades.csv")
    except FileNotFoundError:
        print("Error: step5_trades.csv not found in the current directory.")
        return

    # Convert timestamps to datetime
    df['entry_ts'] = pd.to_datetime(df['entry_ts'])
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])
    
    # Calculate duration
    df['duration_hours'] = (df['exit_ts'] - df['entry_ts']).dt.total_seconds() / 3600.0

    # Set overall aesthetic
    plt.style.use('dark_background')
    sns.set_palette("husl")
    
    # Create a 2x2 grid figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Varanus v4.0 Tier 2 Strategy Performance (Mock Data)', fontsize=24, y=0.95)

    # ==========================================
    # 1. Equity Curve (Cumulative PnL)
    # ==========================================
    ax1 = plt.subplot(2, 2, 1)
    
    # Sort trades chronologically to build cumulative PnL
    equity_df = df.sort_values(by='exit_ts').copy()
    equity_df['cumulative_pnl'] = equity_df['pnl_usd'].cumsum()
    
    # Adding an initial starting point
    ts_start = equity_df['entry_ts'].min() - pd.Timedelta(hours=4)
    equity_curve = pd.concat([
        pd.DataFrame({'exit_ts': [ts_start], 'cumulative_pnl': [0]}),
        equity_df[['exit_ts', 'cumulative_pnl']]
    ])
    
    ax1.plot(equity_curve['exit_ts'], equity_curve['cumulative_pnl'], color='#00ffcc', linewidth=2.5)
    ax1.fill_between(equity_curve['exit_ts'], equity_curve['cumulative_pnl'], 0, alpha=0.1, color='#00ffcc')
    
    ax1.set_title("Equity Curve (Cumulative PnL in USD)", fontsize=16)
    ax1.set_ylabel("PnL ($)", fontsize=12)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Optional watermarks for major drawdowns/runups
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)

    # ==========================================
    # 2. Outcome Distribution (Triple-Barrier Check)
    # ==========================================
    ax2 = plt.subplot(2, 2, 2)
    
    outcome_counts = df['outcome'].value_counts()
    
    colors = {'tp': '#00ff00', 'sl': '#ff3333', 'time': '#ffaa00'}
    plot_colors = [colors.get(x, '#888') for x in outcome_counts.index]
    
    wedges, texts, autotexts = ax2.pie(
        outcome_counts, 
        labels=[o.upper() for o in outcome_counts.index], 
        colors=plot_colors,
        autopct='%1.1f%%', 
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='k') # Donut shape
    )
    plt.setp(autotexts, size=12, weight="bold")
    plt.setp(texts, size=14)
    ax2.set_title("Trade Outcome Distribution", fontsize=16)

    # ==========================================
    # 3. Risk/Reward Heatmap by Asset
    # ==========================================
    ax3 = plt.subplot(2, 2, 3)
    
    # Group by asset and outcome to calculate total PnL
    asset_pnl = df.groupby('asset')['pnl_usd'].sum().sort_values()
    
    # Create horizontal bar chart
    bars = ax3.barh(asset_pnl.index, asset_pnl.values, color=['#ff3333' if v < 0 else '#00ffcc' for v in asset_pnl.values])
    
    ax3.set_title("Total PnL by Asset", fontsize=16)
    ax3.set_xlabel("Net PnL ($)", fontsize=12)
    ax3.axvline(0, color='white', linestyle='-', alpha=0.3)
    
    # Add values to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x = width + (10 if width >= 0 else -10)
        ha = 'left' if width >= 0 else 'right'
        ax3.text(label_x, bar.get_y() + bar.get_height()/2, 
                 f"${width:.2f}", 
                 va='center', ha=ha, color='white', fontweight='bold')
        
        # Add avg confidence logic next to asset name
        avg_conf = df[df['asset'] == asset_pnl.index[i]]['confidence'].mean()
        ax3.text(min(0, min(asset_pnl.values)*1.1) if min(asset_pnl.values) < 0 else 0, bar.get_y() + bar.get_height()/2, 
                 f" (Conf: {avg_conf:.2f})", 
                 va='center', ha='left', color='#aaa')

    # ==========================================
    # 4. Trade Duration Analysis
    # ==========================================
    ax4 = plt.subplot(2, 2, 4)
    
    # Scatter plot: Duration vs PnL, colored by outcome
    colors_map = {'tp': '#00ff00', 'sl': '#ff3333', 'time': '#ffaa00'}
    
    for outcome in df['outcome'].unique():
        subset = df[df['outcome'] == outcome]
        ax4.scatter(subset['duration_hours'], subset['pnl_usd'], 
                    alpha=0.7, s=80, 
                    label=outcome.upper(), color=colors_map.get(outcome, '#888'),
                    edgecolor='black')
        
    ax4.axhline(0, color='white', linestyle='--', alpha=0.3)
    
    # Add trend line for TP vs SL bounds? No, let's keep it scatter for clear cluster view
    ax4.set_title("Trade Duration vs. Profit/Loss", fontsize=16)
    ax4.set_xlabel("Trade Duration (Hours)", fontsize=12)
    ax4.set_ylabel("PnL ($)", fontsize=12)
    ax4.legend(title="Outcome", title_fontsize='12', fontsize='10')
    ax4.grid(True, alpha=0.1)
    
    # Highlight max holding barrier
    max_duration = df['duration_hours'].max()
    ax4.axvline(max_duration, color='#ffaa00', linestyle=':', alpha=0.6, label="Time Barrier")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/performance_dashboard.png", dpi=300, bbox_inches='tight')
    print("[+] Saved comprehensive dashboard to plots/performance_dashboard.png")
    
    # Generate standalone R:R heatmap by confidence matrix
    plt.figure(figsize=(10, 8))
    # Create bins for confidence
    df['conf_bin'] = pd.cut(df['confidence'], bins=[0.80, 0.85, 0.92, 1.0], labels=['Low (1x)', 'Med (2x)', 'High (3x)'])
    
    heatmap_data = df.pivot_table(
        values='pnl_usd', 
        index='asset', 
        columns='conf_bin', 
        aggfunc='sum',
        fill_value=0
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", center=0, 
                cbar_kws={'label': 'Total PnL ($)'},
                linewidths=1, linecolor='black')
    
    plt.title("PnL Heatmap: Asset vs. Execution Confidence", fontsize=16)
    plt.ylabel("Asset")
    plt.xlabel("Confidence Tier (Leverage)")
    plt.savefig("plots/asset_confidence_heatmap.png", dpi=300, bbox_inches='tight')
    print("[+] Saved confidence heatmap to plots/asset_confidence_heatmap.png")


if __name__ == "__main__":
    generate_performance_report()
