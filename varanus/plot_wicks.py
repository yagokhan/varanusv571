import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from varanus.universe import TIER2_UNIVERSE
from varanus.pa_features import compute_atr

CACHE = "/home/yagokhan/chameleon/claude_code_project/data/cache"

def load_data(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(f"{CACHE}/{symbol}_USDT.parquet")
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

def compute_wick_intensity(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Computes Wick Intensity metrics.
    1. Average Relative Wick Size: (Wick Length / Candle Range) representing % of candle that is wick
    2. Average Wick ATR Ratio: (Wick Length / ATR) representing wick size normalized by volatility
    3. Flash Wick Frequency: % of candles where wick > 0.5 * ATR
    """
    # Calculate upper and lower wicks
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    
    # Total wick size
    total_wick = upper_wick + lower_wick
    
    # 1. Relative Wick Size (Wick / High-Low range)
    candle_range = df['high'] - df['low']
    # Avoid div by zero
    valid_range = candle_range > 0
    rel_wick_size = (total_wick[valid_range] / candle_range[valid_range]).mean()
    
    # 2. Wick / ATR Ratio
    atr = compute_atr(df, 14)
    valid_atr = atr > 0
    wick_atr_ratio = (total_wick[valid_atr] / atr[valid_atr]).mean()
    
    # 3. Flash Wick Frequency (Wick > 0.5 ATR)
    flash_wicks = (total_wick[valid_atr] > 0.5 * atr[valid_atr]).mean()
    
    return rel_wick_size, wick_atr_ratio, flash_wicks


def generate_wick_plot():
    os.makedirs("plots", exist_ok=True)
    
    results = []
    print("Computing wick intensity for Tier 2 Universe...")
    
    for asset in TIER2_UNIVERSE:
        try:
            df = load_data(asset)
            # Use last 1000 candles to match the backtest context
            df = df.tail(1000)
            
            rel_wick, wick_atr, flash_freq = compute_wick_intensity(df)
            
            results.append({
                'asset': asset,
                'rel_wick_size': rel_wick,
                'wick_atr_ratio': wick_atr,
                'flash_freq_pct': flash_freq * 100
            })
        except Exception as e:
            print(f"Skipping {asset}: {e}")
            
    res_df = pd.DataFrame(results)
    
    # Plotting
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Tier 2 Asset Wick Intensity Analysis (Last 1000 4H Candles)', fontsize=20, y=0.95)
    
    # 1. Wick to ATR Ratio Bar Chart
    ax1 = plt.subplot(2, 1, 1)
    # Sort by Wick:ATR descending
    sorted_df = res_df.sort_values('wick_atr_ratio', ascending=False)
    
    sns.barplot(
        x='asset', y='wick_atr_ratio', 
        data=sorted_df, 
        palette='magma',
        ax=ax1
    )
    
    ax1.set_title("Average Total Wick Length vs. ATR(14)", fontsize=15)
    ax1.set_ylabel("Wick Size (ATR Multiplier)", fontsize=12)
    ax1.set_xlabel("")
    ax1.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax1.axhline(sorted_df['wick_atr_ratio'].mean(), color='white', linestyle='--', alpha=0.5, label='Universe Average')
    ax1.legend()
    
    # Add data labels
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.2f}x", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='white', xytext=(0, 5), textcoords='offset points')

    # 2. Flash Wick Frequency
    ax2 = plt.subplot(2, 1, 2)
    # Sort by Flash Freq descending
    sorted_df_flash = res_df.sort_values('flash_freq_pct', ascending=False)
    
    sns.barplot(
        x='asset', y='flash_freq_pct', 
        data=sorted_df_flash, 
        palette='viridis',
        ax=ax2
    )
    
    ax2.set_title("Flash Wick Frequency (% of candles where total wicks > 0.5 ATR)", fontsize=15)
    ax2.set_ylabel("Frequency (%)", fontsize=12)
    ax2.set_xlabel("Asset", fontsize=12)
    ax2.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax2.axhline(sorted_df_flash['flash_freq_pct'].mean(), color='white', linestyle='--', alpha=0.5, label='Universe Average')
    ax2.legend()
    
    # Add data labels
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='white', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "plots/wick_intensity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved Wick Intensity plots to {save_path}")

if __name__ == "__main__":
    generate_wick_plot()
