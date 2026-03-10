import pandas as pd

from varanus.universe import HIGH_VOL_SUBTIER

RISK_CONFIG = {
    "initial_capital":          5_000.0,
    "max_portfolio_leverage":   2.5,      # Weighted avg across all open positions
    "max_concurrent_positions": 4,
    "corr_block_threshold":     0.75,     # Block if rolling corr > 0.75 with open asset
    "corr_lookback_days":       20,
    "position_size_scalar": {
        "standard": 1.00,
        "high_vol": 0.75,                 # TAO, ASTR, KITE, ICP
    },
    "leverage_map": {
        (0.80, 0.85): 1.0,
        (0.85, 0.92): 2.0,
        (0.92, 1.00): 3.0,               # Tier 2 absolute cap
    },
    "daily_loss_limit_pct":     0.05,    # Halt all signals if -5% in 24h
    "portfolio_stop_pct":       0.15,    # Halt all signals if -15% from peak
}

def check_portfolio_health(equity_curve: pd.Series,
                            cfg: dict = RISK_CONFIG) -> dict:
    """
    Evaluate portfolio-level circuit breakers.
    Takes a pd.Series of the portfolio equity indexed by strictly monotonic timestamps.
    """
    if len(equity_curve) == 0:
        return {
            "current_equity":  cfg["initial_capital"],
            "daily_loss_pct":  0.0,
            "drawdown_pct":    0.0,
            "halt_signals":    False,
        }

    current = equity_curve.iloc[-1]
    peak    = equity_curve.cummax().iloc[-1]
    
    # Calculate daily loss. We look for the equity value exactly 24 hours ago.
    # If the curve spans less than 24h, we use the very first entry as day_start.
    current_time = equity_curve.index[-1]
    cutoff_time = current_time - pd.Timedelta(days=1)
    
    past_24h = equity_curve[equity_curve.index >= cutoff_time]
    
    if len(past_24h) > 0:
        day_start = past_24h.iloc[0]
    else:
        day_start = equity_curve.iloc[0]

    daily_loss = (current - day_start) / day_start
    drawdown   = (current - peak) / peak if peak > 0 else 0

    return {
        "current_equity":  current,
        "daily_loss_pct":  round(daily_loss * 100, 2),
        "drawdown_pct":    round(drawdown * 100, 2),
        "halt_signals":    bool((daily_loss <= -cfg['daily_loss_limit_pct']) or
                                (drawdown <= -cfg['portfolio_stop_pct'])),
    }


def get_leverage(confidence: float, cfg: dict = RISK_CONFIG) -> float:
    """Map a model confidence score to the appropriate leverage tier."""
    for (lo, hi), lev in cfg['leverage_map'].items():
        if lo <= confidence < hi:
            return lev
    return 1.0  # Safe default below the lowest threshold


def get_position_size(confidence: float, capital: float, asset: str,
                      cfg: dict = RISK_CONFIG) -> float:
    """
    Calculate position size in USD.
    High-vol sub-tier assets receive a 0.75× size scalar.
    """
    lev    = get_leverage(confidence, cfg)
    scalar = (cfg['position_size_scalar']['high_vol']
              if asset in HIGH_VOL_SUBTIER
              else cfg['position_size_scalar']['standard'])
    return (capital * lev * scalar) / cfg['max_concurrent_positions']


def compute_portfolio_leverage(open_trades: dict, capital: float) -> float:
    """
    Return current portfolio leverage = total notional / capital.
    Returns 0.0 when no positions are open or capital is zero.
    """
    if capital <= 0 or not open_trades:
        return 0.0
    total_notional = sum(t['position_usd'] for t in open_trades.values())
    return total_notional / capital


def would_breach_leverage(open_trades: dict, capital: float,
                          new_sig: dict, cfg: dict = RISK_CONFIG) -> bool:
    """Return True if adding the new signal would exceed max_portfolio_leverage."""
    if capital <= 0:
        return True
    new_size = get_position_size(
        new_sig['confidence'], capital, new_sig.get('asset', ''), cfg)
    current_notional = sum(t['position_usd'] for t in open_trades.values())
    return (current_notional + new_size) / capital > cfg['max_portfolio_leverage']


def is_correlated_to_open(asset: str, open_trades: dict,
                           data: dict, cfg: dict = RISK_CONFIG) -> bool:
    """
    Block entry if the candidate asset's recent returns are too highly correlated
    with any currently open position (|corr| >= corr_block_threshold).
    Requires at least 20 overlapping bars to compute; skips if data is insufficient.
    """
    if not open_trades or asset not in data:
        return False

    lookback = cfg['corr_lookback_days'] * 6   # 6 × 4h bars per day
    asset_returns = data[asset]['close'].pct_change().dropna().tail(lookback)

    for open_asset in open_trades:
        if open_asset not in data or open_asset == asset:
            continue
        open_returns = data[open_asset]['close'].pct_change().dropna().tail(lookback)
        combined = pd.concat([asset_returns, open_returns], axis=1, join='inner').dropna()
        if len(combined) < 20:
            continue
        corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        if pd.notna(corr) and abs(corr) >= cfg['corr_block_threshold']:
            return True

    return False
