"""
varanus/backtest.py — Simulation Engine
Varanus v5.2 Dual-Engine.

Supports directional-specific parameter sets:
  - Short Hunter (direction == -1): frozen Trial #183 params (tp=5.768x, sl=0.709x, conf=0.786)
  - Long Runner  (direction ==  1): optimised Long params (tp=2.5-4.5x, sl=0.60-1.0x, conf=0.680-0.850)
"""
import pandas as pd
import numpy as np

from varanus.tbm_labeler import calculate_barriers, TBM_CONFIG
from varanus.model import get_leverage_v51
from varanus.universe import HIGH_VOL_SUBTIER
from varanus.risk import is_correlated_to_open as _risk_is_correlated
from varanus.pa_features import detect_mss

# ── v5.2 Short Hunter frozen params (Trial #183 — DO NOT MODIFY) ───────────────
V52_SHORT_FROZEN_PARAMS = {
    "conf_thresh_short":   0.786,   # confidence_thresh from Trial #183
    "tp_atr_mult":         5.768,   # tp_atr_mult from Trial #183
    "sl_atr_mult":         0.709,   # sl_atr_mult from Trial #183
    "leverage_5x_trigger": 0.968,   # leverage_5x_trigger from Trial #183
}

BACKTEST_CONFIG = {
    # Execution realism
    "initial_capital":     5_000.0,   # USD
    "maker_fee":           0.0002,    # 0.02% — limit order assumption
    "taker_fee":           0.0005,    # 0.05% — market order fallback
    "slippage_pct":        0.0008,    # 0.08% avg mid-cap slippage on 4h bar open
    "entry_on_bar":        "open",    # Enter on next bar open after signal candle

    # Flash-wick handling
    "use_flash_wick_guard":     True,
    "wick_body_close_required": True,

    # Portfolio constraints (enforced every bar)
    "max_concurrent_positions": 4,
    "max_portfolio_leverage":   2.5,
    "corr_block_threshold":     0.75, # Block new trade if open asset corr > 0.75
    "corr_lookback_days":       20,

    # Reporting
    "equity_curve_freq": "4h",
    "trade_log":         True,
}

BACKTEST_PASS_CRITERIA = {
    "min_trades":        50,    # Statistical significance floor
    "min_win_rate":      0.43,
    "min_calmar":        0.50,
    "max_drawdown":     -0.35,  # Hard reject above 35% drawdown
    "min_profit_factor": 1.30,
    "min_sharpe":        0.80,
}

HUNTER_ACTIVE_CONFIG = {
    "signal_decay": {
        "enabled":         True,
        "decay_threshold": 0.35,   # Exit if entry_confidence - current_confidence >= 0.35
    },
    "dynamic_breakeven": {
        "enabled":          True,
        "trigger_pct":      0.75,  # Move SL to entry when 75% of TP distance reached
        "buffer_atr_ratio": 0.05,  # SL = entry +/- (0.05 x ATR) beyond entry
    },
    "mss_invalidation": {
        "enabled": True,           # Exit immediately if 4h MSS flips against trade
    },
}

def _check_barriers(bar: pd.Series, trade: dict, cfg: dict) -> dict | None:
    """
    Check TP, SL, and time barrier for a bar.
    Flash-wick guard: SL requires body close beyond level, not wick touch.
    """
    d = trade['direction']

    # Time barrier (checked first — prevents holding decaying positions)
    if bar.name >= trade['max_hold_bar']:
        return {'type': 'time', 'price': bar['close']}

    # Take-Profit — wick touch is sufficient (we want the gain)
    if d ==  1 and bar['high'] >= trade['take_profit']:
        return {'type': 'tp', 'price': trade['take_profit']}
    if d == -1 and bar['low']  <= trade['take_profit']:
        return {'type': 'tp', 'price': trade['take_profit']}

    # Stop-Loss — flash-wick guard requires body close beyond SL
    if cfg['use_flash_wick_guard'] and cfg['wick_body_close_required']:
        if d ==  1 and bar['close'] < trade['stop_loss']:
            return {'type': 'sl', 'price': trade['stop_loss']}
        if d == -1 and bar['close'] > trade['stop_loss']:
            return {'type': 'sl', 'price': trade['stop_loss']}
    else:
        if d ==  1 and bar['low']  <= trade['stop_loss']:
            return {'type': 'sl', 'price': trade['stop_loss']}
        if d == -1 and bar['high'] >= trade['stop_loss']:
            return {'type': 'sl', 'price': trade['stop_loss']}

    return None

def _apply_hunter_active_management(
    bar:            pd.Series,
    trade:          dict,
    current_proba:  float,
    current_mss:    int,
    atr:            float,
    cfg:            dict = HUNTER_ACTIVE_CONFIG,
) -> dict | None:
    """
    Evaluate Hunter v5.1 active management conditions for a single bar.
    Called BEFORE _check_barriers on every open trade.

    Priority: MSS Invalidation > Signal Decay > Dynamic Breakeven

    Returns an outcome dict if an exit is triggered, else None.
    Dynamic Breakeven mutates trade['stop_loss'] in place — not an exit.
    """
    direction = trade['direction']

    # 1. MSS Invalidation — structure flipped, thesis is broken
    if cfg['mss_invalidation']['enabled']:
        if direction == 1 and current_mss == -1:
            return {'type': 'mss_invalidation', 'price': bar['open']}
        if direction == -1 and current_mss == 1:
            return {'type': 'mss_invalidation', 'price': bar['open']}

    # 2. Signal Decay Exit — model confidence collapsed since entry
    if cfg['signal_decay']['enabled']:
        decay = trade['entry_confidence'] - current_proba
        if decay >= cfg['signal_decay']['decay_threshold']:
            return {'type': 'signal_decay', 'price': bar['open']}

    # 3. Dynamic Breakeven — lock in floor once 75% of TP distance reached
    if cfg['dynamic_breakeven']['enabled'] and atr > 0:
        buffer      = cfg['dynamic_breakeven']['buffer_atr_ratio'] * atr
        trigger_pct = cfg['dynamic_breakeven']['trigger_pct']
        if direction == 1:
            target_dist = trade['take_profit'] - trade['entry_price']
            if bar['high'] >= trade['entry_price'] + trigger_pct * target_dist:
                new_sl = trade['entry_price'] + buffer
                if new_sl > trade['stop_loss']:
                    trade['stop_loss']           = new_sl
                    trade['breakeven_activated'] = True
        elif direction == -1:
            target_dist = trade['entry_price'] - trade['take_profit']
            if bar['low'] <= trade['entry_price'] - trigger_pct * target_dist:
                new_sl = trade['entry_price'] - buffer
                if new_sl < trade['stop_loss']:
                    trade['stop_loss']           = new_sl
                    trade['breakeven_activated'] = True

    return None


def _calculate_pnl(trade: dict, outcome: dict, cfg: dict) -> float:
    """Net PnL after fees and slippage."""
    raw_ret  = trade['direction'] * (outcome['price'] - trade['entry_price']) \
               / trade['entry_price']
    # signal_decay and mss_invalidation are market exits — taker fee applies
    taker_exits = ('sl', 'signal_decay', 'mss_invalidation', 'trailing_sl_hit')
    fee      = cfg['taker_fee'] if outcome['type'] in taker_exits else cfg['maker_fee']
    net_ret  = raw_ret - fee - cfg['slippage_pct']
    return trade['position_usd'] * net_ret

def _would_breach_leverage(open_trades: dict, capital: float, new_sig: dict,
                           cfg: dict, leverage_5x_trigger: float = 0.96) -> bool:
    """Check if adding new trade breaches max portfolio leverage."""
    if capital <= 0: return True
    current_notional = sum(t['position_usd'] for t in open_trades.values())
    new_notional = (capital
                    * get_leverage_v51(new_sig['confidence'], leverage_5x_trigger)
                    * (0.75 if new_sig.get('asset') in HIGH_VOL_SUBTIER else 1.0)
                    / cfg['max_concurrent_positions'])
    return (current_notional + new_notional) / capital > cfg['max_portfolio_leverage']

def _is_correlated_to_open(asset: str, open_trades: dict, corr_cache: dict, data: dict, cfg: dict) -> bool:
    """Block entry if correlated to currently open positions."""
    return _risk_is_correlated(asset, open_trades, data, cfg)

def run_backtest(
    data:    dict[str, pd.DataFrame],   # {asset: OHLCV DataFrame}
    signals: dict[str, pd.DataFrame],   # {asset: signal DataFrame from STEP 2}
    model,                              # Trained model from STEP 4
    params:  dict,
    cfg:     dict = BACKTEST_CONFIG,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simulate the full Varanus Tier 2 strategy over historical data.

    Returns
    -------
    equity_curve : pd.Series  (indexed by timestamp)
    trade_log    : pd.DataFrame
    """
    capital             = cfg['initial_capital']
    equity              = {}
    open_trades         = {}   # {asset: trade_dict}
    trade_log           = []
    corr_cache          = {}
    leverage_5x_trigger = params.get('leverage_5x_trigger', 0.96)
    mss_lookback        = params.get('mss_lookback', 40)

    # Precompute MSS series for all assets (used by MSS Invalidation check)
    mss_cache = {
        asset: detect_mss(df, mss_lookback)
        for asset, df in data.items()
    }

    all_timestamps = sorted(set().union(*[df.index for df in data.values()]))

    for ts in all_timestamps:

        # ── 1. Check barrier outcomes for all open trades ─────────────────
        for asset, trade in list(open_trades.items()):
            if ts not in data[asset].index:
                continue
            bar = data[asset].loc[ts]

            # Resolve current confidence for signal decay check
            sig_df = signals.get(asset)
            if sig_df is not None and ts in sig_df.index:
                row = sig_df.loc[ts]
                current_proba = (row['confidence']
                                 if row['direction'] == trade['direction']
                                 else trade['entry_confidence'])
            else:
                current_proba = trade['entry_confidence']  # no new signal = no decay

            # Resolve current MSS for invalidation check
            current_mss = int(mss_cache.get(asset, pd.Series(dtype=int)).get(ts, 0))

            # Resolve current ATR for dynamic breakeven buffer
            atr_series = data[asset]['high'].rolling(14).max() - data[asset]['low'].rolling(14).min()
            current_atr = float(atr_series.get(ts, 0) or 0)

            # ── Dynamic Trailing Stop — runs BEFORE hunter and standard barriers
            TRAIL_TRIGGER_PCT  = 0.01147
            TRAIL_DISTANCE_PCT = 0.01147

            d  = trade['direction']
            ep = trade['entry_price']

            if d == 1:  # LONG
                profit_pct = (bar['high'] - ep) / ep
                if profit_pct >= TRAIL_TRIGGER_PCT:
                    trade['trail_active'] = True
                if trade['trail_active']:
                    peak = trade['trail_peak']
                    if peak is None or bar['high'] > peak:
                        trade['trail_peak'] = bar['high']
                        trade['trail_stop'] = bar['high'] * (1 - TRAIL_DISTANCE_PCT)
                    if bar['close'] < trade['trail_stop']:
                        outcome = {'type': 'trailing_sl_hit', 'price': trade['trail_stop']}
                        pnl = _calculate_pnl(trade, outcome, cfg)
                        capital += pnl
                        trade_log.append({
                            **trade,
                            'exit_ts':    ts,
                            'exit_price': outcome['price'],
                            'outcome':    outcome['type'],
                            'pnl_usd':    pnl,
                        })
                        del open_trades[asset]
                        continue

            elif d == -1:  # SHORT
                profit_pct = (ep - bar['low']) / ep
                if profit_pct >= TRAIL_TRIGGER_PCT:
                    trade['trail_active'] = True
                if trade['trail_active']:
                    peak = trade['trail_peak']
                    if peak is None or bar['low'] < peak:
                        trade['trail_peak'] = bar['low']
                        trade['trail_stop'] = bar['low'] * (1 + TRAIL_DISTANCE_PCT)
                    if bar['close'] > trade['trail_stop']:
                        outcome = {'type': 'trailing_sl_hit', 'price': trade['trail_stop']}
                        pnl = _calculate_pnl(trade, outcome, cfg)
                        capital += pnl
                        trade_log.append({
                            **trade,
                            'exit_ts':    ts,
                            'exit_price': outcome['price'],
                            'outcome':    outcome['type'],
                            'pnl_usd':    pnl,
                        })
                        del open_trades[asset]
                        continue

            # Hunter active management — runs BEFORE standard barriers
            outcome = _apply_hunter_active_management(
                bar, trade, current_proba, current_mss, current_atr
            )

            # Standard barrier check if hunter didn't trigger
            if outcome is None:
                outcome = _check_barriers(bar, trade, cfg)

            if outcome:
                pnl     = _calculate_pnl(trade, outcome, cfg)
                capital += pnl
                trade_log.append({
                    **trade,
                    'exit_ts':    ts,
                    'exit_price': outcome['price'],
                    'outcome':    outcome['type'],
                    'pnl_usd':    pnl,
                })
                del open_trades[asset]

        # ── 2. Evaluate new signals ────────────────────────────────────────
        # Gather all signals for this timestamp
        current_signals = []
        for asset, sig_df in signals.items():
            if ts in sig_df.index and asset not in open_trades:
                sig = sig_df.loc[ts].copy()
                sig['asset'] = asset
                direction = int(sig.get('direction', 0))
                # v5.2 Dual-Engine: direction-specific confidence thresholds
                if direction == 1:   # Long Runner
                    thresh = params.get('conf_thresh_long',
                                        params.get('confidence_thresh', 0.750))
                elif direction == -1:  # Short Hunter (frozen)
                    thresh = params.get('conf_thresh_short',
                                        V52_SHORT_FROZEN_PARAMS['conf_thresh_short'])
                else:
                    continue
                if sig.get('confidence', 0) >= thresh:
                    current_signals.append(sig)
        
        # Sort signals by confidence (highest first) to prevent list-order bias
        current_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        for sig in current_signals:
            if len(open_trades) >= cfg['max_concurrent_positions']:
                break # Portfolio full
                
            asset = sig['asset']
            if _would_breach_leverage(open_trades, capital, sig, cfg, leverage_5x_trigger):
                continue
            if _is_correlated_to_open(asset, open_trades, corr_cache, data, cfg):
                continue

            # R:R gate before entry — v5.2 directional TP/SL multipliers
            tbm_cfg   = TBM_CONFIG.copy()
            sig_dir   = int(sig.get('direction', 0))
            if sig_dir == 1:   # Long Runner params
                tbm_cfg['take_profit_atr'] = params.get(
                    'tp_mult_long',  params.get('tp_atr_mult', 3.5))
                tbm_cfg['stop_loss_atr']   = params.get(
                    'sl_mult_long',  params.get('sl_atr_mult', 0.80))
            else:               # Short Hunter frozen params
                tbm_cfg['take_profit_atr'] = params.get(
                    'tp_atr_mult', V52_SHORT_FROZEN_PARAMS['tp_atr_mult'])
                tbm_cfg['stop_loss_atr']   = params.get(
                    'sl_atr_mult', V52_SHORT_FROZEN_PARAMS['sl_atr_mult'])
            barriers = calculate_barriers(
                sig['entry_price'], sig['atr'], sig['direction'], tbm_cfg, asset)
            if not barriers.get('min_rr_satisfied', True):
                continue

            # Position sizing — v5.1 leverage schedule with 5x tier
            lev          = get_leverage_v51(sig['confidence'], leverage_5x_trigger)
            size_scalar  = 0.75 if asset in HIGH_VOL_SUBTIER else 1.0
            position_usd = (capital * lev * size_scalar) / cfg['max_concurrent_positions']

            open_trades[asset] = {
                'asset':            asset,
                'entry_ts':         ts,
                'entry_price':      sig['entry_price'],
                'direction':        sig['direction'],
                'take_profit':      barriers['take_profit'],
                'stop_loss':        barriers['stop_loss'],
                'position_usd':     position_usd,
                'leverage':         lev,
                'confidence':       sig['confidence'],
                'entry_confidence': sig['confidence'],  # v5.1: stored for decay check
                'breakeven_activated': False,
                'rr_ratio':         barriers.get('rr_ratio', 0),
                'max_hold_bar':     ts + pd.Timedelta(hours=4 * params.get('max_holding', 30)),
                'trail_active':     False,
                'trail_peak':       None,
                'trail_stop':       None,
            }

        equity[ts] = capital

    # ── 3. Force-close remaining open trades at end-of-test ────────────
    if len(all_timestamps) > 0:
        last_ts = all_timestamps[-1]
        for asset, trade in list(open_trades.items()):
            if last_ts in data[asset].index:
                last_price = data[asset].loc[last_ts, 'close']
                outcome = {'type': 'time', 'price': last_price} # Classify EOT forced-close as time exit
                pnl = _calculate_pnl(trade, outcome, cfg)
                capital += pnl
                trade_log.append({
                    **trade,
                    'exit_ts':    last_ts,
                    'exit_price': last_price,
                    'outcome':    'time',
                    'pnl_usd':    pnl,
                })
        equity[last_ts] = capital

    return pd.Series(equity), pd.DataFrame(trade_log)


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    return ((equity - roll_max) / roll_max).min()


def compute_metrics(equity_curve: pd.Series,
                    trade_log: pd.DataFrame) -> dict:
    """Full performance report for one backtest run."""
    returns   = equity_curve.pct_change().dropna()
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_days    = (equity_curve.index[-1] - equity_curve.index[0]).days
    cagr      = (1 + total_ret) ** (365 / n_days) - 1 if n_days > 0 else 0
    max_dd    = _max_drawdown(equity_curve)
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0
    sharpe    = returns.mean() / returns.std() * (365 * 6) ** 0.5  # 4h = 6 bars/day

    wins  = trade_log['pnl_usd'] > 0 if len(trade_log) else pd.Series(dtype=bool)
    loss  = trade_log['pnl_usd'] < 0 if len(trade_log) else pd.Series(dtype=bool)
    profit_factor = (
        abs(trade_log.loc[wins, 'pnl_usd'].sum() /
            trade_log.loc[loss, 'pnl_usd'].sum())
        if loss.any() else float('inf')
    )
    by_outcome = trade_log['outcome'].value_counts() if len(trade_log) else {}

    # ── v5.2 Directional breakdown (Long Runner vs Short Hunter) ─────────────
    if len(trade_log) > 0 and 'direction' in trade_log.columns:
        lt = trade_log[trade_log['direction'] == 1]
        st = trade_log[trade_log['direction'] == -1]
        l_wins = lt['pnl_usd'] > 0 if len(lt) else pd.Series(dtype=bool)
        s_wins = st['pnl_usd'] > 0 if len(st) else pd.Series(dtype=bool)
        l_pnl  = lt['pnl_usd'] if len(lt) > 1 else pd.Series([0.0])
        long_sharpe  = (l_pnl.mean() / l_pnl.std() * np.sqrt(252)
                        if l_pnl.std() > 0 else 0.0)
        long_net_pnl  = round(lt['pnl_usd'].sum(), 2) if len(lt) else 0.0
        short_net_pnl = round(st['pnl_usd'].sum(), 2) if len(st) else 0.0
    else:
        l_wins = s_wins = pd.Series(dtype=bool)
        long_sharpe = long_net_pnl = short_net_pnl = 0.0
        lt = st = pd.DataFrame()

    return {
        # ── Portfolio metrics (% returns primary) ──────────────────────────
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct":         round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio":     round(calmar, 3),
        "sharpe_ratio":     round(sharpe, 3),
        "win_rate_pct":     round(wins.mean() * 100, 2) if len(trade_log) else 0,
        "profit_factor":    round(profit_factor, 2),
        "total_trades":     len(trade_log),
        "tp_hits":                by_outcome.get('tp', 0),
        "sl_hits":                by_outcome.get('sl', 0),
        "time_exits":             by_outcome.get('time', 0),
        "signal_decay_exits":     by_outcome.get('signal_decay', 0),
        "mss_invalidation_exits": by_outcome.get('mss_invalidation', 0),
        "trailing_sl_exits":      by_outcome.get('trailing_sl_hit', 0),
        "avg_win_usd":      round(trade_log.loc[wins, 'pnl_usd'].mean(), 2) if wins.any() else 0,
        "avg_loss_usd":     round(trade_log.loc[loss, 'pnl_usd'].mean(), 2) if loss.any() else 0,
        # ── v5.2 Directional breakdown ─────────────────────────────────────
        "long_trades":         len(lt),
        "short_trades":        len(st),
        "long_win_rate_pct":   round(l_wins.mean() * 100, 2) if l_wins.any() else 0.0,
        "short_win_rate_pct":  round(s_wins.mean() * 100, 2) if s_wins.any() else 0.0,
        "long_net_pnl_pct":    round(long_net_pnl / equity_curve.iloc[0] * 100, 2)
                               if len(equity_curve) > 0 else 0.0,
        "short_net_pnl_pct":   round(short_net_pnl / equity_curve.iloc[0] * 100, 2)
                               if len(equity_curve) > 0 else 0.0,
        "long_sharpe":         round(long_sharpe, 3),
    }

def passes_backtest_gate(metrics: dict) -> bool:
    c = BACKTEST_PASS_CRITERIA
    return (
        metrics['total_trades']      >= c['min_trades']         and
        metrics['win_rate_pct']      >= c['min_win_rate'] * 100 and
        metrics['calmar_ratio']      >= c['min_calmar']         and
        metrics['max_drawdown_pct']  >= c['max_drawdown'] * 100 and
        metrics['profit_factor']     >= c['min_profit_factor']  and
        metrics['sharpe_ratio']      >= c['min_sharpe']
    )
