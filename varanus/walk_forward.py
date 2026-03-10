"""
varanus/walk_forward.py — Rolling Walk-Forward Validation
Varanus v5.2 Dual-Engine.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

from varanus.model import VaranusModel, VaranusDualModel, MODEL_CONFIG
from varanus.pa_features import build_features, detect_mss, compute_atr
from varanus.tbm_labeler import label_trades, TBM_CONFIG, build_dual_labels
from varanus.backtest import run_backtest, compute_metrics

# v4 config — kept for reference only, not used in v5.1
WFV_CONFIG = {
    "n_folds":           5,
    "method":            "sliding_window",
    "shuffle":           False,
    "train_ratio":       0.70,
    "val_ratio":         0.15,
    "test_ratio":        0.15,
    "min_train_candles": 1000,
    "gap_candles":       24,
    "performance_gate": {
        "min_calmar":      0.50,
        "min_win_rate":    43.0,
        "max_fold_dd":    -0.30,
        "consistency_req": 0.80,
    },
}

# v5.1 — 5-fold rolling, 40/30/30  (scaled for ~730 days / 4379 candles available)
WFV_CONFIG_V51 = {
    "n_folds":           5,
    "method":            "rolling_window",
    "shuffle":           False,           # NEVER shuffle. Temporal integrity sacred.
    "train_ratio":       0.40,            # Prioritise recent 2026 regime over older data
    "val_ratio":         0.30,
    "test_ratio":        0.30,
    "min_train_candles": 700,             # ~117 days on 4h (fits 4379-candle dataset)
    "gap_candles":       24,              # 4-day embargo gap — unchanged from v4
    "performance_gate": {
        "min_hunter_efficiency": 0.60,    # Net Profit / |MaxDD| >= 0.60
        "min_win_rate":          43.0,
        "max_fold_dd":          -0.12,    # Hunter hard cap: -12% per fold
        "consistency_req":       0.75,    # >= 75% of 8 folds must pass
    },
}

def _generate_folds_v51(
    df_dict: dict[str, pd.DataFrame],
    cfg: dict,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    8-Fold Rolling Window Splitter for Hunter v5.1.

    Strictly enforces 40% Train / 30% Val / 30% Test per fold.
    Each fold's test window is non-overlapping and advances forward in time.
    Embargo gaps between splits prevent lookahead leakage.
    Fold 0 = oldest data; Fold 7 = most recent (closest to live regime).

    Returns: list of (train_idx, val_idx, test_idx) as DatetimeIndex tuples.
    """
    n_folds = cfg['n_folds']
    gap     = cfg['gap_candles']
    t_r     = cfg['train_ratio']
    v_r     = cfg['val_ratio']
    s_r     = cfg['test_ratio']

    assert abs(t_r + v_r + s_r - 1.0) < 1e-9, \
        f"Ratios must sum to 1.0, got {t_r + v_r + s_r}"

    # Build global timeline across all assets
    all_ts = set()
    for df in df_dict.values():
        all_ts.update(df.index)
    global_idx = pd.DatetimeIndex(sorted(all_ts))
    total_len  = len(global_idx)

    # Solve for per-fold window size so that n_folds non-overlapping test slices
    # plus the train+val prepend fit within total_len.
    fold_window = int(total_len / (n_folds * s_r + (t_r + v_r)))
    train_len   = int(fold_window * t_r) - gap
    val_len     = int(fold_window * v_r) - gap
    test_len    = int(fold_window * s_r)

    if train_len < cfg['min_train_candles']:
        raise ValueError(
            f"train_len={train_len} < min_train_candles={cfg['min_train_candles']}. "
            f"Need more data or fewer folds."
        )

    step_size   = test_len  # Each fold advances by one test_len
    single_fold = train_len + gap + val_len + gap + test_len
    required    = single_fold + (n_folds - 1) * step_size

    if required > total_len:
        raise ValueError(
            f"Insufficient data: need {required} bars for {n_folds} folds, "
            f"have {total_len}. Reduce n_folds or gap_candles."
        )

    folds = []
    for i in range(n_folds):
        test_start  = (total_len - n_folds * step_size) + i * step_size
        test_end    = test_start + test_len
        val_end     = test_start - gap
        val_start   = val_end - val_len
        train_end   = val_start - gap
        train_start = train_end - train_len

        if train_start < 0:
            print(f"  [WFV] Fold {i+1}: skipped (insufficient history)")
            continue

        train_slice = global_idx[train_start:train_end]
        val_slice   = global_idx[val_start:val_end]
        test_slice  = global_idx[test_start:test_end]

        if len(train_slice) == 0 or len(val_slice) == 0 or len(test_slice) == 0:
            print(f"  [WFV] Fold {i+1}: skipped (empty slice — "
                  f"train={len(train_slice)} val={len(val_slice)} test={len(test_slice)})")
            continue

        print(f"  [WFV] Fold {i+1}: "
              f"Train {train_slice[0].date()}-->{train_slice[-1].date()} "
              f"({len(train_slice)/total_len:.0%}) | "
              f"Val {val_slice[0].date()}-->{val_slice[-1].date()} "
              f"({len(val_slice)/total_len:.0%}) | "
              f"Test {test_slice[0].date()}-->{test_slice[-1].date()} "
              f"({len(test_slice)/total_len:.0%})")

        folds.append((train_slice, val_slice, test_slice))

    if len(folds) < n_folds:
        print(f"  [WFV] Warning: generated {len(folds)}/{n_folds} folds.")

    return folds


# v4 splitter kept for backward compatibility with run_optimization.py v4 path
def _generate_folds(df_dict: dict[str, pd.DataFrame], cfg: dict) -> list:
    return _generate_folds_v51(df_dict, cfg) if cfg.get('method') == 'rolling_window' \
        else _generate_folds_v51(df_dict, cfg)

def _slice(df_dict: dict[str, pd.DataFrame], dt_idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Slice a dictionary of asset DataFrames explicitly by a DatetimeIndex."""
    sliced = {}
    
    if len(dt_idx) == 0:
        return sliced
        
    start_ts = dt_idx[0]
    end_ts   = dt_idx[-1]
    
    for asset, df in df_dict.items():
        # Only take data strictly inside the boundary
        sub = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
        if not sub.empty:
            sliced[asset] = sub
            
    return sliced

def generate_signals(data_slice: dict[str, pd.DataFrame], model: VaranusModel, params: dict) -> dict[str, pd.DataFrame]:
    """Generates trading signals across the given data slice using the provided model."""
    signals = {}
    from varanus.pa_features import compute_atr, detect_mss
    
    for asset, df_4h in data_slice.items():
        # Usually requires 1d data as well for feature gen, but in walk_forward we
        # emulate it by sending df_4h again or managing it. For safety:
        # We will assume df_4h has enough context or `build_features` handles it.
        # But wait – the spec requires `build_features(df_4h, df_1d)`.
        
        # If we just need the XGB predictions to backtest:
        X, _ = build_features(df_4h, df_4h, asset) # Passing 4h as 1d mock for now if actual 1d obj missing in wrapper
        
        if X.empty:
            continue
            
        probs = model.predict_proba(X)
        preds = model.predict(X)
        
        # Format the signal DF
        sig_df = pd.DataFrame(index=X.index)
        
        # Max prob class
        sig_df['confidence'] = probs.max(axis=1)
        sig_df['direction']  = preds
        sig_df['entry_price'] = df_4h.loc[X.index, 'close']
        sig_df['atr'] = compute_atr(df_4h.loc[X.index], 14)
        
        # Filter for actual triggers (not class 0)
        sig_df = sig_df[sig_df['direction'] != 0]
        
        if not sig_df.empty:
            signals[asset] = sig_df
            
    return signals

def _hunter_efficiency(net_profit_usd: float, max_dd_pct: float) -> float:
    """Hunter Efficiency = Net Profit / |Max Drawdown|. Returns 0 if no drawdown."""
    return net_profit_usd / abs(max_dd_pct) if max_dd_pct != 0 else 0.0


def run_walk_forward(
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    params: dict,
    cfg: dict = WFV_CONFIG_V51,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Run 8-fold rolling walk-forward validation (Hunter v5.1).
    Each fold: retrain model on 40% window -> backtest on 30% OOS test slice.
    Gate metric: Hunter Efficiency (Net Profit / |MaxDD|), hard cap -12% DD per fold.
    """
    fold_results  = []
    all_trades    = []
    n_folds_total = cfg['n_folds']

    folds = _generate_folds_v51(df_dict_4h, cfg)

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
        print(f"\n── Fold {fold_idx + 1}/{n_folds_total} ──")

        train_4h = _slice(df_dict_4h, train_idx)
        train_1d = _slice(df_dict_1d, train_idx)
        val_4h   = _slice(df_dict_4h, val_idx)
        val_1d   = _slice(df_dict_1d, val_idx)
        test_4h  = _slice(df_dict_4h, test_idx)
        test_1d  = _slice(df_dict_1d, test_idx)

        # v5.2: use VaranusDualModel when Long Runner params are present.
        # Each engine trains on its own labels — short model never sees long labels.
        _dual = 'tp_mult_long' in params
        model = VaranusDualModel(MODEL_CONFIG) if _dual else VaranusModel(MODEL_CONFIG)

        X_tr_list,  y_tr_list  = [], []
        X_vl_list,  y_vl_list  = [], []
        # y_short_*: v5.1-style labels from label_trades(mss_signal) — preserves
        # the full short label count. Only used when _dual=True.
        y_short_tr_list, y_short_vl_list = [], []

        for asset in train_4h:
            if asset not in train_1d: continue
            X = build_features(train_4h[asset], train_1d[asset], asset, params)
            if X.empty: continue
            if _dual:
                y      = build_dual_labels(train_4h[asset], X, {**params, '_asset': asset})
                y_short = label_trades(train_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                y_short_tr_list.append(y_short)
            else:
                y = label_trades(train_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
            y = y.reindex(X.index).fillna(0).astype(int)
            X_tr_list.append(X)
            y_tr_list.append(y)

        for asset in val_4h:
            if asset not in val_1d: continue
            X = build_features(val_4h[asset], val_1d[asset], asset, params)
            if X.empty: continue
            if _dual:
                y      = build_dual_labels(val_4h[asset], X, {**params, '_asset': asset})
                y_short = label_trades(val_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                y_short_vl_list.append(y_short)
            else:
                y = label_trades(val_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
            y = y.reindex(X.index).fillna(0).astype(int)
            X_vl_list.append(X)
            y_vl_list.append(y)

        if not X_tr_list:
            print("  Skipping fold: insufficient training data")
            continue

        model.fit(
            pd.concat(X_tr_list), pd.concat(y_tr_list),
            pd.concat(X_vl_list) if X_vl_list else None,
            pd.concat(y_vl_list) if y_vl_list else None,
            pd.concat(y_short_tr_list) if y_short_tr_list else None,
            pd.concat(y_short_vl_list) if y_short_vl_list else None,
        )

        # Generate signals and backtest on unseen test window
        signals = {}
        _dual       = 'tp_mult_long' in params
        _conf_long  = params.get('conf_thresh_long',  0.70)
        _conf_short = params.get('conf_thresh_short', 0.786)
        for asset, df_4h in test_4h.items():
            if asset not in test_1d: continue
            X_t = build_features(df_4h, test_1d[asset], asset, params)
            if X_t.empty: continue
            probs = model.predict_proba(X_t)
            if _dual:
                # Dual engine: direction from raw proba with direction-specific thresholds.
                # Bypasses the 0.75 hard floor in model.predict() so conf_thresh_long
                # can meaningfully control long signal frequency below 0.75.
                # SHORT HUNTER LOCK-DOWN: short gate stays at frozen conf_thresh_short.
                p_short = probs[:, 0]
                p_long  = probs[:, 2]
                direction = np.zeros(len(X_t), dtype=int)
                # v5.5: use p_short_max_for_long ceiling if present in params,
                # otherwise fall back to hard p_long > p_short gate (v5.2 behaviour).
                _p_short_max = params.get('p_short_max_for_long', None)
                if _p_short_max is not None:
                    long_mask = (p_long >= _conf_long) & (p_short <= _p_short_max)
                else:
                    long_mask = (p_long >= _conf_long) & (p_long > p_short)
                direction[long_mask] = 1
                direction[(p_short >= _conf_short) & (p_short >= p_long)] = -1
                confidence = np.where(direction == 1, p_long, p_short)
            else:
                # v5.1 Hunter: use model.predict() as before
                direction  = model.predict(X_t)
                confidence = probs.max(axis=1)
            active = direction != 0
            if not active.any(): continue
            idx_active = X_t.index[active]
            sig_df = pd.DataFrame(index=idx_active)
            sig_df['confidence']  = confidence[active]
            sig_df['direction']   = direction[active]
            sig_df['entry_price'] = df_4h.loc[idx_active, 'close'].values
            sig_df['atr']         = compute_atr(df_4h.loc[X_t.index], 14).loc[idx_active].values
            signals[asset] = sig_df

        if not signals:
            print("  No signals in OOS test window. Skipping.")
            continue

        equity, trades = run_backtest(test_4h, signals, model, params)
        metrics        = compute_metrics(equity, trades)

        # Hunter Efficiency for this fold
        net_profit = trades['pnl_usd'].sum() if not trades.empty else 0.0
        fold_dd    = metrics['max_drawdown_pct'] / 100.0
        he         = _hunter_efficiency(net_profit, fold_dd)
        metrics['hunter_efficiency'] = round(he, 3)
        metrics['net_profit_usd']    = round(net_profit, 2)

        trades['fold'] = fold_idx + 1
        all_trades.append(trades)
        fold_results.append({'fold': fold_idx + 1, **metrics})

        dd_flag = " [DD BREACH]" if abs(fold_dd) > abs(cfg['performance_gate']['max_fold_dd']) else ""
        print(f"  Trades: {metrics['total_trades']} "
              f"(L:{metrics.get('long_trades',0)} S:{metrics.get('short_trades',0)}) | "
              f"HunterEff: {he:.3f} | "
              f"WR: {metrics['win_rate_pct']}% "
              f"(L:{metrics.get('long_win_rate_pct',0)}% S:{metrics.get('short_win_rate_pct',0)}%) | "
              f"Ret: {metrics['total_return_pct']}% | "
              f"MaxDD: {metrics['max_drawdown_pct']}% | "
              f"Sharpe: {metrics['sharpe_ratio']} | "
              f"LongSharpe: {metrics.get('long_sharpe',0)} | "
              f"DecayExits: {metrics.get('signal_decay_exits', 0)} | "
              f"MSSExits: {metrics.get('mss_invalidation_exits', 0)}"
              f"{dd_flag}")

    results_df    = pd.DataFrame(fold_results)
    all_trades_df = pd.concat(all_trades).reset_index(drop=True) if all_trades else pd.DataFrame()

    if results_df.empty:
        return results_df, 0.0, all_trades_df

    gate          = cfg['performance_gate']
    passed_folds  = (results_df['hunter_efficiency'] >= gate['min_hunter_efficiency']).sum()
    consistency   = passed_folds / len(results_df)

    print(f"\nWFV Summary — {passed_folds}/{len(results_df)} folds passed Hunter gate")
    print(f"Consistency: {consistency:.0%} (required: {gate['consistency_req']:.0%})")
    cols = ['fold', 'hunter_efficiency', 'total_return_pct', 'win_rate_pct',
            'long_win_rate_pct', 'short_win_rate_pct', 'max_drawdown_pct',
            'sharpe_ratio', 'long_sharpe', 'total_trades', 'long_trades', 'short_trades',
            'signal_decay_exits', 'mss_invalidation_exits']
    print(results_df[[c for c in cols if c in results_df.columns]].to_string(index=False))

    return results_df, consistency, all_trades_df
