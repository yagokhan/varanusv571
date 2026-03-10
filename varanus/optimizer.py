"""
varanus/optimizer.py — Optuna Hyperparameter Search
Varanus v5.2 Dual-Engine.

Two objective modes:
  1. optuna_objective_hunter     — v5.1 Hunter (legacy, full param search)
  2. optuna_objective_dual_engine — v5.2 Long Runner only
       - Short Hunter params FROZEN from Trial #183
       - Searches: conf_thresh_long, tp_mult_long, sl_mult_long
       - Objective: Density Score = (WR × Count) / DD_Impact
"""
import math
import optuna
import pandas as pd
import numpy as np
import json

from varanus.walk_forward import WFV_CONFIG_V51, _generate_folds_v51, _slice
from varanus.backtest import run_backtest, compute_metrics, V52_SHORT_FROZEN_PARAMS
from varanus.model import VaranusModel, VaranusDualModel, MODEL_CONFIG
from varanus.pa_features import build_features, compute_atr
from varanus.tbm_labeler import label_trades, TBM_CONFIG, build_dual_labels
import pandas as pd

HUNTER_OPTUNA_CONFIG = {
    "n_trials":              300,
    "direction":             "maximize",
    "sampler":               "TPESampler",
    "pruner":                "HyperbandPruner",
    "min_trades_per_fold":   8,
    "min_total_trades":      30,
    "dd_penalty_threshold":  0.12,   # Folds breaching -12% DD get penalised
    "dd_penalty_multiplier": 0.40,
}

# v4 frozen params — not searched by Optuna, loaded from best_params.json
V4_FROZEN_PARAMS = {
    "mss_lookback":      31,
    "fvg_min_atr_ratio": 0.392,
    "sweep_min_pct":     0.00641,
    "fvg_max_age":       22,
    "rvol_threshold":    1.287,
    "rsi_oversold":      36,
    "rsi_overbought":    58,
    "max_holding":       31,
    "xgb_n_estimators":  218,
    "xgb_subsample":     0.957,
}

# ── v5.2 Dual-Engine config ────────────────────────────────────────────────────

DUAL_ENGINE_OPTUNA_CONFIG = {
    "n_trials":                300,
    "direction":               "maximize",
    "sampler":                 "TPESampler",
    "pruner":                  "HyperbandPruner",
    "min_long_trades_per_fold": 3,    # Per fold minimum (relaxed — OOS windows are ~14% of data)
    "min_total_long_trades":    60,   # v5.2.2 Deep Search: gate restored to target after
                                      # training imbalance fix (scale_pos_weight on long arm).
                                      # Trials producing < 60 long trades total → -999.0 penalty.
                                      # Target range: 60–100.
    "min_long_win_rate":        0.35, # Hard floor: 35% long win rate to avoid noise
    "dd_penalty_threshold":     0.12,
    "dd_penalty_multiplier":    0.40,
}

# Long Runner search space (v5.2 Deep Search)
# SHORT HUNTER LOCK-DOWN: These ranges apply to direction==1 only.
# sl_mult_long is REMOVED from search space — fixed hard stop at 1.0x ATR (see objective).
LONG_RUNNER_SEARCH_SPACE = {
    "conf_thresh_long": (0.55, 0.72),   # v5.2 Deep Search: lowered from (0.60, 0.75)
                                         # Floor 0.55 maximises signal density;
                                         # ceiling 0.72 maintains quality gate.
    "tp_mult_long":     (2.5,  4.5),    # Unchanged — step-like targets
}


def _hunter_efficiency(net_profit_usd: float, max_dd_pct: float) -> float:
    """Hunter Efficiency = Net Profit / |Max Drawdown|. Returns 0 if no drawdown."""
    return net_profit_usd / abs(max_dd_pct) if max_dd_pct != 0 else 0.0


def optuna_objective_hunter(
    trial: optuna.Trial,
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    cfg: dict = WFV_CONFIG_V51,
) -> float:
    """
    Hunter Efficiency Optuna Objective for Varanus v5.1.

    Searches 6 Hunter parameters across 8-fold rolling walk-forward.
    Reports per-fold intermediate scores so HyperbandPruner can prune early.
    Applies 0.40x penalty on any fold where MaxDD exceeds -12%.

    Search Space (6 Hunter Parameters)
    -----------------------------------
    1. confidence_thresh   — entry gate          [0.750 – 0.880]
    2. sl_atr_mult         — stop loss ATR mult  [0.700 – 1.200]
    3. tp_atr_mult         — take profit ATR mult [3.500 – 6.000]
    4. leverage_5x_trigger — 5x lev gate         [0.930 – 0.980]
    5. xgb_lr              — XGBoost learn rate  [0.005 – 0.080]
    6. xgb_max_depth       — XGBoost tree depth  [3 – 6]
    """
    params = {
        # 1. Entry Gate
        "confidence_thresh":   trial.suggest_float("confidence_thresh",   0.750, 0.880),
        # 2. Stop Loss
        "sl_atr_mult":         trial.suggest_float("sl_atr_mult",         0.700, 1.200),
        # 3. Take Profit
        "tp_atr_mult":         trial.suggest_float("tp_atr_mult",         3.500, 6.000),
        # 4. 5x Leverage Trigger
        "leverage_5x_trigger": trial.suggest_float("leverage_5x_trigger", 0.930, 0.980),
        # 5 & 6. XGBoost — tuned for 40% train windows
        "xgb_lr":              trial.suggest_float("xgb_lr",              0.005, 0.080, log=True),
        "xgb_max_depth":       trial.suggest_int("xgb_max_depth",         3,     6),
        # Frozen v4 params
        **V4_FROZEN_PARAMS,
    }

    print(f"\n>>> Hunter Trial {trial.number} | "
          f"conf={params['confidence_thresh']:.3f} "
          f"sl={params['sl_atr_mult']:.2f}x "
          f"tp={params['tp_atr_mult']:.2f}x "
          f"5xlev@{params['leverage_5x_trigger']:.3f} "
          f"lr={params['xgb_lr']:.4f} "
          f"depth={params['xgb_max_depth']}")

    try:
        folds = _generate_folds_v51(df_dict_4h, cfg)
        if not folds:
            return -999.0

        fold_scores     = []
        total_trades    = 0
        penalty_applied = False

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
            train_4h = _slice(df_dict_4h, train_idx)
            train_1d = _slice(df_dict_1d, train_idx)
            val_4h   = _slice(df_dict_4h, val_idx)
            val_1d   = _slice(df_dict_1d, val_idx)
            test_4h  = _slice(df_dict_4h, test_idx)
            test_1d  = _slice(df_dict_1d, test_idx)

            X_tr_list, y_tr_list = [], []
            X_vl_list, y_vl_list = [], []

            for asset in train_4h:
                if asset not in train_1d: continue
                X = build_features(train_4h[asset], train_1d[asset], asset, params)
                if X.empty: continue
                y = label_trades(train_4h[asset].loc[X.index], X['mss_signal'],
                                 TBM_CONFIG, asset, params)
                y = y.reindex(X.index).fillna(0).astype(int)
                X_tr_list.append(X)
                y_tr_list.append(y)

            for asset in val_4h:
                if asset not in val_1d: continue
                X = build_features(val_4h[asset], val_1d[asset], asset, params)
                if X.empty: continue
                y = label_trades(val_4h[asset].loc[X.index], X['mss_signal'],
                                 TBM_CONFIG, asset, params)
                y = y.reindex(X.index).fillna(0).astype(int)
                X_vl_list.append(X)
                y_vl_list.append(y)

            if not X_tr_list:
                continue

            # Re-tune XGBoost params for this trial's depth/lr
            model_cfg = {**MODEL_CONFIG}
            model_cfg['xgb_params'] = {
                **MODEL_CONFIG['xgb_params'],
                'max_depth':     params['xgb_max_depth'],
                'learning_rate': params['xgb_lr'],
                'n_estimators':  params['xgb_n_estimators'],
                'subsample':     params['xgb_subsample'],
            }
            model = VaranusModel(model_cfg)
            model.fit(
                pd.concat(X_tr_list), pd.concat(y_tr_list),
                pd.concat(X_vl_list) if X_vl_list else None,
                pd.concat(y_vl_list) if y_vl_list else None,
            )

            signals = {}
            for asset, df_4h in test_4h.items():
                if asset not in test_1d: continue
                X_t = build_features(df_4h, test_1d[asset], asset, params)
                if X_t.empty: continue
                probs  = model.predict_proba(X_t)
                preds  = model.predict(X_t)
                sig_df = pd.DataFrame(index=X_t.index)
                sig_df['confidence']  = probs.max(axis=1)
                sig_df['direction']   = preds
                sig_df['entry_price'] = df_4h.loc[X_t.index, 'close']
                sig_df['atr']         = compute_atr(df_4h.loc[X_t.index], 14)
                sig_df = sig_df[sig_df['direction'] != 0]
                if not sig_df.empty:
                    signals[asset] = sig_df

            if not signals:
                continue

            equity, trades = run_backtest(test_4h, signals, model, params)
            metrics        = compute_metrics(equity, trades)

            fold_trades = metrics['total_trades']
            fold_dd     = metrics['max_drawdown_pct'] / 100.0
            net_profit  = trades['pnl_usd'].sum() if not trades.empty else 0.0

            if fold_trades < HUNTER_OPTUNA_CONFIG['min_trades_per_fold']:
                continue

            fold_he = _hunter_efficiency(net_profit, fold_dd)

            # Hard DD penalty
            if abs(fold_dd) > HUNTER_OPTUNA_CONFIG['dd_penalty_threshold']:
                fold_he *= HUNTER_OPTUNA_CONFIG['dd_penalty_multiplier']
                penalty_applied = True

            fold_scores.append(fold_he)
            total_trades += fold_trades

            # Report intermediate value for HyperbandPruner
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if total_trades < HUNTER_OPTUNA_CONFIG['min_total_trades']:
            print(f"  -> Penalty: only {total_trades} total trades.")
            return -999.0

        if not fold_scores:
            return -999.0

        mean_he = float(np.mean(fold_scores))
        flag    = " [DD PENALTY]" if penalty_applied else ""
        print(f"  -> Trial {trial.number} | Hunter Efficiency: {mean_he:.3f} | "
              f"Folds scored: {len(fold_scores)}/{len(folds)} | Trades: {total_trades}{flag}")
        return mean_he

    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"  -> Trial {trial.number} failed: {exc}")
        return -999.0


def run_hunter_optimization(
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    n_trials: int = 300,
    study_name: str = "varanus_v51_hunter",
) -> optuna.Study:
    """
    Launch the Hunter Efficiency Optuna study (v5.1 legacy).

    Usage
    -----
    study = run_hunter_optimization(data_4h, data_1d, n_trials=300)
    print(f"Best Hunter Efficiency: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")
    """
    study = optuna.create_study(
        study_name = study_name,
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.HyperbandPruner(
            min_resource=2, max_resource=8, reduction_factor=3
        ),
    )
    study.optimize(
        lambda t: optuna_objective_hunter(t, df_dict_4h, df_dict_1d),
        n_trials          = n_trials,
        show_progress_bar = True,
    )
    return study


# build_dual_labels lives in tbm_labeler to avoid circular imports


# ═══════════════════════════════════════════════════════════════════════════════
# v5.2 Dual-Engine: Long Runner Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

def optuna_objective_dual_engine(
    trial: optuna.Trial,
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    cfg: dict = WFV_CONFIG_V51,
) -> float:
    """
    v5.2 Dual-Engine Optuna Objective — Long Runner Only.

    Short Hunter parameters are FROZEN from Trial #183:
      conf_thresh_short=0.786, tp_atr_mult=5.768, sl_atr_mult=0.709,
      leverage_5x_trigger=0.968

    Searches 2 Long Runner parameters (v5.2 Deep Search space):
      conf_thresh_long  [0.55  - 0.72]   — lowered further to maximise signal density
      tp_mult_long      [2.5   - 4.5]    — smaller step-like targets (unchanged)
      sl_mult_long = 1.0 (FIXED — Hard Stop guardrail, not searched)

    Objective: Density-weighted score = (WR × Count) / DD_Impact
      where DD_Impact = max(abs(max_drawdown), 0.01)
      Rewards signal density AND quality; penalises drawdown.
    Hard constraint: long_win_rate < 35% → return -999.0
    Signal density target: 60–100 long trades across all folds.
    """
    sp = LONG_RUNNER_SEARCH_SPACE
    long_params = {
        "conf_thresh_long": trial.suggest_float("conf_thresh_long", *sp["conf_thresh_long"]),
        "tp_mult_long":     trial.suggest_float("tp_mult_long",     *sp["tp_mult_long"]),
        "sl_mult_long":     1.0,   # HARD STOP — fixed, not searched (Deep Search guardrail)
    }

    # Full parameter set: Long Runner + frozen Short Hunter + frozen v4 signal core
    params = {
        **V4_FROZEN_PARAMS,
        **V52_SHORT_FROZEN_PARAMS,
        **long_params,
        # XGBoost — v5.1 best values, not re-searched
        "xgb_lr":        0.0609,
        "xgb_max_depth": 6,
    }

    print(f"\n>>> DualEngine Trial {trial.number} | "
          f"conf_long={params['conf_thresh_long']:.3f} "
          f"tp_long={params['tp_mult_long']:.2f}x "
          f"sl_long=1.00x[FIXED]")

    try:
        folds = _generate_folds_v51(df_dict_4h, cfg)
        if not folds:
            return -999.0

        fold_scores  = []
        total_long   = 0
        penalty_flag = False

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
            train_4h = _slice(df_dict_4h, train_idx)
            train_1d = _slice(df_dict_1d, train_idx)
            val_4h   = _slice(df_dict_4h, val_idx)
            val_1d   = _slice(df_dict_1d, val_idx)
            test_4h  = _slice(df_dict_4h, test_idx)
            test_1d  = _slice(df_dict_1d, test_idx)

            X_tr_list,  y_tr_list  = [], []
            X_vl_list,  y_vl_list  = [], []
            y_short_tr_list, y_short_vl_list = [], []

            for asset in train_4h:
                if asset not in train_1d: continue
                X = build_features(train_4h[asset], train_1d[asset], asset, params)
                if X.empty: continue
                y = build_dual_labels(train_4h[asset], X, {**params, '_asset': asset})
                y = y.reindex(X.index).fillna(0).astype(int)
                # v5.1-style short labels: label_trades(mss_signal) preserves full count
                y_short = label_trades(train_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                X_tr_list.append(X); y_tr_list.append(y); y_short_tr_list.append(y_short)

            for asset in val_4h:
                if asset not in val_1d: continue
                X = build_features(val_4h[asset], val_1d[asset], asset, params)
                if X.empty: continue
                y = build_dual_labels(val_4h[asset], X, {**params, '_asset': asset})
                y = y.reindex(X.index).fillna(0).astype(int)
                y_short = label_trades(val_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                X_vl_list.append(X); y_vl_list.append(y); y_short_vl_list.append(y_short)

            if not X_tr_list:
                continue

            model_cfg = {**MODEL_CONFIG}
            model_cfg['xgb_params'] = {
                **MODEL_CONFIG['xgb_params'],
                'max_depth':     params['xgb_max_depth'],
                'learning_rate': params['xgb_lr'],
                'n_estimators':  params['xgb_n_estimators'],
                'subsample':     params['xgb_subsample'],
            }
            model = VaranusDualModel(model_cfg)
            model.fit(
                pd.concat(X_tr_list), pd.concat(y_tr_list),
                pd.concat(X_vl_list) if X_vl_list else None,
                pd.concat(y_vl_list) if y_vl_list else None,
                pd.concat(y_short_tr_list),
                pd.concat(y_short_vl_list) if y_short_vl_list else None,
            )

            signals = {}
            _conf_long  = params.get('conf_thresh_long',  0.70)
            _conf_short = params.get('conf_thresh_short', 0.786)
            for asset, df_4h in test_4h.items():
                if asset not in test_1d: continue
                X_t = build_features(df_4h, test_1d[asset], asset, params)
                if X_t.empty: continue
                probs   = model.predict_proba(X_t)       # (N, 3): [p_short, p_neutral, p_long]
                p_short = probs[:, 0]
                p_long  = probs[:, 2]
                # Direction from direction-specific raw-proba thresholds — NOT model.predict().
                # model.predict() has a hard 0.75 floor; using raw proba lets
                # conf_thresh_long < 0.75 actually increase long signal count.
                # SHORT HUNTER LOCK-DOWN: short gate stays at frozen conf_thresh_short=0.786.
                direction = np.zeros(len(X_t), dtype=int)
                long_mask  = (p_long  >= _conf_long)  & (p_long  > p_short)
                short_mask = (p_short >= _conf_short) & (p_short >= p_long)
                direction[long_mask]  =  1
                direction[short_mask] = -1
                active = direction != 0
                if not active.any(): continue
                idx_active = X_t.index[active]
                sig_df = pd.DataFrame(index=idx_active)
                sig_df['confidence']  = np.where(
                    direction[active] == 1, p_long[active], p_short[active])
                sig_df['direction']   = direction[active]
                sig_df['entry_price'] = df_4h.loc[idx_active, 'close'].values
                sig_df['atr']         = compute_atr(df_4h.loc[X_t.index], 14).loc[idx_active].values
                signals[asset] = sig_df

            if not signals:
                continue

            equity, trades = run_backtest(test_4h, signals, model, params)
            metrics        = compute_metrics(equity, trades)

            fold_long   = metrics.get('long_trades', 0)
            fold_lwr    = metrics.get('long_win_rate_pct', 0.0) / 100.0
            fold_lsharpe = metrics.get('long_sharpe', 0.0)
            fold_dd     = metrics.get('max_drawdown_pct', 0.0) / 100.0

            if fold_long < DUAL_ENGINE_OPTUNA_CONFIG['min_long_trades_per_fold']:
                continue

            # Hard win rate gate — penalise underperforming long configs
            if fold_lwr < DUAL_ENGINE_OPTUNA_CONFIG['min_long_win_rate']:
                fold_scores.append(-10.0)
                total_long += fold_long
                trial.report(-10.0, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                continue

            # Long Runner Score (Deep Search) = (WR × Count) / DD_Impact
            # Density-focused objective: rewards more trades at good win rate,
            # penalises drawdown naturally via the denominator.
            dd_impact  = max(abs(fold_dd), 0.01)
            fold_score = (fold_lwr * fold_long) / dd_impact

            # Extra DD penalty for extreme drawdown (same threshold as Hunter)
            if abs(fold_dd) > DUAL_ENGINE_OPTUNA_CONFIG['dd_penalty_threshold']:
                fold_score *= DUAL_ENGINE_OPTUNA_CONFIG['dd_penalty_multiplier']
                penalty_flag = True

            fold_scores.append(fold_score)
            total_long  += fold_long

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if total_long < DUAL_ENGINE_OPTUNA_CONFIG['min_total_long_trades']:
            print(f"  -> Penalty: only {total_long} long trades total.")
            return -999.0
        if not fold_scores:
            return -999.0

        mean_score = float(np.mean(fold_scores))
        flag = " [DD PENALTY]" if penalty_flag else ""
        print(f"  -> Trial {trial.number} | LongRunner Score: {mean_score:.3f} | "
              f"Folds: {len(fold_scores)}/{len(folds)} | LongTrades: {total_long}{flag}")
        return mean_score

    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"  -> Trial {trial.number} failed: {exc}")
        return -999.0


def run_dual_engine_optimization(
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    n_trials: int = 300,
    study_name: str = "varanus_v52_dual_engine",
) -> optuna.Study:
    """
    Launch the v5.2 Dual-Engine Long Runner Optuna study.

    Short Hunter params remain frozen (Trial #183).
    Only 3 Long Runner parameters are searched.

    Usage
    -----
    study = run_dual_engine_optimization(data_4h, data_1d, n_trials=300)
    print(f"Best Long Runner Score: {study.best_value:.3f}")
    print(f"Long Runner params: {study.best_params}")
    """
    study = optuna.create_study(
        study_name = study_name,
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=52),
        pruner     = optuna.pruners.HyperbandPruner(
            min_resource=2, max_resource=8, reduction_factor=3
        ),
    )
    study.optimize(
        lambda t: optuna_objective_dual_engine(t, df_dict_4h, df_dict_1d),
        n_trials          = n_trials,
        show_progress_bar = True,
    )
    return study


# ═══════════════════════════════════════════════════════════════════════════════
# v5.5 Long Runner — Density-Bias Optimization
# ═══════════════════════════════════════════════════════════════════════════════
#
# Key changes vs v5.2:
#   1. p_short_max_for_long — parameterized long/short competition gate.
#      Replaces hard "p_long > p_short" with a tunable p_short ceiling.
#      Root cause: short model (trained on 2x data) dominates in mixed markets,
#      suppressing valid long signals below the v5.2 confidence threshold.
#   2. rsi_1d_long_limit — injectable RSI bypass ceiling (search [45, 65]).
#      pa_features.build_features() now reads this from params.
#   3. sl_mult_long unfrozen — search [0.70, 1.50]. Tighter SL improves R:R
#      at lower confidence entries.
#   4. Density gate lowered: 60 → 45. Model ceiling ~56 at conf=0.55;
#      gate at 60 created a dead zone where TPE never learned useful gradients.
#   5. MedianPruner replaces HyperbandPruner. Long runner performance is
#      back-loaded (fold 5 typically strongest); Hyperband pruned too early.
#   6. Density score: (WR × Count × ln(Count+1)) / DD_Impact
#      Log factor provides diminishing density bonus, rewarding n>60 without
#      overweighting noise at n>120.
# ═══════════════════════════════════════════════════════════════════════════════

# v5.5 search space — Long Runner only. Short Hunter remains FROZEN.
LONG_RUNNER_SEARCH_SPACE_V55 = {
    "conf_thresh_long":     (0.50, 0.70),   # Lowered floor: 0.55 → 0.50
    "tp_mult_long":         (2.0,  5.0),    # Wider: 2.5–4.5 → 2.0–5.0
    "sl_mult_long":         (0.70, 1.50),   # Unfrozen from 1.0 fixed
    "rsi_1d_long_limit":    (45,   65),     # Bias-bypass RSI ceiling (int)
    "p_short_max_for_long": (0.55, 0.90),   # Max p_short allowed for a long signal
}

DUAL_ENGINE_OPTUNA_CONFIG_V55 = {
    "n_trials":                 300,
    "direction":                "maximize",
    "sampler":                  "TPESampler",
    "pruner":                   "MedianPruner",   # Less aggressive than Hyperband
    "n_startup_trials":         20,               # Random warmup before TPE engages
    "min_long_trades_per_fold": 2,                # Relaxed from 3
    "min_total_long_trades":    45,               # Lowered from 60 (model ceiling ~56)
    "min_long_win_rate":        0.33,             # Relaxed from 0.35
    "dd_penalty_threshold":     0.12,
    "dd_penalty_multiplier":    0.40,
}


def optuna_objective_v55(
    trial: optuna.Trial,
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    cfg: dict = WFV_CONFIG_V51,
) -> float:
    """
    v5.5 Long Runner Optuna Objective — Density-Bias Search.

    Searches 5 Long Runner parameters:
      conf_thresh_long     [0.50 – 0.70]   Entry gate (lowered floor)
      tp_mult_long         [2.0  – 5.0]    TP multiplier (wider)
      sl_mult_long         [0.70 – 1.50]   SL multiplier (unfrozen)
      rsi_1d_long_limit    [45  – 65]      Bias-bypass RSI ceiling
      p_short_max_for_long [0.55 – 0.90]   Max short-model score for long entry

    Short Hunter params FROZEN (Trial #183). Short gate unchanged.

    Density Score = (WR × Count × ln(Count+1)) / DD_Impact
    Hard density gate: total_long < 45 → -999.0 penalty.
    """
    cfg_v55 = DUAL_ENGINE_OPTUNA_CONFIG_V55

    long_params = {
        "conf_thresh_long":     trial.suggest_float("conf_thresh_long",     0.50, 0.70),
        "tp_mult_long":         trial.suggest_float("tp_mult_long",         2.0,  5.0),
        "sl_mult_long":         trial.suggest_float("sl_mult_long",         0.70, 1.50),
        "rsi_1d_long_limit":    trial.suggest_int(  "rsi_1d_long_limit",    45,   65),
        "p_short_max_for_long": trial.suggest_float("p_short_max_for_long", 0.55, 0.90),
    }

    params = {
        **V4_FROZEN_PARAMS,
        **V52_SHORT_FROZEN_PARAMS,
        **long_params,
        "xgb_lr":        0.0609,
        "xgb_max_depth": 6,
    }

    print(f"\n>>> v5.5 Trial {trial.number} | "
          f"conf={long_params['conf_thresh_long']:.3f} "
          f"tp={long_params['tp_mult_long']:.2f}x "
          f"sl={long_params['sl_mult_long']:.2f}x "
          f"rsi1d<={long_params['rsi_1d_long_limit']} "
          f"p_short_max={long_params['p_short_max_for_long']:.3f}")

    try:
        folds = _generate_folds_v51(df_dict_4h, cfg)
        if not folds:
            return -999.0

        fold_scores  = []
        total_long   = 0
        penalty_flag = False

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
            train_4h = _slice(df_dict_4h, train_idx)
            train_1d = _slice(df_dict_1d, train_idx)
            val_4h   = _slice(df_dict_4h, val_idx)
            val_1d   = _slice(df_dict_1d, val_idx)
            test_4h  = _slice(df_dict_4h, test_idx)
            test_1d  = _slice(df_dict_1d, test_idx)

            X_tr_list, y_tr_list = [], []
            X_vl_list, y_vl_list = [], []
            y_short_tr_list, y_short_vl_list = [], []

            for asset in train_4h:
                if asset not in train_1d: continue
                X = build_features(train_4h[asset], train_1d[asset], asset, params)
                if X.empty: continue
                y = build_dual_labels(train_4h[asset], X, {**params, '_asset': asset})
                y = y.reindex(X.index).fillna(0).astype(int)
                y_short = label_trades(train_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                X_tr_list.append(X); y_tr_list.append(y); y_short_tr_list.append(y_short)

            for asset in val_4h:
                if asset not in val_1d: continue
                X = build_features(val_4h[asset], val_1d[asset], asset, params)
                if X.empty: continue
                y = build_dual_labels(val_4h[asset], X, {**params, '_asset': asset})
                y = y.reindex(X.index).fillna(0).astype(int)
                y_short = label_trades(val_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                X_vl_list.append(X); y_vl_list.append(y); y_short_vl_list.append(y_short)

            if not X_tr_list:
                continue

            model_cfg = {**MODEL_CONFIG}
            model_cfg['xgb_params'] = {
                **MODEL_CONFIG['xgb_params'],
                'max_depth':     params['xgb_max_depth'],
                'learning_rate': params['xgb_lr'],
                'n_estimators':  params['xgb_n_estimators'],
                'subsample':     params['xgb_subsample'],
            }
            model = VaranusDualModel(model_cfg)
            model.fit(
                pd.concat(X_tr_list), pd.concat(y_tr_list),
                pd.concat(X_vl_list) if X_vl_list else None,
                pd.concat(y_vl_list) if y_vl_list else None,
                pd.concat(y_short_tr_list),
                pd.concat(y_short_vl_list) if y_short_vl_list else None,
            )

            signals = {}
            _conf_long    = params['conf_thresh_long']
            _conf_short   = params.get('conf_thresh_short', 0.786)
            _p_short_max  = params['p_short_max_for_long']   # v5.5: tunable ceiling

            for asset, df_4h in test_4h.items():
                if asset not in test_1d: continue
                X_t = build_features(df_4h, test_1d[asset], asset, params)
                if X_t.empty: continue
                probs   = model.predict_proba(X_t)  # (N, 3): [p_short, p_neutral, p_long]
                p_short = probs[:, 0]
                p_long  = probs[:, 2]

                direction = np.zeros(len(X_t), dtype=int)
                # v5.5: long gate — p_short ceiling replaces hard p_long > p_short.
                # Allows longs in neutral markets where short model is moderately active.
                # SHORT HUNTER LOCK-DOWN: short gate unchanged.
                long_mask  = (p_long  >= _conf_long) & (p_short <= _p_short_max)
                short_mask = (p_short >= _conf_short) & (p_short >= p_long)
                direction[long_mask]  =  1
                direction[short_mask] = -1

                active = direction != 0
                if not active.any(): continue
                idx_active = X_t.index[active]
                sig_df = pd.DataFrame(index=idx_active)
                sig_df['confidence']  = np.where(
                    direction[active] == 1, p_long[active], p_short[active])
                sig_df['direction']   = direction[active]
                sig_df['entry_price'] = df_4h.loc[idx_active, 'close'].values
                sig_df['atr']         = compute_atr(df_4h.loc[X_t.index], 14).loc[idx_active].values
                signals[asset] = sig_df

            if not signals:
                continue

            equity, trades = run_backtest(test_4h, signals, model, params)
            metrics        = compute_metrics(equity, trades)

            fold_long = metrics.get('long_trades', 0)
            fold_lwr  = metrics.get('long_win_rate_pct', 0.0) / 100.0
            fold_dd   = metrics.get('max_drawdown_pct', 0.0) / 100.0

            if fold_long < cfg_v55['min_long_trades_per_fold']:
                continue

            if fold_lwr < cfg_v55['min_long_win_rate']:
                fold_scores.append(-10.0)
                total_long += fold_long
                trial.report(-10.0, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                continue

            # v5.5 Density Score: (WR × Count × ln(Count+1)) / DD_Impact
            # ln(Count+1) rewards density with diminishing returns:
            #   n=10  → ln(11)=2.40  | n=30  → ln(31)=3.43
            #   n=60  → ln(61)=4.11  | n=100 → ln(101)=4.62
            # Naturally favors 60–100 trades over a "perfect" n=10 sample.
            dd_impact    = max(abs(fold_dd), 0.01)
            density_log  = math.log(fold_long + 1)
            fold_score   = (fold_lwr * fold_long * density_log) / dd_impact

            if abs(fold_dd) > cfg_v55['dd_penalty_threshold']:
                fold_score *= cfg_v55['dd_penalty_multiplier']
                penalty_flag = True

            fold_scores.append(fold_score)
            total_long  += fold_long

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if total_long < cfg_v55['min_total_long_trades']:
            print(f"  -> Penalty: only {total_long} long trades (gate: {cfg_v55['min_total_long_trades']}).")
            return -999.0
        if not fold_scores:
            return -999.0

        mean_score = float(np.mean(fold_scores))
        flag = " [DD PENALTY]" if penalty_flag else ""
        print(f"  -> Trial {trial.number} | v5.5 Density Score: {mean_score:.3f} | "
              f"Folds: {len(fold_scores)}/{len(folds)} | LongTrades: {total_long}{flag}")
        return mean_score

    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"  -> Trial {trial.number} failed: {exc}")
        return -999.0


def run_v55_optimization(
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    n_trials: int = 300,
    study_name: str = "varanus_v55_long_runner",
) -> optuna.Study:
    """
    Launch the v5.5 Long Runner Optuna study.

    Short Hunter params remain FROZEN (Trial #183).
    5 Long Runner parameters searched (vs 2 in v5.2).
    """
    study = optuna.create_study(
        study_name = study_name,
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(
            seed             = 55,
            n_startup_trials = DUAL_ENGINE_OPTUNA_CONFIG_V55["n_startup_trials"],
        ),
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps   = 2,
            interval_steps   = 1,
        ),
    )
    study.optimize(
        lambda t: optuna_objective_v55(t, df_dict_4h, df_dict_1d),
        n_trials          = n_trials,
        show_progress_bar = True,
    )
    return study


# ═══════════════════════════════════════════════════════════════════════════════
# v5.6 Long Runner — The Golden Ratio
# ═══════════════════════════════════════════════════════════════════════════════
#
# Fixes v5.5 Long WR slippage (39.6% → target >41%):
#   1. p_short_max_for_long ceiling tightened: 0.90 → 0.75
#      Prevents longs from firing when the short model is highly confident.
#      v5.5 winning param was 0.869 — allowed too many ambiguous setups.
#   2. Hard per-fold WR gate raised: 0.33 → 0.41
#      Forces TPE to find configs that maintain quality, not just density.
#   3. Density target band 120–180: min_total_long_trades raised 45 → 80.
#      Keeps pressure on density while the tighter p_short ceiling caps noise.
#   4. sl_mult_long remains tunable [0.70–1.50] to protect R:R at higher volume.
# ═══════════════════════════════════════════════════════════════════════════════

LONG_RUNNER_SEARCH_SPACE_V56 = {
    "conf_thresh_long":     (0.55, 0.70),   # RAISED lower bound from 0.50 — filters noise
    "tp_mult_long":         (2.0,  3.5),    # TIGHTENED upper from 5.0 — targets 120-180 band
    "sl_mult_long":         (0.70, 1.20),   # TIGHTENED upper from 1.50 — tighter R:R control
    "rsi_1d_long_limit":    (45,   65),     # Unchanged from v5.5
    "p_short_max_for_long": (0.55, 0.75),   # TIGHTENED from (0.55, 0.90) — blocks short-bias bars
}

DUAL_ENGINE_OPTUNA_CONFIG_V56 = {
    "n_trials":                 300,
    "direction":                "maximize",
    "sampler":                  "TPESampler",
    "pruner":                   "MedianPruner",
    "n_startup_trials":         20,
    "min_long_trades_per_fold": 2,
    "min_total_long_trades":    80,           # Raised from 45 — targets 120–180 band
    "min_long_win_rate":        0.41,         # RAISED from 0.33 — hard quality gate
    "dd_penalty_threshold":     0.12,
    "dd_penalty_multiplier":    0.40,
}


def optuna_objective_v56(
    trial: optuna.Trial,
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    cfg: dict = WFV_CONFIG_V51,
) -> float:
    """
    v5.6 Long Runner Optuna Objective — The Golden Ratio.

    Searches 5 Long Runner parameters:
      conf_thresh_long     [0.55 – 0.70]   raised lower bound — filters noise entries
      tp_mult_long         [2.0  – 3.5]    tightened upper — drives 120-180 trade density
      sl_mult_long         [0.70 – 1.20]   tightened upper — tighter R:R control
      rsi_1d_long_limit    [45  – 65]      unchanged from v5.5
      p_short_max_for_long [0.55 – 0.75]   below conf_thresh_short=0.786 dead zone

    Key changes vs v5.5:
      - conf_thresh_long lower: 0.50 → 0.55 (filters low-confidence noise)
      - tp_mult_long upper: 5.0 → 3.5 (tighter TP targets more frequent closes)
      - sl_mult_long upper: 1.50 → 1.20 (tighter SL improves R:R)
      - p_short_max ceiling: 0.90 → 0.75 (blocks high-confidence short bars)
      - Per-fold WR gate: 0.33 → 0.41 (hard quality enforcement)
      - Density gate: 45 → 80 (targets 120–180 long trades total)
      - Pruner: HyperbandPruner → MedianPruner (prevents back-loaded fold killing)

    Density Score = (WR × Count × ln(Count+1)) / DD_Impact
    """
    cfg_v56 = DUAL_ENGINE_OPTUNA_CONFIG_V56

    long_params = {
        "conf_thresh_long":     trial.suggest_float("conf_thresh_long",     0.55, 0.70),
        "tp_mult_long":         trial.suggest_float("tp_mult_long",         2.0,  3.5),
        "sl_mult_long":         trial.suggest_float("sl_mult_long",         0.70, 1.20),
        "rsi_1d_long_limit":    trial.suggest_int(  "rsi_1d_long_limit",    45,   65),
        "p_short_max_for_long": trial.suggest_float("p_short_max_for_long", 0.55, 0.75),
    }

    params = {
        **V4_FROZEN_PARAMS,
        **V52_SHORT_FROZEN_PARAMS,
        **long_params,
        "xgb_lr":        0.0609,
        "xgb_max_depth": 6,
    }

    print(f"\n>>> v5.6 Trial {trial.number} | "
          f"conf={long_params['conf_thresh_long']:.3f} "
          f"tp={long_params['tp_mult_long']:.2f}x "
          f"sl={long_params['sl_mult_long']:.2f}x "
          f"rsi1d<={long_params['rsi_1d_long_limit']} "
          f"p_short_max={long_params['p_short_max_for_long']:.3f}")

    try:
        folds = _generate_folds_v51(df_dict_4h, cfg)
        if not folds:
            return -999.0

        fold_scores  = []
        total_long   = 0
        penalty_flag = False

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
            train_4h = _slice(df_dict_4h, train_idx)
            train_1d = _slice(df_dict_1d, train_idx)
            val_4h   = _slice(df_dict_4h, val_idx)
            val_1d   = _slice(df_dict_1d, val_idx)
            test_4h  = _slice(df_dict_4h, test_idx)
            test_1d  = _slice(df_dict_1d, test_idx)

            X_tr_list, y_tr_list = [], []
            X_vl_list, y_vl_list = [], []
            y_short_tr_list, y_short_vl_list = [], []

            for asset in train_4h:
                if asset not in train_1d: continue
                X = build_features(train_4h[asset], train_1d[asset], asset, params)
                if X.empty: continue
                y = build_dual_labels(train_4h[asset], X, {**params, '_asset': asset})
                y = y.reindex(X.index).fillna(0).astype(int)
                y_short = label_trades(train_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                X_tr_list.append(X); y_tr_list.append(y); y_short_tr_list.append(y_short)

            for asset in val_4h:
                if asset not in val_1d: continue
                X = build_features(val_4h[asset], val_1d[asset], asset, params)
                if X.empty: continue
                y = build_dual_labels(val_4h[asset], X, {**params, '_asset': asset})
                y = y.reindex(X.index).fillna(0).astype(int)
                y_short = label_trades(val_4h[asset].loc[X.index], X['mss_signal'], TBM_CONFIG, asset, params)
                y_short = y_short.reindex(X.index).fillna(0).astype(int)
                X_vl_list.append(X); y_vl_list.append(y); y_short_vl_list.append(y_short)

            if not X_tr_list:
                continue

            model_cfg = {**MODEL_CONFIG}
            model_cfg['xgb_params'] = {
                **MODEL_CONFIG['xgb_params'],
                'max_depth':     params['xgb_max_depth'],
                'learning_rate': params['xgb_lr'],
                'n_estimators':  params['xgb_n_estimators'],
                'subsample':     params['xgb_subsample'],
            }
            model = VaranusDualModel(model_cfg)
            model.fit(
                pd.concat(X_tr_list), pd.concat(y_tr_list),
                pd.concat(X_vl_list) if X_vl_list else None,
                pd.concat(y_vl_list) if y_vl_list else None,
                pd.concat(y_short_tr_list),
                pd.concat(y_short_vl_list) if y_short_vl_list else None,
            )

            signals = {}
            _conf_long   = params['conf_thresh_long']
            _conf_short  = params.get('conf_thresh_short', 0.786)
            _p_short_max = params['p_short_max_for_long']

            for asset, df_4h in test_4h.items():
                if asset not in test_1d: continue
                X_t = build_features(df_4h, test_1d[asset], asset, params)
                if X_t.empty: continue
                probs   = model.predict_proba(X_t)
                p_short = probs[:, 0]
                p_long  = probs[:, 2]

                direction = np.zeros(len(X_t), dtype=int)
                long_mask  = (p_long  >= _conf_long) & (p_short <= _p_short_max)
                short_mask = (p_short >= _conf_short) & (p_short >= p_long)
                direction[long_mask]  =  1
                direction[short_mask] = -1

                active = direction != 0
                if not active.any(): continue
                idx_active = X_t.index[active]
                sig_df = pd.DataFrame(index=idx_active)
                sig_df['confidence']  = np.where(
                    direction[active] == 1, p_long[active], p_short[active])
                sig_df['direction']   = direction[active]
                sig_df['entry_price'] = df_4h.loc[idx_active, 'close'].values
                sig_df['atr']         = compute_atr(df_4h.loc[X_t.index], 14).loc[idx_active].values
                signals[asset] = sig_df

            if not signals:
                continue

            equity, trades = run_backtest(test_4h, signals, model, params)
            metrics        = compute_metrics(equity, trades)

            fold_long = metrics.get('long_trades', 0)
            fold_lwr  = metrics.get('long_win_rate_pct', 0.0) / 100.0
            fold_dd   = metrics.get('max_drawdown_pct', 0.0) / 100.0

            if fold_long < cfg_v56['min_long_trades_per_fold']:
                continue

            # Hard WR gate raised to 41% — enforces Golden Ratio quality standard
            if fold_lwr < cfg_v56['min_long_win_rate']:
                fold_scores.append(-10.0)
                total_long += fold_long
                trial.report(-10.0, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                continue

            # Density Score: (WR × Count × ln(Count+1)) / DD_Impact
            dd_impact   = max(abs(fold_dd), 0.01)
            density_log = math.log(fold_long + 1)
            fold_score  = (fold_lwr * fold_long * density_log) / dd_impact

            if abs(fold_dd) > cfg_v56['dd_penalty_threshold']:
                fold_score *= cfg_v56['dd_penalty_multiplier']
                penalty_flag = True

            fold_scores.append(fold_score)
            total_long  += fold_long

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if total_long < cfg_v56['min_total_long_trades']:
            print(f"  -> Penalty: only {total_long} long trades (gate: {cfg_v56['min_total_long_trades']}).")
            return -999.0
        if not fold_scores:
            return -999.0

        mean_score = float(np.mean(fold_scores))
        flag = " [DD PENALTY]" if penalty_flag else ""
        print(f"  -> Trial {trial.number} | v5.6 Score: {mean_score:.3f} | "
              f"Folds: {len(fold_scores)}/{len(folds)} | LongTrades: {total_long}{flag}")
        return mean_score

    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"  -> Trial {trial.number} failed: {exc}")
        return -999.0


def run_v56_optimization(
    df_dict_4h: dict[str, pd.DataFrame],
    df_dict_1d: dict[str, pd.DataFrame],
    n_trials: int = 300,
    study_name: str = "varanus_v56_golden_ratio",
) -> optuna.Study:
    """
    Launch the v5.6 Golden Ratio Long Runner Optuna study.

    Short Hunter params remain FROZEN (Trial #183).
    5 params searched; p_short_max_for_long ceiling tightened to 0.75.
    """
    study = optuna.create_study(
        study_name = study_name,
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(
            seed             = 56,
            n_startup_trials = DUAL_ENGINE_OPTUNA_CONFIG_V56["n_startup_trials"],
        ),
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps   = 2,
            interval_steps   = 1,
        ),
    )
    study.optimize(
        lambda t: optuna_objective_v56(t, df_dict_4h, df_dict_1d),
        n_trials          = n_trials,
        show_progress_bar = True,
    )
    return study
