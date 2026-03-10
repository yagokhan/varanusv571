import pandas as pd
import numpy as np
import xgboost as xgb

from varanus.pa_features import detect_mss, detect_fvg, compute_atr
from varanus.universe import HIGH_VOL_SUBTIER

FEATURE_LIST = [
    # PA Features
    "mss_signal",           # {-1, 0, 1}
    "fvg_type",             # {-1, 0, 1} — sweep-validated only
    "fvg_distance_atr",     # Distance to nearest valid FVG / ATR(14)
    "fvg_age_candles",      # Candles since FVG formed
    "sweep_occurred",       # Binary: sweep preceded the FVG?
    "htf_bias",             # 1D MSS direction {-1, 0, 1}

    # Chameleon Confirmation
    "relative_volume",      # Current vol / 20-period avg
    "rsi_14",
    "rsi_slope_3",          # RSI delta over last 3 candles
    "ema21_55_alignment",   # {1, -1, 0}
    "atr_percentile_100",   # ATR rank in 100-period window [0, 1]

    # Market Character
    "volatility_rank",      # ATR rank vs 100-period history [0, 1]
    "volume_rank",          # Volume rank vs 100-period history [0, 1]
    "asset_tier_flag",      # 0=standard Tier 2, 1=high-vol sub-tier
    "hour_of_day",          # 4h candle UTC hour (session awareness)
    "day_of_week",          # Market regime proxy
]

MODEL_CONFIG = {
    "type":                "XGBoostClassifier",
    "target_classes":      3,       # {-1, 0, 1}
    "confidence_threshold": 0.75,   # v5.1: lowered from 0.80 — floor of [0.75–0.88] entry gate search

    "xgb_params": {
        "n_estimators":          500,
        "max_depth":             4,      # v5.1: reduced from 6 — shallower trees for 40% train windows
        "learning_rate":         0.03,   # v5.1: reduced from 0.05 — slower learning prevents overfitting
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "scale_pos_weight":      1.0,    # Rebalanced per fold
        "eval_metric":           "mlogloss",
        "early_stopping_rounds": 30,
        "use_label_encoder":     False,
        "objective":             "multi:softprob",
        "num_class":             3,
        "random_state":          42
    },
}


def get_leverage_v51(confidence: float, leverage_5x_trigger: float = 0.96) -> float:
    """
    Hunter v5.1 leverage schedule.

    Below 0.75              -> 0.0  (no trade — below entry gate)
    [0.75, 0.85)            -> 1x
    [0.85, 0.92)            -> 2x
    [0.92, leverage_5x_trigger) -> 3x
    [leverage_5x_trigger, 1.0]  -> 5x  (high-conviction Hunter strike)

    leverage_5x_trigger is an Optuna parameter searched in [0.93–0.98].
    Default 0.96 is used when no params are passed (e.g. paper trading before
    optimization completes).
    """
    if confidence < 0.75:
        return 0.0
    elif confidence < 0.85:
        return 1.0
    elif confidence < 0.92:
        return 2.0
    elif confidence < leverage_5x_trigger:
        return 3.0
    else:
        return 5.0


# Backward-compat alias — backtest.py and paper_trader.py use this until STEP 5 updates them
get_leverage = get_leverage_v51

def build_features(df_4h: pd.DataFrame, df_1d: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Constructs the feature vector for a given asset.
    df_4h: Primary timeframe DataFrame
    df_1d: HTF DataFrame for bias
    """
    df = df_4h.copy()

    # Base indicators
    atr = compute_atr(df, 14)

    # Required Chameleon Features (assuming they're already calculated or we compute proxies if missing)
    # We will compute them here for completeness if not provided.
    
    # 1. PA Features
    df['mss_signal'] = detect_mss(df)
    
    # FVG features
    from varanus.pa_features import FVG_CONFIG
    fvg_df = detect_fvg(df, atr, FVG_CONFIG)
    
    df['fvg_type'] = 0
    df['fvg_distance_atr'] = 0.0
    df['fvg_age_candles'] = 0
    df['sweep_occurred'] = 0
    
    last_fvg_idx = None
    
    # We need to compute fvg age and distance over time
    for i in range(len(df)):
        if i in fvg_df.index and fvg_df.loc[i, 'fvg_valid']:
            last_fvg_idx = i
            df.loc[df.index[i], 'fvg_type'] = fvg_df.loc[i, 'fvg_type']
            df.loc[df.index[i], 'sweep_occurred'] = 1 # We know it occurred because fvg_valid is True and require_sweep is True
            df.loc[df.index[i], 'fvg_age_candles'] = 0
            
            # fvg_distance_atr: distance from current close to nearest fvg bound
            fvg_top = fvg_df.loc[i, 'fvg_top']
            fvg_bot = fvg_df.loc[i, 'fvg_bottom']
            close_price = df['close'].iloc[i]
            
            dist = min(abs(close_price - fvg_top), abs(close_price - fvg_bot))
            df.loc[df.index[i], 'fvg_distance_atr'] = dist / atr.iloc[i] if atr.iloc[i] > 0 else 0
        elif last_fvg_idx is not None:
             age = i - last_fvg_idx
             if age <= FVG_CONFIG['max_gap_age_candles']:
                 df.loc[df.index[i], 'fvg_type'] = fvg_df.loc[last_fvg_idx, 'fvg_type']
                 df.loc[df.index[i], 'sweep_occurred'] = 1
                 df.loc[df.index[i], 'fvg_age_candles'] = age
                 
                 # fvg_distance_atr
                 fvg_top = fvg_df.loc[last_fvg_idx, 'fvg_top']
                 fvg_bot = fvg_df.loc[last_fvg_idx, 'fvg_bottom']
                 close_price = df['close'].iloc[i]
                 dist = min(abs(close_price - fvg_top), abs(close_price - fvg_bot))
                 df.loc[df.index[i], 'fvg_distance_atr'] = dist / atr.iloc[i] if atr.iloc[i] > 0 else 0


    # HTF Bias
    htf_mss = detect_mss(df_1d)
    
    # Forward fill 1D bias to 4H timeframe
    # Reindex HTF MSS to 4H index (forward filled)
    # Note: Requires exact timestamp alignment to avoid lookahead bias.
    # It's safer to map previous day's close bias to today's 4H bars.
    df['date'] = df.index.date
    df_1d['date'] = df_1d.index.date
    # Map previous day's MSS signal
    df_1d['shifted_mss'] = htf_mss.shift(1).fillna(0)
    mapping = df_1d.drop_duplicates(subset=['date'], keep='last').set_index('date')['shifted_mss']
    df['htf_bias'] = df['date'].map(mapping).fillna(0)
    df = df.drop(columns=['date'])

    # 2. Chameleon Confirmation Features
    # relative_volume
    vol_ma = df['volume'].rolling(20).mean()
    df['relative_volume'] = df['volume'] / vol_ma.replace(0, np.nan)
    df['relative_volume'] = df['relative_volume'].fillna(1.0)
    
    # rsi_14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].fillna(50)
    
    # rsi_slope_3
    df['rsi_slope_3'] = df['rsi_14'].diff(3).fillna(0)
    
    # ema21_55_alignment
    ema21 = df['close'].ewm(span=21, adjust=False).mean()
    ema55 = df['close'].ewm(span=55, adjust=False).mean()
    
    df['ema21_55_alignment'] = 0
    bull_align = (df['close'] > ema21) & (ema21 > ema55)
    bear_align = (df['close'] < ema21) & (ema21 < ema55)
    df.loc[bull_align, 'ema21_55_alignment'] = 1
    df.loc[bear_align, 'ema21_55_alignment'] = -1
    
    # atr_percentile_100
    df['atr_percentile_100'] = atr.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).fillna(0.5)

    # 3. Market Character
    # volatility_rank (same as atr_percentile_100 effectively, but let's keep it distinct if intended differently, e.g. Close/Close vol)
    # Using normalized ATR here as rank proxy
    df['volatility_rank'] = df['atr_percentile_100'] 
    
    # volume_rank
    df['volume_rank'] = df['volume'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).fillna(0.5)
    
    # asset_tier_flag
    df['asset_tier_flag'] = 1 if asset in HIGH_VOL_SUBTIER else 0
    
    # hour_of_day, day_of_week
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    return df[FEATURE_LIST]


class VaranusModel:
    def __init__(self, config: dict = MODEL_CONFIG):
        self.config = config
        self.model = None
        self.classes_ = np.array([-1, 0, 1]) # Expected output format

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train the XGBoost model."""
        params = self.config['xgb_params'].copy()
        early_stopping = params.pop('early_stopping_rounds', None)

        # XGBoost requires labels to be 0, 1, 2 for multi:softprob.
        # We map {-1, 0, 1} -> {0, 1, 2} internally.
        y_train_mapped = y_train + 1

        self.model = xgb.XGBClassifier(**params)

        eval_set = [(X_train, y_train_mapped)]
        if X_val is not None and y_val is not None:
             y_val_mapped = y_val + 1
             eval_set.append((X_val, y_val_mapped))
             self.model.fit(
                 X_train, y_train_mapped,
                 eval_set=eval_set,
                 verbose=False
             )
        else:
             self.model.fit(X_train, y_train_mapped, verbose=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probabilities for classes [-1, 0, 1]."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        probs = self.model.predict_proba(X)

        # If model somehow misses a class (e.g. only 2 classes in training), pad it
        # XGBClassifier handles this gracefully usually if num_class=3 is set.
        if probs.shape[1] < 3:
             full_probs = np.zeros((probs.shape[0], 3))
             # We rely on XGBoost returning the active classes, but just in case
             for i, c in enumerate(self.model.classes_):
                 full_probs[:, int(c)] = probs[:, i]
             return full_probs

        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the predicted class [-1, 0, 1] if confidence >= threshold, else 0."""
        probs = self.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1) - 1 # Map back {0,1,2} -> {-1,0,1}

        # Apply confidence gate
        mask = max_probs < self.config['confidence_threshold']
        predictions[mask] = 0

        return predictions


def _fit_binary_xgb(xgb_params: dict, X_train, y_binary_train,
                    X_val=None, y_binary_val=None):
    """Train a single binary XGBClassifier. Returns fitted model."""
    p = xgb_params.copy()
    p.pop('early_stopping_rounds', None)
    p['objective']   = 'binary:logistic'
    p['eval_metric'] = 'logloss'
    p.pop('num_class', None)

    clf = xgb.XGBClassifier(**p)
    if X_val is not None and y_binary_val is not None:
        clf.fit(X_train, y_binary_train,
                eval_set=[(X_val, y_binary_val)],
                verbose=False)
    else:
        clf.fit(X_train, y_binary_train, verbose=False)
    return clf


class VaranusDualModel:
    """
    v5.2 Dual-Engine model.

    Short Hunter: multi-class softprob (same as v5.1) trained on SHORT-ONLY
    labels {-1, 0}. Long labels never contaminate this model, so short signal
    frequency and confidence distribution are identical to v5.1.

    Long Runner: separate binary classifier trained on {+1 vs 0/-1} labels,
    using long-specific TP/SL barriers from build_dual_labels().

    Interface is identical to VaranusModel so all callers are unchanged.
    """

    def __init__(self, config: dict = MODEL_CONFIG):
        self.config      = config
        self.short_model = None   # multi:softprob on {-1, 0}
        self.long_model  = None   # binary on {+1 vs rest}
        self.classes_    = np.array([-1, 0, 1])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            y_short_train: pd.Series = None, y_short_val: pd.Series = None):
        """
        y_train / y_val      — combined dual labels {-1, 0, +1} from build_dual_labels()
                               used for the Long Runner binary model.
        y_short_train / val  — v5.1-style labels from label_trades(mss_signal).
                               If provided, used for the Short Hunter 3-class model.
                               If None, falls back to y_train (backward compat).
        """
        xgb_p = self.config['xgb_params'].copy()

        # ── Short Hunter: 3-class softprob ─────────────────────────────────────
        # Use y_short_train when provided — these come from label_trades(mss_signal)
        # which preserves the v5.1 label count (~2x more short labels than short_sig_only).
        y_s_tr = y_short_train if y_short_train is not None else y_train
        y_s_vl = y_short_val   if y_short_val   is not None else y_val

        p_short = xgb_p.copy()
        p_short.pop('early_stopping_rounds', None)
        y_str = (y_s_tr + 1).astype(int)          # {-1→0, 0→1, +1→2}

        self.short_model = xgb.XGBClassifier(**p_short)
        if X_val is not None and y_s_vl is not None:
            y_svl = (y_s_vl + 1).astype(int)
            self.short_model.fit(X_train, y_str,
                                 eval_set=[(X_val, y_svl)],
                                 verbose=False)
        else:
            self.short_model.fit(X_train, y_str, verbose=False)

        # ── Long Runner: independent binary on {+1 vs rest} ────────────────────
        # scale_pos_weight corrects training label imbalance: long labels are a
        # small minority (~3-5%) of the training set. Without correction, XGBoost
        # suppresses p_long scores and caps OOS long signals at ~56. Setting
        # scale_pos_weight = neg/pos forces the long arm to treat every long
        # candidate as worth neg/pos "neutral" examples — breaking the ceiling.
        y_long_tr = (y_train == 1).astype(int)
        y_long_vl = (y_val   == 1).astype(int) if y_val is not None else None
        n_neg = int((y_long_tr == 0).sum())
        n_pos = int((y_long_tr == 1).sum())
        long_spw = n_neg / n_pos if n_pos > 0 else 1.0
        xgb_p_long = xgb_p.copy()
        xgb_p_long['scale_pos_weight'] = long_spw
        self.long_model = _fit_binary_xgb(
            xgb_p_long, X_train, y_long_tr, X_val, y_long_vl)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns (N, 3) array: [prob_short, prob_neutral, prob_long].
        col 0 = class -1 (short), col 1 = class 0 (neutral), col 2 = class +1 (long).

        Probabilities are NOT cross-normalised between models. Each model is
        independent — p_short comes directly from the 3-class short model (same
        output as v5.1 VaranusModel), p_long from the binary long model.
        Normalising would divide p_short by (1 + p_long) and suppress signals
        that would have fired in v5.1.
        """
        if self.short_model is None or self.long_model is None:
            raise ValueError("VaranusDualModel has not been fitted yet.")

        # Short model: 3-class softprob → col 0 = P(short), col 1 = P(neutral)
        # Architecture identical to v5.1 VaranusModel.
        sp        = self.short_model.predict_proba(X)   # (N, 3)
        p_short   = sp[:, 0]   # class 0 → original -1
        p_neutral = sp[:, 1]   # class 1 → original  0

        # Long model: independent binary → col 1 = P(long)
        p_long = self.long_model.predict_proba(X)[:, 1]

        return np.column_stack([p_short, p_neutral, p_long])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns {-1, 0, +1}.
        Short fires when p_short >= threshold AND p_short > p_long.
        Long fires when p_long >= threshold AND p_long > p_short.
        Direction-specific thresholds (conf_thresh_short / conf_thresh_long)
        are applied by backtest.py; the global floor is applied here.
        """
        probs     = self.predict_proba(X)
        p_short   = probs[:, 0]
        p_long    = probs[:, 2]
        threshold = self.config['confidence_threshold']

        predictions = np.zeros(len(X), dtype=int)
        short_wins  = (p_short >= threshold) & (p_short >= p_long)
        long_wins   = (p_long  >= threshold) & (p_long  >  p_short)

        predictions[short_wins] = -1
        predictions[long_wins]  =  1
        return predictions
