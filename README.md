# Varanus v5.7 — Anti-Overfit Dual-Engine XGBoost Trading System

> **Dual-engine machine learning system for crypto altcoin trading.**
> Short Hunter frozen at peak performance (Trial #183). Long Runner re-optimized with 8-fold
> explicit-date walk-forward validation to eliminate selection bias.

---

## Table of Contents

1. [What Is Varanus?](#1-what-is-varanus)
2. [Architecture Overview](#2-architecture-overview)
3. [Universe](#3-universe)
4. [Feature Engineering](#4-feature-engineering)
5. [Labeling — Triple-Barrier Method](#5-labeling--triple-barrier-method)
6. [Model — XGBoost Classifier](#6-model--xgboost-classifier)
7. [Dual-Engine Signal Logic](#7-dual-engine-signal-logic)
8. [Walk-Forward Validation Methodology](#8-walk-forward-validation-methodology)
9. [Anti-Overfit Design (v5.7 Specific)](#9-anti-overfit-design-v57-specific)
10. [Optimization — Optuna Density Score](#10-optimization--optuna-density-score)
11. [Backtest Results (8-Fold WFV)](#11-backtest-results-8-fold-wfv)
12. [Blind Test Results](#12-blind-test-results)
    - [Blind Test 1 — Nov 2025 – Jan 2026](#blind-test-1--nov-01-2025--jan-24-2026)
    - [Blind Test 2 — Jan 2026 – Mar 2026](#blind-test-2--jan-24-2026--mar-09-2026)
13. [Strengths](#13-strengths)
14. [Weaknesses](#14-weaknesses)
15. [How to Run (Step-by-Step)](#15-how-to-run-step-by-step)
16. [File Structure](#16-file-structure)
17. [Configuration Reference](#17-configuration-reference)

---

## 1. What Is Varanus?

Varanus is a quantitative crypto trading system that uses gradient-boosted trees (XGBoost) to classify price-action set-ups on 4-hour candles across 15 mid-cap altcoin pairs. It is **not** a neural network, indicator crossover system, or generative model — it is a supervised classifier trained on historically labelled trade outcomes with rigorous out-of-sample testing.

**Version history highlights:**

| Version | Key Change |
|---------|-----------|
| v5.1 | 80-pair universe, 3 safety gates, Calmar objective |
| v5.2 | Short Hunter introduced, frozen at Trial #183 |
| v5.5 | Long Runner added alongside Short Hunter |
| v5.6 | Dual-engine combined; 5 ratio-based WFV folds |
| **v5.7** | **8 explicit-date folds, hard Nov-2025 cutoff, selection-bias fix** |

The critical fix in v5.7: v5.6 used 5 ratio-based OOS windows that were seen by Optuna on every trial, causing inflated +890.6% paper returns. v5.7 uses 8 hard-coded calendar folds with a Nov 01 2025 wall that the optimizer never crosses.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     VARANUS v5.7                            │
│                                                             │
│   ┌──────────────────┐    ┌──────────────────────────────┐  │
│   │   SHORT HUNTER   │    │       LONG RUNNER            │  │
│   │   (FROZEN)       │    │   (v5.7 Optimized)           │  │
│   │                  │    │                              │  │
│   │  Trial #183      │    │  Trial #233                  │  │
│   │  conf > 0.786    │    │  conf > 0.681                │  │
│   │  TP: 5.77×ATR    │    │  TP: 2.98× entry range       │  │
│   │  SL: 0.71×ATR    │    │  SL: 1.15× entry range       │  │
│   └────────┬─────────┘    └──────────────┬───────────────┘  │
│            │                             │                  │
│            └──────────┬──────────────────┘                  │
│                       ▼                                     │
│              COMBINED SIGNAL GATE                           │
│   (p_short_max_for_long ≤ 0.739 required for long entry)    │
│                       │                                     │
│                       ▼                                     │
│              TRIPLE-BARRIER EXIT                            │
│      TP hit | SL hit | Time exit | MSS Invalidation         │
└─────────────────────────────────────────────────────────────┘
```

Both engines use the **same XGBoost model** and **same feature set** but classify opposite directions. A Long trade requires both: (1) long classifier confidence above threshold AND (2) short classifier confidence below `p_short_max_for_long`.

---

## 3. Universe

**15 Tier-2 altcoin pairs (USDT-quoted, Binance Spot):**

```
ADA   AVAX  LINK  DOT   TRX
NEAR  UNI   SUI   ARB   OP
POL   APT   ATOM  FIL   ICP
```

**Selection rationale:** Mid-cap assets with sufficient liquidity ($50M+ 24h volume) and price action structure. Excludes BTC/ETH (too efficient) and micro-caps (too noisy/illiquid).

**Data timeframes:**
- Primary: **4h** candles (signal generation)
- Higher-timeframe bias: **1d** candles (RSI filter)

**Data range:** January 2023 – October 2025 (34 months)
- ARB: starts March 2023 (launch date)
- SUI: starts May 2023 (launch date)
- POL: uses MATIC symbol on Binance Vision before September 2024

---

## 4. Feature Engineering

All features are computed in `varanus/pa_features.py` from raw OHLCV 4h candles and higher-timeframe 1d data.

### Price Action Features

| Feature | Description |
|---------|-------------|
| `mss` | Market Structure Shift — swing high/low break with configurable lookback (default 31 bars) |
| `fvg` | Fair Value Gap — imbalance candle with min ATR ratio threshold (0.39×ATR) |
| `fvg_age` | Bars since FVG formed (max age gate: 22 bars) |
| `sweep` | Liquidity sweep — wick extension beyond prior swing, min 0.641% of price |
| `rvol` | Relative Volume — current volume / rolling 20-bar average (threshold: 1.287×) |

### Technical Features

| Feature | Description |
|---------|-------------|
| `rsi_4h` | RSI(14) on 4h candles |
| `rsi_1d` | RSI(14) on daily candles (HTF bias filter for longs: ≤ 61) |
| `ema_alignment` | EMA(20) vs EMA(50) vs EMA(200) trend stack |
| `atr_pct` | ATR(14) as % of close price (volatility normaliser) |
| `body_ratio` | Candle body / full range (momentum quality) |
| `upper_wick` | Upper wick / full range |
| `lower_wick` | Lower wick / full range |
| `close_position` | Close position within high-low range |

### HTF Bias Integration

The 1d RSI is resampled and aligned to 4h timestamps to prevent look-ahead bias. Any long signal where `rsi_1d > rsi_1d_long_limit` (61) is suppressed regardless of classifier confidence.

---

## 5. Labeling — Triple-Barrier Method

Labels are generated in `varanus/tbm_labeler.py` using the **Triple-Barrier Method** (López de Prado, 2018):

```
Entry →  ┌──────────────────── TP barrier (tp_mult × ATR from entry)
         │
         │   Price path
         │
         └──────────────────── SL barrier (sl_mult × ATR from entry)
                              OR max holding bars exceeded → time exit
```

**Label encoding:**
- `+1` (long): price hits TP before SL within `max_holding` bars
- `-1` (short): price hits SL before TP within `max_holding` bars
- `0` (neutral/time): neither barrier hit within `max_holding` bars

Labels are computed **without** any future information leaking into features. The TBM looks forward only in the label column, not in any feature.

---

## 6. Model — XGBoost Classifier

Model defined in `varanus/model.py`.

**Hyperparameters (v5.7 optimized):**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 218 |
| `learning_rate` | 0.0609 |
| `max_depth` | 6 |
| `subsample` | 0.957 |
| `objective` | `multi:softprob` |
| `num_class` | 3 |

**Training procedure:**
- Train on in-sample fold data
- Predict probabilities for OOS validation fold
- Extract class probability: `P(+1)` for long engine, `P(-1)` for short engine
- Confidence thresholds applied post-inference (not during training)

**Data integrity:**
- `X.dropna()` applied before training and inference to prevent NaN propagation
- Features are normalized per fold (no cross-fold data leakage)
- No feature selection — all features are always used

---

## 7. Dual-Engine Signal Logic

### Short Hunter (FROZEN at Trial #183)

```python
short_signal = (
    p_short >= 0.786        # high classifier confidence for short
    AND rsi_4h conditions   # overbought filter
    AND sweep detected      # liquidity sweep confirmation
)
```

TP at `5.768 × ATR` below entry. SL at `0.709 × ATR` above entry. Leverage: up to 5× when `conf ≥ 0.968`.

The Short Hunter is frozen because it reached peak performance at Trial #183 during v5.2 optimization. Re-optimizing it would risk overfitting to short-side patterns that may not repeat.

### Long Runner (v5.7 Optimized)

```python
long_signal = (
    p_long >= 0.681         # long classifier confidence
    AND p_short <= 0.739    # short engine does NOT disagree
    AND rsi_1d <= 61        # HTF not overbought
    AND mss detected        # market structure shift bullish
)
```

TP at `2.98 × entry_range` above entry. SL at `1.15 × entry_range` below entry.

### Combined Gate

The `p_short_max_for_long` parameter (0.739) ensures the two engines are not contradicting each other. When the short engine has moderate short conviction simultaneously with a long signal, the long is suppressed. This cross-engine validation significantly reduces false positives.

---

## 8. Walk-Forward Validation Methodology

Walk-forward validation (WFV) is implemented in `varanus/walk_forward.py`.

### Why WFV Instead of Simple Train/Test Split?

A single split tells you how the model performs on one unseen period. WFV tests across **8 different market regimes** (ranging from 2023's bear market through 2024's bull run and 2025's mixed conditions). This gives a much more reliable estimate of live performance.

### The 8 Explicit Folds

Each fold has three windows: **Train → Validate → Test (OOS)**

| Fold | Train | Validate | Test (OOS) |
|------|-------|----------|------------|
| 1 | Jan–Jun 2023 | Jul–Sep 2023 | Oct–Dec 2023 |
| 2 | Jan–Sep 2023 | Oct–Dec 2023 | Jan–Mar 2024 |
| 3 | Jan–Dec 2023 | Jan–Mar 2024 | Apr–Jun 2024 |
| 4 | Jan–Mar 2024 | Apr–Jun 2024 | Jul–Sep 2024 |
| 5 | Jan–Jun 2024 | Jul–Sep 2024 | Oct–Dec 2024 |
| 6 | Jan–Sep 2024 | Oct–Dec 2024 | Jan–Mar 2025 |
| 7 | Jan–Dec 2024 | Jan–Mar 2025 | Apr–Jun 2025 |
| 8 | Jan–Mar 2025 | Apr–Jun 2025 | Jul–Oct 2025 |

**Hard cutoff: November 01, 2025.** No data after this date is ever used during optimization or backtesting.

### Rolling Architecture

Each fold's training data grows as the system moves forward in time (expanding window). The model is retrained from scratch for each fold — no weights carry over. This mirrors how a live system would periodically retrain on all available history.

---

## 9. Anti-Overfit Design (v5.7 Specific)

v5.7 introduces four specific anti-overfit mechanisms:

### 9.1 Selection-Bias Elimination

**v5.6 Problem:** 5 ratio-based folds were computed from the same data range. Over 300 Optuna trials, the optimizer saw the exact same OOS windows 300 times and learned to tune parameters specifically to those windows. The result was an inflated +890.6% backtest figure that collapsed in live conditions.

**v5.7 Fix:** 8 hard-coded calendar folds with different market regimes. Each fold covers a distinct period of crypto market history. The optimizer cannot overfit to a recurring window pattern.

### 9.2 Per-Fold Trade Count Gate (Hard)

```python
if long_trades_in_fold < 5:
    return -999  # trial immediately discarded
```

Any trial that produces fewer than 5 long trades in any single fold is invalid. This prevents the optimizer from finding "perfect" parameters that only trade once or twice in a fold and always win.

### 9.3 Combined Trade Count Gate (Hard)

```python
if total_long_trades < 80:
    return -999
```

At least 80 long trades must be generated across all 8 folds combined. Ensures statistical significance.

### 9.4 Unprofitable Fold Penalty

```python
if any(fold_net_pnl <= 0 for fold in folds):
    final_score -= 100
```

Any trial where a fold loses money receives a -100 point deduction from its final score. This strongly biases the optimizer toward parameter sets that are profitable across all market regimes, not just cherry-picked periods.

### 9.5 MedianPruner (Early Stopping)

```python
MedianPruner(n_startup_trials=10, n_warmup_steps=3)
```

After the first 10 trials complete (to establish a baseline), any new trial that falls below the median score after 3 folds is pruned. This saves computation on clearly poor parameter sets.

---

## 10. Optimization — Optuna Density Score

### Objective Function

```
Density Score = (WR × Count × ln(Count + 1)) / DD_Impact
```

Where:
- `WR` = long win rate across all folds (0–1)
- `Count` = total long trades
- `ln(Count + 1)` = logarithmic trade volume bonus (rewards more trades, diminishing returns)
- `DD_Impact` = max drawdown penalty multiplier

**Why not Calmar or Sharpe?**
- Calmar (`return / MaxDD`) can be gamed by finding parameters that reduce trade frequency to near-zero
- Sharpe rewards smooth equity curves but ignores absolute trade count
- Density Score balances accuracy (WR), statistical significance (Count), and risk control (DD_Impact) simultaneously

### Search Space

Only 5 Long Runner parameters are optimized (Short Hunter is frozen):

| Parameter | Range | Description |
|-----------|-------|-------------|
| `conf_thresh_long` | 0.55 – 0.70 | Minimum long classifier confidence |
| `tp_mult_long` | 2.0 – 3.5 | Take-profit multiplier vs entry range |
| `sl_mult_long` | 0.70 – 1.20 | Stop-loss multiplier vs entry range |
| `rsi_1d_long_limit` | 45 – 65 | Max daily RSI for long entry |
| `p_short_max_for_long` | 0.55 – 0.75 | Max short confidence allowed on long entry |

**Best found (Trial #233):**

| Parameter | Value |
|-----------|-------|
| `conf_thresh_long` | 0.6809 |
| `tp_mult_long` | 2.9799 |
| `sl_mult_long` | 1.1468 |
| `rsi_1d_long_limit` | 61 |
| `p_short_max_for_long` | 0.7393 |
| **Best Score** | **247.554** |

---

## 11. Backtest Results (8-Fold WFV)

Backtest runs the 8 folds sequentially, starting with $5,000 capital per fold. Each fold model is trained on its in-sample window and traded on its OOS test window.

### Per-Fold Summary

| Fold | Test Period | Return | MaxDD | WR | Trades | Sharpe |
|------|------------|--------|-------|-----|--------|--------|
| 1 | Oct–Dec 2023 | +142.7% | -11.8% | 39.5% | 109 | 4.71 |
| 2 | Jan–Mar 2024 | +281.1% | -7.5% | 48.0% | 98 | 5.78 |
| 3 | Apr–Jun 2024 | +123.4% | -11.5% | 33.3% | 87 | 4.53 |
| 4 | Jul–Sep 2024 | +197.2% | -9.6% | 41.8% | 98 | 6.27 |
| 5 | Oct–Dec 2024 | +127.1% | -3.8% | 48.9% | 45 | 5.00 |
| 6 | Jan–Mar 2025 | +108.9% | -13.0% | 33.7% | 83 | 4.32 |
| 7 | Apr–Jun 2025 | +27.0% | -29.8% | 34.3% | 67 | 1.93 |
| 8 | Jul–Oct 2025 | +60.7% | -9.9% | 38.1% | 42 | 3.81 |

### Aggregate Statistics

| Metric | Value |
|--------|-------|
| **Total Return** | **+1,068%** |
| **Total Trades** | **629** |
| **Overall Win Rate** | **39.6%** |
| **Profit Factor** | **2.99** |
| **Average Sharpe** | **4.54** |
| **Worst Drawdown** | **-29.8% (Fold 7)** |
| **Long Trades** | 231 (43.7% WR) |
| **Short Trades** | 398 (37.2% WR) |
| **Avg Win** | $322.53 |
| **Avg Loss** | -$74.53 |
| **Expectancy** | $82.65 per trade |

### Exit Type Breakdown

| Exit Type | Count | % |
|-----------|-------|---|
| MSS Invalidation | 213 | 33.9% |
| Stop Loss | 227 | 36.1% |
| Take Profit | 108 | 17.2% |
| Time Exit | 81 | 12.9% |

---

## 12. Blind Test Results

Both blind tests use data the optimizer and backtest engine **never saw**. One final model is trained on the full Jan 2023 – Oct 2025 dataset, then traded on each blind window without any retraining or parameter changes.

### Blind Test 1 — Nov 01 2025 – Jan 24 2026

| Metric | Value |
|--------|-------|
| **Period** | Nov 01 2025 – Jan 24 2026 (85 days) |
| **Total Return** | **+64.0%** |
| **Net Profit** | $3,200.02 on $5,000 |
| **Total Trades** | 70 |
| **Long Trades** | 32 |
| **Short Trades** | 38 |
| **Overall Win Rate** | 42.9% |
| **Long Win Rate** | **46.9%** ✅ |
| **Short Win Rate** | 39.5% |
| **Profit Factor** | 2.18 |
| **Max Drawdown** | -9.9% |
| **Sharpe Ratio** | **3.884** |
| **Expectancy** | $44.72 per trade |

### Blind Test 2 — Jan 24 2026 – Mar 09 2026

| Metric | Value |
|--------|-------|
| **Period** | Jan 24 2026 – Mar 09 2026 (~44 days) |
| **Total Return** | **+32.7%** |
| **Net Profit** | $1,633.32 on $5,000 |
| **Total Trades** | 18 |
| **Long Trades** | 11 |
| **Short Trades** | 7 |
| **Overall Win Rate** | 55.6% |
| **Long Win Rate** | **45.5%** ✅ |
| **Short Win Rate** | 71.4% |
| **Profit Factor** | 3.31 |
| **Max Drawdown** | -9.3% |
| **Sharpe Ratio** | **5.122** |
| **Expectancy** | $90.74 per trade |

### Combined Blind Test Summary

| | Blind Test 1 | Blind Test 2 |
|---|---|---|
| Period | Nov 01 2025 – Jan 24 2026 | Jan 24 2026 – Mar 09 2026 |
| Return | +64.0% | +32.7% |
| Win Rate | 42.9% | 55.6% |
| Profit Factor | 2.18 | 3.31 |
| Sharpe | 3.884 | 5.122 |
| Max DD | -9.9% | -9.3% |
| Trades | 70 | 18 |

**Interpretation:** Both blind tests confirm that the WFV results are genuine and not an artifact of optimization. Across ~129 days of completely unseen data, the system achieved positive returns with controlled drawdown and a Sharpe above 3.8 in both periods. Blind Test 2 shows improving short-side performance (71.4% WR) and a higher overall win rate, suggesting the model's edge has not decayed into 2026.

---

## 13. Strengths

### Methodological Strengths

1. **Genuine out-of-sample validation.** The hard Nov-2025 cutoff is never crossed during development. The blind test is a true hold-out.

2. **Selection-bias prevention.** 8 diverse calendar folds spanning 3 years of different crypto market regimes (2023 bear, 2024 bull, 2025 mixed) prevent the optimizer from learning OOS window-specific patterns.

3. **Statistical significance enforced.** Hard 80-trade minimum means reported WR and PF are meaningful statistics, not lucky streaks of 5-10 trades.

4. **Dual-engine cross-validation.** Long entries require the short engine to disagree below a threshold. This is a built-in contradiction filter.

5. **Price-action foundation.** MSS, FVG, and liquidity sweep features are grounded in institutional order flow logic, not arbitrary indicator combinations.

6. **Frozen Short Hunter.** Prevents re-optimizing a working system. The short engine's parameters are immutable after Trial #183.

7. **Robust labeling.** Triple-Barrier Method produces forward-looking labels without look-ahead bias and respects the time-stop boundary.

### Practical Strengths

8. **High profit factor (2.99).** For every $1 lost, $2.99 is earned. Profitable even at 33% win rate.

9. **Asymmetric R:R.** Average win ($322) is 4.3× average loss ($74), meaning the system profits from letting winners run.

10. **Decent trade frequency.** 629 trades over 2 years (~26/month) is sufficient for statistical analysis without over-trading.

---

## 14. Weaknesses

### Performance Weaknesses

1. **Fold 7 (-29.8% DD, 34.3% WR).** April–June 2025 was a particularly difficult period for altcoins. The system degraded significantly. Any live deployment should expect extended drawdown periods.

2. **Lower short WR (37.2% vs 43.7% long).** The Short Hunter, while frozen at a good configuration, underperforms the Long Runner on win rate. The asymmetric profit factor masks this.

3. **Win rate below 50%.** At 39.6% overall WR, the system relies entirely on large winners to be profitable. Any degradation in average winner size would turn the system unprofitable quickly.

4. **Fold-dependent variance.** Returns range from +27% (Fold 7) to +281% (Fold 2). Live performance will be highly dependent on which market regime is active.

### Methodological Weaknesses

5. **Spot data, futures logic.** The model is trained and backtested on Binance Spot OHLCV data but the intended deployment uses futures/perpetuals. Funding rates, open interest, and liquidation cascades are not modeled.

6. **No slippage or fees in backtest.** Real-world execution costs (0.04-0.1% taker fees, 0.02-0.2% slippage on 4h closes) would reduce returns meaningfully, especially for short trades with tight stops.

7. **Fixed position sizing.** The backtest uses fixed $2,500–$6,000 per trade depending on leverage tier. Real portfolio management would need dynamic sizing relative to equity.

8. **No correlation management.** Multiple signals can fire on correlated assets simultaneously (e.g., AVAX, LINK, DOT all dump together). Real deployment needs a portfolio-level correlation gate.

9. **Regime-blind.** The system has no explicit bull/bear market detector. It enters longs during confirmed downtrends and shorts during bull runs if the local pattern qualifies.

10. **Data snooping risk remains.** Even with 8 folds, all hyperparameter decisions (feature selection, fold design, objective function) were made after seeing the overall data distribution. This is an unavoidable limitation of any retrospective study.

---

## 15. How to Run (Step-by-Step)

### Prerequisites

**Python 3.10+ required.** Tested on Python 3.12.

```bash
# Create and activate virtual environment
python3 -m venv varanus_env
source varanus_env/bin/activate  # Linux/Mac
# OR: varanus_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 1 — Fetch Historical Data

Downloads 34 months of 1h OHLCV for all 15 assets from Binance Vision, resamples to 4h, and saves to local cache. Takes 15–20 minutes depending on your connection.

```bash
python fetch_all_data_v57.py
```

This creates `varanus/data/cache/` with 30 parquet files (15 × `_USDT.parquet` 4h + 15 × `_USDT_1h.parquet`).

### Step 2 — Run Optimization (Optional — pre-optimized params included)

> Skip this step if you want to use the pre-optimized `varanus/config/best_params_v57.json` (Trial #233).

Runs 300 Optuna trials across 8 WFV folds. Takes 4–8 hours on a modern CPU.

```bash
python run_dual_engine_optimization_v57.py
```

Outputs: `varanus/config/best_params_v57.json`

### Step 3 — Run 8-Fold Backtest

Re-runs all 8 folds with the best parameters to generate trade logs and performance summaries.

```bash
python run_backtest_v57.py
```

Outputs:
- `varanus/config/varanusv57_backtest_trades.csv` — individual trade records
- `varanus/config/varanusv57_backtest_summary.csv` — per-fold performance stats

### Step 4 — Run Blind Test 1

Downloads Nov 2025 – Jan 2026 data, trains a final model on the full 2023–2025 dataset, and trades the blind window.

```bash
python run_blind_test_v57.py
```

Outputs:
- `varanus/config/varanusv57_blind_trades.csv`
- `varanus/config/varanusv57_blind_summary.csv`

### Step 5 — Run Blind Test 2 (Extended)

Downloads Jan 2026 – Mar 2026 data and trades a second unseen window using the same final model.

```bash
python run_blind_test_v57_jan24_mar9.py
```

Outputs:
- `varanus/config/varanusv57_blind2_trades.csv`
- `varanus/config/varanusv57_blind2_summary.csv`

### Reproducing Results from Scratch

```bash
# 1. Fetch all data
python fetch_all_data_v57.py

# 2. Optimize (or skip and use included best_params_v57.json)
python run_dual_engine_optimization_v57.py

# 3. Backtest
python run_backtest_v57.py

# 4. Blind test
python run_blind_test_v57.py
```

---

## 16. File Structure

```
varanusv57/
│
├── README.md                            ← This file
├── requirements.txt                     ← Python dependencies
├── varanusv57.md                        ← Original design specification
│
├── fetch_all_data_v57.py                ← Step 1: Download Binance Vision data
├── run_dual_engine_optimization_v57.py  ← Step 2: 300-trial Optuna search
├── run_backtest_v57.py                  ← Step 3: 8-fold WFV backtest
├── run_blind_test_v57.py                ← Step 4: Blind test 1 (Nov 2025–Jan 2026)
├── run_blind_test_v57_jan24_mar9.py     ← Step 5: Blind test 2 (Jan–Mar 2026)
│
├── varanus/                             ← Core library
│   ├── __init__.py
│   ├── universe.py          ← 15 Tier-2 assets + volume/exclusion rules
│   ├── pa_features.py       ← Price action + technical feature engineering
│   ├── tbm_labeler.py       ← Triple-Barrier Method labeler
│   ├── model.py             ← XGBoost wrapper (train / predict_proba)
│   ├── backtest.py          ← Trade simulation engine (entry/exit/PnL)
│   ├── walk_forward.py      ← WFV fold generator + runner (v5.7 explicit dates)
│   ├── optimizer.py         ← Optuna objective function + study runner
│   ├── risk.py              ← Position sizing + leverage tiers
│   ├── alerts.py            ← Telegram notification stubs
│   ├── paper_trader.py      ← Paper trading engine
│   ├── save_results.py      ← CSV/summary output helpers
│   ├── plot_performance.py  ← Equity curve + drawdown plots
│   ├── plot_wicks.py        ← Candlestick + signal visualization
│   │
│   ├── config/
│   │   ├── best_params_v57.json         ← Optimized parameters (Trial #233)
│   │   ├── varanusv57_backtest_trades.csv   ← 629 backtest trades
│   │   ├── varanusv57_backtest_summary.csv  ← Per-fold performance
│   │   ├── varanusv57_blind_trades.csv      ← 70 blind test 1 trades (Nov–Jan)
│   │   ├── varanusv57_blind_summary.csv     ← Blind test 1 performance
│   │   ├── varanusv57_blind2_trades.csv     ← 18 blind test 2 trades (Jan–Mar)
│   │   └── varanusv57_blind2_summary.csv    ← Blind test 2 performance
│   │
│   └── data/
│       └── cache/           ← Parquet files (generated by fetch script)
│                              NOT included in repo — run fetch_all_data_v57.py
│
└── .gitignore
```

---

## 17. Configuration Reference

### best_params_v57.json — Full Parameter Set

```json
{
  "mss_lookback":          31,      // Market structure shift lookback bars
  "fvg_min_atr_ratio":     0.392,   // Min FVG size as fraction of ATR
  "sweep_min_pct":         0.00641, // Min wick extension for liquidity sweep
  "fvg_max_age":           22,      // Max bars since FVG before it expires
  "rvol_threshold":        1.287,   // Relative volume minimum
  "rsi_oversold":          36,      // RSI oversold threshold (short filter)
  "rsi_overbought":        58,      // RSI overbought threshold (short filter)
  "max_holding":           31,      // Max bars before time exit
  "xgb_n_estimators":      218,     // XGBoost trees
  "xgb_subsample":         0.957,   // Row subsampling ratio
  "xgb_lr":                0.0609,  // Learning rate
  "xgb_max_depth":         6,       // Tree depth

  // SHORT HUNTER (FROZEN — Trial #183, DO NOT MODIFY)
  "conf_thresh_short":     0.786,   // Min short confidence
  "tp_atr_mult":           5.768,   // Short take-profit (ATR multiples)
  "sl_atr_mult":           0.709,   // Short stop-loss (ATR multiples)
  "leverage_5x_trigger":   0.968,   // Min confidence for 5× leverage

  // LONG RUNNER (v5.7 Optimized — Trial #233)
  "conf_thresh_long":      0.6809,  // Min long confidence
  "tp_mult_long":          2.9799,  // Long take-profit multiplier
  "sl_mult_long":          1.1468,  // Long stop-loss multiplier
  "rsi_1d_long_limit":     61,      // Max daily RSI for long entry
  "p_short_max_for_long":  0.7393   // Max short engine confidence on long entry
}
```

---

## License & Disclaimer

This repository is for educational and research purposes. Past backtest performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. The blind test results (+64% and +32.7% across two unseen periods) are from limited windows and should not be extrapolated as expected live performance.

---

*Varanus v5.7 — Anti-Overfit Edition. Short Hunter FROZEN at Trial #183. Long Runner optimized at Trial #233.*
