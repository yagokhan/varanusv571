# Varanus v5.7 — Claude Session Brief

This document is a briefing for Claude in a new session. Read this fully before doing anything.

---

## What Is Varanus

Varanus is a dual-engine crypto trading system using XGBoost on 4h OHLCV data. It trades 14 Tier-2 altcoins on Binance spot/futures. It generates signals using price action features (MSS, FVG, liquidity sweeps, RVOL, RSI, EMA alignment, HTF bias), sizes positions using confidence-tiered leverage, and manages risk via Triple-Barrier Method (TBM) labeling.

The system has two engines:
- **Short Hunter** — finds high-confidence short setups. Frozen at Trial #183 (v5.6).
- **Long Runner** — optimized long-side parameters via Optuna walk-forward search.

---

## Version History

| Version | Notes |
|---|---|
| v5.1 | Short Hunter established |
| v5.2 | Dual-engine introduced |
| v5.5 | Long WR slippage addressed |
| v5.6 | "The Golden Ratio" — 5-fold WFV, Trial #293, score 431.92, +890.6% on $5,000 |
| **v5.7** | **Current target** — 8-fold WFV, extended data Jan 2023–Oct 2025, true blind test |

---

## v5.6 Best Parameters (starting point for v5.7)

Stored in `varanus/config/best_params_v56.json`:

```json
{
  "conf_thresh_long": 0.6694,
  "tp_mult_long": 2.373,
  "sl_mult_long": 0.931,
  "rsi_1d_long_limit": 61,
  "p_short_max_for_long": 0.746,
  "conf_thresh_short": 0.786,
  "tp_atr_mult": 5.768,
  "sl_atr_mult": 0.709,
  "leverage_5x_trigger": 0.968
}
```

---

## Asset Universe (14 coins — Tier 2)

`ADA, APT, ARB, ATOM, AVAX, DOT, FIL, ICP, LINK, NEAR, OP, SUI, TRX, UNI`

Timeframes: **4H** (signals) + **1D** (RSI filter via 1h resample)

---

## Price Data — Already Fetched

All 14 assets are in `varanus/data/cache/` as parquet files:

| Asset | Start | End | 4h Bars |
|---|---|---|---|
| ADA, APT, ATOM, AVAX, DOT, FIL, ICP, LINK, NEAR, OP, TRX, UNI | 2023-01-01 | 2025-10-31 | 6,210 |
| ARB | 2023-03-23 | 2025-10-31 | 5,721 |
| SUI | 2023-05-03 | 2025-10-31 | 5,475 |

Files: `{ASSET}_USDT.parquet` (4h) and `{ASSET}_USDT_1h.parquet` (1h)

**Hard cutoff: Nov 01, 2025** — nothing after this date is used for training or optimization.

Data was fetched from https://data.binance.vision using `fetch_binance_vision_2025.py`.
Note: 2025 Binance monthly files use microseconds (16-digit open_time). The fetch script auto-detects and divides by 1000. Do not use the old `fetch_historical_data.py` (Yahoo Finance, only ~2 years).

---

## Why v5.7 — The Core Problem with v5.6

v5.6 ran 300 Optuna trials across the **same 5 OOS windows** every trial. Even though each window is out-of-sample per fold, the optimizer sees the same 5 evaluation windows 300 times and gradually finds parameters that happen to work on those specific windows. This is selection bias — the 890.6% return is likely inflated.

**v5.7 fix:**
1. More folds (8 instead of 5) — harder to overfit
2. More historical data (Jan 2023 vs Mar 2024) — more diverse market regimes
3. Hard cutoff at Nov 01, 2025 — optimizer never touches recent data
4. True blind test regions reserved for honest final evaluation

---

## v5.7 Data Split Design

```
|-- OPTIMIZATION REGION (Jan 2023 – Oct 2025) --|-- BLIND TEST --|-- LIVE --|
Jan 2023                                    Oct 2025  Jan 25 2026   now
                                                 ^         ^
                                           Hard cutoff   Blind start
```

### Blind Test Regions (NEVER used during optimization)
| Region | Dates | Length | Purpose |
|---|---|---|---|
| Pseudo-holdout | Nov 01, 2025 – Jan 24, 2026 | ~85 days | Final honest backtest |
| Live blind | Jan 25, 2026 → present | 43+ days | True OOS |

---

## 8-Fold WFV Configuration

Each fold uses rolling windows. Approximate structure (tune exact dates in `walk_forward.py`):

| Fold | Train | Validation | Test (OOS) |
|---|---|---|---|
| 1 | Jan–Jun 2023 | Jun–Sep 2023 | Sep–Dec 2023 |
| 2 | Apr–Sep 2023 | Sep–Dec 2023 | Dec 2023–Mar 2024 |
| 3 | Jul 2023–Jan 2024 | Jan–Apr 2024 | Apr–Jul 2024 |
| 4 | Oct 2023–Apr 2024 | Apr–Jul 2024 | Jul–Oct 2024 |
| 5 | Jan–Jul 2024 | Jul–Oct 2024 | Oct 2024–Jan 2025 |
| 6 | Apr–Oct 2024 | Oct 2024–Jan 2025 | Jan–Apr 2025 |
| 7 | Jul 2024–Jan 2025 | Jan–Apr 2025 | Apr–Jul 2025 |
| 8 | Oct 2024–Apr 2025 | Apr–Jul 2025 | Jul–Oct 2025 |

---

## Objective Function (unchanged from v5.6)

```
Score = (WR × Count × ln(Count+1)) / DD_Impact
```

Where `DD_Impact = 1 + max(0, max_dd - 0.10) * 5`

**Hard gates per fold (all 8 must pass):**
- Long win rate ≥ 41%
- Total long trades ≥ 80
- All folds profitable

---

## What Needs To Be Done (Task List for v5.7)

### Step 1 — Update Walk-Forward Config
File: `varanus/walk_forward.py`

Find `WFV_CONFIG_V51` (or similar dict) and add a new `WFV_CONFIG_V57` with 8 folds using the date ranges above. Each fold entry needs: `train_start`, `train_end`, `val_start`, `val_end`, `test_start`, `test_end`.

### Step 2 — Create Optimization Script
File: `run_dual_engine_optimization_v57.py`

Copy from `run_dual_engine_optimization_v56.py`. Change:
- `WFV_CONFIG` reference → `WFV_CONFIG_V57`
- n_trials = 300 (keep same)
- Output file: `varanus/config/best_params_v57.json`
- Log file: `logs/dual_engine_opt_v57.log`
- Version strings: v5.6 → v5.7 where relevant

### Step 3 — Create Backtest Script
File: `run_backtest_v57.py`

Copy from `run_backtest_v56.py`. Change:
- Load `best_params_v57.json`
- Use `WFV_CONFIG_V57`
- Output CSVs: `varanusv57_backtest_trades.csv`, `varanusv57_backtest_summary.csv`

### Step 4 — Run Optimization
```bash
nohup python3 run_dual_engine_optimization_v57.py >> logs/dual_engine_opt_v57.log 2>&1 &
```
Takes ~4–6 hours on a strong CPU. Monitor with:
```bash
grep -aoP "Best value: \K[0-9.]+" logs/dual_engine_opt_v57.log | tail -5
tail -20 logs/dual_engine_opt_v57.log
```

### Step 5 — Run Full Backtest
```bash
python3 run_backtest_v57.py
```

### Step 6 — Run Blind Test
After backtest, run the system on Nov 01, 2025 → Jan 24, 2026 data using `best_params_v57.json` but **without** retraining. Compare results to v5.6 blind test to validate the edge is real.

### Step 7 — Compare v5.6 vs v5.7
Key metrics to compare:
- Win rate (long/short/overall)
- Profit factor
- Max drawdown
- Sharpe ratio
- Consistency across folds
- Blind test performance

---

## Project Structure

```
v5.7/
├── varanusv57.md                            ← this file
├── requirements.txt
├── fetch_binance_vision_2025.py             ← use this to refresh data
├── fetch_historical_data.py                 ← OLD (Yahoo Finance) — do not use
├── run_dual_engine_optimization_v56.py      ← v5.6 reference
├── run_dual_engine_optimization_v57.py      ← TO CREATE (Step 2)
├── run_backtest_v56.py                      ← v5.6 reference
├── run_backtest_v57.py                      ← TO CREATE (Step 3)
├── run_walk_forward_v56.py                  ← v5.6 reference
├── run_paper.py                             ← paper trading (update after optimization)
└── varanus/
    ├── alerts.py                            ← Telegram alerts (all say VARANUS v5.6 — update to v5.7 after)
    ├── backtest.py                          ← backtesting engine
    ├── model.py                             ← XGBoost model training
    ├── optimizer.py                         ← Optuna objective & search space
    ├── walk_forward.py                      ← WFV engine — ADD WFV_CONFIG_V57 here
    ├── pa_features.py                       ← price action features (do not change)
    ├── risk.py                              ← position sizing & risk management
    ├── universe.py                          ← TIER2_UNIVERSE list
    ├── paper_trader.py                      ← live paper trading engine
    ├── config/
    │   ├── best_params_v56.json             ← v5.6 optimized params (reference)
    │   ├── best_params_v57.json             ← TO BE CREATED by optimization
    │   ├── telegram.env                     ← bot token + chat ID
    │   ├── varanusv56_backtest_trades.csv   ← v5.6 results
    │   └── varanusv56_backtest_summary.csv  ← v5.6 summary
    └── data/cache/
        ├── ADA_USDT.parquet                 ← 4h data Jan 2023–Oct 2025
        ├── ADA_USDT_1h.parquet              ← 1h data (for 1D RSI)
        └── ... (all 14 assets × 2 files)
```

---

## Telegram Setup

Credentials in `varanus/config/telegram.env`:
```
VARANUS_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
VARANUS_CHAT_ID=443025840
```
Bot name: varanus6

---

## Key Files to Understand First

If you need context on how the system works, read in this order:
1. `varanus/universe.py` — asset list
2. `varanus/pa_features.py` — all features used by the model
3. `varanus/walk_forward.py` — fold config and WFV engine
4. `varanus/optimizer.py` — Optuna objective, search space, gates
5. `varanus/backtest.py` — how trades are simulated
6. `varanus/risk.py` — position sizing, leverage tiers, circuit breakers

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Requirements: `ccxt==4.5.40, numpy==2.4.2, optuna==4.7.0, pandas==3.0.1, pyarrow==23.0.1, requests==2.31.0, scikit-learn==1.8.0, scipy==1.17.1, xgboost==3.2.0`

---

## Summary — Start Here

1. Read this file
2. `pip install -r requirements.txt`
3. Read `varanus/walk_forward.py` to understand current fold config
4. Add `WFV_CONFIG_V57` with 8 folds (date ranges in this doc)
5. Create `run_dual_engine_optimization_v57.py` from v5.6 version
6. Run optimization overnight
7. Run backtest with new params
8. Run blind test on Nov 2025 – Jan 2026
9. Compare to v5.6 results
