# Varanus Trading System — Future Works Roadmap

> **Status:** Post v5.7.1 paper-trading phase
> **Scope:** v5.8 incremental upgrades → v6.0 architectural evolution
> **Last Updated:** 2026-03-10

---

## Overview

This document captures the planned evolution of the Varanus system beyond the current v5.7.1 Dynamic Trailing Stop implementation. Each module is self-contained and can be developed, backtested, and blind-tested independently before merging into the main engine.

---

## Module 1 — Re-Entry Logic

**Target Version:** v5.8
**Priority:** 🔴 High

### Objective
Re-open a position in the same direction if the underlying trend continues after a Trailing Stop has fired. A trailing stop hit often signals a temporary pullback, not a trend reversal — this module captures the resumed leg.

### Mechanism
- **Trigger condition:** A `trailing_sl_hit` outcome was recorded on asset `A` within the last **12 hours**.
- **Re-entry gate:** The model must produce a new signal in the same direction at the next 4h candle close with confidence **≥ 0.72** (stricter than the base long threshold of ~0.65).
- **Cooldown:** Only one re-entry per original trade is permitted. A second trailing stop on the re-entry closes the sequence permanently for that cycle.
- **Barrier recalculation:** TP/SL are recalculated fresh from the re-entry price using the standard ATR-based TBM formula. The trailing stop activates immediately (no additional trigger buffer) since the trend is already confirmed.

### Justification
Blind Test 3 showed that 70% of exits in March 2026 were trailing stop hits, suggesting the model identifies genuine directional moves but the 1.147% trail is tight enough to get shaken out mid-trend. A controlled re-entry recovers missed upside without relaxing risk parameters on the initial entry.

### Implementation Notes
- Add `last_trail_hit_ts` and `last_trail_hit_direction` fields to `paper_state.json` per asset.
- In `scan()`, check these fields before applying the stricter confidence gate.
- Backtest validation required: compare base v5.7.1 equity curve vs. v5.8 with re-entry on the full WFV fold set.

---

## Module 2 — Equity Curve Trading (Defense Layer)

**Target Version:** v5.8
**Priority:** 🔴 High

### Objective
Automatically reduce risk exposure during drawdown periods by monitoring the system's own equity curve. If performance degrades below a rolling average, the system shifts to a defensive posture without fully halting.

### Mechanism
- Maintain a rolling log of equity snapshots at each 4h cycle close (persisted in `paper_state.json`).
- Compute the **20-period Simple Moving Average (SMA)** of the equity series.
- **Defense mode activates** when `current_equity < equity_SMA_20`.
- In defense mode:
  - Leverage cap reduced by **70%** (e.g., max leverage drops from 5x → 1.5x).
  - Confidence thresholds tightened by **+0.05** for both long and short entries.
  - Position size remains fixed at $100 (no change to notional).
- **Defense mode deactivates** when equity closes back above the 20-period SMA for **2 consecutive cycles**.

### Justification
The existing circuit breaker (−5% daily / −15% drawdown) is a hard stop. Equity curve trading provides a softer, graduated response that reduces exposure during losing streaks while keeping the system active and ready to recover. This is standard practice in systematic fund management.

### Implementation Notes
- Add `equity_history` list (capped at 100 entries) and `defense_mode` boolean to state.
- Modify `get_leverage()` call in `scan()` to apply the 70% cap when in defense mode.
- Modify confidence threshold lookup in `scan()` to apply the +0.05 offset.
- Add defense mode status to Telegram heartbeat alert.

---

## Module 3 — Dynamic Volatility Trailing

**Target Version:** v5.8 / v5.9
**Priority:** 🟡 Medium

### Objective
Replace the fixed-percentage trailing stop (1.147% trigger, 1.147% distance) with ATR-based distances that adapt to the current volatility regime of each asset.

### Mechanism
- **Trail trigger:** `1.5 × ATR_14` profit from entry (instead of fixed 1.147%).
- **Trail distance:** `1.0 × ATR_14` below the running peak (instead of fixed 1.147%).
- ATR is recomputed at each bar using the live 4h OHLCV data already available in `check_exits()`.
- The ATR multipliers (1.5× trigger, 1.0× distance) are added as Optuna tuning parameters in the next optimization run with search ranges `[0.8, 3.0]`.

### Justification
A fixed 1.147% trail is appropriate for mid-volatility assets like ADA or DOT but may be too tight for high-volatility assets (ARB, OP, SUI) during volatile regimes, causing premature exits. Conversely, it may be too loose for low-volatility periods on assets like TRX. ATR-based trailing adapts automatically — wider trails in choppy markets, tighter in calm trends.

### Expected Impact
- Fewer premature `trailing_sl_hit` exits on high-volatility assets.
- Better trend capture on sustained moves.
- Slight increase in max holding time; may interact with the time barrier.

### Implementation Notes
- Requires storing `atr_at_entry` in the trade dict (already available from `scan()`).
- Alternatively, recompute live ATR in `check_exits()` from the last 14 bars — this is more accurate but slightly more expensive.
- Run full WFV + 3 blind tests before merging to confirm improvement over fixed trail.

---

## Module 4 — Sentiment Analysis Integration

**Target Version:** v6.0
**Priority:** 🟡 Medium

### Objective
Enrich the XGBoost feature set with macro-sentiment signals to improve prediction quality during extreme market conditions (capitulation, euphoria) that pure price-action features miss.

### Feature Candidates

#### 4a. Fear & Greed Index
- **Source:** Alternative.me public API (daily, free).
- **Feature engineering:**
  - Raw index value (0–100) normalized to [−1, +1].
  - 7-day rolling mean (smoothed sentiment trend).
  - Binary regime flag: `extreme_fear` (index < 25), `extreme_greed` (index > 75).
- **Expected signal:** Extreme fear → long bias filter; extreme greed → reduce long confidence, elevate short sensitivity.

#### 4b. Funding Rates
- **Source:** Binance perpetual futures funding rate API (every 8h).
- **Feature engineering:**
  - Current funding rate per asset (positive = longs paying = bullish crowding).
  - 3-period rolling mean of funding rate.
  - Threshold flags: `funding_crowded_long` (rate > 0.05%), `funding_crowded_short` (rate < −0.02%).
- **Expected signal:** Highly positive funding on an asset with a long signal → reduce confidence (crowded trade risk). Negative funding + long signal → confirm entry.

### Integration Plan
1. Build a lightweight `sentiment.py` data fetcher with local caching (daily refresh).
2. Add sentiment features to `build_features()` in `pa_features.py` behind a feature flag.
3. Retrain with Optuna to re-tune XGBoost depth/estimators for the expanded feature space.
4. A/B backtest: v5.x baseline vs. v6.0 with sentiment features on the same WFV folds.

### Justification
The current 19-feature set is entirely price-action derived. During macro-driven sell-offs (e.g., regulatory shock, BTC crash cascade), sentiment divergence from price can provide early warning signals that ATR/RSI/MSS patterns miss.

---

## Module 5 — Feature Noise Reduction

**Target Version:** v5.9 (pre-v6.0 cleanup)
**Priority:** 🟢 Low

### Objective
Remove low-importance features from the XGBoost model to reduce overfitting, improve inference speed, and make the feature space cleaner before the v6.0 sentiment expansion.

### Approach

#### Step 1 — Importance Audit
- Extract `model.long_model.feature_importances_` and `model.short_model.feature_importances_` after a full training run.
- Rank all features by gain-based importance.
- Flag features with importance score below **1.5% of the top feature's score** as candidates for removal.

#### Step 2 — Ablation Testing
For each candidate feature:
- Re-train with that feature excluded.
- Compare Sharpe ratio and win rate on the 8-fold WFV set.
- Remove if excluding it causes ≤ 0.5% Sharpe degradation.

#### Step 3 — Correlation Pruning
- Compute feature correlation matrix on the training set.
- For any pair with Pearson |r| > 0.90, remove the lower-importance one.

### Known Candidates (Preliminary)
Based on domain knowledge, features most likely to be redundant:
- `fvg_valid` (binary, low variance after sweep confirmation)
- `mss_signal` may partially overlap with `htf_bias`
- High-lag EMA derivatives if ATR already captures the same volatility regime

### Justification
The current feature set was grown incrementally across v5.1 → v5.7.1. Some features were added speculatively and never formally audited for marginal contribution. A leaner model will generalize better to unseen regimes (v6.0 and beyond) and is less likely to degrade in blind tests.

### Implementation Notes
- Do not remove features until ablation testing is complete.
- Keep a `features_removed.md` log documenting what was removed, why, and the before/after metrics.
- This module is a prerequisite for the v6.0 sentiment integration to avoid inflating the feature space unnecessarily.

---

## Implementation Priority Summary

| Module | Feature | Target | Priority | Dependency |
|--------|---------|--------|----------|------------|
| 1 | Re-Entry Logic | v5.8 | 🔴 High | None |
| 2 | Equity Curve Defense Layer | v5.8 | 🔴 High | None |
| 3 | Dynamic ATR Trailing | v5.8–5.9 | 🟡 Medium | Module 1 complete |
| 5 | Feature Noise Reduction | v5.9 | 🟢 Low | Stable v5.8 baseline |
| 4 | Sentiment Integration | v6.0 | 🟡 Medium | Module 5 complete |

---

## Release Gate Requirements

Before any module is merged into the live paper trader, it must pass:

1. **Full WFV backtest** — all 8 folds, Sharpe ≥ current v5.7.1 baseline.
2. **Blind Test** — minimum 30-day out-of-sample window not used in optimization.
3. **Long WR ≥ 41%** and **Short WR ≥ 50%** maintained.
4. **Max Drawdown ≤ −20%** in backtest.
5. **Telegram dry-run** — one full paper cycle executed in dry-run mode before going live.

---

*Varanus v5.7.1 — The Golden Ratio | Roadmap authored 2026-03-10*
