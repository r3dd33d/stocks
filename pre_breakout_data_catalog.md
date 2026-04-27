# Pre-Breakout ML Dataset — Data Catalog

> Reference document for `kairognos_combined_2015_2026_pre_breakout.parquet`
> (also loaded as Postgres table `combined_signals_pre_breakout`).
> Built with the `data-catalog-entry` and `semantic-model-builder` skill patterns.

---

## 1. Overview

| Item | Value |
|------|-------|
| Source file | `~/kairognos_combined_2015_2026_pre_breakout.parquet` |
| DB table | `combined_signals_pre_breakout` (host `127.0.0.1:5433`, db `kairognos`) |
| Rows | 4,730,559 |
| Columns | 115 |
| Unique tickers | 3,519 |
| Date range | 2015-01-02 → 2025-12-31 |
| Methodology | `pre_breakout` only |
| Target / label | `gain_10pct_30d` — 1 if close gained ≥10% within next 30 calendar days, 0 if not, NaN if no forward window |
| Class balance | ~13.2% positive |

---

## 2. Column families — quick map

| # | Family | Purpose | Cols |
|---|--------|---------|-----|
| 1 | Identifiers | Row keys | 3 |
| 2 | OHLCV | Raw price + volume | 5 |
| 3 | Moving averages | Trend baselines | 4 |
| 4 | Momentum | Speed of price change | 11 |
| 5 | Trend strength | ADX family + slope | 6 |
| 6 | Volatility | ATR / ADR / Bollinger | 8 |
| 7 | Volume analytics | Volume context & dryup | 6 |
| 8 | Patterns — VCP / Consolidation | Setup detection | 9 |
| 9 | Breakout / Distribution | Breakout detection + selling pressure | 10 |
| 10 | Relative strength | vs market & sector | 8 |
| 11 | Price-action quality | Bar-level quality scoring | 7 |
| 12 | Methodology scores | Engine's own prediction outputs | 9 |
| 13 | Risk / warnings | Safety filters | 9 |
| 14 | Market regime | Overall market state | 4 |
| 15 | Engine internals | History, biotech flag | 4 |
| 16 | Label | Target | 1 |

---

## 3. Detailed catalog by family

> Conventions:
> - **Type** is the in-parquet type. `str` for categorical, `float64`/`int64` for numerics, `bool` for booleans.
> - **% null** is computed across all 4.73M rows.
> - **Counterpart** = another column carrying overlapping or implied information.
> - **ML note** is a recommended action: keep / drop / decide.

### 3.1 Identifiers

| Column | Type | Meaning | % null | ML note |
|--------|------|---------|--------|---------|
| `symbol` | str | Ticker symbol (e.g., AAPL) | 0% | keep as categorical or join key |
| `exchange` | str | Listing exchange (NASDAQ, NYSE, NYSE ARCA, AMEX, NYSE MKT) | 0% | low-cardinality categorical |
| `timestamp` | datetime[us, UTC] | Bar date (daily granularity) | 0% | use for time-based split, not as a feature |

### 3.2 OHLCV — raw bar data

| Column | Type | Meaning | % null |
|--------|------|---------|--------|
| `open` | float | Open price of the bar | 0% |
| `high` | float | High of the bar | 0% |
| `low` | float | Low of the bar | 0% |
| `close` | float | Close of the bar | 0% |
| `volume` | float | Shares traded | 0% |

### 3.3 Moving averages

| Column | Type | Meaning | % null | Counterpart |
|--------|------|---------|--------|-------------|
| `ma_10` | float | 10-day simple moving average | ~14% (history insufficient) | redundant with `ma_20` |
| `ma_20` | float | 20-day SMA | ~14% | base short-term MA |
| `ma_50` | float | 50-day SMA | ~14% | medium trend reference |
| `ma_200` | float | 200-day SMA | ~14% | long-term trend |
| `price_vs_ma_200_pct` | float | (close − ma_200) / ma_200 × 100 | ~14% | derived; useful |

### 3.4 Momentum

| Column | Type | Meaning | % null | Counterpart / note |
|--------|------|---------|--------|--------------------|
| `rsi_14` | float | 14-period Relative Strength Index | ~14% | classic overbought/oversold |
| `rsi_pullback` | str ('t'/'f') | True if RSI is in a pullback condition | ~14% | derived from rsi_14 |
| `rsi_pullback_label` | str | Bucketed label: oversold/neutral/healthy/normal/overbought | <1% | **bucketed version of rsi_14** |
| `macd` | float | MACD line | ~14% | base value |
| `macd_signal` | float | MACD signal line | ~14% | base value |
| `macd_histogram` | float | macd − macd_signal | ~14% | **= macd − macd_signal (linear combo)** |
| `macd_state` | str | bullish / bearish / neutral | <2% | **bucketed version of macd** |
| `stoch_k` | float | Stochastic %K | ~14% | momentum oscillator |
| `stoch_d` | float | Stochastic %D | ~14% | smoothed %K |
| `stoch_state` | str | overbought / oversold / neutral | <2% | **bucketed version of stoch_k** |
| `cci_20` | float | Commodity Channel Index (20) | ~14% | momentum/extreme detector |
| `cci_state` | str | overbought / oversold / neutral | <2% | **bucketed version of cci_20** |

### 3.5 Trend strength

| Column | Type | Meaning | % null | Counterpart |
|--------|------|---------|--------|-------------|
| `adx` | float | Average Directional Index — trend strength | ~14% | base value |
| `plus_di` | float | +DI: directional indicator (up) | ~14% | **adx is derived from these** |
| `minus_di` | float | −DI: directional indicator (down) | ~14% | **adx is derived from these** |
| `adx_state` | str | weak_ranging / strong_trend / very_strong | <2% | **bucketed version of adx** |
| `ma_slope` | float | Slope of the long-term MA | ~14% | numeric slope |
| `slope_category` | str | zero / weak / moderate / strong | ~14% | **bucketed version of ma_slope** |

### 3.6 Volatility

| Column | Type | Meaning | % null | Note |
|--------|------|---------|--------|------|
| `atr_5` | float | 5-day Average True Range | ~14% | **highly correlated with atr_10** |
| `atr_10` | float | 10-day ATR | ~14% | use this one |
| `adr` | float | Average Daily Range (%) | ~14% | volatility in % terms |
| `adr_past` | float | Prior-window ADR | ~27% | for ADR change detection |
| `adr_declining` | str ('t'/'f') | True if ADR is declining | ~14% | derived from adr/adr_past |
| `bb_upper` | float | Bollinger band upper | ~14% | covered by bb_width_pct + close |
| `bb_lower` | float | Bollinger band lower | ~14% | covered by bb_width_pct + close |
| `bb_width_pct` | float | Bollinger band width as % of price | ~14% | the useful one |

### 3.7 Volume analytics

| Column | Type | Meaning | % null | Counterpart |
|--------|------|---------|--------|-------------|
| `avg_volume_30d` | float | 30-day average volume | ~14% | base value |
| `volume_ratio` | float | Today's volume ÷ avg_volume_30d | ~14% | base value |
| `volume_surge_label` | str | normal / high / very_high | ~14% | **bucketed version of volume_ratio** |
| `obv` | float | On-Balance Volume | ~14% | base value |
| `obv_ma_20` | float | 20-day MA of OBV | ~14% | smoothed OBV |
| `obv_state` | str | rising / falling | ~14% | **bucketed version of obv vs obv_ma_20** |
| `volume_days_above_avg_10` | int | # of last 10 days with volume above avg | 0% | discrete count |
| `volume_days_below_avg_10` | int | # of last 10 days with volume below avg | 0% | discrete count |

### 3.8 Patterns — VCP / Consolidation

> VCP = Volatility Contraction Pattern. A multi-touch base where price is forming progressively tighter swings.

| Column | Type | Meaning | % null | Note |
|--------|------|---------|--------|------|
| `vcp_t_count` | float | Number of touches (T1, T2, ...) detected | ~14% | discrete |
| `vcp_t1_depth` | float | First leg's pullback depth (%) | ~26% | only set when in VCP |
| `vcp_t2_depth` | float | Second leg's pullback depth (%) | ~26% | only set when in VCP |
| `vcp_score` | float | Composite VCP quality score | **~99.9% zero/null** | near-empty in pre_breakout slice |
| `vcp_state` | str | building / formed / failed / etc. | ~14% | engine's state machine |
| `vcp_is_valid` | object (bool/None) | True iff vcp_score above threshold | ~26% | implied by vcp_score |
| `vcs_score` | float | Volume Contraction Score | ~14% | numeric |
| `vcs_label` | str | none / weak / moderate / strong | ~14% | **bucketed version of vcs_score** |
| `consolidation_days` | float | # bars in current consolidation (capped) | ~14% | base value |
| `consolidation_range_pct` | float | Range of consolidation as % | ~14% | base value |
| `is_consolidating` | str ('t'/'f') | True if currently consolidating | ~14% | implied by consolidation_days > 0 |

### 3.9 Breakout / Distribution

| Column | Type | Meaning | % null | Counterpart |
|--------|------|---------|--------|-------------|
| `breakout_detected` | str ('t'/'f') | A breakout was detected | ~14% | base flag |
| `breakout_days_since` | float | Days since the most recent breakout | ~14% | base value |
| `breakout_is_strong` | str ('t'/'f') | Breakout classified as strong | ~14% | **3-bool family** |
| `breakout_is_weak` | str ('t'/'f') | Breakout classified as weak | ~14% | **3-bool family** |
| `breakout_is_suspicious` | str ('t'/'f') | Breakout classified as suspicious | ~14% | **3-bool family** |
| `distribution_detected` | str ('t'/'f') | Distribution day detected on this bar | ~14% | implied by `distribution_days > 0` |
| `distribution_days` | float | Count of distribution days in lookback | ~26% | base value |
| `higher_highs` | float | Count of recent higher highs | ~14% | discrete |
| `higher_lows` | float | Count of recent higher lows | ~14% | discrete |
| `lower_highs` | float | Count of recent lower highs | ~14% | discrete |

### 3.10 Relative strength

| Column | Type | Meaning | % null | Note |
|--------|------|---------|--------|------|
| `relative_strength` | float | Stock RS vs SPY benchmark | ~15% | numeric |
| `rs_state` | str | underperforming / inline / outperforming | ~15% | **bucketed version of relative_strength** |
| `sector_rs` | float | Sector ETF RS vs SPY | ~26% | base value |
| `sector_leading` | object (bool/None) | True if sector outperforming market | ~26% | derived from sector_rs |
| `stock_leading_sector` | object (bool/None) | True if stock outperforming its sector | ~26% | derived |
| `stock_vs_sector_rs` | float | Stock RS minus sector RS | ~26% | derived numeric |
| `ath_distance_pct` | float | % distance below all-time high | ~14% | base value |
| `days_since_ath` | int | Bars since the all-time high | 0% | discrete |

### 3.11 Price-action quality

| Column | Type | Meaning | % null | Note |
|--------|------|---------|--------|------|
| `pa_quality_score` | float | Composite price-action score (0–100) | ~14% | base value |
| `pa_quality_label` | str | Strong / Moderate / Weak | ~14% | **bucketed version of pa_quality_score** |
| `pa_body_score` | int | Sub-score: candle body | 0% | discrete |
| `pa_momentum_score` | int | Sub-score: momentum | 0% | discrete |
| `pa_close_score` | int | Sub-score: close position in bar | 0% | discrete |
| `pa_volume_score` | int | Sub-score: volume confirmation | 0% | discrete |
| `weekly_trend_status` | str | Conflicting / Building / Moderate / Neutral / Strong | ~14% | weekly summary |
| `weekly_trend_partial_week` | str ('t'/'f') | Weekly bar still forming | ~14% | metadata |
| `ma_alignment` | str ('t'/'f') | True iff above_fast_ma AND above_long_ma | ~14% | **= above_fast_ma ∧ above_long_ma** |
| `above_fast_ma` | str ('t'/'f') | Close above fast MA | ~46% | base flag |
| `above_long_ma` | str ('t'/'f') | Close above long MA | ~46% | base flag |
| `strong_uptrend` | str ('t'/'f') | All trend criteria met | ~14% | **near-empty: 99.9% false/null** |

### 3.12 Methodology scores — **engine outputs (leakage hazard)**

| Column | Type | Meaning | % null | ML note |
|--------|------|---------|--------|---------|
| `total_score` | float | Engine's raw score (0–100ish) | <1% | **leakage** for new model |
| `final_score` | float | Score after caps/penalties/bonuses | <1% | **leakage** |
| `liquidity_score` | float | Sub-score: liquidity | <1% | **leakage**; also 84% zero |
| `trend_score` | float | Sub-score: trend | <1% | **leakage** |
| `performance_score` | float | Sub-score: performance | <1% | **leakage** |
| `setup_score` | float | Sub-score: setup quality | <1% | **leakage** |
| `risk_penalty` | float | Penalty applied | <1% | **leakage** |
| `signal_type` | str | Signal classification (constant in this slice) | 0% | **all 100% same value — drop** |
| `signal_strength` | str | none / weak / moderate / strong | 0% | engine's classification |

### 3.13 Risk / warnings

| Column | Type | Meaning | % null | Note |
|--------|------|---------|--------|------|
| `support_level` | float | Nearest support price | ~15% | base value |
| `expected_move` | float | Engine's projected move ($) | ~15% | **leakage** |
| `expected_move_pct` | float | Engine's projected move (%) | ~15% | **leakage** |
| `risk_level` | str | Normal / Caution / High Risk | <1% | engine output |
| `warning_count` | int | Sum of the warning_* booleans | 0% | **= sum of 7 warnings** |
| `warning_extended_move` | bool | Stock has run too far | 0% | **99.8% False — near-empty** |
| `warning_gap_down` | bool | Recent gap-down | 0% | **99.1% False** |
| `warning_failed_breakout` | bool | Recent failed breakout | 0% | 98.6% False |
| `warning_broke_fast_ma` | bool | Broke below fast MA | 0% | 95.0% False |
| `warning_below_fast_ma` | bool | Currently below fast MA | 0% | 81.1% False |
| `warning_below_long_ma` | bool | Currently below long MA | 0% | 76.9% False |
| `warning_distribution` | bool | Distribution warning active | 0% | 93.0% False |

### 3.14 Market regime

| Column | Type | Meaning | % null | Note |
|--------|------|---------|--------|------|
| `market_regime` | str | Bullish / Neutral / Bearish | ~14% | overall market context |
| `spy_distance_from_200` | float | SPY's distance (%) from its 200 SMA | ~14% | numeric source for regime |

### 3.15 Engine internals

| Column | Type | Meaning | % null | Note |
|--------|------|---------|--------|------|
| `history_bars` | int | Count of historical bars before this row | 0% | redundant with history_sufficient |
| `history_sufficient` | bool | True if enough history for full computation | 0% | base flag |
| `is_biotech` | bool | True if biotech sector | 0% | **99.9% False — near-empty** |

### 3.16 Label

| Column | Type | Meaning |
|--------|------|---------|
| `gain_10pct_30d` | float32 (1.0 / 0.0 / NaN) | 1 = close ≥ 1.10× current close within 30 calendar days; 0 = not; NaN = forward window incomplete |

---

## 4. Numeric ↔ bucketed-string counterparts

These are pairs where the engine bucketed a numeric into a label. **For ML, prefer the numeric.** The string version carries strictly less information and creates correlated features.

| Numeric (keep) | Bucketed string (drop for ML) | Buckets |
|----------------|-------------------------------|---------|
| `rsi_14` | `rsi_pullback_label` | oversold / normal / neutral / healthy / overbought |
| `macd` | `macd_state` | bullish / bearish / neutral |
| `stoch_k` | `stoch_state` | overbought / oversold / neutral |
| `cci_20` | `cci_state` | overbought / oversold / neutral |
| `adx` | `adx_state` | weak_ranging / strong_trend / very_strong |
| `obv` (vs `obv_ma_20`) | `obv_state` | rising / falling |
| `relative_strength` | `rs_state` | underperforming / inline / outperforming |
| `vcs_score` | `vcs_label` | none / weak / moderate / strong |
| `pa_quality_score` | `pa_quality_label` | Strong / Moderate / Weak |
| `volume_ratio` | `volume_surge_label` | normal / high / very_high |
| `ma_slope` | `slope_category` | zero / weak / moderate / strong |

---

## 5. Pure linear combinations / mathematical implications

| Derived column | Source(s) | Formula |
|----------------|-----------|---------|
| `macd_histogram` | `macd`, `macd_signal` | `macd − macd_signal` |
| `ma_alignment` | `above_fast_ma`, `above_long_ma` | `above_fast_ma ∧ above_long_ma` |
| `warning_count` | 7 individual `warning_*` booleans | `sum(warning_*)` |
| `distribution_detected` | `distribution_days` | `distribution_days > 0` |
| `vcp_is_valid` | `vcp_score` | implied threshold |
| `is_consolidating` | `consolidation_days` | `consolidation_days > 0` |

---

## 6. Cleanup recommendations for ML

### Tier A — drop unconditionally (no info or zero variance)

| Column | Reason |
|--------|--------|
| `signal_type` | 100% same value in this slice |
| `is_biotech` | 99.9% False |
| `warning_extended_move` | 99.8% False |
| `warning_gap_down` | 99.1% False |
| `warning_failed_breakout` | 98.6% False |
| `warning_broke_fast_ma` | 95.0% False |
| `vcp_score` | 99.9% zero or null |
| `strong_uptrend` | 99.9% False or null |
| `liquidity_score` | 84% zero (also engine output) |
| `history_bars` | redundant with `history_sufficient` |

### Tier B — drop redundant (string buckets + linear combos)

All 11 bucketed strings from §4 + all 6 derivations from §5.
**Total: ~17 columns.**

### Tier C — leakage hazards (drop if training a new model)

| Column | Risk |
|--------|------|
| `total_score`, `final_score` | Engine's overall prediction |
| `liquidity_score`, `trend_score`, `performance_score`, `setup_score`, `risk_penalty` | Sub-scores |
| `expected_move`, `expected_move_pct` | Engine's projection |
| `signal_strength`, `risk_level` | Engine's classification |

→ If goal is *new model*: drop. If goal is *evaluating the engine*: keep.

### Tier D — type fixes (cast, don't drop)

| Column | Current | Cast to |
|--------|---------|---------|
| `sector_leading` | object (bool/None) | nullable Boolean |
| `stock_leading_sector` | object (bool/None) | nullable Boolean |
| `vcp_is_valid` | object (bool/None) | nullable Boolean |
| All `*_state`, `*_label`, `'t'/'f'` strings | str | proper categorical / boolean |

---

## 7. Suggested final feature set sizes

| Cleanup level | Cols kept | Use case |
|---------------|-----------|----------|
| Raw (current) | 115 | nothing dropped |
| **Conservative** (Tier A only) | ~105 | safe minimum cleanup |
| **Recommended** (Tier A + B) | ~88 | for ML, keeps engine scores as features |
| **No-leakage** (Tier A + B + C) | ~76 | for *new* ML model that should not see engine outputs |

---

## 8. Train/validation guidance

- **Time-based split** mandatory (random split would leak future info):
  - Train: 2015 – 2022
  - Validation: 2023
  - Test: 2024 – 2025
- **Class weights** (~13% positive) — set `scale_pos_weight ≈ 6.6` for XGBoost or use `class_weight='balanced'`.
- **Stratify** by year and by `market_regime` if you do any sub-sampling.
- **Per-regime models** have historically outperformed a single model on this dataset (memory note).

---

## 9. Glossary

- **VCP (Volatility Contraction Pattern)** — base where price makes a sequence of progressively tighter pullbacks. Indicates supply drying up before a breakout.
- **VCS (Volume Contraction Score)** — analogous to VCP but for volume contraction.
- **ADR (Average Daily Range)** — average bar range as % of price.
- **ATR (Average True Range)** — same idea but in price units, including gap effects.
- **OBV (On-Balance Volume)** — cumulative volume add/subtract by close direction.
- **DI (+DI / −DI)** — directional indicators that feed ADX.
- **Distribution day** — high-volume down day in an uptrend (institutional selling).
- **All-time high (ATH) distance** — % the close is below the symbol's all-time high.
- **Methodology** — the engine has 3: `pre_breakout`, `momentum`, `vcp`. This file is the `pre_breakout` slice.

---

*Last updated: 2026-04-27. Generated from a profile scan of 4.73M rows.*
