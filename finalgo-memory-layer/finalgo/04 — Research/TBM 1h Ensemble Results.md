---
title: "TBM 1-Hour Ensemble — Full Research Results"
type: report
status: dead
verdict: KILLED
updated: 2026-06-12
tags: []
---
# TBM 1-Hour Ensemble — Full Research Results

**Date:** 2026-06-09  
**Owner:** signal-generation (1h layer)  
**Status:** ⚠️ BOTH SIDES KILLED. SHORT = 44.9% net (was mis-reported 56.5% due to a cost-sign bug) | LONG = 43.3%, 0/5 folds

Implementation plan: [[04 — Research/TBM 1h Ensemble Implementation Plan]]

> [!ERROR] CRITICAL CORRECTION (2026-06-09) — cost-sign bug retracts the short "breakthrough"
> The harness computed the short net return as `-(long_gross − COST) = short_gross + COST` — **adding** the 6bps cost instead of subtracting it (long side unaffected). Every 6bps short headline below was inflated. **Corrected short:** raw WR 50.71% (coin flip), net WR @6bps **44.9%**, net expectancy **−6.42 bps**, t = **−4.77**, **0/5 folds** profitable — i.e. reliably net-NEGATIVE. The previously-claimed "+12.31pp selection skill / 5/5 folds / t=4.48" was entirely the bug (true skill +0.63pp). Found via the raw-vs-net check (`scripts/validation/check_short_raw_wr.py`); fixed in `purged_wf_tbm.py`; short WF re-run. **Net result: the TBM 1h ensemble has no post-cost edge in either direction.** Original (buggy) short numbers are struck through below for the record.

---

## Architecture Summary

- **Label engine:** `scripts/labeling/tbm_label_engine.py` → `data/tbm_labels_1h.parquet` (647,298 rows, 172 tickers, 3 years)
- **Labels:** {0=SL, 1=TP, 2=Timeout} with symmetric ATR barriers (m=1.0), stop-first ambiguity resolution
- **Label distribution:** SL 6.3% / TP 7.2% / Timeout 86.4%  ← strongly timeout-dominated
- **Feature views:** `scripts/features/build_feature_views.py` + `build_momentum_view.py`
- **Walk-forward harness:** `scripts/validation/purged_wf_tbm.py` — purged+embargoed splits, CatBoost MultiClass base learners, isotonic calibration, OOF stacked combiner (logistic), EV filter + τ-sweep, Top-K=3

**Walk-Forward Parameters:**
- Min train: 18 months | Val: 4 months | Test: 2 months | Step: 4 months | Embargo: 1 day
- Cost: 6 bps primary (10 bps secondary)
- Target: net WR ≥ 57% @ 6 bps

---

## Feature Views

| View | Features | Description |
|---|---|---|
| A — Mean-Reversion | 21 | IBS, Buy_Pressure, shadow ratios, VWAP_Dist, Stoch, RSI, PercentB, CMF, OBV, Elder, Price_Zscore, Direction_Consistency |
| B — Trend | 31 | Returns, ROC, MOM, PPO, TRIX, DPO, MA distances, Donchian, Vortex, streaks, alpha, RS |
| C — Volatility | 29 | HL_Range, BB/Donchian/Keltner widths, rolling skew/kurt, volume stats, RVOL, CCI, 52W distance, RSI lags |
| D — Momentum (long only) | 26 | ADX_14, +DI/-DI, Breakout_3H/5H/10H, High_Rank_10H/20H, Vol_Trend, Consec_Up, MOM_Accel, RS_1H, RS_Cumul, Gap_Pct, Gap_Up/Down, ORB_High/Low/Width/Breakout, Dist_ORB |

Time features (`Hour`, `DayOfWeek`, `Is_Open_Hour`, `Is_Close_Hour`, `Time_To_Close`) are **dropped** — proven overfit clock in v18/v19.

---

## Short Model Results — 3 Views (A + B + C)

**CORRECTED (cost-bug fixed, re-run 2026-06-09):**

| Metric | ~~Buggy~~ | **TRUE** | Criterion | Pass |
|---|---|---|---|---|
| Raw WR (pre-cost) | — | 50.71% | — | — |
| Net WR @6bps | ~~56.5%~~ | **44.9%** | ≥57% | ❌ |
| Net WR @10bps | 41.2% | 41.4% | — | — |
| Trades/month | ~~309~~ | ~236/mo | ≥1/mo | ✅ |
| Folds profitable | ~~5/5~~ | **0/5** | — | ❌ |
| Net expectancy | ~~+5.18 bps~~ | **−6.42 bps** | — | — |
| Profit factor | ~~1.25~~ | 0.758 | — | — |
| t-stat | ~~+4.48~~ | **−4.77** | — | — |
| Selection skill | ~~+12.31pp~~ | **+0.63pp** | — | — |
| Bootstrap CI @6bps | ~~[+2.8,+7.7]~~ | **[−9.1, −3.9] bps** | CI_lo>0 | ❌ |

**Status: KILLED — reliably net-NEGATIVE.** The short is not an edge; it loses ~6.4 bps/trade after correct costs.

**Fold-by-fold (short) — verified against `wf_results_short.json`:**

**Corrected fold-by-fold (cost-bug fixed re-run) — net WR @6bps:**

| Fold | Test Months | n | Raw WR | Net WR@6 | Exp (bps) | t |
|---|---|---|---|---|---|---|
| 1 | 2024-12 / 2025-01 | 308 | 53.6% | 47.4% | −5.6 | −0.99 |
| 2 | 2025-04 / 2025-05 | 479 | 49.3% | 44.5% | −5.6 | −2.69 |
| 3 | 2025-08 / 2025-09 | 547 | 50.7% | 42.2% | −3.5 | −2.86 |
| 4 | 2025-12 / 2026-01 | 523 | 48.3% | 42.3% | −8.4 | −3.84 |
| 5 | 2026-04 / 2026-05 | 501 | 51.5% | 49.5% | −8.8 | −2.14 |

Raw WR hovers at ~50% (coin flip); net WR @6bps is 42–49% (all net-negative); 4 of 5 folds have significantly negative t-stats. The τ-sweep cannot find any threshold achieving 57% net WR on validation in any fold (falls back to p95).

**Status: KILLED — net-negative in every fold.**

**Why there is no short edge:** the previously-claimed "IBS/Buy_Pressure overbought→reverts-down" story was inferred from the buggy 56.5% WR. With correct costs the short's raw mean return is −0.82 bps (negative even *before* cost) and selection skill is +0.63 pp. The base-learner AUCs (~0.60–0.65) reflect the model separating the {SL,TP,TO} *classes*, but that class-separation does **not** translate into a tradeable directional edge once the realistic 6bps round-trip cost is applied correctly.

---

## Long Model Results — 4 Views (A + B + C + D)

**Run 1 — 2026-06-09 — 3 views (A/B/C)**

Net WR: 44.2%, net expectancy −4.36 bps. Kill criterion triggered.

**Run 2 — 2026-06-09 — 4 views (A/B/C + D_momentum)**

| Metric | Value | Criterion | Pass |
|---|---|---|---|
| Net WR @6bps | **43.3%** | ≥57% | ❌ |
| CI_lo @6bps | **−8.9 bps** | >0 | ❌ |
| Trades/month | **~224/mo** | ≥1/mo | ✅ |
| Folds profitable | **0/5** | — | ❌ |
| Net expectancy | −5.22 bps | — | — |
| Profit factor | 0.847 | — | — |
| t-stat | −2.98 | — | — |

**Fold-by-fold (long, 4-view run):**

| Fold | Test Months | n | WR@6 | Exp (bps) | t | View D test AUC |
|---|---|---|---|---|---|---|
| 1 | 2024-12 / 2025-01 | 520 | 42.9% | −5.3 | −1.50 | 0.487 |
| 2 | 2025-04 / 2025-05 | 240 | 42.9% | −1.3 | −0.18 | 0.478 |
| 3 | 2025-08 / 2025-09 | 329 | 41.0% | −9.0 | −1.93 | 0.517 |
| 4 | 2025-12 / 2026-01 | 602 | 44.4% | −5.5 | −1.88 | 0.495 |
| 5 | 2026-04 / 2026-05 | 547 | 44.1% | −4.2 | −1.24 | 0.491 |

**Status: KILLED — 0/5 folds profitable. Do not revive without new data sources.**

---

## Critical Diagnostic: Drift-vs-Skill Decomposition (2026-06-09)

A rigorous decomposition (`scripts/validation/diagnose_drift_vs_skill.py`) was run to determine **whether the long failure is a removable drift headwind or a genuine absence of signal.** The answer is conclusive.

### 1. Drift is NOT the cause (hypothesis rejected)
- Mean 1h forward (gross) return across all 647k bars: **−0.14 bps** (essentially zero).
- Per-fold drift **flips sign**: folds 2 & 5 are **+2.89 bps** (up-drift), folds 1/3/4 are −0.4 to −2.2 bps.
- There is no persistent directional drift in either direction.

### 2. Selection skill (CORRECTED for cost bug): BOTH sides ~zero
Comparing model-selected WR to the trade-everything (unconditional) WR isolates pure stock-selection skill. The original short row used the **buggy** selected WR (56.54%); corrected below:

| Side | Unconditional net@6 | Model-selected net@6 | **Selection skill** |
|---|---|---|---|
| **SHORT** | 44.23% | ~~56.54%~~ → **44.85%** | ~~+12.31pp~~ → **+0.63 pp** (noise) |
| **LONG**  | 42.07% | 43.30% | **+1.22 pp** (noise) |

Neither side has meaningful cross-sectional skill. Both are coin-flip directional bets that lose to the 6bps cost. (Note: after correction the long skill +1.22pp is nominally *larger* than the short's +0.63pp — both are noise.)

### 3. Conviction ranking is ~uninformative on both sides
From `check_long_skill_ceiling.py` and `check_short_raw_wr.py`:
- Long WR by EV percentile is flat/non-monotonic; **k=1 long (42.5%) is *worse* than k=3 (43.3%)** — ranking carries no info.
- Short k=1-by-EV net@6 = 46.31% vs k=3 44.85% — lifts ~1.5pp but both are **net-negative**. (The "57.3%" figure in the earlier draft was the buggy net.) No tradeable subset exists on either side.

> **Two corrections to earlier drafts:** (a) The short is **not** a genuine alpha — that was the cost-sign bug. (b) The long is **NULL, not inverted** — its picks sit at the 42% baseline, so inverting it yields no usable short signal either.

---

## Why Longs Are Unpredictable: Daily-Trend Test (2026-06-09)

The strongest remaining long thesis — **"buy oversold dips in daily uptrends"** — was tested directly (`scripts/research/test_daily_trend_long.py`). Daily-trend context (daily SMA distance, daily ROC, trend regime, daily RSI) was built from the 15m cache, lagged to the prior complete day (no lookahead), and joined to 632k long-label bars.

**Result: daily trend has zero effect on long WR.**

| Long subpopulation | WR | vs 42.1% baseline |
|---|---|---|
| Above daily SMA20 | 41.76% | −0.34 pp |
| Below daily SMA20 | 42.52% | +0.42 pp |
| Daily ROC10 > 0 | 41.78% | −0.32 pp |
| Oversold (low IBS) + above SMA20 | 42.93% | +0.83 pp |
| Oversold + uptrend + ROC>2% | 43.02% | +0.92 pp |
| Oversold + *below* SMA20 | 43.22% | +1.12 pp |

Every cell is within ~1 pp of baseline, and the thesis is **inverted** (above-SMA20 is *worse* than below). Per-fold for the best cell: 45.3 / 44.4 / 39.7 / 42.7 / 42.1% — never above 50%.

### The mechanism (corrected)
> Originally this section argued downside was "technically structured" (short alpha) vs upside "catalyst-driven" (no long alpha). **That asymmetry was an artifact of the cost-sign bug** — with correct costs the short has no edge either (raw mean −0.82 bps, skill +0.63 pp). The honest mechanism is simpler: **at 1h scale, neither direction is predictable from lagging price/volume features after a realistic 6 bps round-trip cost.** Both raw win-rates sit at ~50% and both bleed the cost.

---

## Barrier-Geometry Test (2026-06-09) — user hypothesis: "smaller target / longer horizon for longs"

**Hypothesis (correct on mechanics):** down-moves are violent (hit −1×ATR within 1h → short TP resolves) but up-moves grind slowly (need 3–4 candles for +1%). So the 1h/1ATR label discards genuine slow climbs as TIMEOUTS (86.4%), training the long model to predict only rare news-spikes. Reducing the long target and/or extending the horizon should let the predictable grind become the label.

`scripts/research/test_long_geometry.py` relabelled longs across 6 (barrier m, horizon H) configs, strictly intraday (no overnight):

| Config | Timeout% | Uncond WR | uptrend vs downtrend | Separation spread |
|---|---|---|---|---|
| m1.0 / 1h (baseline) | 86.4% | 42.1% | 41.8 vs 42.5 | 0.7 pp |
| m0.5 / 1h | 45.3% | 43.3% | 43.0 vs 43.8 | 0.8 pp |
| m1.0 / 2h | 70.9% | 44.0% | 43.7 vs 44.6 | 0.9 pp |
| m0.5 / 2h | **23.3%** | **45.7%** | 45.4 vs 46.4 | 1.0 pp |
| m0.75 / 3h | 35.4% | 45.7% | 45.3 vs 46.3 | 1.0 pp |
| m1.0 / 3h | 56.4% | 45.0% | 44.6 vs 45.7 | 1.1 pp |

**Mechanics CONFIRMED:** smaller/longer barriers cut timeouts 86%→23% and lift unconditional long WR 42%→46%. The wide barrier WAS discarding real climbs.

**Predictability REJECTED:** the separation spread stays flat at ~1 pp at *every* geometry and stays *inverted* (downtrend ≥ uptrend). Reducing the target raises *all* stocks uniformly — it lifts the base rate, creates no selectable signal.

### Definitive confirmation — full retrain at steelman geometry
`scripts/research/retrain_long_m05_h2.py` — full 107-feature CatBoost, purged WF, at the most favorable geometry (m0.5 / 2h, only 23% timeouts):

```
unconditional long WR : 45.87%
selected long WR      : 47.98%   (still BELOW 50% breakeven)
SELECTION SKILL       : +2.11 pp  (per fold: +2.14, +0.31, +2.56, +5.46, -0.18)
```

+2.11 pp ≈ baseline's +1.22 pp, carried by one fold (4). To reach 57% from a 46% base needs +9 pp of skill; the model delivers +2. **Conclusive: the long base rate is tunable via geometry, but selection skill is not — longs are unpredictable at any barrier/horizon.**

### Six independent approaches tested — all confirm no long edge
| # | Approach | Result |
|---|---|---|
| 1 | Drift removal (market-neutral) | Drift ~0, flips sign — not the cause |
| 2 | Conviction tightening (EV / p_tp / k) | Flat 42–46%; k=1 *worse* than k=3 |
| 3 | Momentum / breakout (View D) | AUC ~0.49, anti-predictive |
| 4 | Daily-trend conditioning | All cells ~42%, thesis inverted |
| 5 | Barrier-geometry scan (6 configs) | Separation spread ~1 pp everywhere |
| 6 | Full retrain at steelman geometry (m0.5/2h) | +2.11 pp skill, 48% WR |

---

## Verdict & Next Steps

**The TBM 1h ensemble has no post-cost edge in either direction.** Both raw win-rates are ~50% and both lose the 6 bps round-trip cost (short −6.42 bps, long −5.22 bps). Selection skill is ~zero on both sides (short +0.63 pp, long +1.22 pp). This is not under-tuning — at 1h scale, direction is not predictable from lagging price/volume features in this universe. **Do not deploy either side.**

### Do NOT keep tuning this
TBM-on-price/volume is exhausted (long tested 6 ways incl. barrier-geometry; short re-run with correct costs is net-negative at t=−4.77). Any 1h directional edge would require **new data**: order-flow / Level-2 depth, options flow, news/sentiment, event/earnings calendar — none available 2026-06-09.

### Process lesson (most important takeaway)
A cost-sign bug inflated the short from a true 44.9% to a reported 56.5% and survived initial review because (a) only the short flip was affected, (b) the @10bps metric was computed correctly and looked plausible, (c) the inflated t=4.48 / 5-of-5 looked like a breakthrough. **Always verify `median(net − gross) == −cost` per side, and check RAW vs NET WR, before trusting any headline.** The bug was caught only when the user asked "what is the raw win-rate?"

### Re-audit other models
Apply the same raw-vs-net / cost-sign / overnight checks to v8 / v10 / 15m before trusting their reported edges.

---

## Files

| File | Description |
|---|---|
| `data/tbm_labels_1h.parquet` | 647,298 labeled bars (m=1.0, 3×cost floor) |
| `data/tbm_feature_views/A_meanrev.parquet` | View A — 21 features, 646,258 rows |
| `data/tbm_feature_views/B_trend.parquet` | View B — 31 features |
| `data/tbm_feature_views/C_vol.parquet` | View C — 29 features |
| `data/tbm_feature_views/D_momentum.parquet` | View D — 26 features (93.8 MB) |
| `data/model_analysis/tbm_1h/wf_results_short.json` | Short model fold results |
| `data/model_analysis/tbm_1h/wf_results_long.json` | Long model fold results |
| `data/model_analysis/tbm_1h/test_trades_short.parquet` | Short model OOS trades |
| `data/model_analysis/tbm_1h/test_trades_long.parquet` | Long model OOS trades |
| `scripts/labeling/tbm_label_engine.py` | TBM label generator |
| `scripts/features/build_feature_views.py` | Views A/B/C builder |
| `scripts/features/build_momentum_view.py` | View D builder |
| `scripts/validation/purged_wf_tbm.py` | Full WF harness (supports 3 or 4 views per side) |
| `scripts/validation/check_short_raw_wr.py` | **Raw-vs-net WR check that EXPOSED the cost-sign bug** (true short 50.7% raw / 44.85% net@6 / +0.63pp) |
| `scripts/validation/diagnose_drift_vs_skill.py` | Drift-vs-skill decomposition (drift~0; both skills ~zero after correction) |
| `scripts/validation/check_long_skill_ceiling.py` | Long conviction-ranking analysis (proves ranking uninformative) |
| `scripts/validation/analyze_short_selectivity.py` | Short k/EV/p_tp selectivity (NOTE: its WR numbers use the buggy net field) |
| `scripts/research/test_daily_trend_long.py` | Daily-trend long stratification (disproves buy-dip thesis) |
| `scripts/research/test_long_geometry.py` | Barrier/horizon scan (6 configs) — timeouts drop but spread stays flat |
| `scripts/research/retrain_long_m05_h2.py` | Definitive long retrain at m0.5/2h — +2.11pp skill, 48% WR |
| `data/model_analysis/tbm_1h/long_geometry_scan.txt` | Saved geometry scan output |
