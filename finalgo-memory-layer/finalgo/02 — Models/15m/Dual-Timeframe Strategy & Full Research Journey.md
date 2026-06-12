---
title: "Dual-Timeframe Strategy & Full Research Journey"
type: reference
status: superseded
model: "15m"
updated: 2026-06-12
tags: []
---
# Dual-Timeframe Strategy & Full Research Journey

**Date:** June 7, 2026
**Models:** `v2_15min_3y` (15-minute ranker) + `v8_upstox_3y` (1-hour ranker, current production active model)
**Scope:** End-to-end research arc — from evaluating the 15-min model, through sniper discovery, cost reality-check, time-of-day structure, the dual-timeframe backtest, the short-model audit, and the final short-as-long-filter discovery.
**Friction baseline:** 10 bps flat (STT + brokerage + slippage) unless noted.
**OOS windows:** 15m model = Apr–Jun 2026; 1h model = Mar–May 2026 (1h data ends May).

> This note is the "read me first" synthesis. Each section links to the deeper standalone note.

---

## ⚠️ CRITICAL CORRECTION (June 7, 2026 — added after rigorous re-audit)

**Most of the headline returns below (the sniper's +158%, the dual-TF's +98%–+187%) are ARTIFACTS and must NOT be trusted.** A later forensic audit (script `definitive_backtest.py`) found two data bugs that inflated every trading result in sections 3–7 and Appendix A:

1. **Overnight contamination.** The last intraday bar's "next return" jumps to the **next morning's open**, not the next intraday bar. The 1h `14:30` signal's `Next_Hour_Return` and the 15m `15:15` bar's `Next_15Min_Return` are **overnight returns**, not intraday. They rode the April 2026 rally with gap risk.
2. **Timestamp misalignment.** The 1h file is **left-labeled** (timestamp = bar start), the 15m file is **right-labeled** (bar end). A 1h signal labeled `T` aligns to the 15m bar labeled `T+60`, not `T`. Proven by exact raw-close matching; verified `NHR[T] == compound(15m N15 at T+60,T+75,T+90,T+105)` (median diff 0.00 bps).

**Clean, reconciled, intraday-only results (10 bps, bootstrap 95% CIs):**

| Config | N | Net WR | Net bps (95% CI) | t | Verdict |
|---|---|---|---|---|---|
| Baseline 1H long K=3 | 554 | 44.2% | −3.7 [−8, +1] | −1.6 | no edge |
| LongConfirm rkL>p90 | 270 | 45.6% | −2.3 [−9, +4] | −0.7 | no edge |
| ShortAvoid rkS<p15 | 298 | 47.0% | −2.0 [−8, +4] | −0.6 | no edge |
| Dual strict | 295 | 47.1% | −1.6 [−8, +5] | −0.5 | no edge |

Per-hour edge (1h long, clean): 09:30 **−12.1** / 10:30 −6.7 / 11:30 +1.7 / 12:30 −0.9 / 13:30 −3.8 / **14:30 +31.7 (OVERNIGHT)**. The only positive hour is the overnight one.
15m sniper split: **15:00 (intraday) = −3.0 bps / 43.5% WR**; **15:15 (overnight) = +28.9 bps / +170.5%**. The sniper was the overnight bar.

**Bottom line:** On clean intraday data, the 1h+15m long strategy has **NO statistically significant edge after 10 bps** in this OOS window. The 15m confirmation still *helps relatively* (lifts WR 44%→47%, roughly halves drawdown) but not enough to cross zero. **What remains genuinely true and unaffected:** the model **ranking quality** (within-dataset IC t≈19) and the **short-model audit verdict** (don't retrain — these used within-dataset metrics immune to the bugs). A possible *overnight* long strategy (buy ~15:15 close, hold to next open: +28.9 bps, 58% WR) exists but is regime/gap-risk-dependent, only 3 months of a rally, and is NOT the intraday strategy that was sought.

> Treat sections 3–7 and Appendix A below as the (flawed) research narrative as it unfolded. The numbers there are superseded by this correction box. Definitive script: `scripts/analysis/definitive_backtest.py`.

> **UPDATE (later same day):** A full clean rebuild + native-V3-1h refetch + retrain + **holdout-validated** OOS audit was completed. See **[[Clean Re-Analysis & OOS Validation]]** for the authoritative result: on genuinely out-of-sample data, **no dual-TF config is significant after 10 bps** (the +21.9 bps short-confirm was ~75% in-sample inflation → +5.4 bps, t=0.9 OOS). The 15m model's ranking quality is genuine (calibration +0.84/+1.00); the 1h is weak.

---

## 0. TL;DR — What We Learned

1. **The 15-min model is a strong cross-sectional ranker** (OOS IC ≈ 0.060, t≈19) but its raw per-bar edge is small (~0.25% on a strong candle). After 10 bps friction, **trading the 15-min model alone intraday is not viable** — except at one hour.
2. **The 15-min edge is concentrated entirely at EOD (15h).** Every other hour is net-negative after fees. The "sniper" is `L > 0.0829 @ 15h` → +19.3 bps net, +158% over 3 months OOS.
3. **The real power of the 15-min model is as a confirmation layer for the 1-hour model.** Requiring 15-min agreement at entry transforms the 1h long book: net edge +1.6 → +29 bps, max DD −36% → −10%.
4. **The short side does not benefit from confirmation — and it is NOT a model defect.** The short models rank as well as the long models (equal IC, better bucket monotonicity). The asymmetry is *economic*: in a long-biased market, overbought stocks follow through with far less magnitude than oversold stocks. **Do not retrain.**
5. **The best use of the short model is on the LONG side.** Its bottom decile (D1 = "won't fall") is an independent, high-quality long signal. Gating 1h longs on `rk_short < p10` delivers the same edge as the long-confirm filter with the **lowest drawdown of any scheme (−9.0%)**.

---

## 1. 15-Min Model Evaluation (Baseline)

`v2_15min_3y` — XGBoost `rank:pairwise`, 86 features, 3.19M rows (Jan 2023–Jun 2026), 6-fold walk-forward.

| Metric | Long | Short |
|---|---|---|
| OOS Spearman IC | +0.0602 (t≈19.5) | +0.0597 (t≈19.2) |
| Win Rate @ K=3 | 58.0% | 57.0% |
| Calibration Rho | +0.9879 | +0.9879 |
| Bucket Monotonicity Rho | +0.7576 | +0.9273 |

Dominant features: **IBS, Buy_Pressure, Log_Return, Lower_Shadow** — microstructure mean-reversion signals.

→ Full detail: [[Complete Edge Catalog]], [[Feature Analysis & SHAP]], [[Prediction Bucket & Calibration Deep Dive]]

---

## 2. The Cost Reality Check

A strong 15-min candle moves ~0.25% (25 bps). The model's *gross* edge per selected trade is real, but small. Key correction made during research:

- **STT for equity intraday = 0.025% = 2.5 bps** (sell side), NOT 25 bps.
- Realistic equity-intraday round-trip friction ≈ **6–10 bps** for ₹50K–₹1L positions (brokerage is the variable component — ₹20 flat = 2 bps on ₹1L but 20 bps on ₹10K).
- Our **10 bps model is therefore slightly conservative** and realistic. The earlier "35 bps kills everything" claim was based on a unit error (25 bps STT) and is wrong.

Conclusion: 10 bps is the right friction assumption. The Tier-1 sniper (+29 bps gross) clears it comfortably; small-magnitude signals (like most short trades) do not.

---

## 3. Sniper Discovery & Time-of-Day Structure

Swept 6 signal types × thresholds × 7 hours. **1,834 valid configs.** Key result:

| Tier | Filter | Hour | N | Raw WR | Net WR | Net bps | 3M Ret | Max DD |
|---|---|---|---|---|---|---|---|---|
| Tier 1 | `L>0.0829` (p99) | 15h | 512 | 58.8% | 53.7% | +19.30 | +158.0% | -27.8% |
| Tier 2 | `L>0.0629` (p95) | 15h | 1,073 | 58.1% | 52.3% | +13.05 | +277.0% | -38.7% |
| Tier 3 | `L>0.0514` (p90) | 15h | 1,428 | 59.0% | 51.8% | +11.62 | +379.7% | -47.9% |
| Sniper Short | `S>0.0514 & L<-0.1112` | 10h | 47 | 78.7% | 57.4% | +1.49 | +0.7% | -0.9% |

**The time-of-day finding is the structural core.** Net bps by hour for Tier 1:

| Hour | 9h | 10h | 11h | 12h | 13h | 14h | **15h** |
|---|---|---|---|---|---|---|---|
| Tier 1 net bps | -9.5 | -7.6 | -2.6 | -7.5 | -6.2 | -4.9 | **+19.3** |

**Every hour except 15h is net-negative across all tiers.** This is not data-mining — it is structural: IBS and Buy_Pressure are most information-rich at EOD when the day's range is established and institutional book-squaring expresses real conviction. At 9h the range hasn't formed; midday is chop.

The Sniper Short (78.7% raw WR!) is striking but useless: gross moves are so tiny (+11.49 bps) that 10 bps friction erases the edge (+1.49 net). This is the first hint of the long/short magnitude asymmetry.

→ Full detail: [[Sniper Trade Analysis]]
Assets: `![[assets/11_tier1_equity_curve.png]]` · `![[assets/12_sniper_deep_dive.png]]` · `![[assets/13_hourly_breakdown.png]]`

### Tier 1 Sniper Deep Dive (the +158%)
- Sharpe 8.47 · Sortino 13.31 · Calmar 5.68x
- **Returns front-loaded:** April +94%, May +37%, June −3%. 94% of return came from April.
- Fat tails: best trade +838 bps, worst −755 bps. Skew +0.70, kurtosis 7.8 → the headline return is partly outlier-driven; size small.
- 62.8% of trading days profitable; max 17 consecutive wins / 16 consecutive losses.

![[assets/12_sniper_deep_dive.png]]
![[assets/13_hourly_breakdown.png]]

---

## 4. The Dual-Timeframe Strategy

**Premise (the user's insight):** The 15-min model can't profit alone intraday after fees, BUT it predicts *direction* well (60%+ raw WR in some tiers). So pair it with the 1-hour model — only take 1h signals the 15-min model confirms, and re-check every 15 min that the edge persists.

**Architecture:**
- 1h bar at T → 1h model ranks top-K.
- **Entry confirmation:** check the 15-min model's rank at T+45 (last 15m bar before entry).
- **Hold:** up to 4×15-min bars (1 hour); base return = `Next_Hour_Return`.
- **Early exit:** monitor each 15-min bar; exit if 15m rank drops below a floor.
- Force-exit after one full hour.

### Results — LONG side (confirmation transforms it)

| Config | N | Net WR | Net bps | 3M Ret | Max DD |
|---|---|---|---|---|---|
| Baseline 1H only K=3 | 1,023 | 46.0% | +1.58 | +11.4% | -36.3% |
| Dual-TF conf p90 K=3 | 238 | 50.8% | +29.43 | +96.0% | -10.5% |
| Dual-TF conf p95 K=3 | 180 | 53.3% | +31.92 | +74.2% | -13.0% |
| Dual+Exit p90 K=3 | 238 | 52.9% | +31.31 | +105.2% | -9.1% |
| Dual-TF conf p90 K=5 | 403 | 50.6% | +27.19 | +186.9% | -11.3% |

Confirmation takes the long book from "barely covers fees" (+1.58 bps) to "+29 bps" and cuts max drawdown ~3.5×.

### Results — SHORT side (confirmation does nothing)

| Config | N | Net WR | Net bps | 3M Ret | Max DD |
|---|---|---|---|---|---|
| Baseline 1H only K=3 | 1,023 | 49.4% | +1.99 | +10.0% | -48.5% |
| Dual-TF conf p90 K=3 | 330 | 50.9% | +1.58 | +1.1% | -34.2% |
| Dual-TF conf p95 K=3 | 247 | 50.6% | +1.94 | +1.4% | -32.6% |

Net bps stays pinned near zero regardless of threshold. The short baseline (+1.99) is actually *better* than the long baseline (+1.58) — but it cannot be concentrated.

### Combined Portfolio

| Config | N | Net WR | Net bps | 3M Ret | Max DD |
|---|---|---|---|---|---|
| Combined Baseline K=3 | 2,046 | 47.7% | +1.78 | +22.5% | -33.3% |
| **Combined Dual-TF p90 K=3** | 568 | 50.9% | +13.25 | +98.2% | -16.0% |
| Combined Dual-TF p90 K=5 | 934 | 50.2% | +11.46 | +164.7% | -28.0% |

The combined improvement (+22.5% → +98%) comes **almost entirely from the long side.**

![[assets/dual_tf_full.png]]

---

## 5. Short-Model Audit — Why Confirmation Fails on Shorts

Question: do we need to retrain the short model? Tested 4 hypotheses.

**H1 — Regime?** No. Universe was flat over OOS (1h mean +0.012%, 48.8% positive bars). Shorts weren't fighting a bull market.

**H2 — Ranking power (IC)?** The short models are NOT broken:

| Frame | Side | Mean IC | t-stat |
|---|---|---|---|
| 1h | LONG | +0.0304 | 5.12 |
| 1h | SHORT | +0.0362 | **6.20** (beats long) |
| 15m | LONG | +0.0602 | 19.52 |
| 15m | SHORT | +0.0597 | 19.21 (≈ long) |

**H3 — Bucket monotonicity?** 15m short is the *best* of all four (Rho +0.93). But the D10 (top-conviction shorts) edge is only +0.0133% vs long D10 +0.0459% — **3.5× smaller magnitude at the top.**

**H4 — Confirmation alpha (the mechanism):** Forward edge of 1h trades by 15m confirmation-rank bin:

| 15m rank bin | LONG edge | SHORT edge |
|---|---|---|
| 85–95 | +0.215% | +0.056% |
| **95–100** | **+0.342%** | **+0.038%** |

When both models strongly agree on a long, forward edge **triples to +0.342%** (≫ 10 bps cost). On the short side the top bin is only +0.038% (< cost). **The confirmation filter concentrates magnitude into the top tail — and on the short side there is no magnitude to concentrate.**

### Verdict: DO NOT RETRAIN
The short ranker is statistically as good as the long ranker. The problem is **economic asymmetry**: the microstructure features (IBS, Buy_Pressure, Lower_Shadow) are fundamentally mean-reversion *long* detectors. Run in reverse for shorts, they detect "overbought fade," but in a retail-heavy, up-drifting market, overbought stocks keep grinding higher far more than oversold stocks keep falling. **Equal hit-rate, asymmetric magnitude.** A retrain of the same architecture on the same features reproduces the same asymmetry.

If better shorts are genuinely wanted, the fix is the **target/objective**, not a retrain: train a short-specific model on down-regime data only, or with an asymmetric loss rewarding large-magnitude down moves, plus regime-conditioning features. That is a different model.

![[assets/short_model_audit.png]]

---

## 6. The Payoff — Short Model as a LONG Filter

The audit showed the short model's **D1 (lowest short rank = "won't fall")** had edge −0.049% → those stocks RISE +0.049%, *larger* than the long model's own top decile. So we use a low short rank as an independent "this goes up" vote.

Backtest (all configs require a 15m match; baseline = 656 trades):

| Scheme (K=3) | Signal | N | Net bps | 3M Ret | Max DD |
|---|---|---|---|---|---|
| Baseline | — | 656 | +6.61 | +49.1% | -21.9% |
| LongConfirm p90 | `rk_long > 0.90` | 221 | +23.45 | +64.0% | -10.5% |
| **ShortAvoid D1** | **`rk_short < 0.10`** | 208 | **+23.63** | +59.8% | **-9.0%** |
| ShortAvoid p15 | `rk_short < 0.15` | 265 | +19.30 | +62.7% | -11.2% |
| DualAgree | `L>p80 & S<p20` | 305 | +17.24 | +64.8% | -13.4% |
| Composite (soft blend) | `top by L+(1-S)` | 666 | +4.43 | +30.5% | -26.3% |

K=5 (more compounding): **ShortAvoid p15 K=5** → +19.69 bps, **+129.3%**, −13.8% DD (vs baseline +91.5%, −28.2%).

**Findings:**
1. **ShortAvoid D1 alone matches the long-confirm edge (+23.6 vs +23.5 bps) at the lowest drawdown of any scheme (−9.0%).** The short model contributes alpha equal to the long model's own confirmation — on the long side.
2. **Hard gate beats soft blend.** The Composite (averaging the two ranks) was the worst — averaging lets high-long/mediocre-short names slip in. A hard veto is strictly better.
3. **The two filters (rk_long high, rk_short low) are nearly equal but use independent model outputs.** Stacking them (DualAgree) didn't beat either alone in this sample, but may separate further on a larger universe.

### Production rule
> Take a 1h long pick **only if** the 15-min short model ranks that ticker in its bottom 10–15% (`rk_short < 0.10–0.15`) at the entry bar.

This extracts the short model's full economic value — best risk-adjusted long edge, lowest drawdown — **without ever placing a short trade.**

![[assets/long_filter_comparison.png]]

---

## 7. Recommended System Configuration

**Engine:** Long-biased dual-timeframe.
- **Signal source:** 1h `v8_upstox_3y` top-K long ranking.
- **Entry gate (either or both):**
  - `rk_long_15m > 0.90` (long-confirm), and/or
  - `rk_short_15m < 0.10–0.15` (short-avoid / D1) — lowest drawdown.
- **Sizing:** K=3 for tightest risk (−9 to −11% DD), K=5 for more compounding (+129–187% but −14% DD).
- **EOD overlay (optional):** the standalone 15-min sniper `L>0.0829 @ 15h` as a separate book.
- **Shorts:** only in confirmed down-regimes (monthly mean ret < 0); otherwise use the short model purely as the long filter above.

**Caveats:**
- OOS is 3 months and partly outlier-driven (April was exceptional). Annualising by ×4 is naive — watch rolling monthly net WR; degrade below ~52% = edge fading.
- 1h data ends May 2026; 15m runs to June. Keep both pipelines fed.

---

## 8. Generation Scripts (all in `scripts/analysis/`)

| Script | Purpose |
|---|---|
| `eval_15min_v2.py` | 15-min model OOS metrics, walk-forward, feature importance |
| `visualize_15min_v2.py` | 8-plot diagnostic suite |
| `eval_buckets_calibration.py` | Prediction bucket & calibration deep dive |
| `sniper_15min.py` | Sniper threshold sweep (6 signals × hours) |
| `sniper_deep_dive.py` | Tier-1 sniper monthly/weekly/risk breakdown |
| `hourly_breakdown.py` | Net bps by hour × tier (time-of-day proof) |
| `dual_tf_backtest.py` | Dual-timeframe long + short + combined backtest |
| `short_model_audit.py` | 4-hypothesis short vs long audit |
| `long_filter_backtest.py` | Short-model-as-long-filter schemes |

Plot outputs: `data/model_analysis/v2_15min_3y/` and `data/model_analysis/dual_tf/`.

---

## Appendix A — Full Backtest Results (every threshold tested)

This appendix consolidates **every** configuration run during the research, grouped by analysis type. "Raw" = no fees; "Net" = after 10 bps. All returns are compounded over the OOS window.

### A.1 Raw Conviction Threshold Analysis — 15-min standalone (EOD, hour 15)

Pure single-model score thresholds on the 15-min long ranker, gated to 15h. Shows how raw direction accuracy survives (or doesn't survive) the 10 bps cost.

| Tier | Threshold | N | Raw WR | Net WR | Raw bps | Net bps | Raw Ret% | Net Ret% | Max DD |
|---|---|---|---|---|---|---|---|---|---|
| Tier 1 | `L>0.0829` (p99) | 512 | 58.8% | 53.7% | +29.30 | +19.30 | +330.1% | +158.0% | -27.8% |
| Tier 2 | `L>0.0629` (p95) | 1,073 | 58.1% | 52.3% | +23.05 | +13.05 | +1000.3% | +277.0% | -38.7% |
| Tier 3 | `L>0.0514` (p90) | 1,428 | 59.0% | 51.8% | +21.62 | +11.62 | +1896.0% | +379.7% | -47.9% |

**Reads:** Higher threshold → higher WR & bps but fewer trades. Lower threshold → more compounding (higher total return) but worse per-trade quality and much deeper drawdown. The ~5pp WR gap (raw→net) is the fee cost; the ~10 bps gap (raw→net bps) is the flat friction.

### A.2 Why the EOD gate is mandatory — broad (all-hours) vs gated

| Config | N | Raw WR | Net WR | Raw bps | Net bps |
|---|---|---|---|---|---|
| `L>0.0829` @ 15h (gated) | 512 | 58.8% | 53.7% | +29.30 | +19.30 |
| `L>0.0829` ALL hours | 1,813 | 56.7% | 41.4% | +11.09 | +1.09 |
| `L>0.0629` ALL hours | 5,848 | 55.7% | 39.6% | +7.19 | **-2.81** |

Removing the hour filter collapses net WR by ~17pp and net bps to ~breakeven or negative. The score threshold alone is **not** enough — **score AND hour 15** are both required.

### A.3 Time-of-Day Breakdown — net bps by hour × tier

Every hour except 15h is net-negative across all three tiers. (Net bps shown; raw WR / net WR in parentheses for Tier 1.)

| Hour | Tier 1 net bps (raw/net WR) | Tier 2 net bps | Tier 3 net bps | No-filter net bps |
|---|---|---|---|---|
| 9h  | -9.53 (45.7%/38.0%) | -4.97 | -5.02 | -9.86 |
| 10h | -7.64 (57.5%/44.5%) | -5.54 | -6.25 | -9.63 |
| 11h | -2.59 (61.5%/42.3%) | -6.02 | -5.46 | -7.92 |
| 12h | -7.50 (52.3%/34.3%) | -7.38 | -7.73 | -8.93 |
| 13h | -6.18 (57.0%/33.3%) | -7.04 | -8.54 | -9.74 |
| 14h | -4.86 (57.3%/34.7%) | -6.33 | -6.72 | -10.10 |
| **15h** | **+19.30 (58.8%/53.7%)** | **+13.05** | **+11.62** | **-6.63** |

Note 11h Tier 1 has 61.5% *raw* WR but only +7.41 raw bps — high accuracy, tiny moves → −2.59 net. Magnitude, not accuracy, is what 15h provides.

### A.4 Model-Combined Thresholds — Dual-TF confirmation (1h signal + 15m confirm)

The 1h `v8_upstox_3y` top-K is gated by the 15-min model's rank percentile at entry (T+45). `confXX` = require 15m rank > pXX. `+Exit` adds early-exit if 15m rank drops below p40 during the hold.

**LONG side** (confirmation transforms it):

| Config | N | Raw WR | Net WR | Net bps | 3M Ret | Max DD | Early% |
|---|---|---|---|---|---|---|---|
| Baseline 1H-only K=3 | 1,023 | 53.4% | 46.0% | +1.58 | +11.4% | -36.3% | — |
| Dual-TF conf p85 K=3 | 303 | 55.4% | 49.2% | +21.53 | +86.1% | -13.7% | — |
| Dual-TF conf p90 K=3 | 238 | 56.7% | 50.8% | +29.43 | +96.0% | -10.5% | — |
| Dual-TF conf p95 K=3 | 180 | 58.3% | 53.3% | +31.92 | +74.2% | -13.0% | — |
| Dual+Exit conf p90 K=3 | 238 | **64.7%** | 52.9% | +31.31 | +105.2% | **-9.1%** | 45% |
| Dual-TF conf p90 **K=5** | 403 | 56.6% | 50.6% | +27.19 | **+186.9%** | -11.3% | — |

**SHORT side** (confirmation does nothing — net bps pinned near 0):

| Config | N | Raw WR | Net WR | Net bps | 3M Ret | Max DD | Early% |
|---|---|---|---|---|---|---|---|
| Baseline 1H-only K=3 | 1,023 | 55.5% | 49.4% | +1.99 | +10.0% | -48.5% | — |
| Dual-TF conf p85 K=3 | 362 | 55.8% | 50.0% | -0.15 | -4.6% | -35.9% | — |
| Dual-TF conf p90 K=3 | 330 | 55.5% | 50.9% | +1.58 | +1.1% | -34.2% | — |
| Dual-TF conf p95 K=3 | 247 | 55.1% | 50.6% | +1.94 | +1.4% | -32.6% | — |
| Dual+Exit conf p90 K=3 | 330 | 60.0% | 53.0% | +0.07 | -3.5% | -37.1% | 41% |
| Dual-TF conf p90 K=5 | 531 | 54.6% | 49.9% | -0.48 | -7.7% | -48.1% | — |

**COMBINED portfolio** (long + short pooled, sorted by date, compounded):

| Config | N | Raw WR | Net WR | Net bps | 3M Ret | Max DD | Early% |
|---|---|---|---|---|---|---|---|
| Combined Baseline K=3 | 2,046 | 54.4% | 47.7% | +1.78 | +22.5% | -33.3% | — |
| Combined Dual-TF p90 K=3 | 568 | 56.0% | 50.9% | +13.25 | +98.2% | -16.0% | — |
| Combined Dual-TF p95 K=3 | 427 | 56.4% | 51.8% | +14.58 | +76.7% | -26.5% | — |
| Combined Dual+Exit p90 K=3 | 568 | 62.0% | 53.0% | +13.16 | +98.1% | -14.3% | 42% |
| Combined Dual-TF p90 **K=5** | 934 | 55.5% | 50.2% | +11.46 | **+164.7%** | -28.0% | — |

The combined gain (+22.5% → +98%) is almost entirely from the long side; short adds little.

### A.5 Dual-Combined Thresholds — Short-Model-as-Long-Filter

Long-only book. Four filtering schemes on 1h long picks, all requiring a 15m match at entry (so the comparable baseline is 656 trades, not 1,023). `rk_long` = 15m long rank pct; `rk_short` = 15m short rank pct (low = "won't fall" = D1 signal).

**K=3:**

| Scheme | Rule | N | Raw WR | Net WR | Net bps | 3M Ret | Max DD |
|---|---|---|---|---|---|---|---|
| Baseline | (none) | 656 | 54.1% | 47.4% | +6.61 | +49.1% | -21.9% |
| LongConfirm | `rk_long > 0.90` | 221 | 55.7% | 49.8% | +23.45 | +64.0% | -10.5% |
| ShortAvoid p25 | `rk_short < 0.25` | 354 | 55.1% | 48.0% | +13.64 | +57.8% | -13.4% |
| ShortAvoid p15 | `rk_short < 0.15` | 265 | 54.3% | 47.5% | +19.30 | +62.7% | -11.2% |
| **ShortAvoid D1** | **`rk_short < 0.10`** | 208 | 55.3% | 49.5% | **+23.63** | +59.8% | **-9.0%** |
| DualAgree | `L>p80 & S<p20` | 305 | 55.4% | 48.5% | +17.24 | +64.8% | -13.4% |
| DualAgree strict | `L>p85 & S<p15` | 255 | 54.1% | 47.8% | +19.67 | +61.1% | -12.3% |
| Composite (soft) | `top by L+(1-S)` | 666 | 53.0% | 44.7% | +4.43 | +30.5% | -26.3% |

**K=5:**

| Scheme | Rule | N | Raw WR | Net WR | Net bps | 3M Ret | Max DD |
|---|---|---|---|---|---|---|---|
| Baseline | (none) | 1,096 | 54.7% | 46.8% | +6.43 | +91.5% | -28.2% |
| **ShortAvoid p15** | `rk_short < 0.15` | 441 | 54.6% | 48.3% | +19.69 | **+129.3%** | -13.8% |
| DualAgree | `L>p80 & S<p20` | 501 | 54.9% | 47.7% | +16.36 | +118.0% | -13.8% |
| Composite (soft) | `top by L+(1-S)` | 1,110 | 51.5% | 43.1% | +1.53 | +13.2% | -35.6% |

**Reads:**
- **ShortAvoid D1 (`rk_short<0.10`) is the standout:** +23.63 bps (≈ LongConfirm's +23.45) at the **lowest DD of all schemes, −9.0%** — using a *different, independent* model output.
- **Hard gate > soft blend:** the Composite (averaging ranks) is consistently the worst, because it lets high-long/mediocre-short names through.
- DualAgree stacks both filters but didn't beat either alone in this sample (smaller N).

### A.6 Best-of Leaderboard (by objective)

| Objective | Winner | Net bps | 3M Ret | Max DD | WR |
|---|---|---|---|---|---|
| Highest net bps / trade | Dual-TF LONG conf p95 K=3 | +31.92 | +74.2% | -13.0% | 53.3% net |
| Best risk-adjusted (low DD, high edge) | **ShortAvoid D1 K=3** | +23.63 | +59.8% | **-9.0%** | 49.5% net |
| Highest total return (controlled DD) | **Dual-TF LONG conf p90 K=5** | +27.19 | **+186.9%** | -11.3% | 50.6% net |
| Highest raw WR | Dual+Exit LONG p90 K=3 | +31.31 | +105.2% | -9.1% | **64.7% raw** |
| Standalone EOD book | Tier 1 sniper `L>0.0829@15h` | +19.30 | +158.0% | -27.8% | 53.7% net |
| Best combined long+short | Combined Dual-TF p90 K=5 | +11.46 | +164.7% | -28.0% | 50.2% net |

> Caveat across all tables: OOS is 3 months and partly outlier-driven (April was exceptional). Treat absolute % returns as directional, not annualizable; rank configs by **net bps and max DD**, which are more stable.

---

## Backlinks

- [[Complete Edge Catalog]] — 15-min walk-forward & OOS performance.
- [[Sniper Trade Analysis]] — EOD sniper tiers & 1h comparison.
- [[Prediction Bucket & Calibration Deep Dive]] — decile tables, calibration.
- [[Feature Analysis & SHAP]] — feature hierarchy & SHAP.
- [[Time of Day & Residual Analysis]] — residual MAE by hour (basis for EOD edge).
- [[Model Diagnostics & Visualizations]] — full plot suite.
