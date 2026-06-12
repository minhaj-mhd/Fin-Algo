---
title: "Clean Re-Analysis & OOS Validation (Native Data, Holdout-Verified)"
type: report
status: active
model: "15m"
updated: 2026-06-12
tags: []
---
# Clean Re-Analysis & OOS Validation (Native Data, Holdout-Verified)

**Date:** June 7, 2026
**Scope:** Full rebuild of the 1h + 15m datasets and models on clean, leak-free, production-faithful data, followed by a *genuinely out-of-sample* dual-timeframe backtest with bootstrap CIs and a fee/per-month audit, then a **powered 6-fold walk-forward backtest (~24 months OOS)**.
**Verdict in one line:** **DEFINITIVELY NO TRADEABLE EDGE.** The powered walk-forward (6 folds, ~5,000 trades/config, 2024-09→2026-06) shows every dual-TF config is **significantly NEGATIVE after 10 bps** (t = −3.9 to −8.0, 0–1 of 6 folds positive). The 15m model is a genuinely strong *ranker* (IC 0.06, t≈18) but its gross per-trade edge (~1–2 bps) is an order of magnitude below the 10 bps cost. Earlier "edges" were data artifacts + in-sample overfitting + single-quarter flukes.

## ⭐ DECISIVE: Powered Walk-Forward Dual-TF (6 folds, ~24 months OOS, 10 bps)

Script: `scripts/analysis/wf_dual_tf_backtest.py`. Each fold trains 1h+15m (long+short) on data STRICTLY before its 4-month test block; trades pooled across folds for statistical power.

| Config | N | Net WR | Net bps | 95% CI | t | +folds | Total Ret |
|---|---|---|---|---|---|---|---|
| SHORT baseline | 5,860 | 45.2% | −8.6 | [−10.7,−6.6] | −8.0✷✷✷ | 0/6 | −99.5% |
| SHORT + ShortConfirm p90 | 2,353 | 46.8% | −8.0 | [−11.7,−4.5] | −4.4✷✷✷ | 1/6 | −86.2% |
| SHORT + LongAvoid p15 | 1,449 | 45.5% | −8.6 | [−12.5,−4.8] | −4.4✷✷✷ | 0/6 | −72.5% |
| LONG baseline | 5,853 | 41.0% | −6.4 | [−8.1,−4.8] | −7.6✷✷✷ | 0/6 | −97.9% |
| LONG + LongConfirm p90 | 2,442 | 43.5% | −5.3 | [−8.0,−2.7] | −3.9✷✷✷ | 0/6 | −74.0% |
| LONG + ShortAvoid p15 | 2,733 | 42.4% | −5.4 | [−7.7,−3.1] | −4.6✷✷✷ | 0/6 | −78.4% |

Every config significantly negative across 2 years / multiple regimes. The 3-month +5.4 bps short-confirm was a single-quarter (May) fluke. **Ranking skill ≠ enough magnitude to beat friction.** Confirmation trims the loss (−6.4→−5.3) but cannot cross zero. CONCLUSION: do not deploy this intraday top-K / 1h-hold strategy at 10 bps — it is a confirmed loser. The 15m model's only legitimate role would require a fundamentally lower cost structure, a different horizon, or use as a filter/risk overlay (not a standalone return engine).

### VETO VALUE — the one constructive use (LONG only)

Script: `scripts/analysis/wf_veto_value.py` (candidates saved: `data/dual_tf_veto_candidates.csv`). Host = 1h top-3; 15m model as a veto/filter; WF pooled OOS (2024-09→2026-06). A veto is judged by PASS−VETO *separation* (cost-agnostic — it doesn't pay the spread), not absolute level.

- **LONG: usable veto.** 15m own-agree splits longs into PASS (44.5% WR, +5.1 bps gross) vs VETO (37.3% WR, +0.3 bps) — separation **+4.8 bps / +7.2pp WR, p=0.015✷**. Cross-veto (15m short-rank high ⇒ block) similar: +4.6 bps, p=0.023✷. So feeding 15m agreement into the veto layer reliably weeds out the worst long candidates.
- **SHORT: no reliable veto.** Separation +1.2 bps, p=0.66 (ns). Consistent with the short side's economic weakness.
- **Caveats:** improves trade *quality* not profitability (both PASS & VETO longs still net-negative after 10 bps: PASS −4.9, VETO −9.7) — it lifts an already-profitable host, can't create profit alone. Magnitude modest (~5 bps), significance marginal (single-✷, pooled). Use as a long-side quality tie-breaker, not a hard gate. **This is the ONE statistically-supported constructive role for the models.**

### 1h BASE model standalone — any tradable edge? NO (exhaustive)

Script: `scripts/analysis/wf_1h_base.py`. Native 1h alone (no 15m), 9-fold walk-forward, ~600K OOS rows (2023-08→2026-06), 10 bps. EVERY slice significantly negative, 0/9 folds positive:
- Conviction top-K: LONG top-1 −6.7✷✷✷ … top-10 −7.3✷✷✷; SHORT top-1 −6.3✷✷✷ … top-10 −8.7✷✷✷.
- Absolute rank-pct: even p99 (top 1%) LONG −6.4✷✷✷, SHORT −7.6✷✷✷.
- Per-hour (top-3): every hour 09:15–13:15 both directions significantly negative (best SHORT 13:15 −4.3✷✗). **The old "2 PM crown jewel" is dead** — there is no 14:xx signal in clean data (it was the overnight artifact), and every real intraday hour loses.
- 1h base gross edge ~+2–3 bps, even weaker than the 15m. **1h base is not tradable at any conviction, any hour, either direction.**

### Conviction × Calibration grid (does ANY threshold monetize? — NO)

Script: `scripts/analysis/wf_conviction_calibration.py`. Swept conviction (1h top-K ∈ {1,3,5,10}) × 15m calibration gate (own-rank pct ∈ {none,p85,p90,p95,p99}) × both directions, walk-forward pooled OOS (336K candidates, 10 bps).

LONG net bps (after 10 bps): top-10/none −7.3✷✷✷ → top-1/p99 **−2.7 (t=−0.5, ns, n=206, 3/6 folds)**. SHORT: best cell top-1/p99 −6.7 (t=−0.9, ns); all else negative.

**Across all 40 configs, NOT ONE is significantly positive.** Conviction+calibration help *monotonically* (less negative as you tighten) — confirming the ranking skill is real — but the surface asymptotes at ~−3 to −7 bps and never reaches profitability. Even the single most-convicted best-calibrated long pick earns only ~+7 bps GROSS, below the 10 bps cost. **It is not a threshold-selection problem; there is no profitable region.** This is the final, exhaustive confirmation of no edge.

---

> This note supersedes the trading-edge claims in [[Dual-Timeframe Strategy & Full Research Journey]] (which were contaminated). The model-quality and audit findings here are the trustworthy ones.

---

## 1. Why we rebuilt everything

Two data bugs were found and fixed (see [[reference-ranking-data-conventions]] in agent memory):

1. **Overnight-return leak.** Forward returns were a plain `shift(-1)` with no session mask, so each day's last bar got the *next morning* as its "next return." Present in both the 1h (`14:30` bar) and 15m (`15:15` bar) datasets. This was the source of the old 1h "2 PM crown jewel" (Short @ 2PM, 68% WR) — it was an **overnight hold**, not a 2PM intraday edge.
2. **Timestamp-grid misalignment.** The 1h dataset was resampled from 30-min candles in UTC (`origin='start_day'`); the +5:30 IST offset distorted the hourly grid (a bar labeled 09:30 actually spanned 09:45–10:45, and the opening 15 min was dropped). The 1h and 15m grids did not correspond.

**Fix:** rebuilt both datasets clean (IST, 09:15-anchored, session-masked) and, for the 1h, fetched **native hourly candles from the Upstox V3 API** (`HistoryV3Api`, `hours/1`) so training matches what live inference can fetch (zero train/serve skew).

---

## 2. Clean models trained

| Model | Source | Grid | History | Registry |
|---|---|---|---|---|
| `v3_15min_clean` | 1-min → 15-min | quarter-hours | 2023–26 (3.1M rows) | registered |
| `v9_clean_1h` | 15-min → 1-h (same source as 15m) | 09:15-anchored | 2023–26 (642K rows) | registered (research) |
| `v10_native_1h` | **Upstox V3 native 1h** | 09:15-anchored | 2022–26 (928K rows) | registered (production-faithful) |

Collectors: `scripts/collectors/rebuild_aligned_datasets.py`, `scripts/collectors/collect_upstox_1h_v3.py`. Training: `scripts/training/train_ranking_clean.py`.

### Walk-forward IC (genuinely OOS per fold)

| Model | WF Long IC | WF Short IC |
|---|---|---|
| Original v8 (contaminated, 30min→1h, +overnight) | 0.0396 | 0.0448 |
| `v9_clean_1h` (resampled) | 0.0204 | 0.0245 |
| `v10_native_1h` (native V3) | 0.0261 | 0.0245 |
| **`v3_15min_clean`** | **0.0586** | **0.0593** |

Native fetch recovered ~28% IC vs the resample (0.020→0.026), confirming a small resampling artifact — but the 1h is still **less than half** the 15m's IC, and decaying over time (native 1h 2026 fold ≈ 0.012). The old 0.04 was grid-distortion + overnight inflation.

---

## 3. Model characterization (diagnostic suites + calibration)

8-plot suites in `data/model_analysis/v10_native_1h/` and `data/model_analysis/v3_15min_clean/` (feature importance, SHAP summary/dependence, learning curve, prediction bucket, cumulative return, calibration, residual).

### The decisive contrast — per-query calibration

| Metric | Native 1h (v10) | Clean 15m (v3) |
|---|---|---|
| Bucket Rho — Long | +0.61 | +0.68 |
| Bucket Rho — Short | −0.41 ✗ | +0.99 ✓ |
| **Calibration Rho — Long** | **−0.14 (WEAK)** | **+0.84 (GOOD)** |
| **Calibration Rho — Short** | **−0.39 (WEAK)** | **+1.00 (EXCELLENT)** |
| Top-decile WR (L/S) | 52.1% / 55.9% | 53.5% / 54.1% |

**The 15m is a genuinely excellent cross-sectional ranker** (short calibration is literally perfect, 9/10 deciles in order). **The native 1h is a poor ranker** — per-query calibration ≈ 0; only the extreme top decile carries a sliver of edge. Scripts: `scripts/analysis/visualize_clean_model.py`, `scripts/analysis/eval_buckets_calibration_clean.py`.

---

## 4. The dual-TF backtest — in-sample vs GENUINE OOS (the key lesson)

**Validity bug caught:** the production models were trained on all-but-the-last-month, so a backtest on the "last 3 months" (2026-04/05/06) was **in-sample** (2 months trained on, 1 validated). The fix: train **holdout** models on data strictly before the test window (train ≤ 2026-02, val 2026-03, OOS 2026-04/05/06 fully unseen), then backtest. Script: `scripts/analysis/oos_holdout_backtest.py`.

### Headline config — the inflation it exposed

| | In-sample (prod models) | **Genuine OOS (holdout)** |
|---|---|---|
| SHORT + 15m ShortConfirm p90 | +21.9 bps, t=3.6 ✷✷✷, 60.9% WR | **+5.4 bps, CI[−7,+18], t=0.9, 53% WR** |

~75% of the headline was in-sample overfitting. On unseen months it loses significance.

### Full genuine-OOS audit (holdout models, 10 bps, bootstrap 95% CI)

| Config | N | Net bps | 95% CI | t | Net WR | Verdict |
|---|---|---|---|---|---|---|
| SHORT baseline | 643 | −12.5 | [−20,−5] | −3.2 | 46% | sig **negative** |
| SHORT + ShortConfirm p90 | 219 | +5.4 | [−7,+18] | 0.9 | 53% | not sig |
| SHORT + LongAvoid p15 | 142 | +6.8 | [−5,+20] | 1.1 | 54% | not sig |
| LONG baseline | 636 | −3.3 | [−8,+2] | −1.3 | 44% | not sig |
| LONG + LongConfirm p90 | 302 | +1.1 | [−6,+8] | 0.3 | 47% | not sig |
| LONG + ShortAvoid p15 | 335 | −2.4 | [−9,+4] | −0.7 | 44% | not sig |

**Reads:**
- No config clears significance after 10 bps; every confirmed CI spans zero.
- The 15m confirmation *does* help directionally (short baseline −12.5 → +5.4; long −3.3 → +1.1; +a few pts WR) — consistent with the 15m being a real ranker — but not enough to beat costs to significance here.
- Positives are single-month-driven (short-confirm: Apr −9.3 / **May +21.9** / Jun −3.1 — it's all May).
- Fee breakeven for the best configs ≈ 15–17 bps gross; at 10 bps they're marginal and fragile.

---

## 5. Bottom line & what would change it

- **Tradeable claim:** none, at significance, on clean holdout-validated 3-month OOS. The contaminated headlines (+98%, +158%, +21.9 bps) do not survive.
- **Still true:** the **15m model's ranking quality is genuine** (WF IC 0.059, calibration +0.84/+1.00). It's the real asset. The 1h is weak and the dual-TF doesn't rescue it to significance.
- **To get confidence either way:** (a) extend OOS well beyond 3 months (current window is one regime, positives are one month); (b) if deploying `v10_native_1h`, switch the live broker's 1h fetch to V3 `hours/1` for train/serve consistency; (c) test exits with `v9_clean_1h` (same source as 15m → reconciles to 0.00; native 1h vs 15m reconcile only to ~22 bps, so exit-path P&L on v10 is unreliable).

---

## Methodology guardrails learned (apply to all future backtests)

1. **Holdout validation:** never backtest production models (trained to last month) on a recent window — it's in-sample. Train on data strictly before the test window.
2. **Session-mask forward returns** (NaN at each day's last bar) — no overnight leak.
3. **Align cross-timeframe by verified close prices**, not by label; prefer one shared candle source.
4. **Bootstrap CIs + t-stats** on every config; a point estimate without a CI is not a result.

---

## Backlinks

- [[Dual-Timeframe Strategy & Full Research Journey]] — earlier (contaminated) research arc; superseded by this note for trading claims.
- [[Complete Edge Catalog]] — 15m model walk-forward & OOS performance.
- [[Prediction Bucket & Calibration Deep Dive]] — 15m calibration methodology.
