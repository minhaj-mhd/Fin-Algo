# Model Card — v3_15min_clean

**Date:** June 7, 2026
**Type:** XGBoost `rank:pairwise` cross-sectional ranker (separate Long & Short models)
**Status:** Registered. The genuinely-strong ranker of the suite. NOT tradable as standalone alpha after costs, but **usable as a LONG-side veto/filter** — see Tradability Verdict.

> Definitive companion analysis: [[Clean Re-Analysis & OOS Validation]].

---

## 1. Training Profile

| Attribute | Value |
|---|---|
| Model ID | `v3_15min_clean` |
| Data source | 1-minute Upstox → resampled to 15-min (IST, `origin='start_day'`) |
| Data file | `data/ranking_data_upstox_15min_3y_clean.csv` |
| Collector | `scripts/collectors/rebuild_aligned_datasets.py` |
| Trainer | `scripts/training/train_ranking_clean.py --tf 15min` |
| Total rows | 3,102,436 |
| Span | 2023-01 → 2026-06 (41 months) |
| Grid | quarter-hours 09:15 → 15:00 tradable (15:15 dropped, overnight-masked) |
| Forward target | `Next_15Min_Return`, session-masked (no overnight leak) |
| Features | 86 |
| Trained at | 2026-06-07T16:22 |

**Hyperparameters:** `rank:pairwise`, eta 0.03, max_depth 4, subsample 0.8, colsample_bytree 0.8, alpha 1.0, lambda 2.0, min_child_weight 15, eval_metric ndcg@3, tree_method hist, device cuda.

---

## 2. Walk-Forward Performance (genuinely OOS per fold, 6 folds)

| Metric | Value |
|---|---|
| Avg Long Spearman IC | **0.0586** |
| Avg Short Spearman IC | **0.0593** |
| Avg Long Win Rate @ K=3 | 58.0% |
| Avg Short Win Rate @ K=3 | 57.2% |

### Fold-by-fold Spearman

| Fold | Long Rho | Short Rho |
|---|---|---|
| 1 | 0.0610 | 0.0629 |
| 2 | 0.0520 | 0.0548 |
| 3 | 0.0691 | 0.0697 |
| 4 | 0.0648 | 0.0661 |
| 5 | 0.0522 | 0.0496 |
| 6 | 0.0526 | 0.0530 |

**Consistent across folds** (no decay), Long ≈ Short. IC essentially matches the original (contaminated) v2_15min (0.0602/0.0597) — confirming the 15m ranking quality was always real (its grid was clean; only the overnight target label was affected, ~1/25 of rows). **This is ~2.3× the native 1h's IC.**

---

## 3. OOS Diagnostics (Apr–Jun 2026, 1,032 queries, 175,988 rows)

| Metric | Long | Short |
|---|---|---|
| Prediction-Bucket Rho (global) | +0.68 | **+0.99** |
| **Calibration Rho (per-query)** | **+0.84 (GOOD)** | **+1.00 (EXCELLENT)** |
| Bucket monotonicity | 8/10 in order | 9/10 in order |
| Top-decile (D10) win rate | 53.5% | 54.1% |
| Top-decile edge per bar | +0.030% | +0.0069% (short side magnitude small) |
| MWU top-20% vs bottom-20% | p=1.3e-65, spread +0.028% | p=6.7e-57, spread +0.024% |

**Read:** a **genuinely excellent cross-sectional ranker.** Per-query calibration Rho +0.84 (long) / +1.00 (short, literally perfect monotonic ordering). The model sorts winners from losers within each timestamp reliably and consistently. This is the real asset of the whole suite.

Diagnostic plots (8-plot suite + dashboard): `data/model_analysis/v3_15min_clean/`. Calibration deep-dive: `scripts/analysis/eval_buckets_calibration_clean.py --model v3_15min_clean`.

---

## 4. Feature Importance (gain)

**Long (top 10):** IBS 60.96, Buy_Pressure 51.47, Lower_Shadow 19.73, OC_Range 16.60, Relative_Return 9.72, Log_Return 9.49, Return 8.96, Is_Open_Hour 8.30, Time_To_Close 7.29, Hour 5.83.

**Short (top 10):** IBS 58.37, Return 39.47, Log_Return 20.31, Relative_Return 15.26, Upper_Shadow 11.30, IBS_3 10.91, Dollar_Volume 10.58, Lower_Shadow 10.58, Keltner_Width 7.39, Is_Open_Hour 7.38.

**IBS + Buy_Pressure dominate massively** (microstructure mean-reversion = the model's core alpha). This short-horizon mean-reversion signal is strong at 15-min and decays over an hour (why the 1h is weak).

---

## 5. Tradability Verdict

**NOT a standalone tradable alpha after 10 bps** (despite the excellent ranking) — the gross per-trade magnitude is only ~2–7 bps, below cost. Conviction × calibration grid (`scripts/analysis/wf_conviction_calibration.py`, 6-fold WF, both directions): no profitable region anywhere; best cell (top-1/p99 long) −2.7 bps (ns). See [[Clean Re-Analysis & OOS Validation]].

**USABLE as a LONG-side veto/quality filter** (`scripts/analysis/wf_veto_value.py`): on 1h long candidates, 15m agreement splits PASS 44.5% WR / +5.1 bps gross vs VETO 37.3% WR / +0.3 bps — separation **+4.8 bps / +7.2pp WR, p=0.015✷**. Short-side veto NOT significant (p=0.66). It improves trade *quality* (not standalone profitability); use as a long-side soft filter, not a hard gate.

---

## Backlinks
- [[Clean Re-Analysis & OOS Validation]] — full clean-data re-analysis, dual-TF, conviction grid, veto value.
- [[Model Card - v10 Native 1h]] — the weaker 1-hour sibling.
- [[Complete Edge Catalog]] · [[Prediction Bucket & Calibration Deep Dive]]
