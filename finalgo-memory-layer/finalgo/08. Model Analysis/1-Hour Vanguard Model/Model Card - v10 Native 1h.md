# Model Card — v10_native_1h

**Date:** June 7, 2026
**Type:** XGBoost `rank:pairwise` cross-sectional ranker (separate Long & Short models)
**Status:** Registered (research / production-faithful). NOT the active model (active = v8_upstox_3y). NOT tradable as standalone alpha — see Tradability Verdict.

> Definitive companion analysis: [[Clean Re-Analysis & OOS Validation]].

---

## 1. Training Profile

| Attribute | Value |
|---|---|
| Model ID | `v10_native_1h` |
| Data source | **Upstox V3 native hourly candles** (`HistoryV3Api`, `hours/1`) — no resampling |
| Data file | `data/ranking_data_upstox_1h_v3_3y.csv` |
| Collector | `scripts/collectors/collect_upstox_1h_v3.py` |
| Trainer | `scripts/training/train_ranking_clean.py --tf 1h_v3` |
| Total rows | 928,078 |
| Span | 2022-01 → 2026-06 (54 months) |
| Grid | 09:15-anchored IST; 5 intraday signals/day (09:15,10:15,11:15,12:15,13:15); 14:xx dropped (overnight-masked) |
| Forward target | `Next_Hour_Return`, session-masked (no overnight leak) |
| Features | 86 (same set as v8) |
| Trained at | 2026-06-07T18:18 |

**Hyperparameters:** `rank:pairwise`, eta 0.03, max_depth 5, subsample 0.8, colsample_bytree 0.8, alpha 1.0, lambda 2.0, min_child_weight 10, eval_metric ndcg@3, tree_method hist, device cuda.

---

## 2. Walk-Forward Performance (genuinely OOS per fold)

| Metric | Value |
|---|---|
| Avg Long Spearman IC | **0.0261** |
| Avg Short Spearman IC | **0.0245** |
| Avg Long Win Rate @ K=3 | 52.4% |
| Avg Short Win Rate @ K=3 | 53.6% |

### Fold-by-fold Spearman (9-fold WF in the standalone-1h test; 6-fold in the training metadata)

Training metadata folds (6):
| Fold | Long Rho | Short Rho |
|---|---|---|
| 1 | 0.0438 | 0.0372 |
| 2 | 0.0339 | 0.0357 |
| 3 | 0.0237 | 0.0317 |
| 4 | 0.0301 | 0.0201 |
| 5 | 0.0288 | 0.0288 |
| 6 | 0.0254 | 0.0229 |

**IC is decaying over time** (later folds weaker): the 9-fold standalone test (2023-08→2026-06) showed fold 1 Long 0.044 → fold 9 Long 0.012. The recent regime is hostile to the 1h.

**Context:** IC ~28% higher than the 15min→1h resample (`v9_clean_1h`, 0.0204) but **less than half** the 15m model (0.0586). The old contaminated v8 reported 0.046 — that was grid-distortion + overnight inflation.

---

## 3. OOS Diagnostics (Apr–Jun 2026, 220 queries, 37,840 rows)

| Metric | Long | Short |
|---|---|---|
| Prediction-Bucket Rho (global) | +0.61 | −0.41 |
| **Calibration Rho (per-query)** | **−0.14 (WEAK)** | **−0.39 (WEAK)** |
| Top-decile (D10) win rate | 52.1% | 55.9% |
| Top-decile edge per bar | +0.053% | +0.032% |
| MWU top-20% vs bottom-20% | p=2.9e-4, spread +0.014% | p=1.3e-4, spread +0.006% |

**Read:** the native 1h is a **poor cross-sectional ranker.** Per-query calibration ≈ 0 — within a timestamp it barely orders stocks monotonically. Only the extreme top decile carries a sliver of edge; middle deciles are noise/wrong-signed. MWU is "significant" only due to large N — the actual spread is ~0.6–1.4 bps.

Diagnostic plots (8-plot suite + dashboard): `data/model_analysis/v10_native_1h/`.

---

## 4. Feature Importance (gain)

**Long (top 10):** IBS 9.30, Lower_Shadow 3.54, Keltner_Width 2.71, Log_Return 2.60, Dist_Keltner_Lower 1.80, Stoch_K 1.61, RSI_lag3 1.54, Relative_Return 1.48, Dist_Donchian_Upper 1.42, Dollar_Volume 1.40.

**Short (top 10):** Keltner_Width 4.32, Donchian_Width 3.21, Dist_52W_High 3.18, HL_Range 2.77, Dist_Keltner_Lower 2.60, Relative_Volatility 2.40, Dist_EMA_24 1.95, IBS 1.84, Log_Return 1.78, Dist_Keltner_Upper 1.75.

Note: the 1h's importances are *flatter / more volatility-driven* than the 15m's (whose IBS+Buy_Pressure dominate massively) — consistent with the 1h's weaker, more diffuse signal.

---

## 5. Tradability Verdict (from holdout-validated walk-forward)

**NOT tradable as standalone alpha.** 9-fold WF (`scripts/analysis/wf_1h_base.py`, ~600K OOS rows, 2023-08→2026-06, 10 bps):
- Every conviction (top-1..top-10) both directions: significantly negative (−6 to −9 bps, 0/9 folds positive).
- Absolute rank-pct even p99 (top 1%): LONG −6.4✷✷✷, SHORT −7.6✷✷✷.
- Per-hour: every hour 09:15–13:15 significantly negative — **the old "2 PM crown jewel" was the overnight artifact and does not exist in clean data.**
- Gross edge only ~+2–3 bps, below the 10 bps cost.

**Production note:** to deploy, the live broker's 1h fetch must be switched to V3 `hours/1` to match this training (currently 30min→1h IST `:00` grid → train/serve skew).

---

## Backlinks
- [[Clean Re-Analysis & OOS Validation]] — full clean-data re-analysis, dual-TF, conviction grid, veto value.
- [[Model Card - v3 Clean 15min]] — the stronger sibling ranker.
