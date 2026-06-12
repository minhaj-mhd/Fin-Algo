---
title: "Validation Gauntlet Report: `v18_random_forest_1h"
type: report
status: active
model: "Gauntlet Reports"
verdict: DEAD
updated: 2026-06-12
tags: []
---
# 🛡️ Validation Gauntlet Report: `v18_random_forest_1h`

## 📌 Metadata
- **Run ID**: `20260610T124108Z-5f7d069f`
- **Evaluated At (UTC)**: `2026-06-10T12:59:53.613982+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Model Adapter**: `xgb_binary`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `23`
- **Deflated t-Threshold**: `3.0781` (corrected for `24` total tests)


## 📊 Dataset Label Verification Stats
- **In-File Verified (INTRA) Rows**: 79.93%
- **Unverifiable (Missing Target Bar) Rows**: 0.04%
- **Boundary (Session Terminal) Rows**: 20.03%
- **Unverified Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.
- **Prefix Invariance Waiver Reason**: N/A


## ⚖️ Final Verdicts
- **LONG Side**: <span style='color:red;font-weight:bold'>DEAD</span>
- **SHORT Side**: <span style='color:orange;font-weight:bold'>FILTER_GRADE</span>


---

## 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2023-08, 2023-09 | +0.0094 | +0.0116 | 0 | 0 |
| 2 | 2023-12, 2024-01 | +0.0165 | +0.0192 | 0 | 0 |
| 3 | 2024-04, 2024-05 | +0.0197 | +0.0211 | 0 | 0 |
| 4 | 2024-08, 2024-09 | +0.0173 | +0.0188 | 0 | 0 |
| 5 | 2024-12, 2025-01 | +0.0171 | +0.0150 | 0 | 0 |
| 6 | 2025-04, 2025-05 | +0.0115 | +0.0227 | 0 | 0 |
| 7 | 2025-08, 2025-09 | +0.0034 | +0.0113 | 0 | 0 |
| 8 | 2025-12, 2026-01 | +0.0064 | +0.0020 | 0 | 0 |
| 9 | 2026-04, 2026-05 | +0.0178 | +0.0100 | 0 | 0 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`3.08`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 1858 | +0.17 | -5.83 | 49.2% | 42.2% | 48.1% | -4.84 ✷ |
| full_OOS | Top-1 | 6.0bps | SHORT | 1858 | +0.43 | -5.57 | 53.8% | 47.4% | 50.7% | -3.44 ✷ |
| full_OOS | Top-1 | 10.0bps | LONG | 1858 | +0.17 | -9.83 | 49.2% | 38.3% | 48.1% | -8.15 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 1858 | +0.43 | -9.57 | 53.8% | 43.1% | 50.7% | -5.91 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 5574 | +0.11 | -5.89 | 48.7% | 42.5% | 48.1% | -8.23 ✷ |
| full_OOS | Top-3 | 6.0bps | SHORT | 5574 | -0.07 | -6.07 | 52.5% | 46.4% | 50.7% | -6.96 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 5574 | +0.11 | -9.89 | 48.7% | 38.6% | 48.1% | -13.82 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 5574 | -0.07 | -10.07 | 52.5% | 41.8% | 50.7% | -11.55 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 1234 | +1.59 | -4.41 | 50.4% | 43.3% | 48.2% | -3.18 ✷ |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1234 | +1.16 | -4.84 | 54.9% | 49.0% | 50.8% | -2.41 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1234 | +1.59 | -8.41 | 50.4% | 39.0% | 48.2% | -6.07 ✷ |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1234 | +1.16 | -8.84 | 54.9% | 44.4% | 50.8% | -4.39 ✷ |
| recent_12mo | Top-3 | 6.0bps | LONG | 3702 | +1.25 | -4.75 | 48.9% | 42.6% | 48.2% | -5.63 ✷ |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3702 | +0.21 | -5.79 | 52.9% | 46.8% | 50.8% | -5.38 ✷ |
| recent_12mo | Top-3 | 10.0bps | LONG | 3702 | +1.25 | -8.75 | 48.9% | 38.4% | 48.2% | -10.36 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3702 | +0.21 | -9.79 | 52.9% | 42.2% | 50.8% | -9.09 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 1095 | 50.0% | -3.62 | 1095 | 51.4% | -7.27 |
| 10:15 | 1119 | 49.7% | -3.49 | 1119 | 51.2% | -6.58 |
| 11:15 | 1122 | 48.8% | -5.65 | 1122 | 51.4% | -7.17 |
| 12:15 | 1119 | 47.1% | -9.11 | 1119 | 55.9% | -2.47 |
| 13:15 | 1119 | 48.0% | -7.51 | 1119 | 52.5% | -6.90 |

