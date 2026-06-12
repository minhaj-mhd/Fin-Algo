---
title: "Validation Gauntlet Report: `v13_ndcg_raw_1h"
type: report
status: active
model: "Gauntlet Reports"
verdict: DEAD
updated: 2026-06-12
tags: []
---
# 🛡️ Validation Gauntlet Report: `v13_ndcg_raw_1h`

## 📌 Metadata
- **Run ID**: `20260610T185246Z-d795438c`
- **Evaluated At (UTC)**: `2026-06-10T19:02:39.731765+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `36`
- **Deflated t-Threshold**: `3.2048` (corrected for `37` total tests)


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
| 1 | 2023-08, 2023-09 | +0.0285 | +0.0337 | 47 | 140 |
| 2 | 2023-10, 2023-11 | +0.0155 | +0.0338 | 13 | 23 |
| 3 | 2023-12, 2024-01 | +0.0108 | +0.0306 | 6 | 4 |
| 4 | 2024-02, 2024-03 | +0.0245 | +0.0346 | 11 | 35 |
| 5 | 2024-04, 2024-05 | +0.0198 | +0.0260 | 11 | 6 |
| 6 | 2024-06, 2024-07 | +0.0157 | +0.0278 | 35 | 9 |
| 7 | 2024-08, 2024-09 | +0.0216 | +0.0161 | 0 | 48 |
| 8 | 2024-10, 2024-11 | +0.0139 | +0.0127 | 20 | 14 |
| 9 | 2024-12, 2025-01 | +0.0156 | +0.0218 | 20 | 12 |
| 10 | 2025-02, 2025-03 | +0.0059 | +0.0202 | 25 | 1 |
| 11 | 2025-04, 2025-05 | +0.0185 | +0.0172 | 1 | 22 |
| 12 | 2025-06, 2025-07 | +0.0169 | +0.0137 | 90 | 8 |
| 13 | 2025-08, 2025-09 | +0.0053 | +0.0149 | 2 | 75 |
| 14 | 2025-10, 2025-11 | +0.0341 | +0.0071 | 0 | 26 |
| 15 | 2025-12, 2026-01 | +0.0188 | +0.0121 | 3 | 3 |
| 16 | 2026-02, 2026-03 | +0.0017 | +0.0112 | 7 | 84 |
| 17 | 2026-04, 2026-05 | +0.0047 | +0.0027 | 108 | 12 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`3.20`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +2.74 | -3.26 | 50.7% | 46.4% | 48.3% | -2.1 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +3.11 | -2.89 | 54.9% | 50.5% | 50.5% | -1.68 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +2.74 | -7.26 | 50.7% | 42.5% | 48.3% | -4.68 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +3.11 | -6.89 | 54.9% | 47.8% | 50.5% | -4.0 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +1.86 | -4.14 | 50.6% | 45.3% | 48.3% | -5.17 ✷ |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +1.92 | -4.08 | 53.5% | 49.0% | 50.5% | -4.29 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +1.86 | -8.14 | 50.6% | 41.7% | 48.3% | -10.15 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +1.92 | -8.08 | 53.5% | 45.9% | 50.5% | -8.51 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +2.91 | -3.09 | 51.1% | 46.1% | 48.4% | -1.49 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +8.95 | +2.95 | 56.9% | 51.9% | 50.4% | 1.2 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +2.91 | -7.09 | 51.1% | 41.6% | 48.4% | -3.43 ✷ |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +8.95 | -1.05 | 56.9% | 49.1% | 50.4% | -0.43 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +1.04 | -4.96 | 49.8% | 44.5% | 48.4% | -4.66 ✷ |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +2.75 | -3.25 | 53.9% | 48.8% | 50.4% | -2.23 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +1.04 | -8.96 | 49.8% | 40.8% | 48.4% | -8.41 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +2.75 | -7.25 | 53.9% | 45.6% | 50.4% | -4.97 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 51.1% | -2.25 | 2040 | 52.1% | -8.91 |
| 10:15 | 2085 | 48.6% | -6.99 | 2085 | 54.2% | +0.19 |
| 11:15 | 2091 | 49.1% | -7.77 | 2091 | 54.5% | -6.06 |
| 12:15 | 2085 | 50.8% | -3.40 | 2085 | 53.1% | -3.33 |
| 13:15 | 2085 | 53.1% | -0.23 | 2085 | 53.4% | -2.38 |

