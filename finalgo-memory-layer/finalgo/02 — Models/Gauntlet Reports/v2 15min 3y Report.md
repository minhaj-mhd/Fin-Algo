---
title: "Validation Gauntlet Report: `v2_15min_3y"
type: report
status: active
model: "Gauntlet Reports"
verdict: FILTER_GRADE
updated: 2026-06-12
tags: []
---
# 🛡️ Validation Gauntlet Report: `v2_15min_3y`

## 📌 Metadata
- **Run ID**: `20260610T173707Z-d795438c`
- **Evaluated At (UTC)**: `2026-06-10T18:19:42.297583+00:00`
- **Dataset Path**: `data/ranking_data_upstox_15min_3y_clean.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `4`
- **Deflated t-Threshold**: `2.5758` (corrected for `5` total tests)


## 📊 Dataset Label Verification Stats
- **In-File Verified (INTRA) Rows**: 95.81%
- **Unverifiable (Missing Target Bar) Rows**: 0.01%
- **Boundary (Session Terminal) Rows**: 4.18%
- **Unverified Label Waiver Reason**: 15m older intraday parquet source files not available in raw history directory.
- **Prefix Invariance Waiver Reason**: N/A


## ⚖️ Final Verdicts
- **LONG Side**: <span style='color:orange;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:orange;font-weight:bold'>FILTER_GRADE</span>


---

## 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2024-09, 2024-10 | +0.0617 | +0.0625 | 284 | 26 |
| 2 | 2024-11, 2024-12 | +0.0601 | +0.0625 | 7 | 36 |
| 3 | 2025-01, 2025-02 | +0.0539 | +0.0548 | 137 | 2 |
| 4 | 2025-03, 2025-04 | +0.0560 | +0.0581 | 25 | 59 |
| 5 | 2025-05, 2025-06 | +0.0691 | +0.0687 | 1 | 3 |
| 6 | 2025-07, 2025-08 | +0.0631 | +0.0605 | 254 | 13 |
| 7 | 2025-09, 2025-10 | +0.0644 | +0.0676 | 10 | 49 |
| 8 | 2025-11, 2025-12 | +0.0622 | +0.0650 | 32 | 54 |
| 9 | 2026-01, 2026-02 | +0.0520 | +0.0493 | 74 | 4 |
| 10 | 2026-03, 2026-04 | +0.0576 | +0.0532 | 127 | 5 |
| 11 | 2026-05, 2026-06 | +0.0529 | +0.0529 | 31 | 2 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`2.58`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 9448 | +3.38 | -2.62 | 55.8% | 44.5% | 47.6% | -6.52 ✷ |
| full_OOS | Top-1 | 6.0bps | SHORT | 9448 | +3.47 | -2.53 | 56.9% | 48.0% | 50.2% | -4.29 ✷ |
| full_OOS | Top-1 | 10.0bps | LONG | 9448 | +3.38 | -6.62 | 55.8% | 37.2% | 47.6% | -16.47 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 9448 | +3.47 | -6.53 | 56.9% | 41.6% | 50.2% | -11.07 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 28344 | +3.25 | -2.75 | 55.3% | 43.6% | 47.6% | -12.73 ✷ |
| full_OOS | Top-3 | 6.0bps | SHORT | 28344 | +2.94 | -3.06 | 56.8% | 46.9% | 50.2% | -10.58 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 28344 | +3.25 | -6.75 | 55.3% | 36.2% | 47.6% | -31.28 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 28344 | +2.94 | -7.06 | 56.8% | 40.3% | 50.2% | -24.42 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 4989 | +3.51 | -2.49 | 55.9% | 43.8% | 47.5% | -4.92 ✷ |
| recent_12mo | Top-1 | 6.0bps | SHORT | 4989 | +3.12 | -2.88 | 57.7% | 47.9% | 50.0% | -3.87 ✷ |
| recent_12mo | Top-1 | 10.0bps | LONG | 4989 | +3.51 | -6.49 | 55.9% | 36.3% | 47.5% | -12.84 ✷ |
| recent_12mo | Top-1 | 10.0bps | SHORT | 4989 | +3.12 | -6.88 | 57.7% | 40.4% | 50.0% | -9.23 ✷ |
| recent_12mo | Top-3 | 6.0bps | LONG | 14967 | +3.28 | -2.72 | 55.6% | 42.9% | 47.5% | -10.21 ✷ |
| recent_12mo | Top-3 | 6.0bps | SHORT | 14967 | +2.88 | -3.12 | 57.2% | 46.6% | 50.0% | -8.58 ✷ |
| recent_12mo | Top-3 | 10.0bps | LONG | 14967 | +3.28 | -6.72 | 55.6% | 34.9% | 47.5% | -25.24 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 14967 | +2.88 | -7.12 | 57.2% | 39.0% | 50.0% | -19.6 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 1149 | 50.7% | -4.06 | 1149 | 55.1% | -0.68 |
| 09:30 | 1182 | 52.9% | -4.98 | 1182 | 54.6% | -1.73 |
| 09:45 | 1182 | 52.7% | -2.57 | 1182 | 53.6% | -5.63 |
| 10:00 | 1182 | 57.2% | -0.34 | 1182 | 52.8% | -5.24 |
| 10:15 | 1182 | 55.2% | -1.91 | 1182 | 58.0% | -3.03 |
| 10:30 | 1182 | 52.1% | -5.24 | 1182 | 56.0% | -1.87 |
| 10:45 | 1182 | 56.0% | -2.19 | 1182 | 58.4% | -4.42 |
| 11:00 | 1182 | 52.8% | -3.92 | 1182 | 56.3% | -3.73 |
| 11:15 | 1182 | 53.5% | -3.63 | 1182 | 58.1% | -3.30 |
| 11:30 | 1182 | 51.4% | -5.50 | 1182 | 56.3% | -3.73 |
| 11:45 | 1182 | 56.9% | -2.26 | 1182 | 56.1% | -5.05 |
| 12:00 | 1182 | 55.8% | -4.43 | 1182 | 59.5% | -2.61 |
| 12:15 | 1182 | 54.3% | -1.36 | 1182 | 56.3% | -5.20 |
| 12:30 | 1182 | 54.7% | -4.35 | 1182 | 58.0% | -2.41 |
| 12:45 | 1182 | 55.2% | -3.59 | 1182 | 55.3% | -7.06 |
| 13:00 | 1182 | 53.2% | -5.17 | 1182 | 59.1% | -1.73 |
| 13:15 | 1182 | 57.4% | -2.41 | 1182 | 56.9% | -2.35 |
| 13:30 | 1182 | 57.6% | -2.44 | 1182 | 56.3% | -4.27 |
| 13:45 | 1185 | 57.0% | -3.72 | 1185 | 56.8% | -2.43 |
| 14:00 | 1185 | 57.0% | -2.35 | 1185 | 58.6% | -2.91 |
| 14:15 | 1185 | 58.4% | -1.92 | 1185 | 56.1% | -4.97 |
| 14:30 | 1182 | 60.7% | -0.13 | 1182 | 55.8% | -2.93 |
| 14:45 | 1182 | 57.4% | +0.46 | 1182 | 55.8% | -5.12 |
| 15:00 | 1182 | 56.9% | +2.05 | 1182 | 62.4% | +9.14 |

