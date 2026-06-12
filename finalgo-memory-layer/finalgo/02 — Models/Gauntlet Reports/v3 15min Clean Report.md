---
title: "Validation Gauntlet Report: `v3_15min_clean"
type: report
status: active
model: "Gauntlet Reports"
verdict: FILTER_GRADE
updated: 2026-06-12
tags: []
---
# 🛡️ Validation Gauntlet Report: `v3_15min_clean`

## 📌 Metadata
- **Run ID**: `20260610T113721Z-5f7d069f`
- **Evaluated At (UTC)**: `2026-06-10T12:01:59.662669+00:00`
- **Dataset Path**: `data/ranking_data_upstox_15min_3y_clean.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `3`
- **Deflated t-Threshold**: `2.4977` (corrected for `4` total tests)


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
| 2 | 2025-01, 2025-02 | +0.0539 | +0.0548 | 137 | 2 |
| 3 | 2025-05, 2025-06 | +0.0691 | +0.0687 | 1 | 3 |
| 4 | 2025-09, 2025-10 | +0.0644 | +0.0676 | 10 | 49 |
| 5 | 2026-01, 2026-02 | +0.0520 | +0.0493 | 74 | 4 |
| 6 | 2026-05, 2026-06 | +0.0529 | +0.0529 | 31 | 2 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`2.50`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 5253 | +3.52 | -2.48 | 54.9% | 43.9% | 47.5% | -4.48 ✷ |
| full_OOS | Top-1 | 6.0bps | SHORT | 5253 | +3.74 | -2.26 | 57.0% | 48.8% | 50.3% | -2.79 ✷ |
| full_OOS | Top-1 | 10.0bps | LONG | 5253 | +3.52 | -6.48 | 54.9% | 36.7% | 47.5% | -11.7 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 5253 | +3.74 | -6.26 | 57.0% | 42.7% | 50.3% | -7.73 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 15759 | +3.28 | -2.72 | 54.9% | 43.4% | 47.5% | -9.21 ✷ |
| full_OOS | Top-3 | 6.0bps | SHORT | 15759 | +3.27 | -2.73 | 57.0% | 47.4% | 50.3% | -6.9 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 15759 | +3.28 | -6.72 | 54.9% | 36.2% | 47.5% | -22.78 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 15759 | +3.27 | -6.73 | 57.0% | 41.0% | 50.3% | -17.03 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 5253 | +3.52 | -2.48 | 54.9% | 43.9% | 47.5% | -4.48 ✷ |
| recent_12mo | Top-1 | 6.0bps | SHORT | 5253 | +3.74 | -2.26 | 57.0% | 48.8% | 50.3% | -2.79 ✷ |
| recent_12mo | Top-1 | 10.0bps | LONG | 5253 | +3.52 | -6.48 | 54.9% | 36.7% | 47.5% | -11.7 ✷ |
| recent_12mo | Top-1 | 10.0bps | SHORT | 5253 | +3.74 | -6.26 | 57.0% | 42.7% | 50.3% | -7.73 ✷ |
| recent_12mo | Top-3 | 6.0bps | LONG | 15759 | +3.28 | -2.72 | 54.9% | 43.4% | 47.5% | -9.21 ✷ |
| recent_12mo | Top-3 | 6.0bps | SHORT | 15759 | +3.27 | -2.73 | 57.0% | 47.4% | 50.3% | -6.9 ✷ |
| recent_12mo | Top-3 | 10.0bps | LONG | 15759 | +3.28 | -6.72 | 54.9% | 36.2% | 47.5% | -22.78 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 15759 | +3.27 | -6.73 | 57.0% | 41.0% | 50.3% | -17.03 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 639 | 51.2% | -3.62 | 639 | 53.4% | -1.44 |
| 09:30 | 657 | 53.4% | -3.10 | 657 | 53.4% | -4.10 |
| 09:45 | 657 | 52.8% | -0.98 | 657 | 54.3% | -4.91 |
| 10:00 | 657 | 54.8% | -0.36 | 657 | 54.2% | -4.84 |
| 10:15 | 657 | 56.9% | -0.92 | 657 | 57.1% | -2.87 |
| 10:30 | 657 | 51.3% | -4.51 | 657 | 56.3% | -1.33 |
| 10:45 | 657 | 58.3% | -0.89 | 657 | 56.6% | -5.73 |
| 11:00 | 657 | 51.4% | -4.69 | 657 | 58.4% | -1.87 |
| 11:15 | 657 | 53.6% | -3.75 | 657 | 60.1% | -1.49 |
| 11:30 | 657 | 49.8% | -7.25 | 657 | 54.3% | -4.24 |
| 11:45 | 657 | 55.9% | -3.52 | 657 | 57.8% | -3.58 |
| 12:00 | 657 | 53.1% | -6.35 | 657 | 61.5% | -2.26 |
| 12:15 | 657 | 53.9% | +0.04 | 657 | 56.3% | -4.58 |
| 12:30 | 657 | 54.0% | -5.20 | 657 | 59.1% | -2.81 |
| 12:45 | 657 | 53.0% | -3.67 | 657 | 54.3% | -7.01 |
| 13:00 | 657 | 52.5% | -5.68 | 657 | 61.3% | +1.47 |
| 13:15 | 657 | 56.3% | -2.52 | 657 | 57.4% | -2.32 |
| 13:30 | 657 | 55.3% | -3.45 | 657 | 56.5% | -3.35 |
| 13:45 | 660 | 56.2% | -4.82 | 660 | 57.0% | -2.57 |
| 14:00 | 660 | 57.6% | -2.20 | 660 | 60.2% | -1.59 |
| 14:15 | 660 | 61.4% | -0.34 | 660 | 57.9% | -3.91 |
| 14:30 | 657 | 61.8% | +0.43 | 657 | 53.9% | -3.19 |
| 14:45 | 657 | 56.6% | +0.27 | 657 | 56.5% | -4.81 |
| 15:00 | 657 | 56.9% | +1.85 | 657 | 60.7% | +7.89 |

