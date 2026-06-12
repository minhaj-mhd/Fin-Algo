---
title: "Validation Gauntlet Report: `daily_macro_v2"
type: report
status: active
model: "Gauntlet Reports"
verdict: FILTER_GRADE
updated: 2026-06-12
tags: []
---
# 🛡️ Validation Gauntlet Report: `daily_macro_v2`

## 📌 Metadata
- **Run ID**: `20260610T135608Z-5f7d069f`
- **Evaluated At (UTC)**: `2026-06-10T13:58:46.931541+00:00`
- **Dataset Path**: `data/ranking_data_daily_macro_v2.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `3`
- **Deflated t-Threshold**: `2.4977` (corrected for `4` total tests)


## 📊 Dataset Label Verification Stats
- **In-File Verified (INTRA) Rows**: 0.00%
- **Unverifiable (Missing Target Bar) Rows**: 0.00%
- **Boundary (Session Terminal) Rows**: 100.00%
- **Unverified Label Waiver Reason**: Daily close-to-close returns have no intraday target bars.
- **Prefix Invariance Waiver Reason**: Daily macro dataset contains cross-asset and global features that cannot be computed per-ticker in isolation.


## ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>TRIGGER_GRADE</span>
- **SHORT Side**: <span style='color:orange;font-weight:bold'>FILTER_GRADE</span>


---

## 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2018-01, 2018-02 | +0.0612 | +0.0904 | 2 | 15 |
| 2 | 2018-05, 2018-06 | +0.0526 | +0.0624 | 10 | 6 |
| 3 | 2018-09, 2018-10 | +0.0314 | +0.0715 | 0 | 15 |
| 4 | 2019-01, 2019-02 | +0.0166 | -0.0037 | 25 | 7 |
| 5 | 2019-05, 2019-06 | +0.0518 | +0.0498 | 49 | 2 |
| 6 | 2019-09, 2019-10 | +0.0403 | +0.0325 | 5 | 56 |
| 7 | 2020-01, 2020-02 | +0.0210 | -0.0064 | 2 | 1 |
| 8 | 2020-05, 2020-06 | +0.0349 | +0.0283 | 144 | 3 |
| 9 | 2020-09, 2020-10 | +0.0405 | +0.0474 | 11 | 9 |
| 10 | 2021-01, 2021-02 | +0.0146 | -0.0009 | 59 | 11 |
| 11 | 2021-05, 2021-06 | +0.0272 | +0.0950 | 22 | 0 |
| 12 | 2021-09, 2021-10 | +0.0364 | +0.0497 | 1 | 0 |
| 13 | 2022-01, 2022-02 | -0.0047 | -0.0226 | 1 | 88 |
| 14 | 2022-05, 2022-06 | +0.0016 | +0.0468 | 1 | 4 |
| 15 | 2022-09, 2022-10 | +0.0204 | +0.0348 | 1 | 41 |
| 16 | 2023-01, 2023-02 | -0.0049 | +0.0436 | 1 | 7 |
| 17 | 2023-05, 2023-06 | +0.0656 | +0.0640 | 6 | 76 |
| 18 | 2023-09, 2023-10 | +0.0613 | +0.0718 | 2 | 0 |
| 19 | 2024-01, 2024-02 | +0.0358 | +0.0507 | 26 | 25 |
| 20 | 2024-05, 2024-06 | +0.0678 | +0.0375 | 0 | 33 |
| 21 | 2024-09, 2024-10 | +0.0261 | +0.0346 | 30 | 7 |
| 22 | 2025-01, 2025-02 | +0.0167 | +0.0590 | 74 | 9 |
| 23 | 2025-05, 2025-06 | +0.0010 | +0.0105 | 13 | 1 |
| 24 | 2025-09, 2025-10 | +0.0428 | +0.0659 | 6 | 3 |
| 25 | 2026-01, 2026-02 | +0.0225 | +0.0337 | 28 | 11 |
| 26 | 2026-05, 2026-06 | +0.0747 | +0.0620 | 9 | 0 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`2.50`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 985 | +22.67 | +16.67 | 52.1% | 50.9% | 49.5% | 0.95 |
| full_OOS | Top-1 | 6.0bps | SHORT | 985 | +20.89 | +14.89 | 54.5% | 54.2% | 50.2% | 0.94 |
| full_OOS | Top-1 | 10.0bps | LONG | 985 | +22.67 | +12.67 | 52.1% | 50.4% | 49.5% | 0.72 |
| full_OOS | Top-1 | 10.0bps | SHORT | 985 | +20.89 | +10.89 | 54.5% | 54.1% | 50.2% | 0.69 |
| full_OOS | Top-3 | 6.0bps | LONG | 2955 | +34.22 | +28.22 | 52.8% | 51.8% | 49.5% | 3.13 ✷ |
| full_OOS | Top-3 | 6.0bps | SHORT | 2955 | +19.10 | +13.10 | 55.5% | 54.8% | 50.2% | 1.49 |
| full_OOS | Top-3 | 10.0bps | LONG | 2955 | +34.22 | +24.22 | 52.8% | 51.5% | 49.5% | 2.69 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 2955 | +19.10 | +9.10 | 55.5% | 54.4% | 50.2% | 1.04 |
| recent_24mo | Top-1 | 6.0bps | LONG | 448 | +33.69 | +27.69 | 53.8% | 51.8% | 49.7% | 0.96 |
| recent_24mo | Top-1 | 6.0bps | SHORT | 448 | +26.51 | +20.51 | 52.9% | 52.7% | 50.1% | 0.91 |
| recent_24mo | Top-1 | 10.0bps | LONG | 448 | +33.69 | +23.69 | 53.8% | 51.1% | 49.7% | 0.82 |
| recent_24mo | Top-1 | 10.0bps | SHORT | 448 | +26.51 | +16.51 | 52.9% | 52.7% | 50.1% | 0.73 |
| recent_24mo | Top-3 | 6.0bps | LONG | 1344 | +28.61 | +22.61 | 53.9% | 52.5% | 49.7% | 1.65 |
| recent_24mo | Top-3 | 6.0bps | SHORT | 1344 | +32.77 | +26.77 | 55.6% | 54.8% | 50.1% | 2.15 |
| recent_24mo | Top-3 | 10.0bps | LONG | 1344 | +28.61 | +18.61 | 53.9% | 52.2% | 49.7% | 1.36 |
| recent_24mo | Top-3 | 10.0bps | SHORT | 1344 | +32.77 | +22.77 | 55.6% | 54.7% | 50.1% | 1.83 |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 00:00 | 2955 | 52.8% | +28.22 | 2955 | 55.5% | +13.10 |

