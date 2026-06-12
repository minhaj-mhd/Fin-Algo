---
title: "Validation Gauntlet Report: `daily_macro_v3"
type: report
status: active
model: "Gauntlet Reports"
verdict: DEAD
updated: 2026-06-12
tags: []
---
# 🛡️ Validation Gauntlet Report: `daily_macro_v3`

## 📌 Metadata
- **Run ID**: `20260610T144343Z-5f7d069f`
- **Evaluated At (UTC)**: `2026-06-10T14:46:16.338856+00:00`
- **Dataset Path**: `data/ranking_data_daily_macro_v3.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `1`
- **Deflated t-Threshold**: `2.2414` (corrected for `2` total tests)


## 📊 Dataset Label Verification Stats
- **In-File Verified (INTRA) Rows**: 0.00%
- **Unverifiable (Missing Target Bar) Rows**: 0.00%
- **Boundary (Session Terminal) Rows**: 100.00%
- **Unverified Label Waiver Reason**: Daily close-to-close returns have no intraday target bars.
- **Prefix Invariance Waiver Reason**: Daily macro dataset contains cross-asset and global features that cannot be computed per-ticker in isolation.


## ⚖️ Final Verdicts
- **LONG Side**: <span style='color:red;font-weight:bold'>DEAD</span>
- **SHORT Side**: <span style='color:orange;font-weight:bold'>FILTER_GRADE</span>


---

## 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2018-01, 2018-02 | +0.0679 | +0.0677 | 23 | 35 |
| 2 | 2018-05, 2018-06 | +0.0513 | +0.0591 | 5 | 3 |
| 3 | 2018-09, 2018-10 | +0.0542 | +0.0511 | 2 | 15 |
| 4 | 2019-01, 2019-02 | +0.0210 | +0.0081 | 4 | 6 |
| 5 | 2019-05, 2019-06 | +0.0462 | +0.0450 | 144 | 0 |
| 6 | 2019-09, 2019-10 | +0.0455 | +0.0419 | 4 | 7 |
| 7 | 2020-01, 2020-02 | -0.0269 | -0.0041 | 47 | 0 |
| 8 | 2020-05, 2020-06 | +0.0251 | +0.0448 | 37 | 46 |
| 9 | 2020-09, 2020-10 | +0.0506 | +0.0632 | 83 | 10 |
| 10 | 2021-01, 2021-02 | +0.0318 | +0.0178 | 41 | 13 |
| 11 | 2021-05, 2021-06 | +0.0433 | +0.0688 | 4 | 29 |
| 12 | 2021-09, 2021-10 | +0.0242 | +0.0441 | 0 | 3 |
| 13 | 2022-01, 2022-02 | -0.0041 | +0.0003 | 33 | 7 |
| 14 | 2022-05, 2022-06 | +0.0389 | +0.0452 | 3 | 81 |
| 15 | 2022-09, 2022-10 | +0.0064 | +0.0253 | 107 | 1 |
| 16 | 2023-01, 2023-02 | +0.0147 | +0.0287 | 9 | 49 |
| 17 | 2023-05, 2023-06 | +0.0466 | +0.0379 | 9 | 60 |
| 18 | 2023-09, 2023-10 | +0.0785 | +0.0771 | 17 | 95 |
| 19 | 2024-01, 2024-02 | +0.0282 | +0.0303 | 15 | 56 |
| 20 | 2024-05, 2024-06 | +0.0264 | +0.0355 | 13 | 2 |
| 21 | 2024-09, 2024-10 | +0.0176 | +0.0179 | 23 | 20 |
| 22 | 2025-01, 2025-02 | +0.0021 | +0.0210 | 65 | 8 |
| 23 | 2025-05, 2025-06 | +0.0243 | +0.0281 | 76 | 0 |
| 24 | 2025-09, 2025-10 | +0.0336 | +0.0513 | 3 | 0 |
| 25 | 2026-01, 2026-02 | +0.0155 | +0.0217 | 2 | 18 |
| 26 | 2026-05, 2026-06 | +0.0513 | +0.0429 | 0 | 61 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`2.24`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 1039 | +14.51 | +8.51 | 50.1% | 49.1% | 49.0% | 0.97 |
| full_OOS | Top-1 | 6.0bps | SHORT | 1039 | +15.67 | +9.67 | 54.3% | 53.5% | 50.6% | 1.08 |
| full_OOS | Top-1 | 10.0bps | LONG | 1039 | +14.51 | +4.51 | 50.1% | 47.7% | 49.0% | 0.51 |
| full_OOS | Top-1 | 10.0bps | SHORT | 1039 | +15.67 | +5.67 | 54.3% | 52.7% | 50.6% | 0.63 |
| full_OOS | Top-3 | 6.0bps | LONG | 3117 | +15.38 | +9.38 | 51.4% | 50.3% | 49.0% | 2.0 |
| full_OOS | Top-3 | 6.0bps | SHORT | 3117 | +15.67 | +9.67 | 53.9% | 52.8% | 50.6% | 1.84 |
| full_OOS | Top-3 | 10.0bps | LONG | 3117 | +15.38 | +5.38 | 51.4% | 49.4% | 49.0% | 1.15 |
| full_OOS | Top-3 | 10.0bps | SHORT | 3117 | +15.67 | +5.67 | 53.9% | 52.2% | 50.6% | 1.08 |
| recent_24mo | Top-1 | 6.0bps | LONG | 474 | -9.80 | -15.80 | 47.9% | 46.4% | 49.4% | -1.51 |
| recent_24mo | Top-1 | 6.0bps | SHORT | 474 | +14.45 | +8.45 | 52.5% | 51.5% | 50.3% | 0.65 |
| recent_24mo | Top-1 | 10.0bps | LONG | 474 | -9.80 | -19.80 | 47.9% | 44.5% | 49.4% | -1.89 |
| recent_24mo | Top-1 | 10.0bps | SHORT | 474 | +14.45 | +4.45 | 52.5% | 50.6% | 50.3% | 0.34 |
| recent_24mo | Top-3 | 6.0bps | LONG | 1422 | +6.63 | +0.63 | 51.3% | 50.1% | 49.4% | 0.1 |
| recent_24mo | Top-3 | 6.0bps | SHORT | 1422 | +17.41 | +11.41 | 52.7% | 51.6% | 50.3% | 1.55 |
| recent_24mo | Top-3 | 10.0bps | LONG | 1422 | +6.63 | -3.37 | 51.3% | 48.7% | 49.4% | -0.55 |
| recent_24mo | Top-3 | 10.0bps | SHORT | 1422 | +17.41 | +7.41 | 52.7% | 51.0% | 50.3% | 1.01 |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 00:00 | 3117 | 51.4% | +9.38 | 3117 | 53.9% | +9.67 |

