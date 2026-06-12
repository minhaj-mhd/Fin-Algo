---
title: "Validation Gauntlet Report: `daily_xgb"
type: report
status: active
model: "Gauntlet Reports"
verdict: DEAD
updated: 2026-06-12
tags: []
---
# 🛡️ Validation Gauntlet Report: `daily_xgb`

## 📌 Metadata
- **Run ID**: `20260610T102743Z-5f7d069f`
- **Evaluated At (UTC)**: `2026-06-10T10:29:15.245406+00:00`
- **Dataset Path**: `data/ranking_data_upstox_daily_5y.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `65acdc6424858d96bd1832310254c1c78a5d16ac`
- **Multiple Testing Context**: Prior runs for dataset family = `3`
- **Deflated t-Threshold**: `2.4977` (corrected for `4` total tests)


## 📊 Dataset Label Verification Stats
- **In-File Verified (INTRA) Rows**: 75.68%
- **Unverifiable (Missing Target Bar) Rows**: 0.00%
- **Boundary (Session Terminal) Rows**: 24.32%
- **Unverified Label Waiver Reason**: Daily close-to-close returns have no intraday target bars.
- **Prefix Invariance Waiver Reason**: N/A


## ⚖️ Final Verdicts
- **LONG Side**: <span style='color:red;font-weight:bold'>DEAD</span>
- **SHORT Side**: <span style='color:red;font-weight:bold'>DEAD</span>


---

## 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2022-12, 2023-01 | +0.0126 | +0.0214 | 8 | 47 |
| 2 | 2023-04, 2023-05 | +0.0244 | +0.0333 | 52 | 12 |
| 3 | 2023-08, 2023-09 | +0.0262 | +0.0239 | 21 | 95 |
| 4 | 2023-12, 2024-01 | +0.0379 | +0.0196 | 8 | 0 |
| 5 | 2024-04, 2024-05 | +0.0039 | -0.0255 | 5 | 13 |
| 6 | 2024-08, 2024-09 | -0.0095 | +0.0196 | 44 | 6 |
| 7 | 2024-12, 2025-01 | -0.0041 | +0.0037 | 12 | 47 |
| 8 | 2025-04, 2025-05 | +0.0479 | +0.0423 | 15 | 0 |
| 9 | 2025-08, 2025-09 | +0.0133 | +0.0379 | 177 | 5 |
| 10 | 2025-12, 2026-01 | +0.0541 | +0.0421 | 34 | 0 |
| 11 | 2026-04, 2026-05 | +0.0280 | -0.0024 | 3 | 18 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`2.50`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 443 | +4.84 | -1.16 | 51.2% | 49.7% | 50.9% | -0.11 |
| full_OOS | Top-1 | 6.0bps | SHORT | 443 | -6.17 | -12.17 | 52.1% | 51.0% | 48.8% | -0.96 |
| full_OOS | Top-1 | 10.0bps | LONG | 443 | +4.84 | -5.16 | 51.2% | 49.0% | 50.9% | -0.49 |
| full_OOS | Top-1 | 10.0bps | SHORT | 443 | -6.17 | -16.17 | 52.1% | 49.2% | 48.8% | -1.27 |
| full_OOS | Top-3 | 6.0bps | LONG | 1329 | +11.74 | +5.74 | 50.8% | 49.1% | 50.9% | 0.89 |
| full_OOS | Top-3 | 6.0bps | SHORT | 1329 | -3.09 | -9.09 | 50.6% | 49.4% | 48.8% | -1.3 |
| full_OOS | Top-3 | 10.0bps | LONG | 1329 | +11.74 | +1.74 | 50.8% | 47.7% | 50.9% | 0.27 |
| full_OOS | Top-3 | 10.0bps | SHORT | 1329 | -3.09 | -13.09 | 50.6% | 48.1% | 48.8% | -1.87 |
| recent_12mo | Top-1 | 6.0bps | LONG | 240 | -13.68 | -19.68 | 51.2% | 49.2% | 50.4% | -1.39 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 240 | +6.84 | +0.84 | 52.1% | 50.8% | 49.4% | 0.05 |
| recent_12mo | Top-1 | 10.0bps | LONG | 240 | -13.68 | -23.68 | 51.2% | 48.8% | 50.4% | -1.68 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 240 | +6.84 | -3.16 | 52.1% | 49.2% | 49.4% | -0.19 |
| recent_12mo | Top-3 | 6.0bps | LONG | 720 | +0.33 | -5.67 | 51.1% | 49.3% | 50.4% | -0.66 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 720 | +13.20 | +7.20 | 51.2% | 50.6% | 49.4% | 0.79 |
| recent_12mo | Top-3 | 10.0bps | LONG | 720 | +0.33 | -9.67 | 51.1% | 48.3% | 50.4% | -1.13 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 720 | +13.20 | +3.20 | 51.2% | 49.4% | 49.4% | 0.35 |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 00:00 | 1329 | 50.8% | +5.74 | 1329 | 50.6% | -9.09 |

