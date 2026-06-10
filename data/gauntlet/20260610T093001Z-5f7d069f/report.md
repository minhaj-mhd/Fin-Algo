# 🛡️ Validation Gauntlet Report: `v10_native_1h`

## 📌 Metadata
- **Run ID**: `20260610T093001Z-5f7d069f`
- **Evaluated At (UTC)**: `2026-06-10T09:33:33.754353+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `65acdc6424858d96bd1832310254c1c78a5d16ac`
- **Multiple Testing Context**: Prior runs for dataset family = `2`
- **Deflated t-Threshold**: `2.3940` (corrected for `3` total tests)


## 📊 Dataset Label Verification Stats
- **In-File Verified (INTRA) Rows**: 79.93%
- **Unverifiable (Missing Target Bar) Rows**: 0.04%
- **Boundary (Session Terminal) Rows**: 20.03%
- **Unverified Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.
- **Prefix Invariance Waiver Reason**: N/A


## ⚖️ Final Verdicts
- **LONG Side**: <span style='color:orange;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:orange;font-weight:bold'>FILTER_GRADE</span>


---

## 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2023-08, 2023-09 | +0.0428 | +0.0380 | 147 | 56 |
| 2 | 2023-12, 2024-01 | +0.0345 | +0.0378 | 3 | 3 |
| 3 | 2024-04, 2024-05 | +0.0217 | +0.0326 | 0 | 5 |
| 4 | 2024-08, 2024-09 | +0.0263 | +0.0193 | 26 | 9 |
| 5 | 2024-12, 2025-01 | +0.0248 | +0.0244 | 0 | 2 |
| 6 | 2025-04, 2025-05 | +0.0269 | +0.0220 | 0 | 14 |
| 7 | 2025-08, 2025-09 | +0.0198 | +0.0229 | 38 | 66 |
| 8 | 2025-12, 2026-01 | +0.0233 | +0.0201 | 3 | 52 |
| 9 | 2026-04, 2026-05 | +0.0071 | +0.0016 | 9 | 109 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`2.39`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 1858 | +5.00 | -1.00 | 53.9% | 47.1% | 48.1% | -0.58 |
| full_OOS | Top-1 | 6.0bps | SHORT | 1858 | +3.05 | -2.95 | 53.8% | 48.8% | 50.7% | -1.35 |
| full_OOS | Top-1 | 10.0bps | LONG | 1858 | +5.00 | -5.00 | 53.9% | 42.6% | 48.1% | -2.91 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 1858 | +3.05 | -6.95 | 53.8% | 46.2% | 50.7% | -3.17 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 5574 | +4.29 | -1.71 | 52.0% | 45.9% | 48.1% | -1.93 |
| full_OOS | Top-3 | 6.0bps | SHORT | 5574 | +2.26 | -3.74 | 54.2% | 49.0% | 50.7% | -3.12 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 5574 | +4.29 | -5.71 | 52.0% | 41.8% | 48.1% | -6.46 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 5574 | +2.26 | -7.74 | 54.2% | 46.1% | 50.7% | -6.46 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 1234 | +5.85 | -0.15 | 54.3% | 47.9% | 48.2% | -0.07 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1234 | +4.20 | -1.80 | 53.6% | 48.5% | 50.8% | -0.71 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1234 | +5.85 | -4.15 | 54.3% | 43.3% | 48.2% | -1.97 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1234 | +4.20 | -5.80 | 53.6% | 45.9% | 50.8% | -2.29 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3702 | +4.94 | -1.06 | 52.5% | 46.7% | 48.2% | -0.98 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3702 | +2.11 | -3.89 | 53.8% | 48.2% | 50.8% | -2.72 ✷ |
| recent_12mo | Top-3 | 10.0bps | LONG | 3702 | +4.94 | -5.06 | 52.5% | 42.8% | 48.2% | -4.67 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3702 | +2.11 | -7.89 | 53.8% | 45.2% | 50.8% | -5.52 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 1095 | 51.8% | +2.43 | 1095 | 54.1% | -4.23 |
| 10:15 | 1119 | 53.4% | -1.31 | 1119 | 54.2% | -4.32 |
| 11:15 | 1122 | 49.3% | -6.05 | 1122 | 52.3% | -4.40 |
| 12:15 | 1119 | 51.5% | -5.49 | 1119 | 54.5% | -3.93 |
| 13:15 | 1119 | 53.8% | +1.99 | 1119 | 56.0% | -1.84 |

