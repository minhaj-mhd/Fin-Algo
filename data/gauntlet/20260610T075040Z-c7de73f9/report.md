# 🛡️ Validation Gauntlet Report: `v10_native_1h`

## 📌 Metadata
- **Run ID**: `20260610T075040Z-c7de73f9`
- **Evaluated At (UTC)**: `2026-06-10T07:50:40.955105+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `93cc34f74d4b770dc6eac29ab6206c84b2434538`
- **Multiple Testing Context**: Prior runs for dataset family = `1`
- **Deflated t-Threshold**: `2.2414` (corrected for `2` total tests)

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
> Deflated t-threshold is **`2.24`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 1858 | +5.00 | -1.00 | 53.9% | 47.1% | -0.58 |
| full_OOS | Top-1 | 6.0bps | SHORT | 1858 | +3.05 | -2.95 | 53.8% | 48.8% | -1.35 |
| full_OOS | Top-1 | 10.0bps | LONG | 1858 | +5.00 | -5.00 | 53.9% | 42.6% | -2.91 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 1858 | +3.05 | -6.95 | 53.8% | 46.2% | -3.17 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 5574 | +4.29 | -1.71 | 52.0% | 45.9% | -1.93 |
| full_OOS | Top-3 | 6.0bps | SHORT | 5574 | +2.26 | -3.74 | 54.2% | 49.0% | -3.12 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 5574 | +4.29 | -5.71 | 52.0% | 41.8% | -6.46 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 5574 | +2.26 | -7.74 | 54.2% | 46.1% | -6.46 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 712 | +4.22 | -1.78 | 54.2% | 48.6% | -0.73 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 712 | +4.49 | -1.51 | 54.1% | 49.0% | -0.47 |
| recent_12mo | Top-1 | 10.0bps | LONG | 712 | +4.22 | -5.78 | 54.2% | 43.0% | -2.38 ✷ |
| recent_12mo | Top-1 | 10.0bps | SHORT | 712 | +4.49 | -5.51 | 54.1% | 45.2% | -1.71 |
| recent_12mo | Top-3 | 6.0bps | LONG | 2136 | +3.64 | -2.36 | 52.5% | 46.6% | -1.78 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 2136 | +1.04 | -4.96 | 53.3% | 47.5% | -2.75 ✷ |
| recent_12mo | Top-3 | 10.0bps | LONG | 2136 | +3.64 | -6.36 | 52.5% | 42.3% | -4.79 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 2136 | +1.04 | -8.96 | 53.3% | 44.3% | -4.97 ✷ |


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

