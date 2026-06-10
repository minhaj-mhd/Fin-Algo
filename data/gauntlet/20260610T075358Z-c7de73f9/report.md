# 🛡️ Validation Gauntlet Report: `daily_xgb`

## 📌 Metadata
- **Run ID**: `20260610T075358Z-c7de73f9`
- **Evaluated At (UTC)**: `2026-06-10T07:53:58.956348+00:00`
- **Dataset Path**: `data/ranking_data_upstox_daily_5y.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `93cc34f74d4b770dc6eac29ab6206c84b2434538`
- **Multiple Testing Context**: Prior runs for dataset family = `0`
- **Deflated t-Threshold**: `1.9600` (corrected for `1` total tests)

## ⚖️ Final Verdicts
- **LONG Side**: <span style='color:orange;font-weight:bold'>FILTER_GRADE</span>
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
> Deflated t-threshold is **`1.96`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 443 | +4.84 | -1.16 | 51.2% | 49.7% | -0.11 |
| full_OOS | Top-1 | 6.0bps | SHORT | 443 | -6.17 | -12.17 | 52.1% | 51.0% | -0.96 |
| full_OOS | Top-1 | 10.0bps | LONG | 443 | +4.84 | -5.16 | 51.2% | 49.0% | -0.49 |
| full_OOS | Top-1 | 10.0bps | SHORT | 443 | -6.17 | -16.17 | 52.1% | 49.2% | -1.27 |
| full_OOS | Top-3 | 6.0bps | LONG | 1329 | +11.74 | +5.74 | 50.8% | 49.1% | 0.89 |
| full_OOS | Top-3 | 6.0bps | SHORT | 1329 | -3.09 | -9.09 | 50.6% | 49.4% | -1.3 |
| full_OOS | Top-3 | 10.0bps | LONG | 1329 | +11.74 | +1.74 | 50.8% | 47.7% | 0.27 |
| full_OOS | Top-3 | 10.0bps | SHORT | 1329 | -3.09 | -13.09 | 50.6% | 48.1% | -1.87 |
| recent_12mo | Top-1 | 6.0bps | LONG | 138 | +10.65 | +4.65 | 59.4% | 55.8% | 0.27 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 138 | +20.80 | +14.80 | 56.5% | 55.8% | 0.86 |
| recent_12mo | Top-1 | 10.0bps | LONG | 138 | +10.65 | +0.65 | 59.4% | 55.1% | 0.04 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 138 | +20.80 | +10.80 | 56.5% | 53.6% | 0.62 |
| recent_12mo | Top-3 | 6.0bps | LONG | 414 | +14.98 | +8.98 | 55.1% | 52.4% | 0.9 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 414 | +11.72 | +5.72 | 51.7% | 51.2% | 0.49 |
| recent_12mo | Top-3 | 10.0bps | LONG | 414 | +14.98 | +4.98 | 55.1% | 51.0% | 0.5 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 414 | +11.72 | +1.72 | 51.7% | 49.5% | 0.15 |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 00:00 | 1329 | 50.8% | +5.74 | 1329 | 50.6% | -9.09 |

