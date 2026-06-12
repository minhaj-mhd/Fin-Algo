# 🛡️ Validation Gauntlet Report: `v16_binary_breakout_1h`

## 📌 Metadata
- **Run ID**: `20260610T190244Z-d795438c`
- **Evaluated At (UTC)**: `2026-06-10T19:12:35.630591+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `39`
- **Deflated t-Threshold**: `3.2272` (corrected for `40` total tests)


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
| 1 | 2023-08, 2023-09 | +0.0413 | +0.0360 | 58 | 13 |
| 2 | 2023-10, 2023-11 | +0.0338 | +0.0407 | 17 | 32 |
| 3 | 2023-12, 2024-01 | +0.0318 | +0.0397 | 96 | 8 |
| 4 | 2024-02, 2024-03 | +0.0325 | +0.0306 | 0 | 8 |
| 5 | 2024-04, 2024-05 | +0.0290 | +0.0327 | 20 | 18 |
| 6 | 2024-06, 2024-07 | +0.0369 | +0.0350 | 4 | 47 |
| 7 | 2024-08, 2024-09 | +0.0237 | +0.0196 | 25 | 36 |
| 8 | 2024-10, 2024-11 | +0.0221 | +0.0204 | 54 | 103 |
| 9 | 2024-12, 2025-01 | +0.0249 | +0.0238 | 0 | 53 |
| 10 | 2025-02, 2025-03 | +0.0211 | +0.0250 | 4 | 12 |
| 11 | 2025-04, 2025-05 | +0.0222 | +0.0218 | 70 | 55 |
| 12 | 2025-06, 2025-07 | +0.0163 | +0.0212 | 10 | 16 |
| 13 | 2025-08, 2025-09 | +0.0184 | +0.0229 | 49 | 21 |
| 14 | 2025-10, 2025-11 | +0.0357 | +0.0226 | 36 | 0 |
| 15 | 2025-12, 2026-01 | +0.0276 | +0.0179 | 28 | 5 |
| 16 | 2026-02, 2026-03 | +0.0151 | +0.0155 | 38 | 31 |
| 17 | 2026-04, 2026-05 | +0.0205 | +0.0097 | 30 | 10 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`3.23`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +2.24 | -3.76 | 51.5% | 45.3% | 48.3% | -2.72 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +1.93 | -4.07 | 54.3% | 49.3% | 50.5% | -2.6 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +2.24 | -7.76 | 51.5% | 41.0% | 48.3% | -5.62 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +1.93 | -8.07 | 54.3% | 46.4% | 50.5% | -5.16 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +2.74 | -3.26 | 51.7% | 45.4% | 48.3% | -4.62 ✷ |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +1.01 | -4.99 | 53.4% | 48.4% | 50.5% | -5.61 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +2.74 | -7.26 | 51.7% | 41.1% | 48.3% | -10.29 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +1.01 | -8.99 | 53.4% | 45.2% | 50.5% | -10.11 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +0.18 | -5.82 | 51.0% | 45.6% | 48.4% | -3.21 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +3.20 | -2.80 | 54.6% | 49.0% | 50.4% | -1.26 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +0.18 | -9.82 | 51.0% | 40.6% | 48.4% | -5.42 ✷ |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +3.20 | -6.80 | 54.6% | 45.9% | 50.4% | -3.06 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +2.05 | -3.95 | 52.0% | 45.9% | 48.4% | -4.1 ✷ |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +1.37 | -4.63 | 53.1% | 47.7% | 50.4% | -3.84 ✷ |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +2.05 | -7.95 | 52.0% | 41.1% | 48.4% | -8.25 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +1.37 | -8.63 | 53.1% | 44.4% | 50.4% | -7.15 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.9% | +0.29 | 2040 | 53.0% | -6.31 |
| 10:15 | 2085 | 49.9% | -5.97 | 2085 | 53.8% | -2.70 |
| 11:15 | 2091 | 49.5% | -7.72 | 2091 | 53.9% | -4.84 |
| 12:15 | 2085 | 52.1% | -1.72 | 2085 | 51.9% | -6.85 |
| 13:15 | 2085 | 53.9% | -1.12 | 2085 | 54.4% | -4.26 |

