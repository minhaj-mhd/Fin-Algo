# 🛡️ Validation Gauntlet Report: `v10_depth4_1h`

## 📌 Metadata
- **Run ID**: `20260610T184210Z-d795438c`
- **Evaluated At (UTC)**: `2026-06-10T18:52:58.944060+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `31`
- **Deflated t-Threshold**: `3.1628` (corrected for `32` total tests)


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
| 1 | 2023-08, 2023-09 | +0.0444 | +0.0372 | 12 | 41 |
| 2 | 2023-10, 2023-11 | +0.0375 | +0.0385 | 143 | 1 |
| 3 | 2023-12, 2024-01 | +0.0371 | +0.0389 | 6 | 217 |
| 4 | 2024-02, 2024-03 | +0.0320 | +0.0369 | 15 | 73 |
| 5 | 2024-04, 2024-05 | +0.0244 | +0.0290 | 1 | 50 |
| 6 | 2024-06, 2024-07 | +0.0432 | +0.0365 | 6 | 15 |
| 7 | 2024-08, 2024-09 | +0.0288 | +0.0205 | 10 | 35 |
| 8 | 2024-10, 2024-11 | +0.0215 | +0.0211 | 43 | 59 |
| 9 | 2024-12, 2025-01 | +0.0275 | +0.0326 | 9 | 63 |
| 10 | 2025-02, 2025-03 | +0.0186 | +0.0294 | 3 | 8 |
| 11 | 2025-04, 2025-05 | +0.0244 | +0.0213 | 3 | 135 |
| 12 | 2025-06, 2025-07 | +0.0251 | +0.0236 | 31 | 164 |
| 13 | 2025-08, 2025-09 | +0.0180 | +0.0226 | 29 | 44 |
| 14 | 2025-10, 2025-11 | +0.0352 | +0.0218 | 9 | 18 |
| 15 | 2025-12, 2026-01 | +0.0214 | +0.0180 | 8 | 0 |
| 16 | 2026-02, 2026-03 | +0.0140 | +0.0159 | 6 | 49 |
| 17 | 2026-04, 2026-05 | +0.0139 | +0.0101 | 5 | 106 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`3.16`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +3.70 | -2.30 | 53.7% | 47.5% | 48.3% | -2.05 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +3.66 | -2.34 | 55.6% | 51.2% | 50.5% | -1.39 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +3.70 | -6.30 | 53.7% | 43.0% | 48.3% | -5.61 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +3.66 | -6.34 | 55.6% | 48.6% | 50.5% | -3.75 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +3.37 | -2.63 | 52.7% | 46.2% | 48.3% | -4.48 ✷ |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +2.30 | -3.70 | 54.4% | 49.9% | 50.5% | -4.01 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +3.37 | -6.63 | 52.7% | 41.6% | 48.3% | -11.29 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +2.30 | -7.70 | 54.4% | 47.1% | 50.5% | -8.36 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +4.03 | -1.97 | 55.0% | 49.1% | 48.4% | -1.2 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +4.47 | -1.53 | 54.9% | 50.4% | 50.4% | -0.64 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +4.03 | -5.97 | 55.0% | 44.8% | 48.4% | -3.64 ✷ |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +4.47 | -5.53 | 54.9% | 47.7% | 50.4% | -2.32 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +3.68 | -2.32 | 53.4% | 47.1% | 48.4% | -2.73 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +2.87 | -3.13 | 54.2% | 49.6% | 50.4% | -2.29 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +3.68 | -6.32 | 53.4% | 42.4% | 48.4% | -7.43 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +2.87 | -7.13 | 54.2% | 46.6% | 50.4% | -5.22 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.8% | -1.56 | 2040 | 52.4% | -8.33 |
| 10:15 | 2085 | 52.5% | -3.89 | 2085 | 55.4% | -0.36 |
| 11:15 | 2091 | 51.7% | -4.37 | 2091 | 55.3% | -3.07 |
| 12:15 | 2085 | 53.1% | -2.55 | 2085 | 54.6% | -4.67 |
| 13:15 | 2085 | 53.2% | -0.77 | 2085 | 54.4% | -2.15 |

