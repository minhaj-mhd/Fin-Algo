# 🛡️ Validation Gauntlet Report: `v17_random_forest_1h`

## 📌 Metadata
- **Run ID**: `20260610T121944Z-5f7d069f`
- **Evaluated At (UTC)**: `2026-06-10T12:41:06.493209+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Model Adapter**: `xgb_binary`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `22`
- **Deflated t-Threshold**: `3.0654` (corrected for `23` total tests)


## 📊 Dataset Label Verification Stats
- **In-File Verified (INTRA) Rows**: 79.93%
- **Unverifiable (Missing Target Bar) Rows**: 0.04%
- **Boundary (Session Terminal) Rows**: 20.03%
- **Unverified Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.
- **Prefix Invariance Waiver Reason**: N/A


## ⚖️ Final Verdicts
- **LONG Side**: <span style='color:red;font-weight:bold'>DEAD</span>
- **SHORT Side**: <span style='color:red;font-weight:bold'>DEAD</span>


---

## 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2023-08, 2023-09 | -0.0171 | +0.0268 | 0 | 0 |
| 2 | 2023-12, 2024-01 | -0.0154 | +0.0305 | 0 | 0 |
| 3 | 2024-04, 2024-05 | -0.0023 | +0.0193 | 0 | 0 |
| 4 | 2024-08, 2024-09 | -0.0033 | +0.0120 | 0 | 0 |
| 5 | 2024-12, 2025-01 | -0.0087 | +0.0275 | 0 | 0 |
| 6 | 2025-04, 2025-05 | -0.0095 | +0.0102 | 0 | 0 |
| 7 | 2025-08, 2025-09 | -0.0124 | +0.0186 | 0 | 0 |
| 8 | 2025-12, 2026-01 | -0.0051 | +0.0061 | 0 | 0 |
| 9 | 2026-04, 2026-05 | +0.0072 | -0.0055 | 0 | 0 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`3.07`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 1858 | +1.23 | -4.77 | 50.0% | 45.6% | 48.1% | -2.3 |
| full_OOS | Top-1 | 6.0bps | SHORT | 1858 | +0.28 | -5.72 | 53.9% | 50.5% | 50.7% | -2.16 |
| full_OOS | Top-1 | 10.0bps | LONG | 1858 | +1.23 | -8.77 | 50.0% | 42.4% | 48.1% | -4.23 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 1858 | +0.28 | -9.72 | 53.9% | 47.9% | 50.7% | -3.68 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 5574 | +0.90 | -5.10 | 49.3% | 45.4% | 48.1% | -4.44 ✷ |
| full_OOS | Top-3 | 6.0bps | SHORT | 5574 | +0.35 | -5.65 | 52.0% | 48.3% | 50.7% | -4.1 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 5574 | +0.90 | -9.10 | 49.3% | 42.3% | 48.1% | -7.92 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 5574 | +0.35 | -9.65 | 52.0% | 45.7% | 50.7% | -7.0 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 1234 | +3.10 | -2.90 | 50.9% | 46.9% | 48.2% | -1.13 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1234 | +1.44 | -4.56 | 53.8% | 50.6% | 50.8% | -1.42 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1234 | +3.10 | -6.90 | 50.9% | 44.2% | 48.2% | -2.7 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1234 | +1.44 | -8.56 | 53.8% | 48.0% | 50.8% | -2.66 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3702 | +1.18 | -4.82 | 49.0% | 45.1% | 48.2% | -3.45 ✷ |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3702 | +0.02 | -5.98 | 51.6% | 47.8% | 50.8% | -3.67 ✷ |
| recent_12mo | Top-3 | 10.0bps | LONG | 3702 | +1.18 | -8.82 | 49.0% | 42.2% | 48.2% | -6.32 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3702 | +0.02 | -9.98 | 51.6% | 45.0% | 50.8% | -6.13 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 1095 | 50.6% | -4.10 | 1095 | 50.7% | -7.14 |
| 10:15 | 1119 | 49.4% | -2.32 | 1119 | 49.4% | -10.57 |
| 11:15 | 1122 | 48.5% | -7.77 | 1122 | 53.6% | -4.61 |
| 12:15 | 1119 | 48.9% | -6.34 | 1119 | 55.2% | +1.96 |
| 13:15 | 1119 | 49.2% | -4.96 | 1119 | 50.8% | -7.92 |

