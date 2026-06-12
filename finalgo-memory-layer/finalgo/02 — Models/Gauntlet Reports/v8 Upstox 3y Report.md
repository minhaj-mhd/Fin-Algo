---
title: "Validation Gauntlet Report: `v8_upstox_3y"
type: report
status: active
model: "Gauntlet Reports"
verdict: FILTER_GRADE
updated: 2026-06-12
tags: []
---
# 🛡️ Validation Gauntlet Report: `v8_upstox_3y`

## 📌 Metadata
- **Run ID**: `20260610T172623Z-d795438c`
- **Evaluated At (UTC)**: `2026-06-10T17:32:53.606647+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Model Adapter**: `xgb_ranker`
- **Git Commit**: `58fac2f405685eee1d91900277a11555efd997e9`
- **Multiple Testing Context**: Prior runs for dataset family = `27`
- **Deflated t-Threshold**: `3.1237` (corrected for `28` total tests)


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
| 2 | 2023-10, 2023-11 | +0.0337 | +0.0387 | 39 | 13 |
| 3 | 2023-12, 2024-01 | +0.0345 | +0.0378 | 3 | 3 |
| 4 | 2024-02, 2024-03 | +0.0288 | +0.0346 | 4 | 14 |
| 5 | 2024-04, 2024-05 | +0.0217 | +0.0326 | 0 | 5 |
| 6 | 2024-06, 2024-07 | +0.0404 | +0.0350 | 7 | 0 |
| 7 | 2024-08, 2024-09 | +0.0263 | +0.0193 | 26 | 9 |
| 8 | 2024-10, 2024-11 | +0.0195 | +0.0209 | 1 | 81 |
| 9 | 2024-12, 2025-01 | +0.0248 | +0.0244 | 0 | 2 |
| 10 | 2025-02, 2025-03 | +0.0230 | +0.0318 | 3 | 26 |
| 11 | 2025-04, 2025-05 | +0.0269 | +0.0220 | 0 | 14 |
| 12 | 2025-06, 2025-07 | +0.0176 | +0.0205 | 5 | 38 |
| 13 | 2025-08, 2025-09 | +0.0198 | +0.0229 | 38 | 66 |
| 14 | 2025-10, 2025-11 | +0.0402 | +0.0216 | 0 | 23 |
| 15 | 2025-12, 2026-01 | +0.0233 | +0.0201 | 3 | 52 |
| 16 | 2026-02, 2026-03 | +0.0093 | +0.0207 | 4 | 95 |
| 17 | 2026-04, 2026-05 | +0.0071 | +0.0016 | 9 | 109 |


---

## 💻 Top-K Returns (Walk-Forward Pooled)
> [!IMPORTANT]
> Deflated t-threshold is **`3.12`**. Values exceeding this are marked with ✷.

| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | Baseline WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +5.16 | -0.84 | 53.2% | 47.3% | 48.3% | -0.64 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +3.23 | -2.77 | 54.0% | 49.0% | 50.5% | -1.72 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +5.16 | -4.84 | 53.2% | 43.2% | 48.3% | -3.65 ✷ |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +3.23 | -6.77 | 54.0% | 46.5% | 50.5% | -4.21 ✷ |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +4.59 | -1.41 | 52.3% | 46.4% | 48.3% | -2.07 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +2.43 | -3.57 | 53.8% | 48.9% | 50.5% | -4.01 ✷ |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +4.59 | -5.41 | 52.3% | 42.4% | 48.3% | -7.92 ✷ |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +2.43 | -7.57 | 53.8% | 45.9% | 50.5% | -8.51 ✷ |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +5.52 | -0.48 | 53.9% | 48.7% | 48.4% | -0.28 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +3.33 | -2.67 | 53.8% | 48.5% | 50.4% | -1.15 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +5.52 | -4.48 | 53.9% | 44.0% | 48.4% | -2.6 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +3.33 | -6.67 | 53.8% | 45.0% | 50.4% | -2.87 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +4.66 | -1.34 | 52.6% | 46.9% | 48.4% | -1.41 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +1.96 | -4.04 | 53.5% | 48.1% | 50.4% | -3.02 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +4.66 | -5.34 | 52.6% | 42.9% | 48.4% | -5.62 ✷ |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +1.96 | -8.04 | 53.5% | 44.7% | 50.4% | -6.01 ✷ |


---

## 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
> [!NOTE]
> Time-of-day tables are **diagnostic only** and do not feed the verdict engine unless a specific slice was pre-registered in the config. Cost level applied is informational `@6bps`.

| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.4% | +1.25 | 2040 | 54.0% | -4.89 |
| 10:15 | 2085 | 51.5% | -4.05 | 2085 | 53.9% | -2.06 |
| 11:15 | 2091 | 49.4% | -4.95 | 2091 | 53.8% | -5.14 |
| 12:15 | 2085 | 53.9% | -1.44 | 2085 | 53.1% | -5.72 |
| 13:15 | 2085 | 54.5% | +2.19 | 2085 | 54.4% | -0.07 |

