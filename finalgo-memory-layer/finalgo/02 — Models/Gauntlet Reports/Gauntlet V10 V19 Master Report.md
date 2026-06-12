---
title: "Validation Gauntlet Master Evaluation Report (v10 to v19)"
type: report
status: active
model: "Gauntlet Reports"
verdict: FILTER_GRADE
updated: 2026-06-12
tags: []
---
# 📊 Validation Gauntlet Master Evaluation Report (v10 to v19)
**Generated At**: 11-06-2026 10:27
This master report compiles the full, detailed walk-forward evaluation outputs for all models evaluated in the latest decontaminated batch run under `step_months=2` parameters.

---
## 📑 Table of Contents
- [v2_15min_3y](#v2-15min-3y)
- [v10_depth4_1h](#v10-depth4-1h)
- [v10_native_1h](#v10-native-1h)
- [v12_lambdamart_1h](#v12-lambdamart-1h)
- [v13_ndcg_raw_1h](#v13-ndcg-raw-1h)
- [v14_lambdamart_no_es_1h](#v14-lambdamart-no-es-1h)
- [v15_lambdamart_es_1h](#v15-lambdamart-es-1h)
- [v16_binary_breakout_1h](#v16-binary-breakout-1h)
- [v19_catboost_1h](#v19-catboost-1h)
- [v8_upstox_3y](#v8-upstox-3y)

---
## 🛡️ Model: `v2_15min_3y`
### 📌 Metadata
- **Run ID**: `20260610T173707Z-d795438c`
- **Evaluated At**: `2026-06-10T18:19:42.295895+00:00`
- **Dataset Path**: `data/ranking_data_upstox_15min_3y_clean.csv`
- **Dataset SHA-256**: `2ca161c9afa17d776d0929edfd47c373104e72f1331bd2c71455c3de1f919abf`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `2.5758293035489004`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `95.81%`
- **Unverifiable Rows**: `0.01%`
- **Boundary Rows**: `4.18%`
- **Label Waiver Reason**: 15m older intraday parquet source files not available in raw history directory.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2024-09, 2024-10 | +0.0617 | +0.0625 | 284 | 26 |
| 2 | 2024-11, 2024-12 | +0.0601 | +0.0625 | 7 | 36 |
| 3 | 2025-01, 2025-02 | +0.0539 | +0.0548 | 137 | 2 |
| 4 | 2025-03, 2025-04 | +0.0560 | +0.0581 | 25 | 59 |
| 5 | 2025-05, 2025-06 | +0.0691 | +0.0687 | 1 | 3 |
| 6 | 2025-07, 2025-08 | +0.0631 | +0.0605 | 254 | 13 |
| 7 | 2025-09, 2025-10 | +0.0644 | +0.0676 | 10 | 49 |
| 8 | 2025-11, 2025-12 | +0.0622 | +0.0650 | 32 | 54 |
| 9 | 2026-01, 2026-02 | +0.0520 | +0.0493 | 74 | 4 |
| 10 | 2026-03, 2026-04 | +0.0576 | +0.0532 | 127 | 5 |
| 11 | 2026-05, 2026-06 | +0.0529 | +0.0529 | 31 | 2 |

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 9448 | +3.38 | -2.62 | 55.8% | 44.5% | -6.52 |
| full_OOS | Top-1 | 6.0bps | SHORT | 9448 | +3.47 | -2.53 | 56.9% | 48.0% | -4.29 |
| full_OOS | Top-1 | 10.0bps | LONG | 9448 | +3.38 | -6.62 | 55.8% | 37.2% | -16.47 |
| full_OOS | Top-1 | 10.0bps | SHORT | 9448 | +3.47 | -6.53 | 56.9% | 41.6% | -11.07 |
| full_OOS | Top-3 | 6.0bps | LONG | 28344 | +3.25 | -2.75 | 55.3% | 43.6% | -12.73 |
| full_OOS | Top-3 | 6.0bps | SHORT | 28344 | +2.94 | -3.06 | 56.8% | 46.9% | -10.58 |
| full_OOS | Top-3 | 10.0bps | LONG | 28344 | +3.25 | -6.75 | 55.3% | 36.2% | -31.28 |
| full_OOS | Top-3 | 10.0bps | SHORT | 28344 | +2.94 | -7.06 | 56.8% | 40.3% | -24.42 |
| recent_12mo | Top-1 | 6.0bps | LONG | 4989 | +3.51 | -2.49 | 55.9% | 43.8% | -4.92 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 4989 | +3.12 | -2.88 | 57.7% | 47.9% | -3.87 |
| recent_12mo | Top-1 | 10.0bps | LONG | 4989 | +3.51 | -6.49 | 55.9% | 36.3% | -12.84 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 4989 | +3.12 | -6.88 | 57.7% | 40.4% | -9.23 |
| recent_12mo | Top-3 | 6.0bps | LONG | 14967 | +3.28 | -2.72 | 55.6% | 42.9% | -10.21 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 14967 | +2.88 | -3.12 | 57.2% | 46.6% | -8.58 |
| recent_12mo | Top-3 | 10.0bps | LONG | 14967 | +3.28 | -6.72 | 55.6% | 34.9% | -25.24 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 14967 | +2.88 | -7.12 | 57.2% | 39.0% | -19.60 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 1149 | 50.7% | -4.06 | 1149 | 55.1% | -0.68 |
| 09:30 | 1182 | 52.9% | -4.98 | 1182 | 54.6% | -1.73 |
| 09:45 | 1182 | 52.7% | -2.57 | 1182 | 53.6% | -5.63 |
| 10:00 | 1182 | 57.2% | -0.34 | 1182 | 52.8% | -5.24 |
| 10:15 | 1182 | 55.2% | -1.91 | 1182 | 58.0% | -3.03 |
| 10:30 | 1182 | 52.1% | -5.24 | 1182 | 56.0% | -1.87 |
| 10:45 | 1182 | 56.0% | -2.19 | 1182 | 58.4% | -4.42 |
| 11:00 | 1182 | 52.8% | -3.92 | 1182 | 56.3% | -3.73 |
| 11:15 | 1182 | 53.5% | -3.63 | 1182 | 58.1% | -3.30 |
| 11:30 | 1182 | 51.4% | -5.50 | 1182 | 56.3% | -3.73 |
| 11:45 | 1182 | 56.9% | -2.26 | 1182 | 56.1% | -5.05 |
| 12:00 | 1182 | 55.8% | -4.43 | 1182 | 59.5% | -2.61 |
| 12:15 | 1182 | 54.3% | -1.36 | 1182 | 56.3% | -5.20 |
| 12:30 | 1182 | 54.7% | -4.35 | 1182 | 58.0% | -2.41 |
| 12:45 | 1182 | 55.2% | -3.59 | 1182 | 55.3% | -7.06 |
| 13:00 | 1182 | 53.2% | -5.17 | 1182 | 59.1% | -1.73 |
| 13:15 | 1182 | 57.4% | -2.41 | 1182 | 56.9% | -2.35 |
| 13:30 | 1182 | 57.6% | -2.44 | 1182 | 56.3% | -4.27 |
| 13:45 | 1185 | 57.0% | -3.72 | 1185 | 56.8% | -2.43 |
| 14:00 | 1185 | 57.0% | -2.35 | 1185 | 58.6% | -2.91 |
| 14:15 | 1185 | 58.4% | -1.92 | 1185 | 56.1% | -4.97 |
| 14:30 | 1182 | 60.7% | -0.13 | 1182 | 55.8% | -2.93 |
| 14:45 | 1182 | 57.4% | +0.46 | 1182 | 55.8% | -5.12 |
| 15:00 | 1182 | 56.9% | +2.05 | 1182 | 62.4% | +9.14 |

================================================================================

## 🛡️ Model: `v10_depth4_1h`
### 📌 Metadata
- **Run ID**: `20260610T184210Z-d795438c`
- **Evaluated At**: `2026-06-10T18:52:58.942060+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.1628179656213016`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
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

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +3.70 | -2.30 | 53.7% | 47.5% | -2.05 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +3.66 | -2.34 | 55.6% | 51.2% | -1.39 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +3.70 | -6.30 | 53.7% | 43.0% | -5.61 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +3.66 | -6.34 | 55.6% | 48.6% | -3.75 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +3.37 | -2.63 | 52.7% | 46.2% | -4.48 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +2.30 | -3.70 | 54.4% | 49.9% | -4.01 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +3.37 | -6.63 | 52.7% | 41.6% | -11.29 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +2.30 | -7.70 | 54.4% | 47.1% | -8.36 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +4.03 | -1.97 | 55.0% | 49.1% | -1.20 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +4.47 | -1.53 | 54.9% | 50.4% | -0.64 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +4.03 | -5.97 | 55.0% | 44.8% | -3.64 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +4.47 | -5.53 | 54.9% | 47.7% | -2.32 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +3.68 | -2.32 | 53.4% | 47.1% | -2.73 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +2.87 | -3.13 | 54.2% | 49.6% | -2.29 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +3.68 | -6.32 | 53.4% | 42.4% | -7.43 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +2.87 | -7.13 | 54.2% | 46.6% | -5.22 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.8% | -1.56 | 2040 | 52.4% | -8.33 |
| 10:15 | 2085 | 52.5% | -3.89 | 2085 | 55.4% | -0.36 |
| 11:15 | 2091 | 51.7% | -4.37 | 2091 | 55.3% | -3.07 |
| 12:15 | 2085 | 53.1% | -2.55 | 2085 | 54.6% | -4.67 |
| 13:15 | 2085 | 53.2% | -0.77 | 2085 | 54.4% | -2.15 |

================================================================================

## 🛡️ Model: `v10_native_1h`
### 📌 Metadata
- **Run ID**: `20260610T183618Z-d795438c`
- **Evaluated At**: `2026-06-10T18:43:17.655028+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.180425742656707`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
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

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +5.16 | -0.84 | 53.2% | 47.3% | -0.64 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +3.23 | -2.77 | 54.0% | 49.0% | -1.72 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +5.16 | -4.84 | 53.2% | 43.2% | -3.65 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +3.23 | -6.77 | 54.0% | 46.5% | -4.21 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +4.59 | -1.41 | 52.3% | 46.4% | -2.07 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +2.43 | -3.57 | 53.8% | 48.9% | -4.01 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +4.59 | -5.41 | 52.3% | 42.4% | -7.92 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +2.43 | -7.57 | 53.8% | 45.9% | -8.51 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +5.52 | -0.48 | 53.9% | 48.7% | -0.28 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +3.33 | -2.67 | 53.8% | 48.5% | -1.15 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +5.52 | -4.48 | 53.9% | 44.0% | -2.60 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +3.33 | -6.67 | 53.8% | 45.0% | -2.87 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +4.66 | -1.34 | 52.6% | 46.9% | -1.41 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +1.96 | -4.04 | 53.5% | 48.1% | -3.02 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +4.66 | -5.34 | 52.6% | 42.9% | -5.62 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +1.96 | -8.04 | 53.5% | 44.7% | -6.01 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.4% | +1.25 | 2040 | 54.0% | -4.89 |
| 10:15 | 2085 | 51.5% | -4.05 | 2085 | 53.9% | -2.06 |
| 11:15 | 2091 | 49.4% | -4.95 | 2091 | 53.8% | -5.14 |
| 12:15 | 2085 | 53.9% | -1.44 | 2085 | 53.1% | -5.72 |
| 13:15 | 2085 | 54.5% | +2.19 | 2085 | 54.4% | -0.07 |

================================================================================

## 🛡️ Model: `v12_lambdamart_1h`
### 📌 Metadata
- **Run ID**: `20260610T185243Z-d795438c`
- **Evaluated At**: `2026-06-10T19:02:39.649441+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.2048452050105642`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:red;font-weight:bold'>DEAD</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2023-08, 2023-09 | +0.0285 | +0.0337 | 47 | 140 |
| 2 | 2023-10, 2023-11 | +0.0155 | +0.0338 | 13 | 23 |
| 3 | 2023-12, 2024-01 | +0.0108 | +0.0306 | 6 | 4 |
| 4 | 2024-02, 2024-03 | +0.0245 | +0.0346 | 11 | 35 |
| 5 | 2024-04, 2024-05 | +0.0198 | +0.0260 | 11 | 6 |
| 6 | 2024-06, 2024-07 | +0.0157 | +0.0278 | 35 | 9 |
| 7 | 2024-08, 2024-09 | +0.0216 | +0.0161 | 0 | 48 |
| 8 | 2024-10, 2024-11 | +0.0139 | +0.0127 | 20 | 14 |
| 9 | 2024-12, 2025-01 | +0.0156 | +0.0218 | 20 | 12 |
| 10 | 2025-02, 2025-03 | +0.0059 | +0.0202 | 25 | 1 |
| 11 | 2025-04, 2025-05 | +0.0185 | +0.0172 | 1 | 22 |
| 12 | 2025-06, 2025-07 | +0.0169 | +0.0137 | 90 | 8 |
| 13 | 2025-08, 2025-09 | +0.0053 | +0.0149 | 2 | 75 |
| 14 | 2025-10, 2025-11 | +0.0341 | +0.0071 | 0 | 26 |
| 15 | 2025-12, 2026-01 | +0.0188 | +0.0121 | 3 | 3 |
| 16 | 2026-02, 2026-03 | +0.0017 | +0.0112 | 7 | 84 |
| 17 | 2026-04, 2026-05 | +0.0047 | +0.0027 | 108 | 12 |

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +2.74 | -3.26 | 50.7% | 46.4% | -2.10 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +3.11 | -2.89 | 54.9% | 50.5% | -1.68 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +2.74 | -7.26 | 50.7% | 42.5% | -4.68 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +3.11 | -6.89 | 54.9% | 47.8% | -4.00 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +1.86 | -4.14 | 50.6% | 45.3% | -5.17 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +1.92 | -4.08 | 53.5% | 49.0% | -4.29 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +1.86 | -8.14 | 50.6% | 41.7% | -10.15 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +1.92 | -8.08 | 53.5% | 45.9% | -8.51 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +2.91 | -3.09 | 51.1% | 46.1% | -1.49 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +8.95 | +2.95 | 56.9% | 51.9% | +1.20 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +2.91 | -7.09 | 51.1% | 41.6% | -3.43 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +8.95 | -1.05 | 56.9% | 49.1% | -0.43 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +1.04 | -4.96 | 49.8% | 44.5% | -4.66 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +2.75 | -3.25 | 53.9% | 48.8% | -2.23 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +1.04 | -8.96 | 49.8% | 40.8% | -8.41 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +2.75 | -7.25 | 53.9% | 45.6% | -4.97 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 51.1% | -2.25 | 2040 | 52.1% | -8.91 |
| 10:15 | 2085 | 48.6% | -6.99 | 2085 | 54.2% | +0.19 |
| 11:15 | 2091 | 49.1% | -7.77 | 2091 | 54.5% | -6.06 |
| 12:15 | 2085 | 50.8% | -3.40 | 2085 | 53.1% | -3.33 |
| 13:15 | 2085 | 53.1% | -0.23 | 2085 | 53.4% | -2.38 |

================================================================================

## 🛡️ Model: `v13_ndcg_raw_1h`
### 📌 Metadata
- **Run ID**: `20260610T185246Z-d795438c`
- **Evaluated At**: `2026-06-10T19:02:39.729529+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.2048452050105642`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:red;font-weight:bold'>DEAD</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
| Fold | Test Segment | Long Rho | Short Rho | Best Iter Long | Best Iter Short |
|---|---|---|---|---|---|
| 1 | 2023-08, 2023-09 | +0.0285 | +0.0337 | 47 | 140 |
| 2 | 2023-10, 2023-11 | +0.0155 | +0.0338 | 13 | 23 |
| 3 | 2023-12, 2024-01 | +0.0108 | +0.0306 | 6 | 4 |
| 4 | 2024-02, 2024-03 | +0.0245 | +0.0346 | 11 | 35 |
| 5 | 2024-04, 2024-05 | +0.0198 | +0.0260 | 11 | 6 |
| 6 | 2024-06, 2024-07 | +0.0157 | +0.0278 | 35 | 9 |
| 7 | 2024-08, 2024-09 | +0.0216 | +0.0161 | 0 | 48 |
| 8 | 2024-10, 2024-11 | +0.0139 | +0.0127 | 20 | 14 |
| 9 | 2024-12, 2025-01 | +0.0156 | +0.0218 | 20 | 12 |
| 10 | 2025-02, 2025-03 | +0.0059 | +0.0202 | 25 | 1 |
| 11 | 2025-04, 2025-05 | +0.0185 | +0.0172 | 1 | 22 |
| 12 | 2025-06, 2025-07 | +0.0169 | +0.0137 | 90 | 8 |
| 13 | 2025-08, 2025-09 | +0.0053 | +0.0149 | 2 | 75 |
| 14 | 2025-10, 2025-11 | +0.0341 | +0.0071 | 0 | 26 |
| 15 | 2025-12, 2026-01 | +0.0188 | +0.0121 | 3 | 3 |
| 16 | 2026-02, 2026-03 | +0.0017 | +0.0112 | 7 | 84 |
| 17 | 2026-04, 2026-05 | +0.0047 | +0.0027 | 108 | 12 |

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +2.74 | -3.26 | 50.7% | 46.4% | -2.10 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +3.11 | -2.89 | 54.9% | 50.5% | -1.68 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +2.74 | -7.26 | 50.7% | 42.5% | -4.68 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +3.11 | -6.89 | 54.9% | 47.8% | -4.00 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +1.86 | -4.14 | 50.6% | 45.3% | -5.17 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +1.92 | -4.08 | 53.5% | 49.0% | -4.29 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +1.86 | -8.14 | 50.6% | 41.7% | -10.15 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +1.92 | -8.08 | 53.5% | 45.9% | -8.51 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +2.91 | -3.09 | 51.1% | 46.1% | -1.49 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +8.95 | +2.95 | 56.9% | 51.9% | +1.20 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +2.91 | -7.09 | 51.1% | 41.6% | -3.43 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +8.95 | -1.05 | 56.9% | 49.1% | -0.43 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +1.04 | -4.96 | 49.8% | 44.5% | -4.66 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +2.75 | -3.25 | 53.9% | 48.8% | -2.23 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +1.04 | -8.96 | 49.8% | 40.8% | -8.41 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +2.75 | -7.25 | 53.9% | 45.6% | -4.97 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 51.1% | -2.25 | 2040 | 52.1% | -8.91 |
| 10:15 | 2085 | 48.6% | -6.99 | 2085 | 54.2% | +0.19 |
| 11:15 | 2091 | 49.1% | -7.77 | 2091 | 54.5% | -6.06 |
| 12:15 | 2085 | 50.8% | -3.40 | 2085 | 53.1% | -3.33 |
| 13:15 | 2085 | 53.1% | -0.23 | 2085 | 53.4% | -2.38 |

================================================================================

## 🛡️ Model: `v14_lambdamart_no_es_1h`
### 📌 Metadata
- **Run ID**: `20260610T185301Z-d795438c`
- **Evaluated At**: `2026-06-10T19:03:34.685826+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.2048452050105642`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
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

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +2.24 | -3.76 | 51.5% | 45.3% | -2.72 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +1.93 | -4.07 | 54.3% | 49.3% | -2.60 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +2.24 | -7.76 | 51.5% | 41.0% | -5.62 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +1.93 | -8.07 | 54.3% | 46.4% | -5.16 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +2.74 | -3.26 | 51.7% | 45.4% | -4.62 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +1.01 | -4.99 | 53.4% | 48.4% | -5.61 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +2.74 | -7.26 | 51.7% | 41.1% | -10.29 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +1.01 | -8.99 | 53.4% | 45.2% | -10.11 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +0.18 | -5.82 | 51.0% | 45.6% | -3.21 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +3.20 | -2.80 | 54.6% | 49.0% | -1.26 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +0.18 | -9.82 | 51.0% | 40.6% | -5.42 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +3.20 | -6.80 | 54.6% | 45.9% | -3.06 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +2.05 | -3.95 | 52.0% | 45.9% | -4.10 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +1.37 | -4.63 | 53.1% | 47.7% | -3.84 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +2.05 | -7.95 | 52.0% | 41.1% | -8.25 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +1.37 | -8.63 | 53.1% | 44.4% | -7.15 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.9% | +0.29 | 2040 | 53.0% | -6.31 |
| 10:15 | 2085 | 49.9% | -5.97 | 2085 | 53.8% | -2.70 |
| 11:15 | 2091 | 49.5% | -7.72 | 2091 | 53.9% | -4.84 |
| 12:15 | 2085 | 52.1% | -1.72 | 2085 | 51.9% | -6.85 |
| 13:15 | 2085 | 53.9% | -1.12 | 2085 | 54.4% | -4.26 |

================================================================================

## 🛡️ Model: `v15_lambdamart_es_1h`
### 📌 Metadata
- **Run ID**: `20260610T190242Z-d795438c`
- **Evaluated At**: `2026-06-10T19:12:31.140048+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.2272184259631627`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
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

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +2.24 | -3.76 | 51.5% | 45.3% | -2.72 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +1.93 | -4.07 | 54.3% | 49.3% | -2.60 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +2.24 | -7.76 | 51.5% | 41.0% | -5.62 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +1.93 | -8.07 | 54.3% | 46.4% | -5.16 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +2.74 | -3.26 | 51.7% | 45.4% | -4.62 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +1.01 | -4.99 | 53.4% | 48.4% | -5.61 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +2.74 | -7.26 | 51.7% | 41.1% | -10.29 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +1.01 | -8.99 | 53.4% | 45.2% | -10.11 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +0.18 | -5.82 | 51.0% | 45.6% | -3.21 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +3.20 | -2.80 | 54.6% | 49.0% | -1.26 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +0.18 | -9.82 | 51.0% | 40.6% | -5.42 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +3.20 | -6.80 | 54.6% | 45.9% | -3.06 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +2.05 | -3.95 | 52.0% | 45.9% | -4.10 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +1.37 | -4.63 | 53.1% | 47.7% | -3.84 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +2.05 | -7.95 | 52.0% | 41.1% | -8.25 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +1.37 | -8.63 | 53.1% | 44.4% | -7.15 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.9% | +0.29 | 2040 | 53.0% | -6.31 |
| 10:15 | 2085 | 49.9% | -5.97 | 2085 | 53.8% | -2.70 |
| 11:15 | 2091 | 49.5% | -7.72 | 2091 | 53.9% | -4.84 |
| 12:15 | 2085 | 52.1% | -1.72 | 2085 | 51.9% | -6.85 |
| 13:15 | 2085 | 53.9% | -1.12 | 2085 | 54.4% | -4.26 |

================================================================================

## 🛡️ Model: `v16_binary_breakout_1h`
### 📌 Metadata
- **Run ID**: `20260610T190244Z-d795438c`
- **Evaluated At**: `2026-06-10T19:12:35.628476+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.2272184259631627`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
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

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +2.24 | -3.76 | 51.5% | 45.3% | -2.72 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +1.93 | -4.07 | 54.3% | 49.3% | -2.60 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +2.24 | -7.76 | 51.5% | 41.0% | -5.62 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +1.93 | -8.07 | 54.3% | 46.4% | -5.16 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +2.74 | -3.26 | 51.7% | 45.4% | -4.62 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +1.01 | -4.99 | 53.4% | 48.4% | -5.61 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +2.74 | -7.26 | 51.7% | 41.1% | -10.29 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +1.01 | -8.99 | 53.4% | 45.2% | -10.11 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +0.18 | -5.82 | 51.0% | 45.6% | -3.21 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +3.20 | -2.80 | 54.6% | 49.0% | -1.26 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +0.18 | -9.82 | 51.0% | 40.6% | -5.42 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +3.20 | -6.80 | 54.6% | 45.9% | -3.06 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +2.05 | -3.95 | 52.0% | 45.9% | -4.10 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +1.37 | -4.63 | 53.1% | 47.7% | -3.84 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +2.05 | -7.95 | 52.0% | 41.1% | -8.25 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +1.37 | -8.63 | 53.1% | 44.4% | -7.15 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.9% | +0.29 | 2040 | 53.0% | -6.31 |
| 10:15 | 2085 | 49.9% | -5.97 | 2085 | 53.8% | -2.70 |
| 11:15 | 2091 | 49.5% | -7.72 | 2091 | 53.9% | -4.84 |
| 12:15 | 2085 | 52.1% | -1.72 | 2085 | 51.9% | -6.85 |
| 13:15 | 2085 | 53.9% | -1.12 | 2085 | 54.4% | -4.26 |

================================================================================

## 🛡️ Model: `v19_catboost_1h`
### 📌 Metadata
- **Run ID**: `20260610T191237Z-d795438c`
- **Evaluated At**: `2026-06-10T19:20:55.091486+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.2411521655173563`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
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

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +2.24 | -3.76 | 51.5% | 45.3% | -2.72 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +1.93 | -4.07 | 54.3% | 49.3% | -2.60 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +2.24 | -7.76 | 51.5% | 41.0% | -5.62 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +1.93 | -8.07 | 54.3% | 46.4% | -5.16 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +2.74 | -3.26 | 51.7% | 45.4% | -4.62 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +1.01 | -4.99 | 53.4% | 48.4% | -5.61 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +2.74 | -7.26 | 51.7% | 41.1% | -10.29 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +1.01 | -8.99 | 53.4% | 45.2% | -10.11 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +0.18 | -5.82 | 51.0% | 45.6% | -3.21 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +3.20 | -2.80 | 54.6% | 49.0% | -1.26 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +0.18 | -9.82 | 51.0% | 40.6% | -5.42 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +3.20 | -6.80 | 54.6% | 45.9% | -3.06 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +2.05 | -3.95 | 52.0% | 45.9% | -4.10 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +1.37 | -4.63 | 53.1% | 47.7% | -3.84 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +2.05 | -7.95 | 52.0% | 41.1% | -8.25 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +1.37 | -8.63 | 53.1% | 44.4% | -7.15 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.9% | +0.29 | 2040 | 53.0% | -6.31 |
| 10:15 | 2085 | 49.9% | -5.97 | 2085 | 53.8% | -2.70 |
| 11:15 | 2091 | 49.5% | -7.72 | 2091 | 53.9% | -4.84 |
| 12:15 | 2085 | 52.1% | -1.72 | 2085 | 51.9% | -6.85 |
| 13:15 | 2085 | 53.9% | -1.12 | 2085 | 54.4% | -4.26 |

================================================================================

## 🛡️ Model: `v8_upstox_3y`
### 📌 Metadata
- **Run ID**: `20260610T172623Z-d795438c`
- **Evaluated At**: `2026-06-10T17:32:53.603624+00:00`
- **Dataset Path**: `data/ranking_data_upstox_1h_v3_3y.csv`
- **Dataset SHA-256**: `b10b37fcbee368ceb8b43b35888df1d5f72736ec9786b6d681e8b9d43bc8e296`
- **Model Adapter**: `xgb_ranker`
- **Deflated t-Threshold**: `3.1237346303238454`

### 📊 Dataset Label Verification Stats
- **In-File Verified Rows**: `79.93%`
- **Unverifiable Rows**: `0.04%`
- **Boundary Rows**: `20.03%`
- **Label Waiver Reason**: Pre-drop 14:15 target bars omitted from 3y training file but verified consistent.

### ⚖️ Final Verdicts
- **LONG Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>
- **SHORT Side**: <span style='color:green;font-weight:bold'>FILTER_GRADE</span>

### 📈 Fold-Level Spearman Correlation
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

### 💻 Top-K Returns (Walk-Forward Pooled)
| Period | Config | Cost Level | Side | Trades | Raw bps | Net bps | Raw WR | Net WR | t-stat |
|---|---|---|---|---|---|---|---|---|---|
| full_OOS | Top-1 | 6.0bps | LONG | 3462 | +5.16 | -0.84 | 53.2% | 47.3% | -0.64 |
| full_OOS | Top-1 | 6.0bps | SHORT | 3462 | +3.23 | -2.77 | 54.0% | 49.0% | -1.72 |
| full_OOS | Top-1 | 10.0bps | LONG | 3462 | +5.16 | -4.84 | 53.2% | 43.2% | -3.65 |
| full_OOS | Top-1 | 10.0bps | SHORT | 3462 | +3.23 | -6.77 | 54.0% | 46.5% | -4.21 |
| full_OOS | Top-3 | 6.0bps | LONG | 10386 | +4.59 | -1.41 | 52.3% | 46.4% | -2.07 |
| full_OOS | Top-3 | 6.0bps | SHORT | 10386 | +2.43 | -3.57 | 53.8% | 48.9% | -4.01 |
| full_OOS | Top-3 | 10.0bps | LONG | 10386 | +4.59 | -5.41 | 52.3% | 42.4% | -7.92 |
| full_OOS | Top-3 | 10.0bps | SHORT | 10386 | +2.43 | -7.57 | 53.8% | 45.9% | -8.51 |
| recent_12mo | Top-1 | 6.0bps | LONG | 1219 | +5.52 | -0.48 | 53.9% | 48.7% | -0.28 |
| recent_12mo | Top-1 | 6.0bps | SHORT | 1219 | +3.33 | -2.67 | 53.8% | 48.5% | -1.15 |
| recent_12mo | Top-1 | 10.0bps | LONG | 1219 | +5.52 | -4.48 | 53.9% | 44.0% | -2.60 |
| recent_12mo | Top-1 | 10.0bps | SHORT | 1219 | +3.33 | -6.67 | 53.8% | 45.0% | -2.87 |
| recent_12mo | Top-3 | 6.0bps | LONG | 3657 | +4.66 | -1.34 | 52.6% | 46.9% | -1.41 |
| recent_12mo | Top-3 | 6.0bps | SHORT | 3657 | +1.96 | -4.04 | 53.5% | 48.1% | -3.02 |
| recent_12mo | Top-3 | 10.0bps | LONG | 3657 | +4.66 | -5.34 | 52.6% | 42.9% | -5.62 |
| recent_12mo | Top-3 | 10.0bps | SHORT | 3657 | +1.96 | -8.04 | 53.5% | 44.7% | -6.01 |

### 🕒 Time-of-Day Performance Breakdown (Top-3, Pooled OOS)
| Time | Long Trades | Long Raw WR | Long Net bps | Short Trades | Short Raw WR | Short Net bps |
|---|---|---|---|---|---|---|
| 09:15 | 2040 | 52.4% | +1.25 | 2040 | 54.0% | -4.89 |
| 10:15 | 2085 | 51.5% | -4.05 | 2085 | 53.9% | -2.06 |
| 11:15 | 2091 | 49.4% | -4.95 | 2091 | 53.8% | -5.14 |
| 12:15 | 2085 | 53.9% | -1.44 | 2085 | 53.1% | -5.72 |
| 13:15 | 2085 | 54.5% | +2.19 | 2085 | 54.4% | -0.07 |

================================================================================
