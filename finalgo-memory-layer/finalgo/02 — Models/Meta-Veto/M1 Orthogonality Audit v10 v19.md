---
title: "M1 Orthogonality Audit: v10-v19 Stacking Panel"
type: report
status: active
model: "Meta-Veto"
updated: 2026-06-12
tags: []
---
# 📊 M1 Orthogonality Audit: v10-v19 Stacking Panel

- **Date**: 2026-06-11 10:13:53
- **Verdict**: 🟢 PASS
- **Max Absolute Partial IC**: `0.025839`
- **Best Feature**: `daily_v3_sent`
- **Threshold**: `0.005`
- **Total Features Audited**: 33

## 📈 Top 15 Features by Partial Rank-IC
| Feature | Raw Rank-IC | Partial Rank-IC (controlling for own_pct) |
| :--- | :---: | :---: |
| `daily_v3_sent` | +0.0272 | +0.0258 |
| `v10_depth4_1h_score` | +0.0288 | +0.0221 |
| `v11_utility_1h_score` | +0.0288 | +0.0221 |
| `v14_lambdamart_no_es_1h_z` | +0.0226 | +0.0195 |
| `v16_binary_breakout_1h_z` | +0.0226 | +0.0195 |
| `v19_catboost_1h_z` | +0.0226 | +0.0195 |
| `v15_lambdamart_es_1h_z` | +0.0226 | +0.0195 |
| `daily_v2_pct` | +0.0194 | +0.0188 |
| `v10_depth4_1h_z` | +0.0258 | +0.0174 |
| `v11_utility_1h_z` | +0.0258 | +0.0174 |
| `v13_ndcg_raw_1h_z` | +0.0110 | +0.0121 |
| `v12_lambdamart_1h_z` | +0.0110 | +0.0121 |
| `hour` | +0.0132 | +0.0121 |
| `v14_lambdamart_no_es_1h_score` | +0.0129 | +0.0099 |
| `v19_catboost_1h_score` | +0.0129 | +0.0099 |


*This audit verifies whether features beyond the anchor model (`v10_native_1h`) possess orthogonal predictive signal.*