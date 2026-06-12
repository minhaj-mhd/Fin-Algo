---
title: "Meta-Veto Orthogonality & Information Audit Report"
type: report
status: active
model: "Meta-Veto"
updated: 2026-06-12
tags: []
---
# 🧠 Meta-Veto Orthogonality & Information Audit Report

- **Status**: 🟢 PASSED (Proceed to M2)
- **Max Partial Rank-IC**: `0.25361` (Feature: `sibling_d4_pct` on `v2_15min_3y`)
- **DEV Span Rows**: `1,126` candidate trades
- **Timestamp**: `2026-06-10T21:29:05.414345`

## 📊 Pairwise Spearman Correlation Matrix (Features + own_pct)

|                 |          hour |   day_of_week |   sibling_d4_pct |   sibling_v11_pct |   cross_tf_pct |   vix_level |      vix_pct |   daily_v2_pct |   daily_v2_sent |   daily_v3_pct |   daily_v3_sent |   macro_gate |       own_pct |
|:----------------|--------------:|--------------:|-----------------:|------------------:|---------------:|------------:|-------------:|---------------:|----------------:|---------------:|----------------:|-------------:|--------------:|
| hour            |   1           |   -0.0113975  |       0.0684706  |         0.0696944 |     0.0423688  |         nan |  -0.00396657 |      0.0924231 |      0.0027726  |     0.106562   |     -0.00278829 |  -0.00166876 |   0.000618398 |
| day_of_week     |  -0.0113975   |    1          |       0.00822085 |         0.0114571 |    -0.0116123  |         nan |   0.222243   |      0.0115901 |      0.128846   |     0.034652   |     -0.04038    |   0.0370101  |  -0.00850782  |
| sibling_d4_pct  |   0.0684706   |    0.00822085 |       1          |         0.785309  |     0.0678365  |         nan |   0.00156262 |      0.0882786 |      0.0161307  |     0.127198   |      0.136265   |   0.0372934  |   0.130641    |
| sibling_v11_pct |   0.0696944   |    0.0114571  |       0.785309   |         1         |     0.0427341  |         nan |  -0.0133308  |      0.0945028 |      0.0426316  |     0.133437   |      0.132988   |   0.0123615  |   0.119931    |
| cross_tf_pct    |   0.0423688   |   -0.0116123  |       0.0678365  |         0.0427341 |     1          |         nan |  -0.0071851  |      0.0537899 |     -0.00452208 |     0.0829404  |      0.126976   |   0.0235312  |   0.111992    |
| vix_level       | nan           |  nan          |     nan          |       nan         |   nan          |         nan | nan          |    nan         |    nan          |   nan          |    nan          | nan          | nan           |
| vix_pct         |  -0.00396657  |    0.222243   |       0.00156262 |        -0.0133308 |    -0.0071851  |         nan |   1          |     -0.006122  |     -0.344693   |     0.00845653 |     -0.245072   |   0.0014694  |  -0.0209273   |
| daily_v2_pct    |   0.0924231   |    0.0115901  |       0.0882786  |         0.0945028 |     0.0537899  |         nan |  -0.006122   |      1         |     -0.0202823  |     0.750789   |      0.141531   |  -0.0179492  |   0.0288768   |
| daily_v2_sent   |   0.0027726   |    0.128846   |       0.0161307  |         0.0426316 |    -0.00452208 |         nan |  -0.344693   |     -0.0202823 |      1          |    -0.0871401  |     -0.00253844 |   0.0187919  |   0.00752447  |
| daily_v3_pct    |   0.106562    |    0.034652   |       0.127198   |         0.133437  |     0.0829404  |         nan |   0.00845653 |      0.750789  |     -0.0871401  |     1          |      0.157551   |   0.00752868 |   0.0428123   |
| daily_v3_sent   |  -0.00278829  |   -0.04038    |       0.136265   |         0.132988  |     0.126976   |         nan |  -0.245072   |      0.141531  |     -0.00253844 |     0.157551   |      1          |  -0.0101423  |   0.00911554  |
| macro_gate      |  -0.00166876  |    0.0370101  |       0.0372934  |         0.0123615 |     0.0235312  |         nan |   0.0014694  |     -0.0179492 |      0.0187919  |     0.00752868 |     -0.0101423  |   1          |  -0.00915602  |
| own_pct         |   0.000618398 |   -0.00850782 |       0.130641   |         0.119931  |     0.111992   |         nan |  -0.0209273  |      0.0288768 |      0.00752447 |     0.0428123  |      0.00911554 |  -0.00915602 |   1           |

## 📈 Incremental Rank-IC (Partial Correlation)

This table reports the raw rank-IC (Spearman correlation with trade returns) and the partial rank-IC (controlling for the downstream own-model prediction score percentile):

| model        | feature         |   n_trades |        raw_ic |    partial_ic |
|:-------------|:----------------|-----------:|--------------:|--------------:|
| v2_15min_3y  | sibling_d4_pct  |        624 |  -0.255438    |  -0.25361     |
| v2_15min_3y  | sibling_v11_pct |        624 |  -0.160964    |  -0.159036    |
| v8_upstox_3y | daily_v3_sent   |        502 |   0.109653    |   0.109657    |
| v8_upstox_3y | sibling_d4_pct  |        502 |   0.0809853   |   0.0819879   |
| v2_15min_3y  | cross_tf_pct    |        624 |   0.070543    |   0.0769501   |
| v2_15min_3y  | macro_gate      |        624 |  -0.0739287   |  -0.0744356   |
| v8_upstox_3y | daily_v3_pct    |        502 |   0.069817    |   0.0697385   |
| v8_upstox_3y | cross_tf_pct    |        502 |   0.0589396   |   0.058842    |
| v8_upstox_3y | macro_gate      |        502 |   0.0412351   |   0.041242    |
| v8_upstox_3y | vix_level       |        502 | nan           |   0.0300435   |
| v8_upstox_3y | vix_pct         |        502 |   0.0286148   |   0.028615    |
| v2_15min_3y  | vix_pct         |        624 |  -0.0258215   |  -0.0270269   |
| v8_upstox_3y | daily_v2_pct    |        502 |   0.0265064   |   0.02637     |
| v8_upstox_3y | sibling_v11_pct |        502 |   0.0259367   |   0.0257908   |
| v8_upstox_3y | hour            |        502 |   0.0242912   |   0.0242951   |
| v8_upstox_3y | daily_v2_sent   |        502 |  -0.0176844   |  -0.0176838   |
| v2_15min_3y  | daily_v2_sent   |        624 |   0.0122777   |   0.0127172   |
| v2_15min_3y  | daily_v3_pct    |        624 |   0.0103717   |   0.0116793   |
| v2_15min_3y  | hour            |        624 |  -0.00992341  |  -0.0100455   |
| v8_upstox_3y | day_of_week     |        502 |   0.00884661  |   0.00885579  |
| v2_15min_3y  | day_of_week     |        624 |  -0.00646782  |  -0.00687752  |
| v2_15min_3y  | daily_v3_sent   |        624 |  -0.00440027  |  -0.00386585  |
| v2_15min_3y  | daily_v2_pct    |        624 |  -0.000302517 |  -4.74358e-05 |
| v2_15min_3y  | vix_level       |        624 | nan           | nan           |

## 🔍 Findings & Verdict

> [!NOTE]
> The audit **passed** because `sibling_d4_pct` on `v2_15min_3y` has an incremental rank-IC of `0.25361`, which exceeds the required floor of `0.0050`. This indicates that daily sentiment and/or cross-timeframe features carry orthogonal predictive value that can improve downstream trade outcomes.
