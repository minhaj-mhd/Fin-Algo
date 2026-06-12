---
title: "daily_macro_v3 Gating Sweep Report (v10 to v19)"
type: report
status: active
model: "Gauntlet Reports"
updated: 2026-06-12
tags: []
---
# 📊 daily_macro_v3 Gating Sweep Report (v10 to v19)

> [!CAUTION] **MULTIPLICITY WARNING — the two "PASSED" cells are NOT certified (audit annotation, Claude, 2026-06-10).**
> This sweep ran 40 cells at a single-test bar (t≥2 / p<0.05); pure noise is expected to produce ~2 passes in 40, and exactly 2 were found (p=0.018, 0.013). The 40-way corrected bar is ~t≥3.0 — neither cell reaches it. Duplicate rows (v12≡v13; v14≡v15≡v16≡v19) mean only ~7 distinct prediction sets were tested, and the two "passes" are correlated siblings, not independent confirmations. Even in the passing cells the FAVORABLE trades remain net-negative (−0.95/−1.00 bps @6bps; binding cost is 10bps) — gating reduces losses, it does not create an edge. **Do not deploy these cells; do not "pre-register" a cell selected from this table on the same data (laundering).**
> **Legitimate signal in this table**: day-gate LONG uplift is positive in *every* distinct cell (+1.01 to +5.07) across two gatekeeper versions — uniform directional evidence (hypothesis-grade) that daily macro day-quality information helps long intraday trades. Path to certification: time-firewalled meta-model (fit pre-2025, certify 2025+) or re-test one pre-declared cell on freshly accumulated live OOS data. See AGENTS.md "Model Metric Discipline" rule 5.

- **Evaluated At**: 2026-06-10T20:55:48.475285
- **Daily Model**: `daily_macro_v3` (Run ID: `20260610T144343Z-5f7d069f`)
- **Friction Applied**: 6.0 bps round-trip

## Summary Sweep Results Table

| Model                   | Gate Mode   | Side   |   Fav Trades |   Unfav Trades |   Fav Net Return (bps) |   Unfav Net Return (bps) |   Net Uplift (bps) |   T-Statistic |   P-Value | Status   |
|:------------------------|:------------|:-------|-------------:|---------------:|-----------------------:|-------------------------:|-------------------:|--------------:|----------:|:---------|
| v10_native_1h           | symbol      | LONG   |         1684 |           1711 |                   0.21 |                    -3.9  |               4.11 |          1.78 |    0.0759 | FAILED   |
| v10_native_1h           | symbol      | SHORT  |         2456 |           1094 |                  -1.73 |                    -3.53 |               1.8  |          0.57 |    0.5717 | FAILED   |
| v10_native_1h           | day         | LONG   |         1821 |           1584 |                  -1.91 |                    -2.93 |               1.01 |          0.45 |    0.654  | FAILED   |
| v10_native_1h           | day         | SHORT  |         1569 |           1197 |                  -3.98 |                    -4.45 |               0.47 |          0.14 |    0.8921 | FAILED   |
| v10_depth4_1h           | symbol      | LONG   |         1540 |           1792 |                  -1.27 |                    -2.55 |               1.28 |          0.66 |    0.51   | FAILED   |
| v10_depth4_1h           | symbol      | SHORT  |         2520 |           1072 |                  -3.05 |                     1.18 |              -4.23 |         -1.31 |    0.191  | FAILED   |
| v10_depth4_1h           | day         | LONG   |         1821 |           1584 |                  -0.95 |                    -5.34 |               4.39 |          2.37 |    0.018  | PASSED   |
| v10_depth4_1h           | day         | SHORT  |         1569 |           1197 |                  -1.12 |                    -7.26 |               6.14 |          1.63 |    0.1032 | FAILED   |
| v11_utility_1h          | symbol      | LONG   |         1667 |           1700 |                  -3.54 |                    -4.01 |               0.47 |          0.22 |    0.8236 | FAILED   |
| v11_utility_1h          | symbol      | SHORT  |         2543 |           1058 |                  -1.73 |                    -2.68 |               0.95 |          0.28 |    0.7778 | FAILED   |
| v11_utility_1h          | day         | LONG   |         1821 |           1584 |                  -1    |                    -6.06 |               5.07 |          2.47 |    0.0134 | PASSED   |
| v11_utility_1h          | day         | SHORT  |         1569 |           1197 |                  -2.92 |                    -2.22 |              -0.7  |         -0.19 |    0.8476 | FAILED   |
| v12_lambdamart_1h       | symbol      | LONG   |         1596 |           1734 |                  -5.68 |                    -3.97 |              -1.71 |         -0.63 |    0.5295 | FAILED   |
| v12_lambdamart_1h       | symbol      | SHORT  |         2327 |           1165 |                   0.24 |                    -4.6  |               4.84 |          1.46 |    0.1452 | FAILED   |
| v12_lambdamart_1h       | day         | LONG   |         1821 |           1584 |                  -4.11 |                    -7.22 |               3.11 |          1.27 |    0.2039 | FAILED   |
| v12_lambdamart_1h       | day         | SHORT  |         1569 |           1197 |                  -3.34 |                    -5.53 |               2.18 |          0.58 |    0.5625 | FAILED   |
| v13_ndcg_raw_1h         | symbol      | LONG   |         1596 |           1734 |                  -5.68 |                    -3.97 |              -1.71 |         -0.63 |    0.5295 | FAILED   |
| v13_ndcg_raw_1h         | symbol      | SHORT  |         2327 |           1165 |                   0.24 |                    -4.6  |               4.84 |          1.46 |    0.1452 | FAILED   |
| v13_ndcg_raw_1h         | day         | LONG   |         1821 |           1584 |                  -4.11 |                    -7.22 |               3.11 |          1.27 |    0.2039 | FAILED   |
| v13_ndcg_raw_1h         | day         | SHORT  |         1569 |           1197 |                  -3.34 |                    -5.53 |               2.18 |          0.58 |    0.5625 | FAILED   |
| v14_lambdamart_no_es_1h | symbol      | LONG   |         1604 |           1728 |                  -2.3  |                    -6    |               3.7  |          1.47 |    0.141  | FAILED   |
| v14_lambdamart_no_es_1h | symbol      | SHORT  |         2431 |           1157 |                  -4.09 |                    -3.44 |              -0.64 |         -0.22 |    0.8294 | FAILED   |
| v14_lambdamart_no_es_1h | day         | LONG   |         1821 |           1584 |                  -5.1  |                    -6.22 |               1.12 |          0.47 |    0.6351 | FAILED   |
| v14_lambdamart_no_es_1h | day         | SHORT  |         1569 |           1197 |                  -4.01 |                    -5.36 |               1.35 |          0.4  |    0.6926 | FAILED   |
| v15_lambdamart_es_1h    | symbol      | LONG   |         1604 |           1728 |                  -2.3  |                    -6    |               3.7  |          1.47 |    0.141  | FAILED   |
| v15_lambdamart_es_1h    | symbol      | SHORT  |         2431 |           1157 |                  -4.09 |                    -3.44 |              -0.64 |         -0.22 |    0.8294 | FAILED   |
| v15_lambdamart_es_1h    | day         | LONG   |         1821 |           1584 |                  -5.1  |                    -6.22 |               1.12 |          0.47 |    0.6351 | FAILED   |
| v15_lambdamart_es_1h    | day         | SHORT  |         1569 |           1197 |                  -4.01 |                    -5.36 |               1.35 |          0.4  |    0.6926 | FAILED   |
| v16_binary_breakout_1h  | symbol      | LONG   |         1604 |           1728 |                  -2.3  |                    -6    |               3.7  |          1.47 |    0.141  | FAILED   |
| v16_binary_breakout_1h  | symbol      | SHORT  |         2431 |           1157 |                  -4.09 |                    -3.44 |              -0.64 |         -0.22 |    0.8294 | FAILED   |
| v16_binary_breakout_1h  | day         | LONG   |         1821 |           1584 |                  -5.1  |                    -6.22 |               1.12 |          0.47 |    0.6351 | FAILED   |
| v16_binary_breakout_1h  | day         | SHORT  |         1569 |           1197 |                  -4.01 |                    -5.36 |               1.35 |          0.4  |    0.6926 | FAILED   |
| v17_random_forest_1h    | symbol      | LONG   |         1654 |           1771 |                  -4.43 |                    -4.66 |               0.22 |          0.08 |    0.9387 | FAILED   |
| v17_random_forest_1h    | symbol      | SHORT  |         2528 |           1082 |                  -3.7  |                    -6.79 |               3.09 |          0.86 |    0.3922 | FAILED   |
| v17_random_forest_1h    | day         | LONG   |         1821 |           1584 |                  -3.76 |                    -7.2  |               3.44 |          1.2  |    0.2301 | FAILED   |
| v17_random_forest_1h    | day         | SHORT  |         1569 |           1197 |                  -3.92 |                    -9    |               5.08 |          1.23 |    0.2189 | FAILED   |
| v18_random_forest_1h    | symbol      | LONG   |         1566 |           1823 |                  -6.37 |                    -6.33 |              -0.03 |         -0.02 |    0.9859 | FAILED   |
| v18_random_forest_1h    | symbol      | SHORT  |         1789 |           1687 |                  -3.99 |                    -7.89 |               3.9  |          1.7  |    0.0884 | FAILED   |
| v18_random_forest_1h    | day         | LONG   |         1821 |           1584 |                  -5.49 |                    -7.1  |               1.61 |          0.91 |    0.3637 | FAILED   |
| v18_random_forest_1h    | day         | SHORT  |         1569 |           1197 |                  -5.71 |                    -8.27 |               2.56 |          0.99 |    0.3207 | FAILED   |
| v19_catboost_1h         | symbol      | LONG   |         1604 |           1728 |                  -2.3  |                    -6    |               3.7  |          1.47 |    0.141  | FAILED   |
| v19_catboost_1h         | symbol      | SHORT  |         2431 |           1157 |                  -4.09 |                    -3.44 |              -0.64 |         -0.22 |    0.8294 | FAILED   |
| v19_catboost_1h         | day         | LONG   |         1821 |           1584 |                  -5.1  |                    -6.22 |               1.12 |          0.47 |    0.6351 | FAILED   |
| v19_catboost_1h         | day         | SHORT  |         1569 |           1197 |                  -4.01 |                    -5.36 |               1.35 |          0.4  |    0.6926 | FAILED   |

---
*Report generated programmatically via evaluate_v3_gating_sweeps.py.*
