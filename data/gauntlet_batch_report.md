# Gauntlet Parallel Batch Run (v10 to v19 with step_months=2)
**Date**: 2026-06-11 00:54:32
**Total Elapsed Time**: 2543.2 seconds

| Model Name | Status | Run ID | Long Verdict | Short Verdict | Time (s) | Notes/Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `v10_native_1h` | SUCCESS | `20260610T184210Z-d795438c` | FILTER_GRADE | FILTER_GRADE | 632.9 |  |
| `v10_depth4_1h` | SUCCESS | `20260610T184210Z-d795438c` | FILTER_GRADE | FILTER_GRADE | 650.9 |  |
| `v11_utility_1h` | SUCCESS | `20260610T184210Z-d795438c` | FILTER_GRADE | FILTER_GRADE | 636.4 |  |
| `v12_lambdamart_1h` | SUCCESS | `20260610T185243Z-d795438c` | DEAD | FILTER_GRADE | 599.1 |  |
| `v13_ndcg_raw_1h` | SUCCESS | `20260610T185246Z-d795438c` | DEAD | FILTER_GRADE | 595.6 |  |
| `v14_lambdamart_no_es_1h` | SUCCESS | `20260610T185301Z-d795438c` | FILTER_GRADE | FILTER_GRADE | 636.1 |  |
| `v15_lambdamart_es_1h` | SUCCESS | `20260610T190242Z-d795438c` | FILTER_GRADE | FILTER_GRADE | 591.1 |  |
| `v15_lambdamart_map5_1h` | FAIL | `N/A` | N/A | N/A | 1.3 | FileNotFoundError: Model metadata not found at models\v15_lambdamart_map5_1h\metadata.json |
| `v16_binary_breakout_1h` | SUCCESS | `20260610T190244Z-d795438c` | FILTER_GRADE | FILTER_GRADE | 594.2 |  |
| `v17_random_forest_1h` | TIMEOUT | `N/A` | N/A | N/A | 720.1 | Execution timed out (12 minutes) |
| `v18_random_forest_1h` | TIMEOUT | `N/A` | N/A | N/A | 720.1 | Execution timed out (12 minutes) |
| `v19_catboost_1h` | SUCCESS | `20260610T191237Z-d795438c` | FILTER_GRADE | FILTER_GRADE | 499.4 |  |