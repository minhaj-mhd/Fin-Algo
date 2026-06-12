# M1 Orthogonality Kill-Gate Report (MV2)

> [!NOTE]
> **Gate status**: PASS - Proceed to R1  
> **Max |partial IC|**: `0.05760` (feature: `cross_tf_pct` on `v8_upstox_3y`)  
> **Required**: >= 0.005  
> **Panel**: `data\gauntlet\meta\mv2_clean\trade_panel.parquet`  
> **Timestamp**: `2026-06-10T23:51:23.306354`

## Gate Qualifying Features

Features qualify if they are NOT own_score/own_z/own_pct or identity indicators (model_is_v8, side_is_long). Hour/ToD, cross-TF, daily scores, VIX/macro all qualify.

## Incremental Rank-IC Table

| model        | feature       |     n |   raw_ic |   partial_ic |
|:-------------|:--------------|------:|---------:|-------------:|
| v8_upstox_3y | cross_tf_pct  |  1937 |  0.05757 |      0.0576  |
| v2_15min_3y  | hour          | 11796 |  0.03007 |      0.03008 |
| v8_upstox_3y | daily_v2_pct  | 10410 |  0.02729 |      0.02713 |
| v8_upstox_3y | daily_v3_sent | 10410 |  0.02258 |      0.02246 |
| v2_15min_3y  | daily_v3_sent | 11796 |  0.01904 |      0.01913 |
| v8_upstox_3y | daily_v3_pct  | 10410 |  0.01741 |      0.01749 |
| v8_upstox_3y | daily_v2_sent | 10410 |  0.01672 |      0.01693 |
| v2_15min_3y  | cross_tf_pct  | 11779 |  0.01602 |      0.01563 |
| v8_upstox_3y | hour          | 10410 |  0.013   |      0.013   |
| v8_upstox_3y | day_of_week   | 10410 | -0.01276 |     -0.01275 |
| v8_upstox_3y | macro_gate    | 10410 |  0.01062 |      0.01063 |
| v2_15min_3y  | macro_gate    | 11796 | -0.00774 |     -0.008   |
| v2_15min_3y  | vix_pct       | 11796 | -0.00549 |     -0.00664 |
| v2_15min_3y  | daily_v2_sent | 11796 |  0.00391 |      0.00457 |
| v2_15min_3y  | daily_v3_pct  | 11796 |  0.00358 |      0.00345 |
| v8_upstox_3y | vix_pct       | 10410 |  0.00308 |      0.00321 |
| v2_15min_3y  | daily_v2_pct  | 11796 |  0.00308 |      0.00302 |
| v2_15min_3y  | day_of_week   | 11796 |  0.0023  |      0.00251 |

## Finding

> [!NOTE]
> Gate **PASSED**: `cross_tf_pct` on `v8_upstox_3y` has incremental IC = `0.05760` >= threshold. Stacking has a demonstrated premise. Proceed to R1 (dev_run.py).

## Backlinks

- [[02 — Models/Meta-Veto/Meta-Veto Rectification Plan MV2]]
- [[06 — Logs/Active Board]]
