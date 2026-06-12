# Vanguard Trades Database Schema & Feature Set Analysis

**Total Trades in Database:** 1466

Below is the complete list of columns in the `trades` table, along with their fill rates, unique value counts, and sample values.

| Column Name | Data Type | Missing Count (%) | Fill Rate (%) | Unique Values | Samples |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `id` | int64 | 0 (0.00%) | 100.00% | 1466 | 37, 38, 39 |
| `trade_id` | str | 0 (0.00%) | 100.00% | 1466 | V-1778561145-ESCORTS.NS, V-1778561152-TCS.NS, V-1778561160-LAURUSLABS.NS |
| `timestamp` | str | 0 (0.00%) | 100.00% | 1466 | 2026-05-12T10:15:45.706405, 2026-05-12T10:15:52.310504, 2026-05-12T10:16:00.266784 |
| `ticker` | str | 0 (0.00%) | 100.00% | 171 | ESCORTS.NS, TCS.NS, LAURUSLABS.NS |
| `side` | str | 0 (0.00%) | 100.00% | 2 | LONG, LONG, SHORT |
| `tech_score` | float64 | 0 (0.00%) | 100.00% | 1338 | 0.20789316296577454, 0.17691192030906677, 0.2229892611503601 |
| `nlp_sentiment` | float64 | 157 (10.71%) | 89.29% | 5 | 0.0, 0.0, 1.0 |
| `entry_price` | float64 | 0 (0.00%) | 100.00% | 1422 | 2938.10009765625, 2301.0, 1271.0999755859375 |
| `exit_price` | float64 | 17 (1.16%) | 98.84% | 1404 | 2970.699951171875, 2303.800048828125, 1270.0999755859375 |
| `peak_price` | float64 | 0 (0.00%) | 100.00% | 1414 | 2970.699951171875, 2303.800048828125, 1269.300048828125 |
| `peak_profit_pct` | float64 | 0 (0.00%) | 100.00% | 1297 | 1.1095555778249424, 0.12168834542046936, 0.14160387006401995 |
| `final_profit_pct` | float64 | 0 (0.00%) | 100.00% | 1340 | 1.1096, 0.1217, 0.0787 |
| `exit_time` | str | 0 (0.00%) | 100.00% | 1465 | 2026-05-12T11:15:45.706405, 2026-05-12T11:15:52.310504, 2026-05-12T11:16:00.266784 |
| `status` | str | 0 (0.00%) | 100.00% | 5 | VETOED, VETOED, VETOED |
| `comment` | str | 0 (0.00%) | 100.00% | 1392 | VETO: The hourly return of -320.98% indicates an extreme and catastrophic price collapse over the last hour, signaling overwhelming selling pressure and severe negative momentum. This far outweighs any other metric and makes a long position incredibly risky. The very low technical conviction score (0.2079) and below-average relative volume (0.54) further confirm a weak setup for an upward move., VETO: The hourly return of -1189.98% indicates an extreme and catastrophic price decline within the last hour. This, combined with a very low technical conviction score (0.1769) for a long position, a significant distance from the 52-week high (-34.4%), and high relative volume (3.17) suggesting heavy selling pressure, points to a profoundly bearish market sentiment for the next hour. The technicals provide no support for a long trade; rather, they signal a severe downtrend., VETO: The stock has displayed extremely strong bullish momentum with an 18.87% hourly return and is currently trading above its 52-week high. The technical conviction score for a short trade is very low (0.2230), indicating technicals do not support a bearish position. Despite RVOL and Dollar Volume being reported as 0.0, suggesting potential data anomalies or extreme illiquidity, the significant price action metrics unequivocally point to robust buying pressure and a strong upward trend. |
| `one_hour_prob` | str | 113 (7.71%) | 92.29% | 69 | 62%, 62%, 42% |
| `quantity` | float64 | 786 (53.62%) | 46.38% | 59 | 325.0, 79.0, 55.0 |
| `net_pnl_amt` | float64 | 1244 (84.86%) | 15.14% | 67 | -376.28718749998524, -129.75609999998562, -172.415 |
| `margin_used` | float64 | 791 (53.96%) | 46.04% | 68 | 19862.4, 19273.239999999998, 19195.96 |
| `buy_brokerage` | float64 | 846 (57.71%) | 42.29% | 2 | 10.0, 0.0, 0.0 |
| `tv_sentiment` | str | 308 (21.01%) | 78.99% | 6 | SELL, NEUTRAL, SELL |
| `pending_since` | str | 1436 (97.95%) | 2.05% | 30 | 2026-05-21T12:10:38.904104, 2026-05-21T11:22:55.785594, 2026-05-21T12:46:07.411581 |
| `extension_count` | float64 | 986 (67.26%) | 32.74% | 1 | 0.0, 0.0, 0.0 |
| `extended_exit_time` | object | 1466 (100.00%) | 0.00% | 0 |  |
| `extension_pending` | float64 | 500 (34.11%) | 65.89% | 2 | 0.0, 0.0, 0.0 |
| `stop_loss_pct` | float64 | 828 (56.48%) | 43.52% | 29 | 0.939, 0.687, 0.838 |
| `take_profit_pct` | float64 | 828 (56.48%) | 43.52% | 29 | 1.879, 1.375, 1.677 |
| `trailing_active` | float64 | 500 (34.11%) | 65.89% | 2 | 0.0, 0.0, 0.0 |
| `breakeven_locked` | float64 | 500 (34.11%) | 65.89% | 2 | 0.0, 0.0, 0.0 |
| `long_score` | float64 | 956 (65.21%) | 34.79% | 503 | 0.08678742498159409, 0.04917588829994202, 0.05077197402715683 |
| `short_score` | float64 | 956 (65.21%) | 34.79% | 492 | -0.2173031121492386, -0.22615061700344086, -0.22436437010765076 |
| `strategy_id` | float64 | 1290 (87.99%) | 12.01% | 2 | 10.0, 2.0, 10.0 |
| `score_15m` | float64 | 1437 (98.02%) | 1.98% | 29 | 0.1642128974199295, 0.19303934276103973, -0.12529483437538147 |
| `score_30m` | float64 | 1437 (98.02%) | 1.98% | 29 | 0.06883291900157928, 0.10182878375053406, -0.07028065621852875 |
| `score_1d` | float64 | 1437 (98.02%) | 1.98% | 29 | 0.0819670557975769, 0.14702798426151276, -0.08256359398365021 |
| `is_ensemble` | float64 | 1437 (98.02%) | 1.98% | 2 | 1.0, 1.0, 0.0 |
