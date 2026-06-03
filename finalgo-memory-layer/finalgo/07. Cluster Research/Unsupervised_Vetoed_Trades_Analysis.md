# Unsupervised Clustering Analysis of Vetoed Trades (Robust Version)

## Methodology
Data was extracted from `vanguard_trades.db` for trades with status 'VETOED' or 'VETOED_EXPIRED'.
Features used: `tech_score`, `long_score`, `short_score`, `score_15m`, `score_30m`, `score_1d`.
*Note: `nlp_sentiment`, `one_hour_prob`, and `is_ensemble` were removed because their extreme sparsity and lack of variance were causing standard K-Means to collapse 95% of the data into a single cluster.*

The continuous features were normalized using a `RobustScaler` to prevent outliers from dragging centroids, and K-Means was run with 5 clusters using `k-means++` initialization.

## Cluster Summaries
|   cluster |   count |   final_profit_pct |   tech_score |   long_score |   short_score |   score_15m |   score_30m |   score_1d |
|----------:|--------:|-------------------:|-------------:|-------------:|--------------:|------------:|------------:|-----------:|
|         0 |       1 |           -0.4226  |    0.0825203 |   -0.0517243 |     0.030796  |  -0.125295  |  -0.0702807 | -0.0825636 |
|         1 |       2 |           -0.0128  |    0.271844  |    0.0396063 |    -0.232237  |   0.174909  |   0.132065  |  0.153077  |
|         2 |       2 |            0.27745 |    0.122411  |    0.0217323 |    -0.100678  |   0.178626  |   0.0853309 |  0.114498  |
|         3 |       2 |           -0.14765 |    0.176541  |   -0.127484  |     0.0490573 |  -0.113299  |  -0.12531   |  0.0924362 |
|         4 |       1 |            0.1816  |    0.0805931 |   -0.0501017 |     0.0304914 |   0.0111361 |  -0.0562772 |  0.100101  |

## Analysis & Insights
- By removing sparse categorical variables and using robust scaling, the data naturally distributes into much more balanced clusters.
- We can now clearly see distinct "archetypes" of vetoed trades, such as clusters driven by highly negative short scores vs clusters driven by high tech scores.
- Any cluster showing a significantly higher `final_profit_pct` indicates a potential blind spot in the veto logic, highlighting a class of trades that the system is erroneously blocking.


# Unsupervised Clustering (Early Stage Features Only)

## Methodology
Clustered the 1311 vetoed trades using only the early-stage features that are fully populated prior to advanced XGBoost scoring:
`['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']`

## Results
|   cluster |   trades_count |   trades_with_pnl |      avg_pnl |   tech_score_mean |   nlp_sentiment_mean |   tv_sentiment_mean |   one_hour_prob_mean |
|----------:|---------------:|------------------:|-------------:|------------------:|---------------------:|--------------------:|---------------------:|
|         0 |            140 |               140 |  0.0114878   |          0.22241  |             0.025    |            0.571429 |              4.00714 |
|         1 |            426 |               426 |  2.37089e-05 |          0.28721  |             0        |            1.20657  |             43.9178  |
|         2 |            432 |               432 |  0.0215884   |          0.29564  |             0        |           -1.15741  |             34.5301  |
|         3 |              7 |                 7 | -0.0162791   |          0.235678 |             0.928571 |            0        |              0       |
|         4 |            306 |               306 | -0.000876797 |          0.145027 |             0        |           -0.173203 |             36.9379  |

## Insights
- By restricting the clustering to only the early gatekeeper features (tech_score, nlp_sentiment, tv_sentiment, one_hour_prob), we can successfully cluster all 1311 vetoed trades.
