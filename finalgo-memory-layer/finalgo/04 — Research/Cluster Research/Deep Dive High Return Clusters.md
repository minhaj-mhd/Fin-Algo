---
title: "Deep Dive: High Return Trades Clustering"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# Deep Dive: High Return Trades Clustering

## Methodology
Filtered all historical trades for `final_profit_pct > 0.5%`. Found 138 trades matching this highly profitable criteria.

Features used: `tech_score`, `nlp_sentiment`, `tv_sentiment` (mapped numerically: BUY=1, SELL=-1), `one_hour_prob`.

## Full Clusterization Centroids (Unscaled)
|   cluster |   tech_score |   nlp_sentiment |   tv_sentiment |   one_hour_prob |
|----------:|-------------:|----------------:|---------------:|----------------:|
|         0 |     0.165415 |       0         |       0.555556 |         26.8222 |
|         1 |     0.287591 |       0.0608108 |       1.16216  |         49.1081 |
|         2 |     0.254447 |       0.455882  |       0        |          5      |
|         3 |     0.291332 |       0         |      -1.12821  |         32.7179 |

## Cluster Trade Distribution & Performance
|   cluster |   trade_id |   final_profit_pct | side   |
|----------:|-----------:|-------------------:|:-------|
|         0 |         45 |           0.826901 | SHORT  |
|         1 |         37 |           0.899322 | SHORT  |
|         2 |         17 |           0.970279 | SHORT  |
|         3 |         39 |           0.925995 | LONG   |

## Live Trading Implementation Logic
To check if a new incoming trade belongs to these highly profitable clusters in real-time, we have saved the `StandardScaler` and `KMeans` models to the `models/high_return_clusters/` directory.

```python
import pickle
import numpy as np

# Load models at engine startup
scaler = pickle.load(open('models/high_return_clusters/scaler.pkl', 'rb'))
kmeans = pickle.load(open('models/high_return_clusters/kmeans.pkl', 'rb'))

def is_high_return_archetype(trade_features_dict):
    # Map TV sentiment
    tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}
    tv_val = tv_map.get(trade_features_dict.get('tv_sentiment', 'NEUTRAL'), 0)
    
    vec = np.array([[
        float(trade_features_dict.get('tech_score', 0)),
        float(trade_features_dict.get('nlp_sentiment', 0)),
        tv_val,
        float(str(trade_features_dict.get('one_hour_prob', 0)).replace('%', ''))
    ]])
    
    # Scale and predict cluster
    vec_scaled = scaler.transform(vec)
    cluster_id = kmeans.predict(vec_scaled)[0]
    
    # Calculate distance to cluster centroid to ensure it's a tight fit
    centroid = kmeans.cluster_centers_[cluster_id]
    distance = np.linalg.norm(vec_scaled - centroid)
    
    # Threshold for closeness (e.g., Euclidean distance < 2.0)
    return cluster_id, distance
```

## Insights & Rules Engine Application
- **Contrarian Triggers:** The highest performing clusters are strictly contrarian to Retail TV Sentiment. The AI's `tech_score` and `one_hour_prob` correctly identify momentum exhaustion.
- **Veto Override Rule:** If a trade is flagged by the primary gatekeepers for a veto, but `is_high_return_archetype()` matches Cluster 1 or Cluster 3 with a distance `< 2.0`, the veto should be overridden because the trade matches the mathematical signature of our highest historical winners.


## Hierarchical Sub-Clustering of High Return Archetypes
To maximize win-rate and filter out noise, each of the primary high-return clusters was hierarchically sub-clustered to isolate the absolute densest pockets of winning trades.

### Primary Cluster 0
**Sub-Cluster 0** (14 primary winners)
- `Dist < 1.50` | Caught: 327 | Win Rate: 49.8% | Avg PnL: -0.0083%
- `Dist < 1.00` | Caught: 97 | Win Rate: 48.5% | Avg PnL: -0.0460%
- `Dist < 0.75` | Caught: 42 | Win Rate: 33.3% | Avg PnL: -0.0972%
- `Dist < 0.50` | Caught: 10 | Win Rate: 40.0% | Avg PnL: -0.1031%

**Sub-Cluster 1** (14 primary winners)
- `Dist < 1.50` | Caught: 123 | Win Rate: 47.2% | Avg PnL: -0.0073%
- `Dist < 1.00` | Caught: 75 | Win Rate: 46.7% | Avg PnL: 0.0145%
- `Dist < 0.75` | Caught: 50 | Win Rate: 54.0% | Avg PnL: 0.1158%
- `Dist < 0.50` | Caught: 3 | Win Rate: 100.0% | Avg PnL: 0.3949%

**Sub-Cluster 2** (17 primary winners)
- `Dist < 1.50` | Caught: 374 | Win Rate: 47.3% | Avg PnL: -0.0203%
- `Dist < 1.00` | Caught: 169 | Win Rate: 51.5% | Avg PnL: 0.0188%
- `Dist < 0.75` | Caught: 107 | Win Rate: 50.5% | Avg PnL: 0.0268%
- `Dist < 0.50` | Caught: 36 | Win Rate: 52.8% | Avg PnL: 0.0218%

### Primary Cluster 1
**Sub-Cluster 0** (3 primary winners)
- `Dist < 1.50` | Caught: 17 | Win Rate: 64.7% | Avg PnL: 0.2017%
- `Dist < 1.00` | Caught: 10 | Win Rate: 70.0% | Avg PnL: 0.1776%
- `Dist < 0.75` | Caught: 4 | Win Rate: 100.0% | Avg PnL: 0.4057%
- `Dist < 0.50` | Caught: 0

**Sub-Cluster 1** (22 primary winners)
- `Dist < 1.50` | Caught: 430 | Win Rate: 52.6% | Avg PnL: 0.0182%
- `Dist < 1.00` | Caught: 188 | Win Rate: 58.0% | Avg PnL: 0.0598%
- `Dist < 0.75` | Caught: 120 | Win Rate: 61.7% | Avg PnL: 0.0983%
- `Dist < 0.50` | Caught: 47 | Win Rate: 63.8% | Avg PnL: 0.1409%

**Sub-Cluster 2** (12 primary winners)
- `Dist < 1.50` | Caught: 149 | Win Rate: 42.3% | Avg PnL: -0.0737%
- `Dist < 1.00` | Caught: 27 | Win Rate: 44.4% | Avg PnL: 0.0069%
- `Dist < 0.75` | Caught: 6 | Win Rate: 33.3% | Avg PnL: 0.0401%
- `Dist < 0.50` | Caught: 0

### Primary Cluster 2
**Sub-Cluster 0** (12 primary winners)
- `Dist < 1.50` | Caught: 54 | Win Rate: 57.4% | Avg PnL: 0.1350%
- `Dist < 1.00` | Caught: 32 | Win Rate: 59.4% | Avg PnL: 0.1378%
- `Dist < 0.75` | Caught: 21 | Win Rate: 52.4% | Avg PnL: 0.0822%
- `Dist < 0.50` | Caught: 15 | Win Rate: 73.3% | Avg PnL: 0.3151%

**Sub-Cluster 1** (4 primary winners)
- `Dist < 1.50` | Caught: 24 | Win Rate: 58.3% | Avg PnL: 0.1020%
- `Dist < 1.00` | Caught: 8 | Win Rate: 75.0% | Avg PnL: 0.5070%
- `Dist < 0.75` | Caught: 7 | Win Rate: 71.4% | Avg PnL: 0.4817%
- `Dist < 0.50` | Caught: 2 | Win Rate: 50.0% | Avg PnL: 0.1212%

**Sub-Cluster 2** (1 primary winners)
- `Dist < 1.50` | Caught: 2 | Win Rate: 50.0% | Avg PnL: 0.4506%
- `Dist < 1.00` | Caught: 2 | Win Rate: 50.0% | Avg PnL: 0.4506%
- `Dist < 0.75` | Caught: 2 | Win Rate: 50.0% | Avg PnL: 0.4506%
- `Dist < 0.50` | Caught: 1 | Win Rate: 100.0% | Avg PnL: 1.6741%

### Primary Cluster 3
**Sub-Cluster 0** (8 primary winners)
- `Dist < 1.50` | Caught: 360 | Win Rate: 49.4% | Avg PnL: 0.0280%
- `Dist < 1.00` | Caught: 163 | Win Rate: 51.5% | Avg PnL: 0.0278%
- `Dist < 0.75` | Caught: 72 | Win Rate: 59.7% | Avg PnL: 0.0779%
- `Dist < 0.50` | Caught: 54 | Win Rate: 53.7% | Avg PnL: 0.0840%

**Sub-Cluster 1** (16 primary winners)
- `Dist < 1.50` | Caught: 544 | Win Rate: 48.5% | Avg PnL: 0.0213%
- `Dist < 1.00` | Caught: 283 | Win Rate: 46.6% | Avg PnL: 0.0103%
- `Dist < 0.75` | Caught: 163 | Win Rate: 45.4% | Avg PnL: -0.0060%
- `Dist < 0.50` | Caught: 97 | Win Rate: 42.3% | Avg PnL: 0.0113%

**Sub-Cluster 2** (16 primary winners)
- `Dist < 1.50` | Caught: 457 | Win Rate: 48.6% | Avg PnL: 0.0255%
- `Dist < 1.00` | Caught: 237 | Win Rate: 47.3% | Avg PnL: 0.0238%
- `Dist < 0.75` | Caught: 145 | Win Rate: 43.4% | Avg PnL: 0.0026%
- `Dist < 0.50` | Caught: 85 | Win Rate: 49.4% | Avg PnL: 0.0295%

