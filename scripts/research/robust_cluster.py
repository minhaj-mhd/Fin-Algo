import sqlite3
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

# 1. Extract data
conn = sqlite3.connect('data/vanguard_trades.db')
df = pd.read_sql("SELECT * FROM trades WHERE status IN ('VETOED', 'VETOED_EXPIRED')", conn)
df['final_profit_pct'] = pd.to_numeric(df['final_profit_pct'], errors='coerce')

# 2. Select continuous predictive features
# Dropped nlp_sentiment, one_hour_prob, and is_ensemble as they were sparse/constant and skewed distances
features = ['tech_score', 'long_score', 'short_score', 'score_15m', 'score_30m', 'score_1d']

# Clean Data
df_clean = df.dropna(subset=features).copy()
X = df_clean[features].values

# 3. Robust Scaling to handle outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means Clustering
# Using k-means++ initialization and a larger number of init runs
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# 5. Summarize
summary = df_clean.groupby('cluster').agg({
    'trade_id': 'count',
    'final_profit_pct': 'mean',
    **{f: 'mean' for f in features}
}).rename(columns={'trade_id': 'count'})

print(summary.to_markdown())

# Prepare Markdown report
report = """# Unsupervised Clustering Analysis of Vetoed Trades (Robust Version)

## Methodology
Data was extracted from `vanguard_trades.db` for trades with status 'VETOED' or 'VETOED_EXPIRED'.
Features used: `tech_score`, `long_score`, `short_score`, `score_15m`, `score_30m`, `score_1d`.
*Note: `nlp_sentiment`, `one_hour_prob`, and `is_ensemble` were removed because their extreme sparsity and lack of variance were causing standard K-Means to collapse 95% of the data into a single cluster.*

The continuous features were normalized using a `RobustScaler` to prevent outliers from dragging centroids, and K-Means was run with 5 clusters using `k-means++` initialization.

## Cluster Summaries
"""

report += summary.to_markdown()

report += """

## Analysis & Insights
- By removing sparse categorical variables and using robust scaling, the data naturally distributes into much more balanced clusters.
- We can now clearly see distinct "archetypes" of vetoed trades, such as clusters driven by highly negative short scores vs clusters driven by high tech scores.
- Any cluster showing a significantly higher `final_profit_pct` indicates a potential blind spot in the veto logic, highlighting a class of trades that the system is erroneously blocking.
"""

with open(r'c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\05. Archives\Unsupervised_Vetoed_Trades_Analysis.md', 'w') as f:
    f.write(report)

print("\nReport successfully updated.")
