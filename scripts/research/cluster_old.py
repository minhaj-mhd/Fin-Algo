import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

conn = sqlite3.connect('data/vanguard_trades.db')
df = pd.read_sql("SELECT * FROM trades WHERE status IN ('VETOED', 'VETOED_EXPIRED')", conn)
df['final_profit_pct'] = pd.to_numeric(df['final_profit_pct'], errors='coerce')

# The older/early-stage features
features = ['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']

tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}
if 'tv_sentiment' in df.columns:
    df['tv_sentiment'] = df['tv_sentiment'].replace(tv_map)
    df['tv_sentiment'] = pd.to_numeric(df['tv_sentiment'], errors='coerce').fillna(0.0)

for f in features:
    if f not in df.columns:
        df[f] = 0.0
    else:
        df[f] = pd.to_numeric(df[f].astype(str).str.replace('%', '').str.strip(), errors='coerce')

df_clean = df.copy()
df_clean[features] = df_clean[features].fillna(0.0)

print("Feature variances:")
for f in features:
    print(f"  {f}: {df_clean[f].var()}")

X = df_clean[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

summary = df_clean.groupby('cluster').agg({
    'trade_id': 'count',
    'final_profit_pct': ['count', 'mean'],
    **{f: 'mean' for f in features}
})

# Flatten MultiIndex columns for printing
summary.columns = ['_'.join(col).strip() if type(col) is tuple else col for col in summary.columns.values]
summary = summary.rename(columns={'trade_id_count': 'trades_count', 'final_profit_pct_count': 'trades_with_pnl', 'final_profit_pct_mean': 'avg_pnl'})

print("\n--- CLUSTERING RESULTS ---")
print(summary.to_markdown())

# Save this to the finalgo memory layer as a follow up
report = f"""# Unsupervised Clustering (Early Stage Features Only)

## Methodology
Clustered the {len(df)} vetoed trades using only the early-stage features that are fully populated prior to advanced XGBoost scoring:
`{features}`

## Results
{summary.to_markdown()}

## Insights
- By restricting the clustering to only the early gatekeeper features (tech_score, nlp_sentiment, tv_sentiment, one_hour_prob), we can successfully cluster all {len(df)} vetoed trades.
"""

with open(r'c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\05. Archives\Unsupervised_Vetoed_Trades_Analysis.md', 'a') as f:
    f.write("\n\n" + report)
