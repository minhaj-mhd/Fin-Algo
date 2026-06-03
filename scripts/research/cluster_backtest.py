import sqlite3
import pandas as pd
import pickle
import numpy as np

# Load Models
scaler = pickle.load(open('models/high_return_clusters/scaler.pkl', 'rb'))
kmeans = pickle.load(open('models/high_return_clusters/kmeans.pkl', 'rb'))

conn = sqlite3.connect('data/vanguard_trades.db')
df = pd.read_sql("SELECT * FROM trades", conn)
df['final_profit_pct'] = pd.to_numeric(df['final_profit_pct'], errors='coerce')

# We only care about trades that actually have PnL data
df = df.dropna(subset=['final_profit_pct']).copy()

features = ['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']
tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}
if 'tv_sentiment' in df.columns:
    df['tv_sentiment'] = df['tv_sentiment'].replace(tv_map)

for f in features:
    df[f] = pd.to_numeric(df[f].astype(str).str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)

X = df[features].values
X_scaled = scaler.transform(X)

# Predict which high-return archetype every single trade in the database is closest to
df['predicted_cluster'] = kmeans.predict(X_scaled)

# Calculate Euclidean distance to the cluster's centroid
distances = []
for i in range(len(df)):
    cluster_id = df['predicted_cluster'].iloc[i]
    centroid = kmeans.cluster_centers_[cluster_id]
    dist = np.linalg.norm(X_scaled[i] - centroid)
    distances.append(dist)

df['distance_to_centroid'] = distances

print(f"Total Trades Evaluated: {len(df)}")

# Evaluate Purity of each cluster across the entire database
summary = df.groupby('predicted_cluster').agg(
    total_captured=('trade_id', 'count'),
    win_rate_pct=('final_profit_pct', lambda x: (x > 0).mean() * 100),
    avg_pnl=('final_profit_pct', 'mean'),
    number_of_big_winners=('final_profit_pct', lambda x: (x > 0.5).sum())
)

print("\n--- PERFORMANCE OF CLUSTERS ACROSS ALL TRADES ---")
print(summary.to_markdown())

# What if we enforce a strict distance threshold? (e.g. distance < 1.0) to filter out loose matches
print("\n--- PERFORMANCE (STRICT MATCH: DISTANCE < 1.0) ---")
df_tight = df[df['distance_to_centroid'] < 1.0]
summary_tight = df_tight.groupby('predicted_cluster').agg(
    total_captured=('trade_id', 'count'),
    win_rate_pct=('final_profit_pct', lambda x: (x > 0).mean() * 100),
    avg_pnl=('final_profit_pct', 'mean')
)
print(summary_tight.to_markdown())
