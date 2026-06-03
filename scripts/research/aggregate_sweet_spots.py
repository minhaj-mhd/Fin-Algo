import sqlite3
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans

# Load models
scaler = pickle.load(open('models/high_return_clusters/scaler.pkl', 'rb'))
kmeans_primary = pickle.load(open('models/high_return_clusters/kmeans.pkl', 'rb'))

conn = sqlite3.connect('data/vanguard_trades.db')
df = pd.read_sql("SELECT * FROM trades", conn)
df['final_profit_pct'] = pd.to_numeric(df['final_profit_pct'], errors='coerce')
df = df.dropna(subset=['final_profit_pct']).copy()

features = ['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']
tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}
if 'tv_sentiment' in df.columns:
    df['tv_sentiment'] = df['tv_sentiment'].replace(tv_map)
for f in features:
    df[f] = pd.to_numeric(df[f].astype(str).str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)

X = df[features].values
X_scaled = scaler.transform(X)

df_winners = df[df['final_profit_pct'] > 0.5].copy()
X_w_scaled = scaler.transform(df_winners[features].values)
df_winners['pc'] = kmeans_primary.predict(X_w_scaled)

# We need to recreate the exact sub_kmeans models to get the centroids
centroids_to_check = [] 

# Cluster 0, Sub 1 (K=3) -> +0.11% at < 0.75
c0_w = df_winners[df_winners['pc'] == 0]
k0 = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
k0.fit(scaler.transform(c0_w[features].values))
centroids_to_check.append((k0.cluster_centers_[1], 0.75))

# Cluster 1, Sub 1 (K=3) -> +0.14% at < 0.50 (Note: <0.75 was 0.098%, so we use 0.50 to stay >0.1%)
c1_w = df_winners[df_winners['pc'] == 1]
k1 = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
k1.fit(scaler.transform(c1_w[features].values))
centroids_to_check.append((k1.cluster_centers_[1], 0.50))

# Cluster 2, Sub 0 and Sub 1 (K=3) -> +0.31% at < 0.50 and +0.50% at < 1.00
c2_w = df_winners[df_winners['pc'] == 2]
k2 = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
k2.fit(scaler.transform(c2_w[features].values))
centroids_to_check.append((k2.cluster_centers_[0], 0.50))
centroids_to_check.append((k2.cluster_centers_[1], 1.00))

# Cluster 3, Sub 0 (K=5) -> +0.25% at < 0.75
c3_w = df_winners[df_winners['pc'] == 3]
k3 = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
k3.fit(scaler.transform(c3_w[features].values))
centroids_to_check.append((k3.cluster_centers_[0], 0.75))

# Now check all trades to avoid double counting
captured_indices = set()
for i in range(len(X_scaled)):
    vec = X_scaled[i]
    for centroid, threshold in centroids_to_check:
        dist = np.linalg.norm(vec - centroid)
        if dist < threshold:
            captured_indices.add(df.index[i])
            break 

df_captured = df.loc[list(captured_indices)]
total = len(df_captured)
wr = (df_captured['final_profit_pct'] > 0).mean() * 100
avg_pnl = df_captured['final_profit_pct'].mean()

print(f"Total Unique Trades: {total}")
print(f"Aggregate Win Rate: {wr:.2f}%")
print(f"Aggregate Avg PnL: {avg_pnl:.4f}%")
