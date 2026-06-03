import sqlite3
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans

# 1. Load the original models
scaler = pickle.load(open('models/high_return_clusters/scaler.pkl', 'rb'))
kmeans_primary = pickle.load(open('models/high_return_clusters/kmeans.pkl', 'rb'))

# 2. Load all trades
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

# 3. Identify the big winners in Cluster 1
df_winners = df[df['final_profit_pct'] > 0.5].copy()
X_winners_scaled = scaler.transform(df_winners[features].values)
df_winners['primary_cluster'] = kmeans_primary.predict(X_winners_scaled)

# Filter to just Cluster 2 winners
df_c2_winners = df_winners[df_winners['primary_cluster'] == 2].copy()
X_c2_winners_scaled = scaler.transform(df_c2_winners[features].values)

print(f"Total winners in Cluster 2: {len(df_c2_winners)}")

# 4. Sub-cluster these 17 trades into 3 sub-archetypes
sub_kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
df_c2_winners['sub_cluster'] = sub_kmeans.fit_predict(X_c2_winners_scaled)

print("\nSub-clusters found within the winners:")
print(df_c2_winners.groupby('sub_cluster').size())

# 5. Evaluate these new sub-centroids across the ENTIRE database
print("\n--- EVALUATING SUB-CENTROIDS ON ALL 1450 TRADES ---")
for sub_c in range(3):
    sub_centroid = sub_kmeans.cluster_centers_[sub_c]
    
    # Calculate distance for ALL trades in database to this specific sub_centroid
    distances = np.linalg.norm(X_scaled - sub_centroid, axis=1)
    df_temp = df.copy()
    df_temp['dist_to_sub'] = distances
    
    print(f"\n--- Sub-Cluster {sub_c} ---")
    for t in [1.5, 1.0, 0.75, 0.5]:
        dft = df_temp[df_temp['dist_to_sub'] < t]
        if len(dft) > 0:
            wr = (dft['final_profit_pct'] > 0).mean()*100
            print(f"Dist < {t:.2f} | Trades: {len(dft):3d} | WR: {wr:5.1f}% | Avg PnL: {dft['final_profit_pct'].mean():.4f}%")
        else:
            print(f"Dist < {t:.2f} | Trades:   0")
