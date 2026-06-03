import sqlite3
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans

# 1. Load models & data
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

# Get winners
df_winners = df[df['final_profit_pct'] > 0.5].copy()
X_w_scaled = scaler.transform(df_winners[features].values)
df_winners['pc'] = kmeans_primary.predict(X_w_scaled)

# Filter Cluster 3
df_c3_w = df_winners[df_winners['pc'] == 3].copy()
X_c3_w_scaled = scaler.transform(df_c3_w[features].values)
print(f"Total winners in Cluster 3: {len(df_c3_w)}")

# Test different values of K for sub-clustering
for k in [4, 5, 6, 7]:
    print(f"\n=================== TESTING K={k} FOR CLUSTER 3 ===================")
    sub = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    df_c3_w['sc'] = sub.fit_predict(X_c3_w_scaled)
    
    # Evaluate each sub-centroid
    for sub_c in range(k):
        cnt = len(df_c3_w[df_c3_w['sc'] == sub_c])
        centroid = sub.cluster_centers_[sub_c]
        dists = np.linalg.norm(X_scaled - centroid, axis=1)
        
        df_temp = df.copy()
        df_temp['dist'] = dists
        
        # Only print promising results to reduce noise (e.g. WR >= 55% at tight distances)
        found_promising = False
        output_buffer = f"--- Sub-Cluster {sub_c} ({cnt} winners inside) ---\n"
        
        for t in [1.0, 0.75, 0.5]:
            dft = df_temp[df_temp['dist'] < t]
            if len(dft) >= 5:
                wr = (dft['final_profit_pct'] > 0).mean()*100
                pnl = dft['final_profit_pct'].mean()
                output_buffer += f"Dist < {t:.2f} | Trades: {len(dft):3d} | WR: {wr:5.1f}% | Avg PnL: {pnl:.4f}%\n"
                if wr >= 58.0:
                    found_promising = True
                    
        if found_promising:
            print(output_buffer)

print("\nOptimization scan complete.")
