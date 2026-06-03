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

df_winners = df[df['final_profit_pct'] > 0.5].copy()
X_winners_scaled = scaler.transform(df_winners[features].values)
df_winners['primary_cluster'] = kmeans_primary.predict(X_winners_scaled)

markdown_report = "\n\n## Hierarchical Sub-Clustering of High Return Archetypes\n"
markdown_report += "To maximize win-rate and filter out noise, each of the primary high-return clusters was hierarchically sub-clustered to isolate the absolute densest pockets of winning trades.\n\n"

for c in range(4):
    print(f"\n=================== PRIMARY CLUSTER {c} ===================")
    markdown_report += f"### Primary Cluster {c}\n"
    df_c_winners = df_winners[df_winners['primary_cluster'] == c].copy()
    X_c_winners_scaled = scaler.transform(df_c_winners[features].values)
    
    print(f"Total winners in Cluster {c}: {len(df_c_winners)}")
    
    if len(df_c_winners) < 3:
        continue
        
    sub_kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    df_c_winners['sub_cluster'] = sub_kmeans.fit_predict(X_c_winners_scaled)
    
    for sub_c in range(3):
        sub_count = len(df_c_winners[df_c_winners['sub_cluster'] == sub_c])
        sub_centroid = sub_kmeans.cluster_centers_[sub_c]
        distances = np.linalg.norm(X_scaled - sub_centroid, axis=1)
        df_temp = df.copy()
        df_temp['dist_to_sub'] = distances
        
        print(f"\n--- Sub-Cluster {sub_c} ({sub_count} winners) ---")
        markdown_report += f"**Sub-Cluster {sub_c}** ({sub_count} primary winners)\n"
        
        for t in [1.5, 1.0, 0.75, 0.5]:
            dft = df_temp[df_temp['dist_to_sub'] < t]
            if len(dft) > 0:
                wr = (dft['final_profit_pct'] > 0).mean()*100
                pnl = dft['final_profit_pct'].mean()
                print(f"Dist < {t:.2f} | Trades: {len(dft):3d} | WR: {wr:5.1f}% | Avg PnL: {pnl:.4f}%")
                markdown_report += f"- `Dist < {t:.2f}` | Caught: {len(dft)} | Win Rate: {wr:.1f}% | Avg PnL: {pnl:.4f}%\n"
            else:
                markdown_report += f"- `Dist < {t:.2f}` | Caught: 0\n"
        markdown_report += "\n"

with open(r'c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\07. Cluster Research\Deep_Dive_High_Return_Clusters.md', 'a') as f:
    f.write(markdown_report)

print("Report appended to memory layer.")
