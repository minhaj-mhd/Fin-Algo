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
df = df.dropna(subset=['final_profit_pct']).copy()

features = ['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']
tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}
if 'tv_sentiment' in df.columns:
    df['tv_sentiment'] = df['tv_sentiment'].replace(tv_map)

for f in features:
    df[f] = pd.to_numeric(df[f].astype(str).str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)

X = df[features].values
X_scaled = scaler.transform(X)

df['predicted_cluster'] = kmeans.predict(X_scaled)

distances = []
for i in range(len(df)):
    cluster_id = df['predicted_cluster'].iloc[i]
    centroid = kmeans.cluster_centers_[cluster_id]
    dist = np.linalg.norm(X_scaled[i] - centroid)
    distances.append(dist)

df['distance_to_centroid'] = distances

print("--- AGGREGATE SYSTEM BACKTEST (ALL 4 CLUSTERS COMBINED) ---")
thresholds = [2.0, 1.5, 1.0, 0.75, 0.5]
for t in thresholds:
    df_t = df[df['distance_to_centroid'] < t]
    total = len(df_t)
    if total > 0:
        win_rate = (df_t['final_profit_pct'] > 0).mean() * 100
        avg_pnl = df_t['final_profit_pct'].mean()
        print(f"Threshold < {t:.2f}: {total:4d} Trades | Win Rate: {win_rate:5.2f}% | Avg PnL: {avg_pnl:7.4f}%")
    else:
        print(f"Threshold < {t:.2f}:    0 Trades")
