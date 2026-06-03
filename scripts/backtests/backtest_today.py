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
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get "today"
latest_date = df['timestamp'].max().date()
df_today = df[df['timestamp'].dt.date == latest_date].copy()
df_today = df_today.dropna(subset=['final_profit_pct'])

print(f"--- BACKTESTING TODAY: {latest_date} ---")
print(f"Total trades with PnL today: {len(df_today)}")

if len(df_today) == 0:
    print("No closed trades today to backtest.")
    exit()

features = ['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']
tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}
if 'tv_sentiment' in df_today.columns:
    df_today['tv_sentiment'] = df_today['tv_sentiment'].replace(tv_map)
for f in features:
    df_today[f] = pd.to_numeric(df_today[f].astype(str).str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)

X_today_scaled = scaler.transform(df_today[features].values)

# Recreate the exact sweet spot centroids using the full historical winners
df_full = df.dropna(subset=['final_profit_pct']).copy()
if 'tv_sentiment' in df_full.columns:
    df_full['tv_sentiment'] = df_full['tv_sentiment'].replace(tv_map)
for f in features:
    df_full[f] = pd.to_numeric(df_full[f].astype(str).str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)

df_winners = df_full[df_full['final_profit_pct'] > 0.5].copy()
X_w_scaled = scaler.transform(df_winners[features].values)
df_winners['pc'] = kmeans_primary.predict(X_w_scaled)

centroids_to_check = []
# Cluster 1, Sub 1 (< 0.75)
c1_w = df_winners[df_winners['pc'] == 1]
k1 = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42).fit(scaler.transform(c1_w[features].values))
centroids_to_check.append(("Cluster 1 (Sub 1)", k1.cluster_centers_[1], 0.75))

# Cluster 2, Sub 0 (< 0.50) & Sub 1 (< 1.00)
c2_w = df_winners[df_winners['pc'] == 2]
k2 = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42).fit(scaler.transform(c2_w[features].values))
centroids_to_check.append(("Cluster 2 (Sub 0)", k2.cluster_centers_[0], 0.50))
centroids_to_check.append(("Cluster 2 (Sub 1)", k2.cluster_centers_[1], 1.00))

# Cluster 3, Sub 0 (< 0.75)
c3_w = df_winners[df_winners['pc'] == 3]
k3 = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42).fit(scaler.transform(c3_w[features].values))
centroids_to_check.append(("Cluster 3 (Sub 0)", k3.cluster_centers_[0], 0.75))

# Evaluate today's trades against the sweet spots
captured_indices = set()
captured_by_cluster = {name: 0 for name, _, _ in centroids_to_check}

for i in range(len(X_today_scaled)):
    vec = X_today_scaled[i]
    for name, centroid, threshold in centroids_to_check:
        dist = np.linalg.norm(vec - centroid)
        if dist < threshold:
            captured_indices.add(df_today.index[i])
            captured_by_cluster[name] += 1
            break # only assign to first matched to avoid double counting

df_captured = df_today.loc[list(captured_indices)]
total = len(df_captured)

print("\n--- RESULTS FOR TODAY ---")
for name, count in captured_by_cluster.items():
    print(f"Rescued by {name}: {count} trades")

print(f"\nTotal Overridden Trades Today: {total}")
if total > 0:
    wr = (df_captured['final_profit_pct'] > 0).mean() * 100
    avg_pnl = df_captured['final_profit_pct'].mean()
    print(f"Today's Aggregate Win Rate: {wr:.2f}%")
    print(f"Today's Aggregate Avg PnL: {avg_pnl:.4f}%")
    
    # Leveraged math
    print(f"\n--- LEVERAGED & SLIPPAGE-ADJUSTED MATH ---")
    adj_pnl = avg_pnl - 0.06
    lev_pnl = adj_pnl * 5
    tot_lev_ret = total * lev_pnl
    print(f"Leveraged Net PnL per trade: +{lev_pnl:.4f}%")
    print(f"Total Leveraged Net Return for Today: +{tot_lev_ret:.4f}%")
