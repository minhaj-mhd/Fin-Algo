import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

conn = sqlite3.connect('data/vanguard_trades.db')
# Load all trades
df = pd.read_sql("SELECT * FROM trades", conn)
df['final_profit_pct'] = pd.to_numeric(df['final_profit_pct'], errors='coerce')

# Filter for HIGH returns (e.g., > 0.5% profit)
df_high = df[df['final_profit_pct'] > 0.5].copy()

features = ['tech_score', 'nlp_sentiment', 'tv_sentiment', 'one_hour_prob']
tv_map = {'STRONG_SELL': -2, 'SELL': -1, 'NEUTRAL': 0, 'BUY': 1, 'STRONG_BUY': 2}

if 'tv_sentiment' in df_high.columns:
    df_high['tv_sentiment'] = df_high['tv_sentiment'].replace(tv_map)

for f in features:
    if f not in df_high.columns:
        df_high[f] = 0.0
    else:
        df_high[f] = pd.to_numeric(df_high[f].astype(str).str.replace('%', '').str.strip(), errors='coerce').fillna(0.0)

if len(df_high) == 0:
    print("No trades found with > 0.5% return.")
    exit()

X = df_high[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster the winning trades to see the common archetypes
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=20, random_state=42)
df_high['cluster'] = kmeans.fit_predict(X_scaled)

summary = df_high.groupby('cluster').agg({
    'trade_id': 'count',
    'final_profit_pct': 'mean',
    'side': lambda x: x.value_counts().idxmax() if not x.empty else 'N/A',
    **{f: 'mean' for f in features}
})

print(f"Total High Return Trades (>0.5%): {len(df_high)}")
print("\n--- HIGH RETURN ARCHETYPES ---")
print(summary.to_markdown())
