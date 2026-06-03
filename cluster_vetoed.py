import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

db_path = r"c:\Users\loq\Desktop\Trading\finalgo\data\vanguard_trades.db"
conn = sqlite3.connect(db_path)

# Extract vetoed trades
query = "SELECT * FROM trades WHERE status IN ('VETOED', 'VETOED_EXPIRED')"
df = pd.read_sql_query(query, conn)
conn.close()

features = [
    'tech_score', 'nlp_sentiment', 'one_hour_prob', 'long_score', 
    'short_score', 'score_15m', 'score_30m', 'score_1d', 'is_ensemble'
]

# Ensure features exist
available_features = [f for f in features if f in df.columns]

# Convert features to numeric (coerce errors to NaN)
for f in available_features:
    df[f] = pd.to_numeric(df[f], errors='coerce')

# Fill NaNs with median
for f in available_features:
    df[f] = df[f].fillna(df[f].median())

# Handle case where all values are NaN (median is also NaN)
df[available_features] = df[available_features].fillna(0)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[available_features])

# KMeans
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(X_scaled)

# Check final_profit_pct
has_pnl = 'final_profit_pct' in df.columns and df['final_profit_pct'].notna().any()

if has_pnl:
    df['final_profit_pct'] = df['final_profit_pct'].fillna(0)

# Aggregate
agg_dict = {f: 'mean' for f in available_features}
agg_dict['cluster'] = 'count'
if has_pnl:
    agg_dict['final_profit_pct'] = 'mean'

cluster_summary = df.groupby('cluster').agg(agg_dict).rename(columns={'cluster': 'count'})

# Generate Markdown Report
report = []
report.append("# Unsupervised Clustering Analysis of Vetoed Trades")
report.append("\n## Methodology")
report.append("Data was extracted from `vanguard_trades.db` for trades with status 'VETOED' or 'VETOED_EXPIRED'.")
report.append(f"A total of {len(df)} trades were analyzed.")
report.append(f"Features used: {', '.join(available_features)}.")
report.append("Missing values were imputed with the feature medians. Features were then scaled using `StandardScaler`.")
report.append(f"K-Means clustering was applied with k={n_clusters}.")

report.append("\n## Cluster Summaries")
report.append(cluster_summary.to_markdown())

report.append("\n## Analysis & Insights")
for idx, row in cluster_summary.iterrows():
    report.append(f"\n### Cluster {idx}")
    report.append(f"- **Count**: {int(row['count'])} trades")
    if has_pnl:
        report.append(f"- **Average PnL**: {row['final_profit_pct']:.4f}%")
    
    # Identify distinguishing features
    high_feats = []
    low_feats = []
    for f in available_features:
        global_mean = df[f].mean()
        if global_mean != 0:
            if row[f] > global_mean * 1.2:
                high_feats.append(f)
            elif row[f] < global_mean * 0.8:
                low_feats.append(f)
        else:
            if row[f] > 0.1:
                high_feats.append(f)
            elif row[f] < -0.1:
                low_feats.append(f)
            
    if high_feats:
        report.append(f"- **High Features**: {', '.join(high_feats)}")
    if low_feats:
        report.append(f"- **Low Features**: {', '.join(low_feats)}")

report.append("\n## Actionable Insights")
report.append("- Review the clusters to see if certain veto patterns can be identified earlier in the pipeline.")
report.append("- Clusters with extreme values in specific timeframes may indicate conflicting signals that lead to vetoes.")

# Write report
report_path = r"c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\05. Archives\Unsupervised_Vetoed_Trades_Analysis.md"
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(report))

print(f"Report generated successfully. Found {len(df)} trades and clustered into {n_clusters} clusters.")
if has_pnl:
    print("final_profit_pct was present.")
else:
    print("final_profit_pct was NOT present.")
