"""
Quick rebuild of Transformer daily dataset from cached raw data.
Uses compute_features_daily_transformer() with smooth temporal features only.
"""
import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.getcwd())
from scripts.feature_utils import compute_features_daily_transformer_v2
from scripts.tickers import TICKERS
from tqdm import tqdm

CACHE_DIR = 'data/raw_upstox_daily_cache'
OUTPUT = 'data/ranking_data_upstox_daily_5y_transformer.csv'

print('Rebuilding Transformer dataset from cached raw data...')
print('Using compute_features_daily_transformer_v2() — pruned stationary features')
print(f'Tickers: {len(TICKERS)}')

tf_dfs = []
for ticker in tqdm(TICKERS, desc='Features'):
    safe_name = ticker.replace('.NS', '')
    cache_file = os.path.join(CACHE_DIR, f"{safe_name}.csv")
    if not os.path.exists(cache_file):
        continue
    df_raw = pd.read_csv(cache_file, parse_dates=['timestamp'])
    if len(df_raw) < 100:
        continue
    
    df_std = df_raw.rename(columns={
        'timestamp': 'DateTime', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    if 'oi' in df_std.columns:
        df_std = df_std.drop(columns=['oi'])
    df_std['DateTime'] = pd.to_datetime(df_std['DateTime'])
    df_std = df_std.set_index('DateTime')
    
    # Transformer-optimized features (smooth, stationary, pruned)
    df_feat = compute_features_daily_transformer_v2(df_std)
    df_feat['DateTime'] = df_feat.index
    df_feat['Ticker'] = ticker
    df_feat['Next_Day_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1
    tf_dfs.append(df_feat)

print(f'\nFeatures computed for {len(tf_dfs)} tickers')

# Compile cross-sectional dataset
df_all = pd.concat(tf_dfs, ignore_index=True)
df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
df_all = df_all.dropna(subset=['Next_Day_Return'])
df_all = df_all.sort_values('DateTime')
df_all['Query_ID'] = df_all.groupby(df_all['DateTime'].dt.date).ngroup()

query_sizes = df_all.groupby('Query_ID').size()
valid = query_sizes[query_sizes >= 5].index
df_all = df_all[df_all['Query_ID'].isin(valid)].copy()
df_all = df_all.sort_values('DateTime')
df_all['Query_ID'] = df_all.groupby(df_all['DateTime'].dt.date).ngroup()

# Market context
df_all['Market_Mean_Return'] = df_all.groupby('Query_ID')['Return'].transform('mean')
df_all['Relative_Return'] = df_all['Return'] - df_all['Market_Mean_Return']
df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
df_all['Relative_Volatility'] = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)

# Cross-sectional Z-scoring (exclude non-feature columns)
exclude = {
    'DateTime', 'Query_ID', 'Ticker', 'Next_Day_Return',
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Market_Mean_Return', 'Relative_Return',
    'Market_Mean_Volatility', 'Relative_Volatility',
}
feature_cols = [c for c in df_all.columns if c not in exclude]

print(f'Z-scoring {len(feature_cols)} features...')
for col in tqdm(feature_cols, desc='Z-Score'):
    m = df_all.groupby('Query_ID')[col].transform('mean')
    s = df_all.groupby('Query_ID')[col].transform('std')
    df_all[col] = (df_all[col] - m) / (s + 1e-8)
df_all[feature_cols] = df_all[feature_cols].fillna(0)

df_all.to_csv(OUTPUT, index=False)
print(f'\nDone: {len(df_all):,} rows, {len(feature_cols)} features')
print(f'Date span: {df_all["DateTime"].min()} -> {df_all["DateTime"].max()}')
print(f'Saved: {OUTPUT}')
