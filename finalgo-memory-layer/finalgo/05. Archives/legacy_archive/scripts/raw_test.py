import pandas as pd
import xgboost as xgb
import json
import numpy as np

def score_df(df, meta_path, long_model_path, short_model_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    features = meta['features']
    df = df.dropna(subset=features).copy()
    
    dmatrix = xgb.DMatrix(df[features])
    long_model = xgb.Booster()
    long_model.load_model(long_model_path)
    df['long_conv'] = long_model.predict(dmatrix)
    
    short_model = xgb.Booster()
    short_model.load_model(short_model_path)
    df['short_conv'] = short_model.predict(dmatrix)
    
    # Pre-filter to save time: only keep those that are possibly top 3
    # Actually just rank them
    def rank_group(group):
        group['long_rank'] = group['long_conv'].rank(ascending=False)
        group['short_rank'] = group['short_conv'].rank(ascending=False)
        return group
        
    df = df.groupby('DateTime', group_keys=False).apply(rank_group)
    if 'DateTime' not in df.columns:
        df = df.reset_index()
    return df

print('Loading 15M...')
df_15m = pd.read_csv('data/ranking_data_upstox_15min_1y.csv')
df_15m = df_15m[df_15m['DateTime'].str.startswith('2026-05')].copy()
df_15m = score_df(df_15m, 'models/v1_15min/metadata.json', 'models/v1_15min/xgb_long_model.json', 'models/v1_15min/xgb_short_model.json')

# Calculate 4-bar forward return (1 hour) using grouped shift
print('Calculating forward returns...')
df_15m = df_15m.sort_values(['Ticker', 'DateTime'])
# 4 bar forward close
df_15m['Close_T4'] = df_15m.groupby('Ticker')['Close'].shift(-4)
# If T4 crosses into the next day or doesn't exist, we fallback to close at EOD but for simplicity let's just use the strict shift.
# It's an approximation for raw model evaluation
df_15m['Long_Ret_4B'] = (df_15m['Close_T4'] / df_15m['Close']) - 1.0 - 0.0015 # 0.15% friction
df_15m['Short_Ret_4B'] = 1.0 - (df_15m['Close_T4'] / df_15m['Close']) - 0.0015

df_15m = df_15m.dropna(subset=['Close_T4'])

def eval_subset(subset, name):
    l = subset['Long_Ret_4B'].dropna()
    s = subset['Short_Ret_4B'].dropna()
    all_rets = np.concatenate([l, s])
    if len(all_rets) == 0:
        return
    wins = np.sum(all_rets > 0)
    wr = wins / len(all_rets) * 100
    mean_ret = np.mean(all_rets) * 100
    print(f"{name}: Trades={len(all_rets)}, WR={wr:.1f}%, Avg Net Ret={mean_ret:.3f}%")

print('\n--- 15M RAW MODEL PERFORMANCE (Holding 4 bars/1 Hour) ---')
eval_subset(df_15m[df_15m['long_rank'] == 1], "Top 1 Long")
eval_subset(df_15m[df_15m['short_rank'] == 1], "Top 1 Short")
eval_subset(df_15m[df_15m['long_rank'] <= 3], "Top 3 Long")
eval_subset(df_15m[df_15m['short_rank'] <= 3], "Top 3 Short")

print('\nLoading 30M...')
df_1h = pd.read_csv('data/ranking_data_upstox_30min_1y.csv')
df_1h = df_1h[df_1h['DateTime'].str.startswith('2026-05')].copy()
df_1h = score_df(df_1h, 'models/v1_30min/metadata.json', 'models/v1_30min/xgb_long_model.json', 'models/v1_30min/xgb_short_model.json')

df_1h = df_1h.sort_values(['Ticker', 'DateTime'])
df_1h['Close_T1'] = df_1h.groupby('Ticker')['Close'].shift(-1)
df_1h['Long_Ret_1B'] = (df_1h['Close_T1'] / df_1h['Close']) - 1.0 - 0.0015
df_1h['Short_Ret_1B'] = 1.0 - (df_1h['Close_T1'] / df_1h['Close']) - 0.0015

df_1h = df_1h.dropna(subset=['Close_T1'])

def eval_subset_1h(subset, name):
    l = subset['Long_Ret_1B'].dropna()
    s = subset['Short_Ret_1B'].dropna()
    all_rets = np.concatenate([l, s])
    if len(all_rets) == 0:
        return
    wins = np.sum(all_rets > 0)
    wr = wins / len(all_rets) * 100
    mean_ret = np.mean(all_rets) * 100
    print(f"{name}: Trades={len(all_rets)}, WR={wr:.1f}%, Avg Net Ret={mean_ret:.3f}%")

print('\n--- 1H RAW MODEL PERFORMANCE (Holding 1 bar/1 Hour) ---')
eval_subset_1h(df_1h[df_1h['long_rank'] == 1], "Top 1 Long")
eval_subset_1h(df_1h[df_1h['short_rank'] == 1], "Top 1 Short")
eval_subset_1h(df_1h[df_1h['long_rank'] <= 3], "Top 3 Long")
eval_subset_1h(df_1h[df_1h['short_rank'] <= 3], "Top 3 Short")
