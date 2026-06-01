import pandas as pd
import xgboost as xgb
import json
import numpy as np

print("--- 15M RAW MODEL (1 Hour Hold) ---")
df = pd.read_csv('data/ranking_data_upstox_15min_1y.csv')
df = df[df['DateTime'].str.startswith('2026-05')].copy()
with open('models/v1_15min/metadata.json', 'r') as f:
    meta = json.load(f)
features = meta['features']
df = df.dropna(subset=features).copy()
dmatrix = xgb.DMatrix(df[features])

l = xgb.Booster()
l.load_model('models/v1_15min/xgb_long_model.json')
df['long_conv'] = l.predict(dmatrix)

s = xgb.Booster()
s.load_model('models/v1_15min/xgb_short_model.json')
df['short_conv'] = s.predict(dmatrix)

df['long_rank'] = df.groupby('DateTime')['long_conv'].rank(ascending=False)
df['short_rank'] = df.groupby('DateTime')['short_conv'].rank(ascending=False)

df = df.sort_values(['Ticker', 'DateTime'])
df['Close_T4'] = df.groupby('Ticker')['Close'].shift(-4)
df['Long_Ret_4B'] = (df['Close_T4'] / df['Close']) - 1.0 - 0.0015
df['Short_Ret_4B'] = 1.0 - (df['Close_T4'] / df['Close']) - 0.0015
df = df.dropna(subset=['Close_T4'])

for r in [1, 3]:
    for side in ['long', 'short']:
        subset = df[df[f'{side}_rank'] <= r]
        col = 'Long_Ret_4B' if side == 'long' else 'Short_Ret_4B'
        rets = subset[col].values
        if len(rets) > 0:
            wr = np.sum(rets > 0) / len(rets) * 100
            print(f'15M Top {r} {side}: Trades={len(rets)}, WR={wr:.1f}%, Net={np.mean(rets)*100:.3f}%')

print("\n--- 30M RAW MODEL (1 Hour Hold) ---")
df2 = pd.read_csv('data/ranking_data_upstox_30min_1y.csv')
df2 = df2[df2['DateTime'].str.startswith('2026-05')].copy()
with open('models/v1_30min/metadata.json', 'r') as f:
    meta = json.load(f)
features = meta['features']
df2 = df2.dropna(subset=features).copy()
dmatrix2 = xgb.DMatrix(df2[features])

l2 = xgb.Booster()
l2.load_model('models/v1_30min/xgb_long_model.json')
df2['long_conv'] = l2.predict(dmatrix2)

s2 = xgb.Booster()
s2.load_model('models/v1_30min/xgb_short_model.json')
df2['short_conv'] = s2.predict(dmatrix2)

df2['long_rank'] = df2.groupby('DateTime')['long_conv'].rank(ascending=False)
df2['short_rank'] = df2.groupby('DateTime')['short_conv'].rank(ascending=False)

df2 = df2.sort_values(['Ticker', 'DateTime'])
df2['Close_T2'] = df2.groupby('Ticker')['Close'].shift(-2)
df2['Long_Ret_2B'] = (df2['Close_T2'] / df2['Close']) - 1.0 - 0.0015
df2['Short_Ret_2B'] = 1.0 - (df2['Close_T2'] / df2['Close']) - 0.0015
df2 = df2.dropna(subset=['Close_T2'])

for r in [1, 3]:
    for side in ['long', 'short']:
        subset = df2[df2[f'{side}_rank'] <= r]
        col = 'Long_Ret_2B' if side == 'long' else 'Short_Ret_2B'
        rets = subset[col].values
        if len(rets) > 0:
            wr = np.sum(rets > 0) / len(rets) * 100
            print(f'30M Top {r} {side}: Trades={len(rets)}, WR={wr:.1f}%, Net={np.mean(rets)*100:.3f}%')
