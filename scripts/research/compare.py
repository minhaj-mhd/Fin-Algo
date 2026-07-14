import pandas as pd
import json, xgboost as xgb, numpy as np

# Load macro
macro_files = {'brent': 'data/raw_global_daily/BRENT.parquet', 'sp500': 'data/raw_global_daily/SP500.parquet'}
m_df = None
for n, p in macro_files.items():
    d = pd.read_parquet(p)
    d[f'{n}_ret_prev'] = d['close'].pct_change().shift(1)
    d['date'] = d['timestamp'].dt.date
    d = d[['date', f'{n}_ret_prev']].dropna()
    m_df = d if m_df is None else pd.merge(m_df, d, on='date', how='outer')

# Load nifty
nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
nifty['ts'] = pd.to_datetime(nifty['ts']).dt.tz_localize(None)
nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
nifty['date'] = nifty['ts'].dt.date
do = nifty.groupby('date')['open'].first().reset_index().rename(columns={'open': 'd_open'})
nifty = pd.merge(nifty, do, on='date')
nifty['nifty_intraday'] = nifty['close'] / nifty['d_open'] - 1
n2_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))
ni_map = dict(zip(nifty['ts'], nifty['nifty_intraday']))

# Load panel
df = pd.read_parquet('data/research/v20_rolling_1h/panel_backfilled.parquet')
df['DateTime'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
df['date'] = df['DateTime'].dt.date
df = pd.merge(df, m_df, on='date', how='left')
df['n2h'] = df['DateTime'].map(n2_map)
df['nin'] = df['DateTime'].map(ni_map)
df = df.dropna(subset=['n2h','nin','sp500_ret_prev'])

tm = (df['DateTime'].dt.time >= pd.to_datetime('10:15').time()) & (df['DateTime'].dt.time <= pd.to_datetime('14:15').time())
df = df[tm]

feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
df = df.dropna(subset=feats + ['Next_Hour_Return'])
X = xgb.DMatrix(np.nan_to_num(df[feats].values.astype(np.float32)), feature_names=feats)
bl = xgb.Booster(); bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')
bs = xgb.Booster(); bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
df['ls'] = bl.predict(X)
df['ss'] = bs.predict(X)
df['long_conv'] = (df['ls'] - df.groupby('DateTime')['ls'].transform('mean')) - (df['ss'] - df.groupby('DateTime')['ss'].transform('mean'))

# Simulation
baseline = []
filtered = []

for ts, g in df.groupby('DateTime'):
    n2h = g['n2h'].iloc[0]
    nin = g['nin'].iloc[0]
    sp = g['sp500_ret_prev'].iloc[0]
    
    if n2h > 0.0025 and nin > 0.0020:
        c = g[g['ls'] > 0.035].sort_values('long_conv', ascending=False)
        if len(c) > 0:
            tr = c.iloc[0].copy()
            tr['net'] = tr['Next_Hour_Return']*10000 - 6.0
            baseline.append(tr)
            
            # Apply Filter
            # Global: Risk Off (sp < -0.005), Neutral, Risk On (sp > 0.005)
            # Local: Extreme (n2h > 0.0070)
            if sp < -0.005:
                pass # Veto all Risk Off
            elif sp > 0.005:
                if n2h > 0.0070: filtered.append(tr)
            else: # Neutral
                if n2h < 0.0040 or n2h > 0.0070: filtered.append(tr)

b_df = pd.DataFrame(baseline)
f_df = pd.DataFrame(filtered)

print('--- BASELINE LONGS (> 0.035) ---')
print(f'Trades: {len(b_df)} | Win Rate: {(b_df["net"]>0).mean():.1%} | Total BPS: {b_df["net"].sum():.2f} | Avg: {b_df["net"].mean():.2f}')
print('--- FILTERED LONGS ---')
print(f'Trades: {len(f_df)} | Win Rate: {(f_df["net"]>0).mean():.1%} | Total BPS: {f_df["net"].sum():.2f} | Avg: {f_df["net"].mean():.2f}')
