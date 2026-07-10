import numpy as np, pandas as pd

preds = np.load('data/gauntlet/20260615T175149Z-5f7d069f/preds.npz', allow_pickle=True)

df = pd.DataFrame({
    'idx': preds['idx'],
    'ym': preds['ym'],
    'q': preds['q'],
    'y': preds['y'],
    'time': preds['time'],
    'rl': preds['rl'],
    'rs': preds['rs'],
})

print('OOS coverage:')
print('  Months:', sorted(df['ym'].unique()))
print('  Total rows:', len(df))

panel = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet', columns=['DateTime', 'Ticker'])
panel = panel.reset_index(drop=True)

idx_map = panel.iloc[preds['idx']].reset_index(drop=True)
df['DateTime'] = idx_map['DateTime'].values
df['Ticker']   = idx_map['Ticker'].values

print()
print('Sample:')
print(df[['DateTime', 'Ticker', 'rl', 'rs', 'y']].head(5))
print()
print('Date range:', df['DateTime'].min(), '->', df['DateTime'].max())
print('Unique tickers:', df['Ticker'].nunique())
