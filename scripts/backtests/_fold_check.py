import pandas as pd, json

with open('models/research/v20_rolling_1h/metadata.json') as f:
    meta = json.load(f)

print('WF folds:', len(meta['walk_forward_folds']))
for fold in meta['walk_forward_folds']:
    fnum = fold['fold']
    lr = fold['long_rho']
    sr = fold['short_rho']
    print(f'  Fold {fnum}: Long={lr:.4f}  Short={sr:.4f}')

panel = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet', columns=['DateTime'])
ts = sorted(panel['DateTime'].unique())
n = len(ts)
fold_size = n // 8
print(f'\nTotal unique timestamps: {n}')
print(f'Est fold size: {fold_size}')
print()
for i in range(8):
    s = i * fold_size
    e = min((i+1)*fold_size, n) - 1
    print(f'Fold {i+1}: {pd.Timestamp(ts[s]).date()} -> {pd.Timestamp(ts[e]).date()}')
