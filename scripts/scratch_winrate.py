import pandas as pd
import numpy as np
import xgboost as xgb

df = pd.read_csv('data/ranking_data_upstox_1h_v3_3y.csv')
features = [c for c in df.columns if c not in ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Return', 'YearMonth']]
X = df[features].values

nan_mask = ~np.isfinite(X)
if nan_mask.any():
    for ci in range(X.shape[1]):
        bad = ~np.isfinite(X[:, ci])
        if bad.any():
            good = X[~bad, ci]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

bst = xgb.Booster()
bst.load_model('models/v15_lambdamart_es_1h/xgb_long_model.json')
df['preds'] = bst.predict(xgb.DMatrix(X))

df['YearMonth'] = df['DateTime'].str[:7]
unique_months = sorted(df['YearMonth'].unique())
split_idx = int(len(unique_months) * 0.8)
test_months = unique_months[split_idx:]
df_test = df[df['YearMonth'].isin(test_months)].copy()

gross_ret = []
for qid in df_test['Query_ID'].unique():
    q = df_test[df_test['Query_ID'] == qid]
    if len(q) < 4: continue
    idx = np.argsort(q['preds'].values)[::-1][:3]
    picked = q['Next_Hour_Return'].values[idx]
    gross_ret.extend(picked.tolist())

win_raw = np.mean(np.array(gross_ret) > 0)
win_fee = np.mean(np.array(gross_ret) > 0.0010)
avg_gross = np.mean(gross_ret)

print(f"OOS Raw Winrate (Gross > 0) : {win_raw:.2%}")
print(f"OOS Fee Winrate (Gross > 10bps) : {win_fee:.2%}")
print(f"OOS Avg Gross Return: {avg_gross*10000:.2f} bps")
