import pandas as pd
import numpy as np
import xgboost as xgb

# Configuration
DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL = 'Next_Hour_Return'
COST = 0.0010
TRADE_PROB = 0.52

print("=" * 64)
print("BACKTEST: LAST AVAILABLE DAY (2026-06-05) - LOGIC A3")
print("=" * 64)

# 1. Load Data
df = pd.read_csv(DATA_FILE)
df_test = df[df['DateTime'].str.startswith('2026-06-05')].copy()

if len(df_test) == 0:
    print("No data found for 2026-06-05 in the dataset.")
    exit(0)

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth', 'Date']
feature_cols = [c for c in df_test.columns if c not in exclude_cols]

X_test = df_test[feature_cols].values.astype(np.float64)
y_returns = df_test[RET_COL].values.astype(np.float64)
query_ids = df_test['Query_ID'].values

# Handle NaNs
nan_mask = ~np.isfinite(X_test)
if nan_mask.any():
    for ci in range(X_test.shape[1]):
        bad = ~np.isfinite(X_test[:, ci])
        if bad.any():
            good = X_test[~bad, ci]
            X_test[bad, ci] = float(good.mean()) if len(good) else 0.0

dte = xgb.DMatrix(X_test)

# 2. Load Models
v10_l = xgb.Booster(); v10_l.load_model('models/v10_native_1h/xgb_long_model.json')
v10_s = xgb.Booster(); v10_s.load_model('models/v10_native_1h/xgb_short_model.json')
v18_l = xgb.Booster(); v18_l.load_model('models/v18_random_forest_1h/xgb_long_model.json')
v18_s = xgb.Booster(); v18_s.load_model('models/v18_random_forest_1h/xgb_short_model.json')

# 3. Predict
pred_l_v10 = v10_l.predict(dte)
pred_s_v10 = v10_s.predict(dte)
pred_l_v18 = v18_l.predict(dte)
pred_s_v18 = v18_s.predict(dte)

# 4. Evaluate Logic A3
logicA_k3_l_rets, logicA_k3_s_rets = [], []
trades_L = []
trades_S = []

for qid in np.unique(query_ids):
    m = query_ids == qid
    if m.sum() < 3: continue
    
    r_l_v10 = pred_l_v10[m]
    r_s_v10 = pred_s_v10[m]
    p_l_v18 = pred_l_v18[m]
    p_s_v18 = pred_s_v18[m]
    actual  = y_returns[m]
    
    q_df = df_test[m].reset_index(drop=True)

    # Top 3 Longs
    top3_l_idx = np.argsort(r_l_v10)[-3:]
    for idx in top3_l_idx:
        if p_l_v18[idx] > TRADE_PROB:
            logicA_k3_l_rets.append(actual[idx])
            trades_L.append({
                'Time': q_df.iloc[idx]['DateTime'],
                'Ticker': q_df.iloc[idx]['Ticker'],
                'V10_Rank': r_l_v10[idx],
                'V18_Prob': p_l_v18[idx],
                'Return_bps': actual[idx] * 10000
            })
            
    # Top 3 Shorts
    top3_s_idx = np.argsort(r_s_v10)[-3:]
    for idx in top3_s_idx:
        if p_s_v18[idx] > TRADE_PROB:
            logicA_k3_s_rets.append(-actual[idx])
            trades_S.append({
                'Time': q_df.iloc[idx]['DateTime'],
                'Ticker': q_df.iloc[idx]['Ticker'],
                'V10_Rank': r_l_v10[idx],
                'V18_Prob': p_l_v18[idx],
                'Return_bps': -actual[idx] * 10000
            })

# 5. Print Results
def print_logic_stats(name, rets):
    if len(rets) == 0:
        print(f"      {name:<6} :    0 trades | raw  +0.0bps | net  +0.0bps | raw win 0.0% | net hit 0.0%")
        return
    r = np.array(rets)
    raw_ret = float(np.mean(r))
    net_ret = raw_ret - COST
    raw_win = float(np.mean(r > 0))
    net_hit = float(np.mean(r > COST))
    print(f"      {name:<6} : {len(rets):>4} trades | raw {raw_ret*10000:>+5.1f}bps | net {net_ret*10000:>+5.1f}bps | raw win {raw_win:.1%} | net hit {net_hit:.1%}")

print("\nRESULTS FOR 2026-06-05:")
print_logic_stats("Longs", logicA_k3_l_rets)
print_logic_stats("Shorts", logicA_k3_s_rets)

print("\n=== SPECIFIC LONG TRADES ===")
for t in trades_L:
    print(f"{t['Time']} | {t['Ticker']:<12} | Rank: {t['V10_Rank']:.4f} | Prob: {t['V18_Prob']:.1%} | Ret: {t['Return_bps']:+5.1f} bps")

print("\n=== SPECIFIC SHORT TRADES ===")
for t in trades_S:
    print(f"{t['Time']} | {t['Ticker']:<12} | Rank: {t['V10_Rank']:.4f} | Prob: {t['V18_Prob']:.1%} | Ret: {t['Return_bps']:+5.1f} bps")

print("=" * 64)
