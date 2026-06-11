import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import rankdata

# Configuration
DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL = 'Next_Hour_Return'
COST = 0.0010
TRADE_PROB = 0.52

print("=" * 64)
print("HYBRID INFERENCE BACKTEST: V10 Ranker + V18 Classifier (Veto)")
print("Period: Last 12 Months (2025-07 to 2026-06) - Withheld Test Set")
print("=" * 64)

# 1. Load Data
print("Loading data...")
df = pd.read_csv(DATA_FILE)
df['YearMonth'] = df['DateTime'].str[:7]
df_test = df[df['YearMonth'] >= '2025-07'].copy()
print(f"Test period: {df_test['YearMonth'].min()} to {df_test['YearMonth'].max()}")
print(f"Number of rows: {len(df_test)}")

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']
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
print("\nLoading pre-trained V10 models...")
v10_l = xgb.Booster()
v10_l.load_model('models/v10_native_1h/xgb_long_model.json')
v10_s = xgb.Booster()
v10_s.load_model('models/v10_native_1h/xgb_short_model.json')

print("Loading pre-trained V18 models...")
v18_l = xgb.Booster()
v18_l.load_model('models/v18_random_forest_1h/xgb_long_model.json')
v18_s = xgb.Booster()
v18_s.load_model('models/v18_random_forest_1h/xgb_short_model.json')

# 3. Predict
print("\nGenerating predictions...")
pred_l_v10 = v10_l.predict(dte)
pred_s_v10 = v10_s.predict(dte)
pred_l_v18 = v18_l.predict(dte)
pred_s_v18 = v18_s.predict(dte)

# 4. Evaluate Logic & Export Trades
logicA_k1_l_rets, logicA_k1_s_rets = [], []
logicA_k3_l_rets, logicA_k3_s_rets = [], []
logicA_k5_l_rets, logicA_k5_s_rets = [], []
logicB_l_rets, logicB_s_rets = [], []

trade_logs_A1_L = []
trade_logs_B_L = []

for qid in np.unique(query_ids):
    m = query_ids == qid
    if m.sum() < 3: continue
    
    r_l_v10 = pred_l_v10[m]
    r_s_v10 = pred_s_v10[m]
    p_l_v18 = pred_l_v18[m]
    p_s_v18 = pred_s_v18[m]
    actual  = y_returns[m]
    
    # Context data for logs
    q_df = df_test[m].reset_index(drop=True)

    # LOGIC A (Rank then Veto)
    # Top 1
    top1_l_idx = np.argsort(r_l_v10)[-1]
    if p_l_v18[top1_l_idx] > TRADE_PROB:
        logicA_k1_l_rets.append(actual[top1_l_idx])
        trade_logs_A1_L.append({
            'DateTime': q_df.iloc[top1_l_idx]['DateTime'],
            'Ticker': q_df.iloc[top1_l_idx]['Ticker'],
            'Return': actual[top1_l_idx],
            'V10_Rank_Score': r_l_v10[top1_l_idx],
            'V18_Prob': p_l_v18[top1_l_idx]
        })
        
    top1_s_idx = np.argsort(r_s_v10)[-1]
    if p_s_v18[top1_s_idx] > TRADE_PROB:
        logicA_k1_s_rets.append(-actual[top1_s_idx])

    # Top 3
    top3_l_idx = np.argsort(r_l_v10)[-3:]
    for idx in top3_l_idx:
        if p_l_v18[idx] > TRADE_PROB:
            logicA_k3_l_rets.append(actual[idx])
            
    top3_s_idx = np.argsort(r_s_v10)[-3:]
    for idx in top3_s_idx:
        if p_s_v18[idx] > TRADE_PROB:
            logicA_k3_s_rets.append(-actual[idx])

    # Top 5
    top5_l_idx = np.argsort(r_l_v10)[-5:]
    for idx in top5_l_idx:
        if p_l_v18[idx] > TRADE_PROB:
            logicA_k5_l_rets.append(actual[idx])
            
    top5_s_idx = np.argsort(r_s_v10)[-5:]
    for idx in top5_s_idx:
        if p_s_v18[idx] > TRADE_PROB:
            logicA_k5_s_rets.append(-actual[idx])

    # LOGIC B (Filter then Rank)
    # Long
    pass_l_idx = np.where(p_l_v18 > TRADE_PROB)[0]
    if len(pass_l_idx) > 0:
        best_pass_l_idx = pass_l_idx[np.argmax(r_l_v10[pass_l_idx])]
        logicB_l_rets.append(actual[best_pass_l_idx])
        trade_logs_B_L.append({
            'DateTime': q_df.iloc[best_pass_l_idx]['DateTime'],
            'Ticker': q_df.iloc[best_pass_l_idx]['Ticker'],
            'Return': actual[best_pass_l_idx],
            'V10_Rank_Score': r_l_v10[best_pass_l_idx],
            'V18_Prob': p_l_v18[best_pass_l_idx]
        })
        
    # Short
    pass_s_idx = np.where(p_s_v18 > TRADE_PROB)[0]
    if len(pass_s_idx) > 0:
        best_pass_s_idx = pass_s_idx[np.argmax(r_s_v10[pass_s_idx])]
        logicB_s_rets.append(-actual[best_pass_s_idx])

# Save specific trades to CSV
pd.DataFrame(trade_logs_A1_L).to_csv('data/A1_Long_Trades_Last12M.csv', index=False)
pd.DataFrame(trade_logs_B_L).to_csv('data/B_Long_Trades_Last12M.csv', index=False)

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

print("\nRESULTS (Threshold > 52% Prob):")
print("    Logic A (Top 1) - Rank then Veto")
print_logic_stats("Longs", logicA_k1_l_rets)
print_logic_stats("Shorts", logicA_k1_s_rets)
print("    Logic A (Top 3) - Rank then Veto")
print_logic_stats("Longs", logicA_k3_l_rets)
print_logic_stats("Shorts", logicA_k3_s_rets)
print("    Logic A (Top 5) - Rank then Veto")
print_logic_stats("Longs", logicA_k5_l_rets)
print_logic_stats("Shorts", logicA_k5_s_rets)
print("    Logic B (Filter then Rank)")
print_logic_stats("Longs", logicB_l_rets)
print_logic_stats("Shorts", logicB_s_rets)

print("=" * 64)
