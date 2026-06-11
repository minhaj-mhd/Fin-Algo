import pandas as pd
import numpy as np
import xgboost as xgb
from collections import defaultdict

# Configuration
DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL = 'Next_Hour_Return'
TRADE_PROB = 0.52

print("=" * 64)
print("TRADE FREQUENCY ANALYSIS: Last 12 Months (2025-07 to 2026-06)")
print("=" * 64)

# 1. Load Data
df = pd.read_csv(DATA_FILE)
df['YearMonth'] = df['DateTime'].str[:7]
df['Date'] = df['DateTime'].str[:10]
df_test = df[df['YearMonth'] >= '2025-07'].copy()

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth', 'Date']
feature_cols = [c for c in df_test.columns if c not in exclude_cols]

X_test = df_test[feature_cols].values.astype(np.float64)
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

# 4. Evaluate Logic & Collect Dates
dates = {'A1_L': [], 'A1_S': [], 'A3_L': [], 'A3_S': [], 'B_L': [], 'B_S': []}

for qid in np.unique(query_ids):
    m = query_ids == qid
    if m.sum() < 3: continue
    
    r_l_v10 = pred_l_v10[m]
    r_s_v10 = pred_s_v10[m]
    p_l_v18 = pred_l_v18[m]
    p_s_v18 = pred_s_v18[m]
    
    q_df = df_test[m].reset_index(drop=True)

    # LOGIC A1
    top1_l_idx = np.argsort(r_l_v10)[-1]
    if p_l_v18[top1_l_idx] > TRADE_PROB:
        dates['A1_L'].append(q_df.iloc[top1_l_idx]['Date'])
        
    top1_s_idx = np.argsort(r_s_v10)[-1]
    if p_s_v18[top1_s_idx] > TRADE_PROB:
        dates['A1_S'].append(q_df.iloc[top1_s_idx]['Date'])

    # LOGIC A3
    top3_l_idx = np.argsort(r_l_v10)[-3:]
    for idx in top3_l_idx:
        if p_l_v18[idx] > TRADE_PROB:
            dates['A3_L'].append(q_df.iloc[idx]['Date'])
            
    top3_s_idx = np.argsort(r_s_v10)[-3:]
    for idx in top3_s_idx:
        if p_s_v18[idx] > TRADE_PROB:
            dates['A3_S'].append(q_df.iloc[idx]['Date'])

    # LOGIC B
    pass_l_idx = np.where(p_l_v18 > TRADE_PROB)[0]
    if len(pass_l_idx) > 0:
        best_pass_l_idx = pass_l_idx[np.argmax(r_l_v10[pass_l_idx])]
        dates['B_L'].append(q_df.iloc[best_pass_l_idx]['Date'])
        
    pass_s_idx = np.where(p_s_v18 > TRADE_PROB)[0]
    if len(pass_s_idx) > 0:
        best_pass_s_idx = pass_s_idx[np.argmax(r_s_v10[pass_s_idx])]
        dates['B_S'].append(q_df.iloc[best_pass_s_idx]['Date'])

# 5. Print Frequency Stats
unique_trading_days = df_test['Date'].nunique()
print(f"Total Trading Days in 12-month period: {unique_trading_days}")

logics = [
    ('Logic A1 Long', 'A1_L'), ('Logic A1 Short', 'A1_S'),
    ('Logic A3 Long', 'A3_L'), ('Logic A3 Short', 'A3_S'),
    ('Logic B Long', 'B_L'), ('Logic B Short', 'B_S')
]

for label, key in logics:
    trade_dates = dates[key]
    total_trades = len(trade_dates)
    if total_trades == 0:
        continue
        
    # Group by month
    trade_dates_series = pd.Series(trade_dates)
    months = trade_dates_series.str[:7]
    trades_per_month = months.value_counts().sort_index()
    
    avg_per_month = total_trades / 12.0
    avg_per_day = total_trades / unique_trading_days
    
    print(f"\n{label}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Avg per Month: {avg_per_month:.1f}")
    print(f"  Avg per Day: {avg_per_day:.2f}")
    
    print("  Monthly Breakdown:")
    for month, count in trades_per_month.items():
        print(f"    {month}: {count} trades")

print("\n" + "=" * 64)
