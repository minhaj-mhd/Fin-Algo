import os
import pandas as pd
import numpy as np
import xgboost as xgb

# Config
DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
V10_DIR = 'models/v10_depth4_1h'
V18_DIR = 'models/v18_random_forest_1h'

print("Loading data...")
df = pd.read_csv(DATA_FILE)
df['YearMonth'] = df['DateTime'].str[:7]

# Filter for the unimpeachable OOS window
mask = (df['YearMonth'] >= '2025-08') & (df['YearMonth'] <= '2026-05')
df_eval = df[mask].copy()

print(f"Evaluation rows (Aug 2025 - May 2026): {len(df_eval)}")
if len(df_eval) == 0:
    print("No data found for the specified period.")
    exit()

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Return', 'YearMonth']
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df_eval[feature_cols].values.astype(np.float64)

dtest = xgb.DMatrix(X)

print("Loading models...")
v10_long = xgb.Booster()
v10_long.load_model(f"{V10_DIR}/xgb_long_model.json")
v10_short = xgb.Booster()
v10_short.load_model(f"{V10_DIR}/xgb_short_model.json")

v18_long = xgb.Booster()
v18_long.load_model(f"{V18_DIR}/xgb_long_model.json")
v18_short = xgb.Booster()
v18_short.load_model(f"{V18_DIR}/xgb_short_model.json")

print("Generating predictions...")
df_eval['v10_long'] = v10_long.predict(dtest)
df_eval['v10_short'] = v10_short.predict(dtest)
df_eval['v18_long'] = v18_long.predict(dtest)
df_eval['v18_short'] = v18_short.predict(dtest)

# Evaluate Top K = 1, Top K = 3
for TOP_K in [1, 3]:
    print(f"\n================================")
    print(f"--- TOP {TOP_K} RANKED TRADES ---")
    print(f"================================")
    
    for prob_th in [0.0, 0.52]:
        gate_name = "NO VETO (v10 standalone)" if prob_th == 0.0 else f"v18 VETO > {prob_th*100}%"
        print(f"\n  [ {gate_name} ]")
        l_returns = []
        s_returns = []
        
        for qid in df_eval['Query_ID'].unique():
            q = df_eval[df_eval['Query_ID'] == qid]
            if len(q) < TOP_K: continue
            
            # Longs
            long_cands = q.sort_values('v10_long', ascending=False).head(TOP_K)
            valid_longs = long_cands[long_cands['v18_long'] > prob_th]
            l_returns.extend(valid_longs['Next_Hour_Return'].values)
            
            # Shorts
            short_cands = q.sort_values('v10_short', ascending=False).head(TOP_K)
            valid_shorts = short_cands[short_cands['v18_short'] > prob_th]
            s_returns.extend(-valid_shorts['Next_Hour_Return'].values) # invert for short
            
        def stats(rets, name):
            if not rets: return f"{name}: 0 trades"
            n = len(rets)
            raw = np.mean(rets)
            net6 = raw - 0.0006
            net10 = raw - 0.0010
            wr = np.mean(np.array(rets) > 0)
            return f"{name}: {n:>4} trades | WR: {wr:.1%} | Raw: {raw*10000:>+5.1f} bps | Net(6bps): {net6*10000:>+5.1f} bps | Net(10bps): {net10*10000:>+5.1f} bps"
            
        print("    " + stats(l_returns, "LONG "))
        print("    " + stats(s_returns, "SHORT"))
