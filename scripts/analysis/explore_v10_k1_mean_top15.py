import os
import pandas as pd
import numpy as np
import xgboost as xgb

# Config
DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
V10_DIR = 'models/v10_native_1h'

print("Loading data...")
df = pd.read_csv(DATA_FILE)
df['YearMonth'] = df['DateTime'].str[:7]

# Filter for the unimpeachable OOS window
# Testing specifically on the last week of data (approx June 5 to June 12, 2026)
mask = (df['DateTime'] >= '2026-06-05') & (df['DateTime'] <= '2026-06-13')
df_eval = df[mask].copy()

print(f"Evaluation rows (OOS window): {len(df_eval)}")
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

print("Generating predictions...")
df_eval['v10_long'] = v10_long.predict(dtest)
df_eval['v10_short'] = v10_short.predict(dtest)

# Analysis thresholds (in percentage: 0 means no threshold beyond > mean)
thresholds = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]

print("\n--- Sweeping Relative Conviction Thresholds (K1 vs Mean of Top 15) ---")

# Pre-calculate Query_ID groups to save time
grouped = df_eval.groupby('Query_ID')

results = {th: {'long_rets': [], 'short_rets': []} for th in thresholds}

for qid, group in grouped:
    if len(group) < 15:
        continue
    
    # Process Long
    group_long = group.sort_values('v10_long', ascending=False)
    top15_long = group_long.head(15)
    mean_top15_long = top15_long['v10_long'].mean()
    k1_long = top15_long.iloc[0]
    
    # Process Short
    group_short = group.sort_values('v10_short', ascending=False)
    top15_short = group_short.head(15)
    mean_top15_short = top15_short['v10_short'].mean()
    k1_short = top15_short.iloc[0]
    
    for th in thresholds:
        # Long condition
        if k1_long['v10_long'] > mean_top15_long * (1 + th / 100.0):
            results[th]['long_rets'].append(k1_long['Next_Hour_Return'])
            
        # Short condition
        if k1_short['v10_short'] > mean_top15_short * (1 + th / 100.0):
            results[th]['short_rets'].append(-k1_short['Next_Hour_Return'])

print("\n================================")
print("RESULTS SUMMARY")
print("================================")

def stats(rets, name):
    if not rets: return f"{name}: 0 trades"
    n = len(rets)
    raw = np.mean(rets)
    net10 = raw - 0.0010
    wr = np.mean(np.array(rets) > 0)
    return f"{name}: {n:>4} trades | WR: {wr:.1%} | Raw: {raw*10000:>+5.1f} bps | Net(10bps): {net10*10000:>+5.1f} bps"

for th in thresholds:
    print(f"\n[ Threshold: +{th}% over Mean Top 15 ]")
    print("  " + stats(results[th]['long_rets'], "LONG "))
    print("  " + stats(results[th]['short_rets'], "SHORT"))

    # Combined
    combined = results[th]['long_rets'] + results[th]['short_rets']
    if combined:
        print("  " + stats(combined, "TOTAL"))
