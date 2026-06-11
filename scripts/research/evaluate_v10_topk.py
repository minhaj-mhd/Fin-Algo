import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

DATA_FILE = "data/ranking_data_upstox_1h_v3_3y.csv"
TOPK = [1, 3, 5]

print(f"Loading {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

unique_qids = np.sort(df['Query_ID'].unique())
split_idx = int(len(unique_qids) * 0.8)
test_qids = unique_qids[split_idx:]
df_test = df[df['Query_ID'].isin(test_qids)].copy()

print(f"Test set: {df_test.shape[0]} rows")

with open("models/v10_native_1h/metadata.json") as f:
    meta = json.load(f)
feature_cols = meta["features"]

missing = [c for c in feature_cols if c not in df_test.columns]
if missing:
    for m in missing:
        df_test[m] = 0.0

X_test = df_test[feature_cols].values
X_test = np.nan_to_num(X_test)

bst_long = xgb.Booster()
bst_long.load_model("models/v10_native_1h/xgb_long_model.json")
bst_short = xgb.Booster()
bst_short.load_model("models/v10_native_1h/xgb_short_model.json")

dmatrix = xgb.DMatrix(X_test)
df_test["long_score"] = bst_long.predict(dmatrix)
df_test["short_score"] = bst_short.predict(dmatrix)

long_precisions = {}
short_precisions = {}

for k in TOPK:
    long_hits = 0
    short_hits = 0
    total_picks = 0
    
    for qid in test_qids:
        q_df = df_test[df_test['Query_ID'] == qid]
        if len(q_df) < k + 1:
            continue
            
        actual = q_df['Next_Hour_Return'].values
        median = np.median(actual)
        
        long_sc = q_df["long_score"].values
        short_sc = q_df["short_score"].values
        
        top_long_idx = np.argsort(long_sc)[::-1][:k]
        top_short_idx = np.argsort(short_sc)[::-1][:k]
        
        long_hits += (actual[top_long_idx] > median).sum()
        short_hits += (actual[top_short_idx] < median).sum()
        total_picks += k
        
    long_precisions[k] = long_hits / total_picks if total_picks > 0 else 0
    short_precisions[k] = short_hits / total_picks if total_picks > 0 else 0

top3_returns = []
for qid in test_qids:
    q_df = df_test[df_test['Query_ID'] == qid]
    if len(q_df) < 4:
        continue
    actual = q_df['Next_Hour_Return'].values
    long_sc = q_df["long_score"].values
    top3_idx = np.argsort(long_sc)[::-1][:3]
    top3_returns.append(actual[top3_idx].mean())

res = {
    "long_prec@1": long_precisions[1],
    "long_prec@3": long_precisions[3],
    "long_prec@5": long_precisions[5],
    "short_prec@3": short_precisions[3],
    "avg_top3_long_return": float(np.mean(top3_returns))
}
with open("data/v10_eval_results.json", "w") as f:
    json.dump(res, f)
print(json.dumps(res, indent=2))
