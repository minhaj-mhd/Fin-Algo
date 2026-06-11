import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb

def load_and_filter_csv(path, month_prefixes):
    print(f"Loading and filtering {path} for {month_prefixes}...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000):
        mask = chunk['DateTime'].str.startswith(tuple(month_prefixes))
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)

def evaluate_model_edge(name, df, model_dir, return_col):
    print(f"\nEvaluating {name} from {model_dir}...")
    meta_path = f"{model_dir}/metadata.json"
    long_path = f"{model_dir}/xgb_long_model.json"
    short_path = f"{model_dir}/xgb_short_model.json"
    
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0
            
    X = df[feature_cols].values
    X = np.nan_to_num(X)
    
    bst_l = xgb.Booster()
    bst_l.load_model(long_path)
    bst_s = xgb.Booster()
    bst_s.load_model(short_path)
    
    dmat = xgb.DMatrix(X)
    df['long_score'] = bst_l.predict(dmat)
    df['short_score'] = bst_s.predict(dmat)
    
    # Extract time of day
    df['Time'] = df['DateTime'].str[11:16]
    
    results = []
    
    for time_val, group in df.groupby('Time'):
        long_returns = []
        short_returns = []
        
        for qid, q_group in group.groupby('Query_ID'):
            if len(q_group) < 3:
                continue
            
            actual = q_group[return_col].values
            
            # Top 3 Longs
            idx_long = np.argsort(q_group['long_score'].values)[::-1][:3]
            long_returns.extend(actual[idx_long])
            
            # Top 3 Shorts
            idx_short = np.argsort(q_group['short_score'].values)[::-1][:3]
            short_returns.extend(actual[idx_short])
            
        if long_returns and short_returns:
            results.append({
                'Model': name,
                'Time': time_val,
                'Long_Return_bps': np.mean(long_returns) * 10000,
                'Long_WinRate': np.mean(np.array(long_returns) > 0) * 100,
                'Short_Return_bps': np.mean(short_returns) * 10000, # Actual returns of shorts (should be negative for a good short)
                'Short_WinRate': np.mean(np.array(short_returns) < 0) * 100,
                'Count': len(long_returns)
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    months = ["2026-05"]
    
    # Data paths
    data_1h_v3 = "data/ranking_data_upstox_1h_v3_3y.csv"
    data_15m_v3 = "data/ranking_data_upstox_15min_3y_clean.csv"
    
    df_1h_v3 = load_and_filter_csv(data_1h_v3, months)
    df_15m_v3 = load_and_filter_csv(data_15m_v3, months)
    
    print(f"Data loaded: 1H_v3: {len(df_1h_v3)}, 15M_v3: {len(df_15m_v3)}")
    
    res_v10 = evaluate_model_edge("v10_native_1h", df_1h_v3, "models/v10_native_1h", "Next_Hour_Return")
    res_v3 = evaluate_model_edge("v3_15min_clean", df_15m_v3, "models/v3_15min_clean", "Next_15Min_Return")
    
    all_res = pd.concat([res_v10, res_v3])
    print("\n--- AUDIT RESULTS ---")
    print(all_res.to_string(index=False))
    
    all_res.to_csv("data/model_edge_audit_latest.csv", index=False)
    print("\nSaved to data/model_edge_audit_latest.csv")
