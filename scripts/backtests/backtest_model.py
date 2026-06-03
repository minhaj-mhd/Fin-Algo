"""
backtest_model.py — Unified Backtester and Evaluator for Active Registry Models

Loads the active model from models/registry.json, loads its corresponding dataset,
runs temporal test set evaluation, and displays ranking, regression, and trading profitability stats.

Usage:
  python scripts/backtest_model.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.getcwd())

from scripts.model_registry import ModelRegistry

def main():
    print("=" * 70)
    print("VANGUARD MODEL UNIFIED BACKTESTER")
    print("=" * 70)

    # 1. Load active model configuration from registry
    try:
        registry = ModelRegistry()
        active = registry.get_active_model()
        model_name = active["name"]
        print(f"\n[OK] Loaded active model config: {model_name}")
        print(f"     Description: {active.get('description', 'No description')}")
    except Exception as e:
        print(f"[FATAL] Failed to load active model from registry: {e}")
        sys.exit(1)

    # 2. Resolve metadata and dataset paths
    meta_path = active["meta"]
    if not os.path.exists(meta_path):
        print(f"[FATAL] Model metadata not found at {meta_path}")
        sys.exit(1)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    feature_cols = metadata.get("features", [])
    data_file = metadata.get("data_file", "data/ranking_data_upstox.csv")

    if not os.path.exists(data_file):
        print(f"[WARN] Data file {data_file} specified in metadata not found.")
        # Fallbacks
        if os.path.exists("data/ranking_data_upstox.csv"):
            data_file = "data/ranking_data_upstox.csv"
        elif os.path.exists("data/ranking_data_full.csv"):
            data_file = "data/ranking_data_full.csv"
        else:
            print("[FATAL] No valid data CSV found in data/ directory.")
            sys.exit(1)

    print(f"[INFO] Using dataset: {data_file}")
    df = pd.read_csv(data_file)
    print(f"       Loaded {len(df):,} rows across {df['Query_ID'].nunique():,} queries")

    # 3. Load long/short boosters
    long_path = active["long_model"]
    short_path = active["short_model"]
    if not os.path.exists(long_path) or not os.path.exists(short_path):
        print(f"[FATAL] Model files not found:\n  Long: {long_path}\n  Short: {short_path}")
        sys.exit(1)

    bst_long = xgb.Booster()
    bst_long.load_model(long_path)
    bst_short = xgb.Booster()
    bst_short.load_model(short_path)

    # 4. Load scaler if configured
    scaler = None
    scaler_path = active.get("scaler")
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as sf:
            scaler = pickle.load(sf)
        print(f"[INFO] Loaded StandardScaler from {scaler_path}")
    else:
        print("[INFO] Scale-invariant mode (no scaler used)")

    # 5. Check feature compatibility
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        print(f"[WARN] {len(missing_features)} features missing from dataset! Fallback mapping...")
        feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"[INFO] Features count: {len(feature_cols)}")

    # 6. Strict Temporal Split (Evaluate ONLY on the last 20% of query periods)
    unique_qids = np.sort(df['Query_ID'].unique())
    split_idx = int(len(unique_qids) * 0.8)
    test_qids = unique_qids[split_idx:]
    df_test = df[df['Query_ID'].isin(test_qids)].copy()
    
    print(f"[INFO] Test Split: {len(df_test):,} rows | {len(test_qids):,} queries (last 20%)")

    # Prepare inputs
    X_test = df_test[feature_cols].values
    X_test = np.nan_to_num(X_test)
    y_actual = df_test['Next_Hour_Return'].values
    query_ids = df_test['Query_ID'].values

    scaler_is_fitted = (
        scaler is not None
        and hasattr(scaler, 'scale_')
        and scaler.scale_ is not None
    )
    if scaler_is_fitted:
        X_test_scaled = scaler.transform(X_test)
        print("[INFO] Scaler applied")
    else:
        X_test_scaled = X_test
        print("[INFO] Scale-invariant or dummy scaler mode (no scaler applied)")

    # Run inference
    dtest = xgb.DMatrix(X_test_scaled)
    long_scores = bst_long.predict(dtest)
    short_scores = bst_short.predict(dtest)

    df_test['long_score'] = long_scores
    df_test['short_score'] = short_scores

    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================

    # 1. Correlation Metrics
    spearman_long = []
    spearman_short = []
    kendall_long = []
    
    for qid in test_qids:
        mask = query_ids == qid
        if mask.sum() < 2:
            continue
        
        act = y_actual[mask]
        l_score = long_scores[mask]
        s_score = short_scores[mask]
        
        # Long Spearman: high score should correlate with positive returns
        sp_l, _ = spearmanr(l_score, act)
        if not np.isnan(sp_l):
            spearman_long.append(sp_l)
            
        # Short Spearman: high score should correlate with negative returns
        sp_s, _ = spearmanr(s_score, -act)
        if not np.isnan(sp_s):
            spearman_short.append(sp_s)

        kt_l, _ = kendalltau(l_score, act)
        if not np.isnan(kt_l):
            kendall_long.append(kt_l)

    print("\n" + "=" * 70)
    print("1. CORRELATION METRICS (Rank correlation per hour)")
    print("=" * 70)
    print(f"  Long Model Spearman Rho  :  mean={np.mean(spearman_long):.4f}  std={np.std(spearman_long):.4f}  median={np.median(spearman_long):.4f}")
    print(f"  Short Model Spearman Rho :  mean={np.mean(spearman_short):.4f}  std={np.std(spearman_short):.4f}  median={np.median(spearman_short):.4f}")
    print(f"  Long Model Kendall Tau   :  mean={np.mean(kendall_long):.4f}  std={np.std(kendall_long):.4f}")

    # 2. Precision At K (P@K)
    print("\n" + "=" * 70)
    print("2. PRECISION AT K (How often selections beat the query median)")
    print("=" * 70)
    
    TOPK = [1, 3, 5]
    for k in TOPK:
        long_hits = 0
        short_hits = 0
        total_picks = 0
        valid_queries = 0
        
        for qid in test_qids:
            q_mask = df_test['Query_ID'] == qid
            q_df = df_test[q_mask]
            
            if len(q_df) < k + 1:
                continue
                
            act_returns = q_df['Next_Hour_Return'].values
            median_ret = np.median(act_returns)
            
            l_sc = q_df['long_score'].values
            s_sc = q_df['short_score'].values
            
            # Select top K indexes
            top_long_idx = np.argsort(l_sc)[::-1][:k]
            top_short_idx = np.argsort(s_sc)[::-1][:k]
            
            # Long hit = return > median
            long_hits += (act_returns[top_long_idx] > median_ret).sum()
            # Short hit = return < median
            short_hits += (act_returns[top_short_idx] < median_ret).sum()
            
            total_picks += k
            valid_queries += 1
            
        p_long = long_hits / total_picks if total_picks > 0 else 0
        p_short = short_hits / total_picks if total_picks > 0 else 0
        
        print(f"  K={k} selections:")
        print(f"    Long  Precision @ {k} : {p_long:6.1%} (Random: 50.0% | Lift: {p_long/0.5 - 1:+.1%})")
        print(f"    Short Precision @ {k} : {p_short:6.1%} (Random: 50.0% | Lift: {p_short/0.5 - 1:+.1%})")

    # 3. Simulate Daily Trading Performance
    print("\n" + "=" * 70)
    print("3. SIMULATED HOURLY PERFORMANCE & WIN RATES")
    print("=" * 70)
    
    portfolio_stats = {
        1: {"long_ret": [], "short_ret": []},
        3: {"long_ret": [], "short_ret": []},
        5: {"long_ret": [], "short_ret": []}
    }
    market_returns = []
    
    for qid in test_qids:
        q_mask = df_test['Query_ID'] == qid
        q_df = df_test[q_mask]
        
        if len(q_df) < 1:
            continue
            
        act_returns = q_df['Next_Hour_Return'].values
        l_sc = q_df['long_score'].values
        s_sc = q_df['short_score'].values
        
        market_returns.append(np.mean(act_returns))
        
        for k in [1, 3, 5]:
            if len(q_df) >= k:
                topk_long_idx = np.argsort(l_sc)[::-1][:k]
                topk_short_idx = np.argsort(s_sc)[::-1][:k]
                portfolio_stats[k]["long_ret"].append(np.mean(act_returns[topk_long_idx]))
                portfolio_stats[k]["short_ret"].append(np.mean(-act_returns[topk_short_idx]))
                
    avg_m = np.mean(market_returns) if market_returns else 0.0
    print(f"  Market Mean hourly return :  {avg_m*100:+.4f}%\n")
    
    for k in [1, 3, 5]:
        long_rets = np.array(portfolio_stats[k]["long_ret"])
        short_rets = np.array(portfolio_stats[k]["short_ret"])
        
        avg_l = np.mean(long_rets) if len(long_rets) > 0 else 0.0
        avg_s = np.mean(short_rets) if len(short_rets) > 0 else 0.0
        
        wr_l = np.sum(long_rets > 0) / len(long_rets) if len(long_rets) > 0 else 0.0
        wr_s = np.sum(short_rets > 0) / len(short_rets) if len(short_rets) > 0 else 0.0
        
        print(f"  Top-{k} Portfolio Performance:")
        print(f"    Longs  : Avg hourly return = {avg_l*100:+.4f}% | Edge = {(avg_l - avg_m)*100:+.4f}% | Win Rate = {wr_l*100:.2f}%")
        print(f"    Shorts : Avg hourly return = {avg_s*100:+.4f}% | Edge = {(avg_s + avg_m)*100:+.4f}% | Win Rate = {wr_s*100:.2f}%")
        print()

    # Save a JSON report
    avg_l3 = np.mean(portfolio_stats[3]["long_ret"]) if portfolio_stats[3]["long_ret"] else 0.0
    avg_s3 = np.mean(portfolio_stats[3]["short_ret"]) if portfolio_stats[3]["short_ret"] else 0.0
    wr_l3 = np.sum(np.array(portfolio_stats[3]["long_ret"]) > 0) / len(portfolio_stats[3]["long_ret"]) if portfolio_stats[3]["long_ret"] else 0.0
    wr_s3 = np.sum(np.array(portfolio_stats[3]["short_ret"]) > 0) / len(portfolio_stats[3]["short_ret"]) if portfolio_stats[3]["short_ret"] else 0.0

    report = {
        "model_name": model_name,
        "spearman_long_mean": float(np.mean(spearman_long)),
        "spearman_short_mean": float(np.mean(spearman_short)),
        "avg_top3_long_return_pct": float(avg_l3 * 100),
        "avg_top3_short_return_pct": float(avg_s3 * 100),
        "wr_top3_long_pct": float(wr_l3 * 100),
        "wr_top3_short_pct": float(wr_s3 * 100),
        "avg_market_return_pct": float(avg_m * 100),
        "dataset_used": data_file,
        "backtested_at": datetime.now().isoformat()
    }
    
    report_path = "data/backtest_report.json"
    with open(report_path, "w") as rf:
        json.dump(report, rf, indent=2)
    print(f"\n[OK] Backtest completed. Results saved to {report_path}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    from datetime import datetime
    main()
