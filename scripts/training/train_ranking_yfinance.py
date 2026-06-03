"""
Train XGBoost ranking models on 2-Year yfinance hourly data.
- Loads ranking_data_yfinance_2y.csv
- Splits data temporally (80% train, 20% test)
- Trains separate Long and Short pairwise ranking models
- Saves new models to models/v7_yfinance_2y/
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import StandardScaler
from datetime import datetime

sys.path.append(os.getcwd())

# Configuration
DATA_FILE = 'data/ranking_data_yfinance_2y.csv'
MODEL_DIR = 'models/v7_yfinance_2y'
LONG_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_long_model.json')
SHORT_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_short_model.json')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
META_PATH = os.path.join(MODEL_DIR, 'metadata.json')

def main():
    print("=" * 70)
    print("YFINANCE 2-YEAR MODEL TRAINING PIPELINE")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        print(f"[FATAL] {DATA_FILE} not found. Run prepare_ranking_data_yfinance.py first.")
        sys.exit(1)

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Loading dataset: {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {df.shape[0]:,} rows")

    # Select features
    exclude_cols = ['DateTime', 'DateTime_Hour', 'Query_ID', 'Ticker',
                    'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Return']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y_returns = df['Next_Hour_Return'].values
    query_ids = df['Query_ID'].values

    print(f"Features count: {len(feature_cols)}")
    print(f"Total samples: {X.shape[0]:,}")
    print(f"Total queries: {len(np.unique(query_ids)):,}")

    # Handle NaN/Inf in features
    nan_mask = np.isnan(X) | np.isinf(X)
    if nan_mask.any():
        print(f"[INFO] Imputing {nan_mask.sum()} NaN/Inf values...")
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            valid_data = col_data[~(np.isnan(col_data) | np.isinf(col_data))]
            if len(valid_data) > 0:
                X[np.isnan(col_data) | np.isinf(col_data), col_idx] = np.mean(valid_data)

    # Temporal split (80% train / 20% test queries)
    unique_qids = np.sort(df['Query_ID'].unique())
    split_idx = int(len(unique_qids) * 0.8)
    train_qids = unique_qids[:split_idx]
    test_qids = unique_qids[split_idx:]

    train_mask = df['Query_ID'].isin(train_qids).values
    test_mask = df['Query_ID'].isin(test_qids).values

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_raw, y_test_raw = y_returns[train_mask], y_returns[test_mask]
    query_ids_train = query_ids[train_mask]
    query_ids_test = query_ids[test_mask]

    print(f"\nTemporal Split Details:")
    print(f"  Train Set: {X_train.shape[0]:,} rows | {len(train_qids):,} queries")
    print(f"  Test Set : {X_test.shape[0]:,} rows | {len(test_qids):,} queries")

    # Helper function to generate integer rank targets per query
    def get_integer_ranks(y_vals, qids, invert=False):
        y_int = np.zeros_like(y_vals, dtype=int)
        for qid in np.unique(qids):
            mask = qids == qid
            if mask.sum() == 0: continue
            vals = -y_vals[mask] if invert else y_vals[mask]
            # rankdata returns 1-indexed ranks, convert to 0-indexed integer
            y_int[mask] = rankdata(vals, method='ordinal') - 1
        return y_int

    print("\nCreating integer rank targets...")
    y_long_train = get_integer_ranks(y_train_raw, query_ids_train, invert=False)
    y_long_test = get_integer_ranks(y_test_raw, query_ids_test, invert=False)
    y_short_train = get_integer_ranks(y_train_raw, query_ids_train, invert=True)
    y_short_test = get_integer_ranks(y_test_raw, query_ids_test, invert=True)

    # Compute group sizes (required by XGBoost rank objectives)
    group_sizes_train = pd.Series(query_ids_train).groupby(query_ids_train).size().values
    group_sizes_test = pd.Series(query_ids_test).groupby(query_ids_test).size().values

    # Detect if CUDA device is available
    try:
        # A tiny test model to see if device 'cuda' works
        dtest_gpu = xgb.DMatrix(np.array([[1.0]]), label=np.array([1.0]))
        dtest_gpu.set_group(np.array([1]))
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, dtest_gpu, num_boost_round=1)
        device = 'cuda'
        print("[INFO] CUDA device detected and working. Training on GPU.")
    except Exception as e:
        print(f"[INFO] CUDA device not available or failed with error: {e}. Falling back to CPU.")
        device = 'cpu'

    # Model training params (optimised for top-3 rankings)
    params = {
        'objective': 'rank:pairwise',
        'eta': 0.03,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 1.0,
        'lambda': 2.0,
        'min_child_weight': 10,
        'random_state': 42,
        'verbosity': 0,
        'eval_metric': 'ndcg@3',
        'ndcg_exp_gain': False,
        'tree_method': 'hist',
        'device': device
    }

    # Training function
    def train_model(X_tr, y_tr, grp_tr, X_te, y_te, grp_te, save_path, model_desc):
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dtrain.set_group(grp_tr)
        dtest = xgb.DMatrix(X_te, label=y_te)
        dtest.set_group(grp_te)

        print(f"\nTraining {model_desc} Model...")
        bst = xgb.train(params, dtrain, num_boost_round=600,
                        evals=[(dtrain, 'train'), (dtest, 'test')],
                        early_stopping_rounds=50,
                        verbose_eval=50)
        bst.save_model(save_path)
        print(f"[SUCCESS] Saved model: {save_path}")
        return bst

    # Evaluate Spearman correlation per query
    def evaluate_model(bst, X_tr, X_te, y_raw_tr, y_raw_te, qids_tr, qids_te, invert=False):
        dtrain = xgb.DMatrix(X_tr)
        dtest = xgb.DMatrix(X_te)
        y_pred_train = bst.predict(dtrain)
        y_pred_test = bst.predict(dtest)

        y_eval_tr = -y_raw_tr if invert else y_raw_tr
        y_eval_te = -y_raw_te if invert else y_raw_te

        train_corrs = []
        for qid in np.unique(qids_tr):
            mask = qids_tr == qid
            if mask.sum() > 1:
                corr, _ = spearmanr(y_pred_train[mask], y_eval_tr[mask])
                if not np.isnan(corr):
                    train_corrs.append(corr)

        test_corrs = []
        for qid in np.unique(qids_te):
            mask = qids_te == qid
            if mask.sum() > 1:
                corr, _ = spearmanr(y_pred_test[mask], y_eval_te[mask])
                if not np.isnan(corr):
                    test_corrs.append(corr)

        return float(np.mean(train_corrs)), float(np.mean(test_corrs))

    # 1. Train Long Model
    bst_long = train_model(X_train, y_long_train, group_sizes_train,
                           X_test, y_long_test, group_sizes_test,
                           LONG_MODEL_PATH, "Long Ranker")
    train_rho_long, test_rho_long = evaluate_model(
        bst_long, X_train, X_test, y_train_raw, y_test_raw, query_ids_train, query_ids_test, invert=False
    )

    # 2. Train Short Model
    bst_short = train_model(X_train, y_short_train, group_sizes_train,
                            X_test, y_short_test, group_sizes_test,
                            SHORT_MODEL_PATH, "Short Ranker")
    train_rho_short, test_rho_short = evaluate_model(
        bst_short, X_train, X_test, y_train_raw, y_test_raw, query_ids_train, query_ids_test, invert=True
    )

    # 3. Save StandardScaler (Dummy/No-Op to match vanguard_signal_engine signature)
    scaler = StandardScaler(with_mean=False, with_std=False)
    with open(SCALER_PATH, 'wb') as sf:
        pickle.dump(scaler, sf)
    print(f"[SUCCESS] Saved dummy StandardScaler to: {SCALER_PATH}")

    # 4. Save metadata JSON
    metadata = {
        'features': feature_cols,
        'num_features': len(feature_cols),
        'data_source': 'yfinance_2y_hourly',
        'data_file': DATA_FILE,
        'train_queries': int(len(train_qids)),
        'test_queries': int(len(test_qids)),
        'total_rows': int(df.shape[0]),
        'long_train_spearman': train_rho_long,
        'long_test_spearman': test_rho_long,
        'short_train_spearman': train_rho_short,
        'short_test_spearman': test_rho_short,
        'params': params,
        'trained_at': datetime.now().isoformat()
    }
    with open(META_PATH, 'w') as mf:
        json.dump(metadata, mf, indent=2)
    print(f"[SUCCESS] Saved model metadata to: {META_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING METRICS SUMMARY")
    print("=" * 70)
    print(f"  Long Model  :  Train Spearman Rho = {train_rho_long:.4f}  |  Test Spearman Rho = {test_rho_long:.4f}")
    print(f"  Short Model :  Train Spearman Rho = {train_rho_short:.4f}  |  Test Spearman Rho = {test_rho_short:.4f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
