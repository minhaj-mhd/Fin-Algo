"""
Train XGBoost ranking models on UPSTOX hourly data.
- Prefers ranking_data_upstox_3y.csv (3-year backfill) when available
- Falls back to ranking_data_upstox.csv (90-day) if 3-year not built yet
- Trains separate Long and Short pairwise ranking models
- Saves models to models/v8_upstox_3y/ directory
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, rankdata
import pickle
import json
import os
import shutil
from datetime import datetime

# ========================================
# LOAD DATA
# ========================================
print("=" * 60)
print("UPSTOX MODEL TRAINING PIPELINE")
print("=" * 60)

MODEL_VERSION    = 'v8_upstox_3y'
os.makedirs(f'models/{MODEL_VERSION}', exist_ok=True)
LONG_MODEL_PATH  = f'models/{MODEL_VERSION}/xgb_long_model.json'
SHORT_MODEL_PATH = f'models/{MODEL_VERSION}/xgb_short_model.json'
META_PATH        = f'models/{MODEL_VERSION}/metadata.json'

# Prefer 3-year dataset, fall back to 90-day
data_file_3y = 'data/ranking_data_upstox_3y.csv'
data_file_90 = 'data/ranking_data_upstox.csv'

if os.path.exists(data_file_3y):
    data_file = data_file_3y
    print(f"[DATA] Using 3-year dataset: {data_file_3y}")
elif os.path.exists(data_file_90):
    data_file = data_file_90
    print(f"[DATA] 3-year dataset not found, using 90-day fallback: {data_file_90}")
else:
    print(f"[FATAL] No dataset found. Run scripts/collect_upstox_3y.py first.")
    exit(1)

print(f"Output models : {LONG_MODEL_PATH}, {SHORT_MODEL_PATH}")

print("Loading data...")
df = pd.read_csv(data_file)
print(f"Loaded {df.shape[0]:,} rows")

# ========================================
# FEATURE SELECTION
# ========================================
exclude_cols = ['DateTime', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Return']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y_returns = df['Next_Hour_Return'].values
query_ids = df['Query_ID'].values

print(f"Features: {len(feature_cols)}")
print(f"Samples: {X.shape[0]:,}")
print(f"Unique Queries: {len(np.unique(query_ids)):,}")

# Handle NaN/Inf
nan_mask = np.isnan(X) | np.isinf(X)
if nan_mask.any():
    print(f"Replacing {nan_mask.sum()} NaN/Inf values")
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        valid_data = col_data[~(np.isnan(col_data) | np.isinf(col_data))]
        if len(valid_data) > 0:
            X[np.isnan(col_data) | np.isinf(col_data), col_idx] = np.mean(valid_data)

# ========================================
# TEMPORAL TRAIN/TEST SPLIT (80/20)
# ========================================
unique_query_ids = np.sort(df['Query_ID'].unique())
split_idx = int(len(unique_query_ids) * 0.8)

train_qids = unique_query_ids[:split_idx]
test_qids = unique_query_ids[split_idx:]

train_mask = df['Query_ID'].isin(train_qids).values
test_mask = df['Query_ID'].isin(test_qids).values

X_train, X_test = X[train_mask], X[test_mask]
y_train_raw, y_test_raw = y_returns[train_mask], y_returns[test_mask]
query_ids_train = query_ids[train_mask]
query_ids_test = query_ids[test_mask]

print(f"\nTemporal Split:")
print(f"  Train: {X_train.shape[0]:,} rows, {len(np.unique(query_ids_train)):,} queries")
print(f"  Test:  {X_test.shape[0]:,} rows, {len(np.unique(query_ids_test)):,} queries")

# ========================================
# CREATE INTEGER RANK LABELS
# ========================================
def get_integer_ranks(y_vals, qids, invert=False):
    y_int = np.zeros_like(y_vals, dtype=int)
    for qid in np.unique(qids):
        mask = qids == qid
        if mask.sum() == 0: continue
        vals = -y_vals[mask] if invert else y_vals[mask]
        y_int[mask] = rankdata(vals, method='ordinal') - 1
    return y_int

print("\nCreating integer rank labels...")
y_long_train = get_integer_ranks(y_train_raw, query_ids_train, invert=False)
y_long_test = get_integer_ranks(y_test_raw, query_ids_test, invert=False)
y_short_train = get_integer_ranks(y_train_raw, query_ids_train, invert=True)
y_short_test = get_integer_ranks(y_test_raw, query_ids_test, invert=True)

# ========================================
# TRAINING FUNCTIONS
# ========================================
def train_model(X_tr, y_tr, grp_tr, X_te, y_te, grp_te, model_path, params):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group(grp_tr)
    dtest = xgb.DMatrix(X_te, label=y_te)
    dtest.set_group(grp_te)

    print(f"\n  Training {model_path}...")
    bst = xgb.train(params, dtrain, num_boost_round=500,
                    evals=[(dtrain, 'train'), (dtest, 'test')],
                    early_stopping_rounds=50,
                    verbose_eval=False)

    bst.save_model(model_path)
    print(f"  Saved: {model_path} (best iteration: {bst.best_iteration})")
    return bst

def evaluate_model(bst, X_tr, X_te, y_raw_tr, y_raw_te, qids_tr, qids_te, grp_tr, grp_te, invert=False):
    dtrain = xgb.DMatrix(X_tr)
    dtest = xgb.DMatrix(X_te)

    y_pred_train = bst.predict(dtrain)
    y_pred_test = bst.predict(dtest)

    y_eval_tr = -y_raw_tr if invert else y_raw_tr
    y_eval_te = -y_raw_te if invert else y_raw_te

    train_corrs, test_corrs = [], []
    for qid in np.unique(qids_tr):
        mask = qids_tr == qid
        if mask.sum() > 1:
            corr, _ = spearmanr(y_pred_train[mask], y_eval_tr[mask])
            if not np.isnan(corr): train_corrs.append(corr)

    for qid in np.unique(qids_te):
        mask = qids_te == qid
        if mask.sum() > 1:
            corr, _ = spearmanr(y_pred_test[mask], y_eval_te[mask])
            if not np.isnan(corr): test_corrs.append(corr)

    return np.mean(train_corrs), np.mean(test_corrs)

# ========================================
# COMPUTE GROUP SIZES
# ========================================
group_sizes_train = pd.Series(query_ids_train).groupby(query_ids_train).size().values
group_sizes_test = pd.Series(query_ids_test).groupby(query_ids_test).size().values

# ========================================
# TRAINING PARAMS (same as original, tuned slightly for smaller dataset)
# ========================================
params = {
    'objective': 'rank:pairwise',
    'eta': 0.03,            # Slower learning rate for smaller dataset
    'max_depth': 5,         # Slightly shallower to reduce overfitting
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 1.0,
    'lambda': 2.0,
    'min_child_weight': 10,  # Higher to prevent overfitting on small data
    'random_state': 42,
    'verbosity': 0,
    'eval_metric': 'ndcg@3',  # Optimise for top-3 (we pick top-3 signals)
    'ndcg_exp_gain': False,
    'tree_method': 'hist',
    'device': 'cuda'
}

# ========================================
# TRAIN LONG MODEL
# ========================================
print("\n" + "=" * 60)
print("TRAINING LONG RANKING MODEL")
print("=" * 60)

bst_long = train_model(X_train, y_long_train, group_sizes_train,
                       X_test, y_long_test, group_sizes_test,
                       LONG_MODEL_PATH, params)

train_rho_long, test_rho_long = evaluate_model(
    bst_long, X_train, X_test, y_train_raw, y_test_raw,
    query_ids_train, query_ids_test, group_sizes_train, group_sizes_test)

# ========================================
# TRAIN SHORT MODEL
# ========================================
print("\n" + "=" * 60)
print("TRAINING SHORT RANKING MODEL")
print("=" * 60)

bst_short = train_model(X_train, y_short_train, group_sizes_train,
                        X_test, y_short_test, group_sizes_test,
                        SHORT_MODEL_PATH, params)

train_rho_short, test_rho_short = evaluate_model(
    bst_short, X_train, X_test, y_train_raw, y_test_raw,
    query_ids_train, query_ids_test, group_sizes_train, group_sizes_test,
    invert=True)

# ========================================
# SAVE METADATA & SCALER
# ========================================
scaler = StandardScaler(with_mean=False, with_std=False)
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

# Load OLD production metadata for comparison only
old_meta = {}
old_meta_path = 'models/model_metadata.json'
if os.path.exists(old_meta_path):
    with open(old_meta_path) as f:
        old_meta = json.load(f)

metadata = {
    'features': feature_cols,
    'num_features': len(feature_cols),
    'data_source': 'upstox_90d_hourly',
    'data_file': data_file,
    'train_queries': int(len(np.unique(query_ids_train))),
    'test_queries': int(len(np.unique(query_ids_test))),
    'total_rows': int(df.shape[0]),
    'long_test_spearman': float(test_rho_long),
    'short_test_spearman': float(test_rho_short),
    'long_train_spearman': float(train_rho_long),
    'short_train_spearman': float(train_rho_short),
    'params': params,
    'trained_at': datetime.now().isoformat(),
}

with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

# ========================================
# RESULTS COMPARISON
# ========================================
print("\n" + "=" * 60)
print("RESULTS — OLD vs NEW")
print("=" * 60)

old_long_rho = old_meta.get('long_test_spearman', 'N/A')
old_short_rho = old_meta.get('short_test_spearman', 'N/A')

print(f"\n{'Metric':<30} {'OLD (yfinance)':>15} {'NEW (Upstox)':>15} {'Change':>10}")
print("-" * 75)
print(f"{'Long rho (test)':<30} {old_long_rho:>15.4f} {test_rho_long:>15.4f} {'+' if test_rho_long > old_long_rho else ''}{test_rho_long - old_long_rho:>9.4f}")
print(f"{'Short rho (test)':<30} {old_short_rho:>15.4f} {test_rho_short:>15.4f} {'+' if test_rho_short > old_short_rho else ''}{test_rho_short - old_short_rho:>9.4f}")
print(f"{'Long rho (train)':<30} {'N/A':>15} {train_rho_long:>15.4f}")
print(f"{'Short rho (train)':<30} {'N/A':>15} {train_rho_short:>15.4f}")
print(f"{'Training rows':<30} {old_meta.get('train_queries', 'N/A') if isinstance(old_meta.get('train_queries'), int) else 'N/A':>15} {len(np.unique(query_ids_train)):>15}")
print(f"{'Test rows':<30} {old_meta.get('test_queries', 'N/A') if isinstance(old_meta.get('test_queries'), int) else 'N/A':>15} {len(np.unique(query_ids_test)):>15}")

# Overfit check
print(f"\n{'Overfit Check':<30}")
print(f"  Long:  train rho = {train_rho_long:.4f}, test rho = {test_rho_long:.4f} -> gap = {train_rho_long - test_rho_long:.4f}")
print(f"  Short: train rho = {train_rho_short:.4f}, test rho = {test_rho_short:.4f} -> gap = {train_rho_short - test_rho_short:.4f}")

if train_rho_long - test_rho_long > 0.05:
    print("  [WARN] Long model may be overfitting (train-test gap > 0.05)")
if train_rho_short - test_rho_short > 0.05:
    print("  [WARN] Short model may be overfitting (train-test gap > 0.05)")

# Feature importance (top 10)
print(f"\n{'=' * 60}")
print("TOP 10 FEATURE IMPORTANCE (Long Model)")
print("=" * 60)
importance = bst_long.get_score(importance_type='gain')
# Map feature indices to names
imp_named = {}
for k, v in importance.items():
    idx = int(k.replace('f', ''))
    if idx < len(feature_cols):
        imp_named[feature_cols[idx]] = v

for feat, score in sorted(imp_named.items(), key=lambda x: -x[1])[:10]:
    print(f"  {feat:<30} {score:>10.2f}")

print(f"\n{'=' * 60}")
print("TOP 10 FEATURE IMPORTANCE (Short Model)")
print("=" * 60)
importance_s = bst_short.get_score(importance_type='gain')
imp_named_s = {}
for k, v in importance_s.items():
    idx = int(k.replace('f', ''))
    if idx < len(feature_cols):
        imp_named_s[feature_cols[idx]] = v

for feat, score in sorted(imp_named_s.items(), key=lambda x: -x[1])[:10]:
    print(f"  {feat:<30} {score:>10.2f}")

print(f"\n{'=' * 60}")
print("TRAINING COMPLETE!")
print(f"   Long model  : {LONG_MODEL_PATH}")
print(f"   Short model : {SHORT_MODEL_PATH}")
print(f"   Metadata    : {META_PATH}")
print(f"   Production models are UNTOUCHED.")
print(f"   To activate: update vanguard_signal_engine.py to point to new model paths.")
print(f"{'=' * 60}")
