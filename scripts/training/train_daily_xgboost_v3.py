"""
Train V3 daily macro XGBoost ranking models using Walk-Forward Validation.
- Loads data/ranking_data_daily_macro_v3.csv (1-day forward return target)
- Performs 4-fold Walk-Forward Validation to ensure honest, leakage-free estimates
- Trains final production models on maximum historical context
- Saves final models, walk-forward metadata, and scaler to models/daily_macro_v3/
"""

import os
import sys
import pickle
import json
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, rankdata

sys.path.append(os.getcwd())

# ========================================
# CONFIG
# ========================================
MODEL_VERSION = 'daily_macro_v3'
MODEL_DIR = f'models/{MODEL_VERSION}'
os.makedirs(MODEL_DIR, exist_ok=True)

LONG_MODEL_PATH  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_MODEL_PATH = f'{MODEL_DIR}/xgb_short_model.json'
META_PATH        = f'{MODEL_DIR}/metadata.json'
SCALER_PATH      = f'{MODEL_DIR}/scaler.pkl'

DATA_FILE = 'data/ranking_data_daily_macro_v3.csv'

print("=" * 60)
print("DAILY MACRO XGBOOST RANKER TRAINING PIPELINE (V3)")
print("Target: 1-Day Forward Close-to-Close Return (Label_1D)")
print("=" * 60)

if not os.path.exists(DATA_FILE):
    print(f"[FATAL] Data file not found: {DATA_FILE}")
    sys.exit(1)

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
print(f"Loaded {df.shape[0]:,} rows")

# Extract the YYYY-MM month string for temporal splits
df['YearMonth'] = df['DateTime'].str[:7]
unique_months = sorted(df['YearMonth'].unique())
print(f"Data spans {len(unique_months)} months: {unique_months[0]} to {unique_months[-1]}")

# ========================================
# FEATURE SELECTION
# ========================================
exclude_cols = ['DateTime', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', 'Label_1D', 'Sector', 'YearMonth']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y_returns = df['Label_1D'].values
query_ids = df['Query_ID'].values
year_months = df['YearMonth'].values

print(f"Features: {len(feature_cols)}")
print(f"Samples: {X.shape[0]:,}")
print(f"Unique Queries (Trading Days): {len(np.unique(query_ids)):,}")

# Handle NaN/Inf in features
nan_mask = np.isnan(X) | np.isinf(X)
if nan_mask.any():
    print(f"Replacing {nan_mask.sum()} NaN/Inf values...")
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        valid_data = col_data[~(np.isnan(col_data) | np.isinf(col_data))]
        if len(valid_data) > 0:
            X[np.isnan(col_data) | np.isinf(col_data), col_idx] = np.mean(valid_data)

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

# ========================================
# GPU DETECTION
# ========================================
device = 'cpu'
try:
    print("Checking CUDA GPU availability...")
    d_dummy = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10))
    d_dummy.set_group([10])
    xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, d_dummy, num_boost_round=1)
    device = 'cuda'
    print("  CUDA GPU detected! Training will use hardware acceleration.")
except Exception:
    print("  CUDA GPU not available. Falling back to CPU training.")

# ========================================
# TRAINING HYPERPARAMETERS (RECTIFIED V3)
# ========================================
params = {
    'objective': 'rank:pairwise',
    'eta': 0.01,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 2.0,
    'lambda': 4.0,
    'min_child_weight': 40,
    'random_state': 42,
    'verbosity': 0,
    'eval_metric': 'ndcg@5',
    'ndcg_exp_gain': False,
    'tree_method': 'hist',
    'device': device
}

# ========================================
# EVALUATION METRIC HELPERS
# ========================================
def compute_spearman_rho(df_eval, score_col, invert=False):
    rhos = []
    for qid in df_eval['Query_ID'].unique():
        q_df = df_eval[df_eval['Query_ID'] == qid]
        if len(q_df) > 1:
            y_eval = -q_df['Label_1D'].values if invert else q_df['Label_1D'].values
            pred = q_df[score_col].values
            rho, _ = spearmanr(pred, y_eval)
            if not np.isnan(rho):
                rhos.append(rho)
    return np.mean(rhos) if rhos else 0.0

def evaluate_ranking_performance(df_eval, long_scores, short_scores):
    df_sub = df_eval.copy()
    df_sub['long_score'] = long_scores
    df_sub['short_score'] = short_scores
    
    unique_qids = df_sub['Query_ID'].unique()
    
    long_rho = compute_spearman_rho(df_sub, 'long_score', invert=False)
    short_rho = compute_spearman_rho(df_sub, 'short_score', invert=True)
    
    topk_list = [1, 3, 5]
    long_win_rates = {}
    short_win_rates = {}
    
    for k in topk_list:
        long_hits = 0
        long_total = 0
        short_hits = 0
        short_total = 0
        
        for qid in unique_qids:
            q_df = df_sub[df_sub['Query_ID'] == qid]
            if len(q_df) < k + 1:
                continue
                
            actual_returns = q_df['Label_1D'].values
            median_return = np.median(actual_returns)
            
            # Long
            long_sc = q_df['long_score'].values
            top_long_idx = np.argsort(long_sc)[::-1][:k]
            long_hits += (actual_returns[top_long_idx] > median_return).sum()
            long_total += k
            
            # Short
            short_sc = q_df['short_score'].values
            top_short_idx = np.argsort(short_sc)[::-1][:k]
            short_hits += (actual_returns[top_short_idx] < median_return).sum()
            short_total += k
            
        long_win_rates[k] = long_hits / long_total if long_total > 0 else 0.0
        short_win_rates[k] = short_hits / short_total if short_total > 0 else 0.0
        
    top3_long_returns = []
    top3_short_returns = []
    random_returns = []
    
    for qid in unique_qids:
        q_df = df_sub[df_sub['Query_ID'] == qid]
        if len(q_df) < 4:
            continue
            
        actual = q_df['Label_1D'].values
        long_sc = q_df['long_score'].values
        short_sc = q_df['short_score'].values
        
        top3_long_idx = np.argsort(long_sc)[::-1][:3]
        top3_short_idx = np.argsort(short_sc)[::-1][:3]
        
        top3_long_returns.append(actual[top3_long_idx].mean())
        top3_short_returns.append(-actual[top3_short_idx].mean())
        random_returns.append(actual.mean())
        
    if top3_long_returns:
        avg_long = np.mean(top3_long_returns)
        avg_short = np.mean(top3_short_returns)
        avg_rand = np.mean(random_returns)
        
        long_edge = avg_long - avg_rand
        short_edge = avg_short - (-avg_rand)
        combined_edge = avg_long + avg_short
    else:
        avg_long, avg_short, avg_rand = 0.0, 0.0, 0.0
        long_edge, short_edge, combined_edge = 0.0, 0.0, 0.0
        
    return {
        'long_rho': long_rho,
        'short_rho': short_rho,
        'long_win_rates': long_win_rates,
        'short_win_rates': short_win_rates,
        'avg_long_return': avg_long,
        'avg_short_return': avg_short,
        'avg_market_return': avg_rand,
        'long_edge': long_edge,
        'short_edge': short_edge,
        'combined_edge': combined_edge
    }

# ========================================
# WALK-FORWARD FOLDS CONFIG
# ========================================
val_size = 6
test_size = 6
folds_config = []

for fold_idx in range(1, 5):
    test_end_idx = len(unique_months) - (4 - fold_idx) * test_size
    test_start_idx = test_end_idx - test_size
    val_end_idx = test_start_idx
    val_start_idx = val_end_idx - val_size
    train_end_idx = val_start_idx
    
    folds_config.append({
        'fold': fold_idx,
        'train_months': unique_months[:train_end_idx],
        'val_months': unique_months[val_start_idx:val_end_idx],
        'test_months': unique_months[test_start_idx:test_end_idx]
    })

walk_forward_results = []

# ========================================
# RUN WALK-FORWARD VALIDATION
# ========================================
print("\n" + "=" * 60)
print("RUNNING WALK-FORWARD VALIDATION ON V3 DAILY MACRO TRENDS")
print("=" * 60)

for cfg in folds_config:
    fold_idx = cfg['fold']
    tr_m, val_m, te_m = cfg['train_months'], cfg['val_months'], cfg['test_months']
    
    print(f"\n--- FOLD {fold_idx} ---")
    print(f"  Train: {tr_m[0]} -> {tr_m[-1]}")
    print(f"  Val:   {val_m[0]} -> {val_m[-1]}")
    print(f"  Test:  {te_m[0]} -> {te_m[-1]}")
    
    tr_mask = df['YearMonth'].isin(tr_m).values
    val_mask = df['YearMonth'].isin(val_m).values
    te_mask = df['YearMonth'].isin(te_m).values
    
    X_tr, y_tr_raw, qids_tr = X[tr_mask], y_returns[tr_mask], query_ids[tr_mask]
    X_val, y_val_raw, qids_val = X[val_mask], y_returns[val_mask], query_ids[val_mask]
    X_te, y_te_raw, qids_te = X[te_mask], y_returns[te_mask], query_ids[te_mask]
    df_te = df[te_mask]
    
    print(f"  Data sizes: Train={X_tr.shape[0]:,}, Val={X_val.shape[0]:,}, Test={X_te.shape[0]:,}")
    
    y_long_tr = get_integer_ranks(y_tr_raw, qids_tr, invert=False)
    y_long_val = get_integer_ranks(y_val_raw, qids_val, invert=False)
    y_short_tr = get_integer_ranks(y_tr_raw, qids_tr, invert=True)
    y_short_val = get_integer_ranks(y_val_raw, qids_val, invert=True)
    
    grp_tr = pd.Series(qids_tr).groupby(qids_tr).size().values
    grp_val = pd.Series(qids_val).groupby(qids_val).size().values
    
    dtrain_long = xgb.DMatrix(X_tr, label=y_long_tr)
    dtrain_long.set_group(grp_tr)
    dval_long = xgb.DMatrix(X_val, label=y_long_val)
    dval_long.set_group(grp_val)
    
    dtrain_short = xgb.DMatrix(X_tr, label=y_short_tr)
    dtrain_short.set_group(grp_tr)
    dval_short = xgb.DMatrix(X_val, label=y_short_val)
    dval_short.set_group(grp_val)
    
    print(f"  Training Long Model...")
    bst_long = xgb.train(
        params, dtrain_long, num_boost_round=1500,
        evals=[(dtrain_long, 'train'), (dval_long, 'val')],
        early_stopping_rounds=150,
        verbose_eval=False
    )
    print(f"    Best iteration: {bst_long.best_iteration}")
    
    print(f"  Training Short Model...")
    bst_short = xgb.train(
        params, dtrain_short, num_boost_round=1500,
        evals=[(dtrain_short, 'train'), (dval_short, 'val')],
        early_stopping_rounds=150,
        verbose_eval=False
    )
    print(f"    Best iteration: {bst_short.best_iteration}")
    
    dmatrix_te = xgb.DMatrix(X_te)
    long_preds = bst_long.predict(dmatrix_te)
    short_preds = bst_short.predict(dmatrix_te)
    
    metrics = evaluate_ranking_performance(df_te, long_preds, short_preds)
    metrics['fold'] = fold_idx
    metrics['long_best_iter'] = bst_long.best_iteration
    metrics['short_best_iter'] = bst_short.best_iteration
    walk_forward_results.append(metrics)
    
    print(f"  [FOLD SUMMARY]")
    print(f"    Best Iter  : Long={bst_long.best_iteration} | Short={bst_short.best_iteration}")
    print(f"    Long Rho   : {metrics['long_rho']:.4f} | Short Rho: {metrics['short_rho']:.4f}")
    print(f"    Long Win Rate @ 3 : {metrics['long_win_rates'][3]:.1%} | Short @ 3: {metrics['short_win_rates'][3]:.1%}")
    print(f"    Long Return Edge  : {metrics['long_edge']*100:+.4f}% | Short Edge: {metrics['short_edge']*100:+.4f}%")
    print(f"    Combined Edge     : {metrics['combined_edge']*100:+.4f}%")

# ========================================
# REPORT AGGREGATE WF RESULTS
# ========================================
avg_long_rho = np.mean([r['long_rho'] for r in walk_forward_results])
avg_short_rho = np.mean([r['short_rho'] for r in walk_forward_results])

avg_long_wr_3 = np.mean([r['long_win_rates'][3] for r in walk_forward_results])
avg_short_wr_3 = np.mean([r['short_win_rates'][3] for r in walk_forward_results])

avg_long_edge = np.mean([r['long_edge'] for r in walk_forward_results])
avg_short_edge = np.mean([r['short_edge'] for r in walk_forward_results])
avg_combined_edge = np.mean([r['combined_edge'] for r in walk_forward_results])

print("\n" + "=" * 60)
print("AGGREGATE WALK-FORWARD VALIDATION RESULTS (V3)")
print("=" * 60)
print(f"Averaged over {len(folds_config)} temporal test folds:")
print(f"  Average Spearman Rho:")
print(f"    Long Model  : {avg_long_rho:.4f}")
print(f"    Short Model : {avg_short_rho:.4f}")
print(f"  Average Win Rates @ 3:")
print(f"    Long Model  : {avg_long_wr_3:.1%}")
print(f"    Short Model : {avg_short_wr_3:.1%}")
print(f"  Average Top-3 Return Edges (over market):")
print(f"    Long Edge  : {avg_long_edge*100:+.4f}%")
print(f"    Short Edge : {avg_short_edge*100:+.4f}%")
print(f"    Combined Edge: {avg_combined_edge*100:+.4f}%")
print("=" * 60)

# ========================================
# TRAIN PRODUCTION MODEL ON ALL HISTORICAL DATA
# ========================================
print("\n" + "=" * 60)
print("TRAINING FINAL RECTIFIED PRODUCTION MODELS")
print("=" * 60)

scaler = StandardScaler(with_mean=False, with_std=False)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

prod_train_months = unique_months[:-2]
prod_val_months = unique_months[-2:]

print(f"  Production Train: {prod_train_months[0]} -> {prod_train_months[-1]}")
print(f"  Production Val:   {prod_val_months[0]} -> {prod_val_months[-1]}")

prod_tr_mask = df['YearMonth'].isin(prod_train_months).values
prod_val_mask = df['YearMonth'].isin(prod_val_months).values

X_prod_tr, y_prod_tr_raw, qids_prod_tr = X[prod_tr_mask], y_returns[prod_tr_mask], query_ids[prod_tr_mask]
X_prod_val, y_prod_val_raw, qids_prod_val = X[prod_val_mask], y_returns[prod_val_mask], query_ids[prod_val_mask]

y_long_prod_tr = get_integer_ranks(y_prod_tr_raw, qids_prod_tr, invert=False)
y_long_prod_val = get_integer_ranks(y_prod_val_raw, qids_prod_val, invert=False)
y_short_prod_tr = get_integer_ranks(y_prod_tr_raw, qids_prod_tr, invert=True)
y_short_prod_val = get_integer_ranks(y_prod_val_raw, qids_prod_val, invert=True)

grp_prod_tr = pd.Series(qids_prod_tr).groupby(qids_prod_tr).size().values
grp_prod_val = pd.Series(qids_prod_val).groupby(qids_prod_val).size().values

# LONG
dtrain_prod_long = xgb.DMatrix(X_prod_tr, label=y_long_prod_tr)
dtrain_prod_long.set_group(grp_prod_tr)
dval_prod_long = xgb.DMatrix(X_prod_val, label=y_long_prod_val)
dval_prod_long.set_group(grp_prod_val)

bst_prod_long = xgb.train(
    params, dtrain_prod_long, num_boost_round=1500,
    evals=[(dtrain_prod_long, 'train'), (dval_prod_long, 'val')],
    early_stopping_rounds=150,
    verbose_eval=100
)
bst_prod_long.save_model(LONG_MODEL_PATH)

# SHORT
dtrain_prod_short = xgb.DMatrix(X_prod_tr, label=y_short_prod_tr)
dtrain_prod_short.set_group(grp_prod_tr)
dval_prod_short = xgb.DMatrix(X_prod_val, label=y_short_prod_val)
dval_prod_short.set_group(grp_prod_val)

bst_prod_short = xgb.train(
    params, dtrain_prod_short, num_boost_round=1500,
    evals=[(dtrain_prod_short, 'train'), (dval_prod_short, 'val')],
    early_stopping_rounds=150,
    verbose_eval=100
)
bst_prod_short.save_model(SHORT_MODEL_PATH)

def extract_importance(bst):
    try:
        importance = bst.get_score(importance_type='gain')
        imp_named = {}
        for k, v in importance.items():
            idx = int(k.replace('f', ''))
            if idx < len(feature_cols):
                imp_named[feature_cols[idx]] = float(v)
        return dict(sorted(imp_named.items(), key=lambda x: -x[1])[:20])
    except Exception:
        return {}

metadata = {
    'features': feature_cols,
    'num_features': len(feature_cols),
    'data_source': 'daily_macro_v3',
    'data_file': DATA_FILE,
    'total_rows': int(df.shape[0]),
    'walk_forward_summary': {
        'avg_long_spearman': float(avg_long_rho),
        'avg_short_spearman': float(avg_short_rho),
        'avg_long_win_rate_k3': float(avg_long_wr_3),
        'avg_short_win_rate_k3': float(avg_short_wr_3),
        'avg_long_return_edge_k3': float(avg_long_edge),
        'avg_short_return_edge_k3': float(avg_short_edge),
        'avg_combined_edge_k3': float(avg_combined_edge),
    },
    'walk_forward_folds': [
        {
            'fold': int(r['fold']),
            'long_rho': float(r['long_rho']),
            'short_rho': float(r['short_rho']),
            'long_win_rates': {str(k): float(v) for k, v in r['long_win_rates'].items()},
            'short_win_rates': {str(k): float(v) for k, v in r['short_win_rates'].items()},
            'long_edge': float(r['long_edge']),
            'short_edge': float(r['short_edge']),
            'combined_edge': float(r['combined_edge']),
            'long_best_iter': int(r['long_best_iter']),
            'short_best_iter': int(r['short_best_iter']),
        }
        for r in walk_forward_results
    ],
    'production_training': {
        'train_months': prod_train_months,
        'val_months': prod_val_months,
        'long_best_iteration': int(bst_prod_long.best_iteration),
        'short_best_iteration': int(bst_prod_short.best_iteration),
    },
    'top_features_long': extract_importance(bst_prod_long),
    'top_features_short': extract_importance(bst_prod_short),
    'params': params,
    'trained_at': datetime.now().isoformat(),
}

with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("V3 DAILY MACRO XGBOOST TRAINING COMPLETE")
print(f"  Long Model   : {LONG_MODEL_PATH}")
print(f"  Short Model  : {SHORT_MODEL_PATH}")
print(f"  Metadata     : {META_PATH}")
print(f"  Scaler       : {SCALER_PATH}")
print("=" * 60)
