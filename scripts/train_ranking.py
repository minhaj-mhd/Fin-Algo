import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import pickle
import json

print("Loading data...")
df = pd.read_csv('data/ranking_data_full.csv')
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

# Feature Engineering: Market Context is now handled in prepare_ranking_data.py
# We just need to define the feature list
exclude_cols = ['DateTime', 'DateTime_Hour', 'Query_ID', 'Ticker', 
                'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Return']
feature_cols = [col for col in df.columns if col not in exclude_cols]


X = df[feature_cols].values
y_returns = df['Next_Hour_Return'].values
query_ids = df['Query_ID'].values

print(f"Features: {len(feature_cols)}")
print(f"Samples: {X.shape[0]}")

# Handle NaN/Inf
nan_mask = np.isnan(X) | np.isinf(X)
if nan_mask.any():
    print(f"Replacing {nan_mask.sum()} NaN/Inf values")
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        valid_data = col_data[~(np.isnan(col_data) | np.isinf(col_data))]
        if len(valid_data) > 0:
            X[np.isnan(col_data) | np.isinf(col_data), col_idx] = np.mean(valid_data)

# Labels are now raw returns (handled in the execution section)
y = y_returns

# Train-test split by query
# Strict Temporal Split (First 80% train, Last 20% test)
unique_query_ids = np.sort(df['Query_ID'].unique())
split_idx = int(len(unique_query_ids) * 0.8)

train_qids = unique_query_ids[:split_idx]
test_qids = unique_query_ids[split_idx:]

train_mask = df['Query_ID'].isin(train_qids)
test_mask = df['Query_ID'].isin(test_qids)

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]
query_ids_train = query_ids[train_mask]
query_ids_test = query_ids[test_mask]

print(f"Train: {X_train.shape[0]} rows, {len(np.unique(query_ids_train))} queries")
print(f"Test: {X_test.shape[0]} rows, {len(np.unique(query_ids_test))} queries")

# Normalize
print("Features are pre-normalized (Cross-Sectional Z-Score).")
# Normalize
print("Features are pre-normalized (Cross-Sectional Z-Score).")


# Create DMatrix
group_sizes_train = df[train_mask].groupby('Query_ID').size().values
group_sizes_test = df[test_mask].groupby('Query_ID').size().values

dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_group(group_sizes_train)

dtest = xgb.DMatrix(X_test, label=y_test)
dtest.set_group(group_sizes_test)

def train_and_save_model(X_train, y_train, group_sizes_train, X_test, y_test, group_sizes_test, model_path, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_sizes_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    dtest.set_group(group_sizes_test)
    
    print(f"Training XGBoost model for {model_path}...")
    bst = xgb.train(params, dtrain, num_boost_round=300, 
                    evals=[(dtrain, 'train'), (dtest, 'test')], 
                    early_stopping_rounds=30,
                    verbose_eval=False)
    
    bst.save_model(model_path)
    return bst

def calculate_relevance(y_returns, query_ids):
    y = np.zeros_like(y_returns, dtype=int)
    from scipy.stats import rankdata
    for qid in np.unique(query_ids):
        mask = query_ids == qid
        vals = y_returns[mask]
        if len(vals) == 0: continue
        dense_ranks = rankdata(vals, method='dense')
        zero_based = dense_ranks - 1
        max_rank = zero_based.max() if zero_based.size > 0 else 0
        if max_rank <= 0:
            y[mask] = 0
        else:
            y[mask] = (zero_based * 31 // max_rank).astype(int)
    return y

def evaluate_and_correlate(bst, X_train, X_test, y_returns_train, y_returns_test, query_ids_train, query_ids_test, group_sizes_train, group_sizes_test):
    dtrain = xgb.DMatrix(X_train)
    dtrain.set_group(group_sizes_train)
    dtest = xgb.DMatrix(X_test)
    dtest.set_group(group_sizes_test)
    
    y_pred_train = bst.predict(dtrain)
    y_pred_test = bst.predict(dtest)
    
    train_corrs = []
    test_corrs = []
    
    for qid in np.unique(query_ids_train):
        mask = query_ids_train == qid
        if mask.sum() > 1:
            corr, _ = spearmanr(y_pred_train[mask], y_returns_train[mask])
            if not np.isnan(corr): train_corrs.append(corr)
            
    for qid in np.unique(query_ids_test):
        mask = query_ids_test == qid
        if mask.sum() > 1:
            corr, _ = spearmanr(y_pred_test[mask], y_returns_test[mask])
            if not np.isnan(corr): test_corrs.append(corr)
            
    return np.mean(train_corrs), np.mean(test_corrs)

# --- Main Execution ---

# Use unique integer ranks for pairwise ranking (satisfies label_is_integer check)
from scipy.stats import rankdata

def get_integer_ranks(y_vals, qids, invert=False):
    y_int = np.zeros_like(y_vals, dtype=int)
    for qid in np.unique(qids):
        mask = qids == qid
        if mask.sum() == 0: continue
        vals = -y_vals[mask] if invert else y_vals[mask]
        # method='ordinal' gives unique integers 1..N
        y_int[mask] = rankdata(vals, method='ordinal') - 1
    return y_int

print("Creating unique integer ranks...")
y_long = get_integer_ranks(y_returns, query_ids, invert=False)
y_short = get_integer_ranks(y_returns, query_ids, invert=True)

# Train-test split (using same masks for consistency)
y_train_long = y_long[train_mask]
y_test_long = y_long[test_mask]

y_train_short = y_short[train_mask]
y_test_short = y_short[test_mask]

params = {
    'objective': 'rank:pairwise',
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 1.0,
    'lambda': 2.0,
    'min_child_weight': 5,
    'random_state': 42,
    'verbosity': 0,
    'eval_metric': 'ndcg@1',
    'ndcg_exp_gain': False,
    'tree_method': 'hist',
    'device': 'cuda'
}

# Train Long Model
bst_long = train_and_save_model(X_train, y_train_long, group_sizes_train, 
                                X_test, y_test_long, group_sizes_test, 
                                'models/xgb_long_model.json', params)

# Train Short Model
bst_short = train_and_save_model(X_train, y_train_short, group_sizes_train, 
                                 X_test, y_test_short, group_sizes_test, 
                                 'models/xgb_short_model.json', params)

# Evaluate Long
train_rho_long, test_rho_long = evaluate_and_correlate(bst_long, X_train, X_test, 
                                                       y_returns[train_mask], y_returns[test_mask], 
                                                       query_ids_train, query_ids_test, 
                                                       group_sizes_train, group_sizes_test)

# Evaluate Short (Evaluate against -Next_Hour_Return because that's what it tries to rank)
train_rho_short, test_rho_short = evaluate_and_correlate(bst_short, X_train, X_test, 
                                                         -y_returns[train_mask], -y_returns[test_mask], 
                                                         query_ids_train, query_ids_test, 
                                                         group_sizes_train, group_sizes_test)

print("\n" + "="*30)
print("LONG MODEL PERFORMANCE")
print(f"Spearman (train): {train_rho_long:.4f}")
print(f"Spearman (test):  {test_rho_long:.4f}")
print("="*30)

print("\n" + "="*30)
print("SHORT MODEL PERFORMANCE")
print(f"Spearman (train): {train_rho_short:.4f}")
print(f"Spearman (test):  {test_rho_short:.4f}")
print("="*30)

# Save No-Op Scaler and Metadata
# We save a no-op scaler to maintain compatibility with vanguard_signal_engine.py
scaler = StandardScaler(with_mean=False, with_std=False)
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

metadata = {
    'features': feature_cols,
    'num_features': len(feature_cols),
    'train_queries': int(len(np.unique(query_ids_train))),
    'test_queries': int(len(np.unique(query_ids_test))),
    'long_test_spearman': float(test_rho_long),
    'short_test_spearman': float(test_rho_short),
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nTRAINING COMPLETE!")
print(f"Files saved: xgb_long_model.json, xgb_short_model.json, scaler.pkl, model_metadata.json")
