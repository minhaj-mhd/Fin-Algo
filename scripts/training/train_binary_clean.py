import os, sys, pickle, json, argparse
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
sys.path.append(os.getcwd())

CFG = {
    '1h_roll_v24': dict(
        data='data/research/v20_rolling_1h/panel.parquet',
        ret_col='Next_Hour_Return',
        model_dir='models/research/v24_binary_1h',
        desc='RESEARCH v24: Binary classification using binary:logistic to output absolute probabilities. Label is Next_Hour_Return > 0 for Longs, < 0 for Shorts.',
        params=dict(max_depth=5, min_child_weight=10),
    ),
    '1h_roll_v24_top20': dict(
        data='data/research/v20_rolling_1h/panel.parquet',
        ret_col='Next_Hour_Return',
        model_dir='models/research/v24_binary_1h_top20',
        desc='RESEARCH v24: Binary classification (Top 20 features) using binary:logistic.',
        params=dict(max_depth=5, min_child_weight=10),
        selected_features_long=[
            'Relative_Return', 'Return', 'Is_Close_Hour', 'Log_Return', 'Time_To_Close', 
            'Hour', 'Market_Mean_Volatility', 'Down_Streak', 'Market_Mean_Return', 
            'DayOfWeek', 'IBS', 'Is_Open_Hour', 'Lower_Shadow', 'Up_Streak', 
            'CMF_20', 'Vortex_Plus', 'OBV_Dist', 'Dist_52W_High', 'Dist_BB_Lower', 'Dollar_Volume'
        ],
        selected_features_short=[
            'Log_Return', 'Relative_Return', 'Return', 'Time_To_Close', 'Is_Close_Hour', 
            'Hour', 'Market_Mean_Volatility', 'IBS', 'DayOfWeek', 'Market_Mean_Return', 
            'Down_Streak', 'Up_Streak', 'CMF_20', 'OBV_Dist', 'Lower_Shadow', 
            'Is_Open_Hour', 'Dollar_Volume', 'Rolling_Skew', 'Price_Zscore', 'Vortex_Plus'
        ]
    ),
    '1h_roll_v25_fat': dict(
        data='data/research/v20_rolling_1h/panel.parquet',
        ret_col='Next_Hour_Return',
        model_dir='models/research/v25_fat_tail_1h',
        desc='RESEARCH v25: Fat-tail binary classification (label > 25 bps).',
        params=dict(max_depth=5, min_child_weight=10),
        label_thresh=0.0025,
    )
}

ap = argparse.ArgumentParser()
ap.add_argument('--tf', required=True, choices=list(CFG.keys()))
ap.add_argument('--regime', default='all', choices=['all', 'bull', 'bear', 'chop'], help='Train only on specific Nifty 500 100-DMA regime')
args = ap.parse_args()
c = CFG[args.tf]

if args.regime != 'all':
    c['model_dir'] = c['model_dir'] + f'_{args.regime}_100dma'

DATA_FILE, RET_COL, MODEL_DIR = c['data'], c['ret_col'], c['model_dir']
os.makedirs(MODEL_DIR, exist_ok=True)
LONG_MODEL_PATH  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_MODEL_PATH = f'{MODEL_DIR}/xgb_short_model.json'
META_PATH        = f'{MODEL_DIR}/metadata.json'
SCALER_PATH      = f'{MODEL_DIR}/scaler.pkl'

print("=" * 64)
print(f"BINARY CLASSIFICATION TRAINING ({args.tf}) — Walk-Forward + Early Stopping")
print("=" * 64)
print(f"Loading {DATA_FILE} ...")
df = pd.read_parquet(DATA_FILE) if DATA_FILE.endswith('.parquet') else pd.read_csv(DATA_FILE)
print(f"Loaded {df.shape[0]:,} rows")

if args.regime != 'all':
    print(f"Applying regime filter: {args.regime.upper()} (Nifty 500 100-DMA)")
    nifty = pd.read_csv('data/raw_index_cache/nifty500_1d.csv')
    nifty['timestamp'] = pd.to_datetime(nifty['timestamp'])
    nifty['date'] = nifty['timestamp'].dt.date
    nifty = nifty.sort_values('date')
    nifty['nifty_100dma'] = nifty['close'].rolling(window=100).mean()
    
    # E0.1 Fix: Shift by 1 day to prevent lookahead bias
    nifty['prev_close'] = nifty['close'].shift(1)
    nifty['prev_100dma'] = nifty['nifty_100dma'].shift(1)
    nifty['routing_timestamp'] = nifty['timestamp'].shift(1)
    
    df['date'] = pd.to_datetime(df['DateTime']).dt.date
    nifty_subset = nifty[['date', 'routing_timestamp', 'prev_close', 'prev_100dma']].rename(
        columns={'prev_close': 'nifty_close', 'prev_100dma': 'nifty_100dma'}
    )
    df = df.merge(nifty_subset, on='date', how='left')
    
    # E0.1 Audit Assertion: Ensure routing data is from BEFORE the current trade's time
    routing_ts_naive = pd.to_datetime(df['routing_timestamp']).dt.tz_localize(None)
    datetime_naive = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
    
    # Drop NaNs before assertion
    valid_mask = routing_ts_naive.notna() & datetime_naive.notna()
    
    assert (routing_ts_naive[valid_mask] <= datetime_naive[valid_mask]).all(), "Lookahead bias detected in routing key!"
    
    df = df.dropna(subset=['nifty_100dma'])
    
    if args.regime == 'bull':
        # E0.2 Fix: Exclude buffer zone during training
        df = df[(df['nifty_close'] > df['nifty_100dma']) & (np.abs((df['nifty_close'] - df['nifty_100dma']) / df['nifty_100dma']) >= 0.015)].copy()
    elif args.regime == 'bear':
        # E0.2 Fix: Exclude buffer zone during training
        df = df[(df['nifty_close'] <= df['nifty_100dma']) & (np.abs((df['nifty_close'] - df['nifty_100dma']) / df['nifty_100dma']) >= 0.015)].copy()
    elif args.regime == 'chop':
        df = df[np.abs((df['nifty_close'] - df['nifty_100dma']) / df['nifty_100dma']) < 0.015].copy()
        
    print(f"Post-regime filter rows: {df.shape[0]:,}")
    df.drop(columns=['date', 'routing_timestamp', 'nifty_close', 'nifty_100dma'], inplace=True, errors='ignore')

df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
unique_months = sorted(df['YearMonth'].unique())
print(f"Spans {len(unique_months)} months: {unique_months[0]} -> {unique_months[-1]}")

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']

if 'selected_features_long' in c and 'selected_features_short' in c:
    feature_cols = list(dict.fromkeys(c['selected_features_long'] + c['selected_features_short']))
    feature_cols = [col for col in feature_cols if col in df.columns]
elif 'selected_features' in c:
    feature_cols = [col for col in c['selected_features'] if col in df.columns]
else:
    feature_cols = [col for col in df.columns if col not in exclude_cols]

has_split_feats = 'selected_features_long' in c and 'selected_features_short' in c
if has_split_feats:
    idx_long = [feature_cols.index(f) for f in c['selected_features_long'] if f in feature_cols]
    idx_short = [feature_cols.index(f) for f in c['selected_features_short'] if f in feature_cols]
else:
    idx_long = list(range(len(feature_cols)))
    idx_short = list(range(len(feature_cols)))

print(f"Features: {len(feature_cols)} | Samples: {df.shape[0]:,}")

X = df[feature_cols].values.astype(np.float64)
y_returns = df[RET_COL].values

nan_mask = np.isnan(X) | np.isinf(X)
if nan_mask.any():
    print(f"Replacing {int(nan_mask.sum())} NaN/Inf values...")
    for ci in range(X.shape[1]):
        col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
        if bad.any():
            good = col[~bad]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

# GPU detection
device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.random.randint(2, size=10))
    xgb.train({'objective': 'binary:logistic', 'device': 'cuda', 'tree_method': 'hist'}, d, num_boost_round=1)
    device = 'cuda'; print("  CUDA GPU detected.")
except Exception:
    print("  CPU training.")

params = {
    'objective': 'binary:logistic', 'eta': 0.03,
    'max_depth': c['params']['max_depth'], 'subsample': 0.8, 'colsample_bytree': 0.8,
    'alpha': 1.0, 'lambda': 2.0, 'min_child_weight': c['params']['min_child_weight'],
    'random_state': 42, 'verbosity': 0, 'eval_metric': 'logloss',
    'tree_method': 'hist', 'device': device,
}

def evaluate_thresholds(y_true_returns, long_probs, short_probs, threshold=0.55):
    # Long Edge
    long_mask = long_probs > threshold
    long_count = long_mask.sum()
    if long_count > 0:
        long_returns = y_true_returns[long_mask]
        long_wr = (long_returns > 0).mean()
        long_edge = long_returns.mean()
    else:
        long_wr = long_edge = 0.0

    # Short Edge
    short_mask = short_probs > threshold
    short_count = short_mask.sum()
    if short_count > 0:
        short_returns = -y_true_returns[short_mask] # Invert returns for shorts
        short_wr = (short_returns > 0).mean()
        short_edge = short_returns.mean()
    else:
        short_wr = short_edge = 0.0
        
    return long_count, long_wr, long_edge, short_count, short_wr, short_edge

# walk-forward folds
min_train_months, horizon = 18, 2
folds = []
for i in range(min_train_months, len(unique_months) - horizon, 4):
    folds.append(dict(fold=len(folds)+1, train=unique_months[:i],
                      val=[unique_months[i]], test=unique_months[i+1:i+horizon+1]))
print(f"\nWalk-forward folds: {len(folds)}")

wf_results = {thresh: {'l_edge': [], 's_edge': [], 'l_count': [], 's_count': []} for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]}

for cfg in folds:
    tr_m, val_m, te_m = cfg['train'], cfg['val'], cfg['test']
    print(f"\n--- FOLD {cfg['fold']} --- train {tr_m[0]}->{tr_m[-1]} ({len(tr_m)}m) | val {val_m[0]} | test {te_m[0]}->{te_m[-1]}")
    trm = df['YearMonth'].isin(tr_m).values
    vam = df['YearMonth'].isin(val_m).values
    tem = df['YearMonth'].isin(te_m).values
    
    Xtr, ytr = X[trm], y_returns[trm]
    Xva, yva = X[vam], y_returns[vam]
    Xte, yte = X[tem], y_returns[tem]

    label_thresh = c.get('label_thresh', 0.0)

    # Labels for binary classifier
    ytr_long = (ytr > label_thresh).astype(int)
    yva_long = (yva > label_thresh).astype(int)
    
    ytr_short = (ytr < -label_thresh).astype(int)
    yva_short = (yva < -label_thresh).astype(int)

    p_long = params.copy()
    p_long['scale_pos_weight'] = (len(ytr_long) - ytr_long.sum()) / max(1, ytr_long.sum())

    p_short = params.copy()
    p_short['scale_pos_weight'] = (len(ytr_short) - ytr_short.sum()) / max(1, ytr_short.sum())

    # long
    dtl = xgb.DMatrix(Xtr[:, idx_long], label=ytr_long)
    dvl = xgb.DMatrix(Xva[:, idx_long], label=yva_long)
    bl = xgb.train(p_long, dtl, num_boost_round=500, evals=[(dvl, 'val')], early_stopping_rounds=50, verbose_eval=False)
    
    # short
    dts = xgb.DMatrix(Xtr[:, idx_short], label=ytr_short)
    dvs = xgb.DMatrix(Xva[:, idx_short], label=yva_short)
    bs = xgb.train(p_short, dts, num_boost_round=500, evals=[(dvs, 'val')], early_stopping_rounds=50, verbose_eval=False)
    
    dte_long = xgb.DMatrix(Xte[:, idx_long])
    dte_short = xgb.DMatrix(Xte[:, idx_short])
    
    l_probs = bl.predict(dte_long)
    s_probs = bs.predict(dte_short)
    
    for thresh in wf_results.keys():
        lc, lwr, ledge, sc, swr, sedge = evaluate_thresholds(yte, l_probs, s_probs, thresh)
        wf_results[thresh]['l_edge'].append(ledge)
        wf_results[thresh]['s_edge'].append(sedge)
        wf_results[thresh]['l_count'].append(lc)
        wf_results[thresh]['s_count'].append(sc)
        
    print(f"    >0.55 | L: {wf_results[0.55]['l_count'][-1]} trades ({wf_results[0.55]['l_edge'][-1]*10000:+.1f} bps) | S: {wf_results[0.55]['s_count'][-1]} trades ({wf_results[0.55]['s_edge'][-1]*10000:+.1f} bps)")
    print(f"    >0.60 | L: {wf_results[0.6]['l_count'][-1]} trades ({wf_results[0.6]['l_edge'][-1]*10000:+.1f} bps) | S: {wf_results[0.6]['s_count'][-1]} trades ({wf_results[0.6]['s_edge'][-1]*10000:+.1f} bps)")

print("\n" + "=" * 64)
print("AGGREGATE THRESHOLD PERFORMANCE (Average across folds)")
for thresh in sorted(wf_results.keys()):
    l_edge = np.mean(wf_results[thresh]['l_edge']) * 10000
    s_edge = np.mean(wf_results[thresh]['s_edge']) * 10000
    l_c = np.mean(wf_results[thresh]['l_count'])
    s_c = np.mean(wf_results[thresh]['s_count'])
    print(f"Threshold >{thresh:.2f} | Long Edge: {l_edge:+.2f} bps ({l_c:.0f} trades) | Short Edge: {s_edge:+.2f} bps ({s_c:.0f} trades)")

# Production models
print("\nTraining production models (Strict 80% Train/Val, 20% untouched Test)...")
split_idx = int(len(unique_months) * 0.8)
ptr = df['YearMonth'].isin(unique_months[:split_idx-1]).values
pva = df['YearMonth'].isin([unique_months[split_idx-1]]).values
Xptr, yptr = X[ptr], y_returns[ptr]
Xpva, ypva = X[pva], y_returns[pva]

label_thresh = c.get('label_thresh', 0.0)
yptr_long = (yptr > label_thresh).astype(int)
ypva_long = (ypva > label_thresh).astype(int)
yptr_short = (yptr < -label_thresh).astype(int)
ypva_short = (ypva < -label_thresh).astype(int)

p_long = params.copy()
p_long['scale_pos_weight'] = (len(yptr_long) - yptr_long.sum()) / max(1, yptr_long.sum())
p_short = params.copy()
p_short['scale_pos_weight'] = (len(yptr_short) - yptr_short.sum()) / max(1, yptr_short.sum())

dptl = xgb.DMatrix(Xptr[:, idx_long], label=yptr_long)
dpvl = xgb.DMatrix(Xpva[:, idx_long], label=ypva_long)
prod_long = xgb.train(p_long, dptl, num_boost_round=500, evals=[(dpvl, 'val')], early_stopping_rounds=50, verbose_eval=50)
prod_long.save_model(LONG_MODEL_PATH)

dpts = xgb.DMatrix(Xptr[:, idx_short], label=yptr_short)
dpvs = xgb.DMatrix(Xpva[:, idx_short], label=ypva_short)
prod_short = xgb.train(p_short, dpts, num_boost_round=500, evals=[(dpvs, 'val')], early_stopping_rounds=50, verbose_eval=50)
prod_short.save_model(SHORT_MODEL_PATH)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(StandardScaler(with_mean=False, with_std=False), f)

def imp(bst):
    try:
        s = bst.get_score(importance_type='gain'); out = {}
        for k, v in s.items():
            i = int(k.replace('f', ''))
            if i < len(feature_cols): out[feature_cols[i]] = float(v)
        return dict(sorted(out.items(), key=lambda x: -x[1])[:20])
    except Exception:
        return {}

metadata = {
    'description': c['desc'], 'type': 'binary_classifier', 
    'features': feature_cols,
    'features_long': c.get('selected_features_long', feature_cols),
    'features_short': c.get('selected_features_short', feature_cols),
    'num_features': len(feature_cols), 'data_source': f'upstox_{args.tf}_clean',
    'data_file': DATA_FILE, 'total_rows': int(df.shape[0]),
    'long_model': LONG_MODEL_PATH, 'short_model': SHORT_MODEL_PATH, 'meta': META_PATH, 'scaler': SCALER_PATH,
    'top_features_long': imp(prod_long), 'top_features_short': imp(prod_short),
    'params': params, 'trained_at': datetime.now().isoformat(),
}
with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 64)
print(f"DONE — {args.tf} binary models saved to {MODEL_DIR}")
print("=" * 64)
