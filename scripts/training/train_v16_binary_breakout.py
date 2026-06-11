"""
v16 - Binary Breakout Classifier.

This model discards the relative ranking approach (LambdaMART) and instead 
frames the problem as an absolute binary classification:
Target: Will the stock move > 20 bps in the next hour? (1 or 0)

Objective: binary:logistic
Evaluation Metric: auc
Trading Threshold: > 85% probability

Usage:
    python scripts/training/train_v16_binary_breakout.py

Outputs:
    models/v16_binary_breakout_1h/
"""

import os, pickle, json
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ── config ─────────────────────────────────────────────────────────────────────
DATA_FILE   = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL     = 'Next_Hour_Return'
MODEL_DIR   = 'models/v16_binary_breakout_1h'
COST        = 0.0010
BREAKOUT_TH = 0.0020  # 20 bps target
TRADE_PROB  = 0.62    # 62% probability threshold

os.makedirs(MODEL_DIR, exist_ok=True)
LONG_MODEL  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_MODEL = f'{MODEL_DIR}/xgb_short_model.json'
META_PATH   = f'{MODEL_DIR}/metadata.json'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'

# ── data ───────────────────────────────────────────────────────────────────────
print("=" * 64)
print(f"v16 BINARY BREAKOUT CLASSIFIER (>{int(BREAKOUT_TH*10000)} bps target)")
print("=" * 64)

print(f"Loading {DATA_FILE} ...")
df = pd.read_csv(DATA_FILE)
print(f"  {df.shape[0]:,} rows")

df['YearMonth'] = df['DateTime'].str[:7]
unique_months   = sorted(df['YearMonth'].unique())
print(f"  {len(unique_months)} months: {unique_months[0]} -> {unique_months[-1]}")

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"  Features: {len(feature_cols)}")

X         = df[feature_cols].values.astype(np.float64)
y_returns = df[RET_COL].values.astype(np.float64)

nan_mask = ~np.isfinite(X)
if nan_mask.any():
    for ci in range(X.shape[1]):
        bad = ~np.isfinite(X[:, ci])
        if bad.any():
            good = X[~bad, ci]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

# Create binary targets
print("  Creating binary breakout targets...")
y_long_binary  = (y_returns >= BREAKOUT_TH).astype(np.int32)
y_short_binary = (y_returns <= -BREAKOUT_TH).astype(np.int32)

print(f"  Long target (>20bps) class balance:  {y_long_binary.mean():.2%}")
print(f"  Short target (<-20bps) class balance: {y_short_binary.mean():.2%}")

# Scale Pos Weight to handle extreme class imbalance
pos_weight_long  = (1.0 - y_long_binary.mean()) / y_long_binary.mean()
pos_weight_short = (1.0 - y_short_binary.mean()) / y_short_binary.mean()

# ── XGBoost params ─────────────────────────────────────────────────────────────
device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.zeros(10, dtype=np.int32))
    xgb.train({'objective': 'binary:logistic', 'device': 'cuda', 'tree_method': 'hist'},
              d, num_boost_round=1)
    device = 'cuda'
    print("  CUDA GPU detected.")
except Exception:
    print("  CPU training.")

def get_params(pos_weight: float) -> dict:
    return {
        'objective':         'binary:logistic',
        'eval_metric':       'auc',
        'scale_pos_weight':  pos_weight, # balance the dataset
        'eta':               0.03,
        'max_depth':         4,          # Previously optimized to 4
        'subsample':         0.8,
        'colsample_bytree':  0.8,
        'alpha':             1.0,
        'lambda':            2.0,
        'min_child_weight':  10,
        'random_state':      42,
        'verbosity':         0,
        'tree_method':       'hist',
        'device':            device,
    }

# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate_fold(df_eval: pd.DataFrame, long_probs: np.ndarray, short_probs: np.ndarray):
    # Long trades
    long_mask = long_probs > TRADE_PROB
    n_long = long_mask.sum()
    if n_long > 0:
        long_returns = df_eval[RET_COL].values[long_mask]
        l_raw_ret = float(np.mean(long_returns))
        l_net = l_raw_ret - COST
        l_hit = float(np.mean(long_returns > COST))
        l_raw_hit = float(np.mean(long_returns > 0))
    else:
        l_raw_ret, l_net, l_hit, l_raw_hit = 0.0, 0.0, 0.0, 0.0

    # Short trades
    short_mask = short_probs > TRADE_PROB
    n_short = short_mask.sum()
    if n_short > 0:
        short_returns = -df_eval[RET_COL].values[short_mask] # Invert returns for shorts
        s_raw_ret = float(np.mean(short_returns))
        s_net = s_raw_ret - COST
        s_hit = float(np.mean(short_returns > COST))
        s_raw_hit = float(np.mean(short_returns > 0))
    else:
        s_raw_ret, s_net, s_hit, s_raw_hit = 0.0, 0.0, 0.0, 0.0

    # Calculate AUC strictly for monitoring predictive power
    try:
        l_auc = roc_auc_score(df_eval['long_target'].values, long_probs)
    except: l_auc = 0.5
    try:
        s_auc = roc_auc_score(df_eval['short_target'].values, short_probs)
    except: s_auc = 0.5

    return dict(
        long_net=l_net, long_raw=l_raw_ret, long_hit=l_hit, long_raw_hit=l_raw_hit, n_long=n_long, long_auc=l_auc,
        short_net=s_net, short_raw=s_raw_ret, short_hit=s_hit, short_raw_hit=s_raw_hit, n_short=n_short, short_auc=s_auc
    )

# ── walk-forward ───────────────────────────────────────────────────────────────
min_train_months, val_months, horizon, step = 18, 4, 2, 4
folds = []
for i in range(min_train_months, len(unique_months) - val_months - horizon, step):
    folds.append(dict(
        fold=len(folds) + 1,
        train=unique_months[:i],
        val=unique_months[i:i + val_months],
        test=unique_months[i + val_months:i + val_months + horizon],
    ))

print(f"\nWalk-forward folds: {len(folds)}")
print(f"Headline: Net return on trades > {TRADE_PROB*100}% probability (cost=10bps)\n")

wf = []
for cfg in folds:
    tr_m, val_m, te_m = cfg['train'], cfg['val'], cfg['test']
    print(f"--- FOLD {cfg['fold']} --- train {tr_m[0]}->{tr_m[-1]} ({len(tr_m)}m) | val {val_m[0]} | test {te_m[0]}->{te_m[-1]}")

    trm = df['YearMonth'].isin(tr_m).values
    vam = df['YearMonth'].isin(val_m).values
    tem = df['YearMonth'].isin(te_m).values

    Xtr, Xva, Xte = X[trm], X[vam], X[tem]
    dfte = df[tem].copy()
    dfte['long_target'] = y_long_binary[tem]
    dfte['short_target'] = y_short_binary[tem]

    # Train Longs
    dtr_l = xgb.DMatrix(Xtr, label=y_long_binary[trm])
    dva_l = xgb.DMatrix(Xva, label=y_long_binary[vam])
    bl = xgb.train(get_params(pos_weight_long), dtr_l, num_boost_round=500,
                   evals=[(dva_l, 'val')], early_stopping_rounds=50, verbose_eval=False)
                   
    # Train Shorts
    dtr_s = xgb.DMatrix(Xtr, label=y_short_binary[trm])
    dva_s = xgb.DMatrix(Xva, label=y_short_binary[vam])
    bs = xgb.train(get_params(pos_weight_short), dtr_s, num_boost_round=500,
                   evals=[(dva_s, 'val')], early_stopping_rounds=50, verbose_eval=False)

    dte = xgb.DMatrix(Xte)
    l_preds = bl.predict(dte)
    s_preds = bs.predict(dte)
    m   = evaluate_fold(dfte, l_preds, s_preds)
    m['fold'] = cfg['fold']
    wf.append(m)

    print(f"    [Max Prob: Long={l_preds.max():.1%}, Short={s_preds.max():.1%}]")
    print(f"    Longs : {m['n_long']:>4} trades | raw ret {m['long_raw']*10000:>+5.1f}bps | net ret {m['long_net']*10000:>+5.1f}bps | raw win {m['long_raw_hit']:.1%} | net hit {m['long_hit']:.1%}")
    print(f"    Shorts: {m['n_short']:>4} trades | raw ret {m['short_raw']*10000:>+5.1f}bps | net ret {m['short_net']*10000:>+5.1f}bps | raw win {m['short_raw_hit']:.1%} | net hit {m['short_hit']:.1%}")

# ── aggregate ──────────────────────────────────────────────────────────────────
total_long_trades = sum(r['n_long'] for r in wf)
total_short_trades = sum(r['n_short'] for r in wf)

avg_long_net = np.average([r['long_net'] for r in wf if r['n_long'] > 0], 
                          weights=[r['n_long'] for r in wf if r['n_long'] > 0]) if total_long_trades > 0 else 0.0

avg_short_net = np.average([r['short_net'] for r in wf if r['n_short'] > 0], 
                           weights=[r['n_short'] for r in wf if r['n_short'] > 0]) if total_short_trades > 0 else 0.0

avg_long_raw = np.average([r['long_raw'] for r in wf if r['n_long'] > 0], weights=[r['n_long'] for r in wf if r['n_long'] > 0]) if total_long_trades > 0 else 0.0
avg_short_raw = np.average([r['short_raw'] for r in wf if r['n_short'] > 0], weights=[r['n_short'] for r in wf if r['n_short'] > 0]) if total_short_trades > 0 else 0.0
avg_long_hit = np.average([r['long_hit'] for r in wf if r['n_long'] > 0], weights=[r['n_long'] for r in wf if r['n_long'] > 0]) if total_long_trades > 0 else 0.0
avg_short_hit = np.average([r['short_hit'] for r in wf if r['n_short'] > 0], weights=[r['n_short'] for r in wf if r['n_short'] > 0]) if total_short_trades > 0 else 0.0
avg_long_raw_hit = np.average([r['long_raw_hit'] for r in wf if r['n_long'] > 0], weights=[r['n_long'] for r in wf if r['n_long'] > 0]) if total_long_trades > 0 else 0.0
avg_short_raw_hit = np.average([r['short_raw_hit'] for r in wf if r['n_short'] > 0], weights=[r['n_short'] for r in wf if r['n_short'] > 0]) if total_short_trades > 0 else 0.0

print("\n" + "=" * 64)
print("WALK-FORWARD AGGREGATE (Weighted by Trade Count)")
print(f"  Total Long Trades  : {total_long_trades}")
print(f"  Total Short Trades : {total_short_trades}")
print(f"  Avg Long Raw Return: {avg_long_raw*10000:+.2f} bps")
print(f"  Avg Long Net Return: {avg_long_net*10000:+.2f} bps")
print(f"  Avg Long Raw Winrate: {avg_long_raw_hit:.2%}")
print(f"  Avg Long Net Hitrate: {avg_long_hit:.2%}")
print(f"  Avg Short Raw Return: {avg_short_raw*10000:+.2f} bps")
print(f"  Avg Short Net Return:{avg_short_net*10000:+.2f} bps")
print(f"  Avg Short Raw Winrate: {avg_short_raw_hit:.2%}")
print(f"  Avg Short Net Hitrate: {avg_short_hit:.2%}")

# ── production models ──────────────────────────────────────────────────────────
print("\nTraining production models (Strict 80% Train, val=4 months before test)...")
split_idx = int(len(unique_months) * 0.8)
ptr = df['YearMonth'].isin(unique_months[:split_idx-4]).values
pva = df['YearMonth'].isin(unique_months[split_idx-4:split_idx]).values

dtr_p_l = xgb.DMatrix(X[ptr], label=y_long_binary[ptr])
dva_p_l = xgb.DMatrix(X[pva], label=y_long_binary[pva])
prod_long = xgb.train(get_params(pos_weight_long), dtr_p_l, num_boost_round=500,
                      evals=[(dva_p_l, 'val')], early_stopping_rounds=50, verbose_eval=50)
prod_long.save_model(LONG_MODEL)

dtr_p_s = xgb.DMatrix(X[ptr], label=y_short_binary[ptr])
dva_p_s = xgb.DMatrix(X[pva], label=y_short_binary[pva])
prod_short = xgb.train(get_params(pos_weight_short), dtr_p_s, num_boost_round=500,
                       evals=[(dva_p_s, 'val')], early_stopping_rounds=50, verbose_eval=50)
prod_short.save_model(SHORT_MODEL)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(StandardScaler(with_mean=False, with_std=False), f)

def top_features(bst, n=20):
    try:
        sc  = bst.get_score(importance_type='gain')
        out = {feature_cols[int(k.replace('f', ''))]: float(v)
               for k, v in sc.items() if int(k.replace('f', '')) < len(feature_cols)}
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:n])
    except Exception: return {}

metadata = {
    'description':  'v16 - Binary Breakout Classifier (>20bps target)',
    'type':         'binary_logistic',
    'threshold':    TRADE_PROB,
    'target_bps':   int(BREAKOUT_TH*10000),
    'features':     feature_cols,
    'num_features': len(feature_cols),
    'top_features_long':  top_features(prod_long),
    'top_features_short': top_features(prod_short)
}
with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"DONE -> {MODEL_DIR}")
