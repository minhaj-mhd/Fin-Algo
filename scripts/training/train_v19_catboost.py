"""
v19 - CatBoost Classifier (Directional).

Model uses CatBoost's oblivious trees to prevent overfitting.
Target: > 0 bps
Features: Temporal features kept.
Depth: 5
Task Type: GPU

Usage:
    python scripts/training/train_v19_catboost.py

Outputs:
    models/v19_catboost_1h/
"""

import os, pickle, json
from datetime import datetime
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# ── config ─────────────────────────────────────────────────────────────────────
DATA_FILE   = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL     = 'Next_Hour_Return'
MODEL_DIR   = 'models/v19_catboost_1h'
COST        = 0.0010
BREAKOUT_TH = 0.0000  # 0 bps target (direction only)
TRADE_PROB  = 0.52    # 52% probability threshold

os.makedirs(MODEL_DIR, exist_ok=True)
LONG_MODEL  = f'{MODEL_DIR}/cb_long_model.cbm'
SHORT_MODEL = f'{MODEL_DIR}/cb_short_model.cbm'
META_PATH   = f'{MODEL_DIR}/metadata.json'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'

# ── data ───────────────────────────────────────────────────────────────────────
print("=" * 64)
print(f"v19 CATBOOST CLASSIFIER (>{int(BREAKOUT_TH*10000)} bps target, GPU)")
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
print("  Creating binary direction targets...")
y_long_binary  = (y_returns > BREAKOUT_TH).astype(np.int32)
y_short_binary = (y_returns < -BREAKOUT_TH).astype(np.int32)

print(f"  Long target (>0bps) class balance:  {y_long_binary.mean():.2%}")
print(f"  Short target (<0bps) class balance: {y_short_binary.mean():.2%}")

pos_weight_long  = (1.0 - y_long_binary.mean()) / y_long_binary.mean()
pos_weight_short = (1.0 - y_short_binary.mean()) / y_short_binary.mean()

def get_cb_model(pos_weight: float) -> CatBoostClassifier:
    # Use GPU explicitly
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        loss_function='Logloss',
        eval_metric='AUC',
        class_weights=[1.0, pos_weight],
        random_seed=42,
        task_type='GPU',
        devices='0', # Explicitly target GPU 0
        verbose=False,
        early_stopping_rounds=50
    )

# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate_fold(df_eval: pd.DataFrame, long_probs: np.ndarray, short_probs: np.ndarray):
    rets = df_eval[RET_COL].values
    y_l_true = df_eval['long_target'].values
    y_s_true = df_eval['short_target'].values
    
    # Long Trades
    long_mask = long_probs > TRADE_PROB
    n_long = long_mask.sum()
    if n_long > 0:
        l_ret = rets[long_mask]
        l_raw_ret = float(np.mean(l_ret))
        l_net = l_raw_ret - COST
        l_hit = float(np.mean(l_ret > COST))
        l_raw_hit = float(np.mean(l_ret > 0))
    else:
        l_raw_ret = l_net = l_hit = l_raw_hit = 0.0
        
    l_tp = int((long_mask & (y_l_true == 1)).sum())
    l_tn = int((~long_mask & (y_l_true == 0)).sum())
    l_fp = int((long_mask & (y_l_true == 0)).sum())
    l_fn = int((~long_mask & (y_l_true == 1)).sum())
    
    l_acc = (l_tp + l_tn) / (l_tp + l_tn + l_fp + l_fn) if (l_tp + l_tn + l_fp + l_fn) > 0 else 0.0
    l_prec = l_tp / (l_tp + l_fp) if (l_tp + l_fp) > 0 else 0.0
    l_rec = l_tp / (l_tp + l_fn) if (l_tp + l_fn) > 0 else 0.0
    
    l_rho, _ = spearmanr(long_probs, rets)
    if np.isnan(l_rho): l_rho = 0.0
    try: l_auc = roc_auc_score(y_l_true, long_probs)
    except: l_auc = 0.5
    
    # Short Trades
    short_mask = short_probs > TRADE_PROB
    n_short = short_mask.sum()
    if n_short > 0:
        s_ret = -rets[short_mask]
        s_raw_ret = float(np.mean(s_ret))
        s_net = s_raw_ret - COST
        s_hit = float(np.mean(s_ret > COST))
        s_raw_hit = float(np.mean(s_ret > 0))
    else:
        s_raw_ret = s_net = s_hit = s_raw_hit = 0.0
        
    s_tp = int((short_mask & (y_s_true == 1)).sum())
    s_tn = int((~short_mask & (y_s_true == 0)).sum())
    s_fp = int((short_mask & (y_s_true == 0)).sum())
    s_fn = int((~short_mask & (y_s_true == 1)).sum())
    
    s_acc = (s_tp + s_tn) / (s_tp + s_tn + s_fp + s_fn) if (s_tp + s_tn + s_fp + s_fn) > 0 else 0.0
    s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
    s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
    
    s_rho, _ = spearmanr(short_probs, -rets) # Invert returns for short correlation
    if np.isnan(s_rho): s_rho = 0.0
    try: s_auc = roc_auc_score(y_s_true, short_probs)
    except: s_auc = 0.5

    return {
        'long': {
            'net': l_net, 'raw': l_raw_ret, 'hit': l_hit, 'raw_hit': l_raw_hit, 'n': n_long,
            'auc': l_auc, 'rho': l_rho, 'acc': l_acc, 'prec': l_prec, 'rec': l_rec,
            'tp': l_tp, 'tn': l_tn, 'fp': l_fp, 'fn': l_fn
        },
        'short': {
            'net': s_net, 'raw': s_raw_ret, 'hit': s_hit, 'raw_hit': s_raw_hit, 'n': n_short,
            'auc': s_auc, 'rho': s_rho, 'acc': s_acc, 'prec': s_prec, 'rec': s_rec,
            'tp': s_tp, 'tn': s_tn, 'fp': s_fp, 'fn': s_fn
        }
    }

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
print(f"Headline: Trades > {TRADE_PROB*100}% probability (cost=10bps)\n")

wf = []
for cfg in folds:
    tr_m, val_m, te_m = cfg['train'], cfg['val'], cfg['test']
    print(f"\n--- FOLD {cfg['fold']} --- train {tr_m[0]}->{tr_m[-1]} | val {val_m[0]} | test {te_m[0]}->{te_m[-1]}")

    trm = df['YearMonth'].isin(tr_m).values
    vam = df['YearMonth'].isin(val_m).values
    tem = df['YearMonth'].isin(te_m).values

    Xtr, Xva, Xte = X[trm], X[vam], X[tem]
    dfte = df[tem].copy()
    dfte['long_target'] = y_long_binary[tem]
    dfte['short_target'] = y_short_binary[tem]

    # Train Longs
    bl = get_cb_model(pos_weight_long)
    bl.fit(Xtr, y_long_binary[trm], eval_set=(Xva, y_long_binary[vam]), verbose=False)
                   
    # Train Shorts
    bs = get_cb_model(pos_weight_short)
    bs.fit(Xtr, y_short_binary[trm], eval_set=(Xva, y_short_binary[vam]), verbose=False)

    l_preds = bl.predict_proba(Xte)[:, 1]
    s_preds = bs.predict_proba(Xte)[:, 1]
    
    m   = evaluate_fold(dfte, l_preds, s_preds)
    m['fold'] = cfg['fold']
    wf.append(m)

    lm = m['long']; sm = m['short']
    print(f"    L -> AUC: {lm['auc']:.3f} | Rho: {lm['rho']:.3f} | Prec: {lm['prec']:.1%} | Rec: {lm['rec']:.1%} | Trades: {lm['n']} | TP:{lm['tp']} FP:{lm['fp']}")
    print(f"         Raw Ret: {lm['raw']*10000:>+5.1f}bps | Net Ret: {lm['net']*10000:>+5.1f}bps | Raw WR: {lm['raw_hit']:.1%} | Net WR: {lm['hit']:.1%}")
    print(f"    S -> AUC: {sm['auc']:.3f} | Rho: {sm['rho']:.3f} | Prec: {sm['prec']:.1%} | Rec: {sm['rec']:.1%} | Trades: {sm['n']} | TP:{sm['tp']} FP:{sm['fp']}")
    print(f"         Raw Ret: {sm['raw']*10000:>+5.1f}bps | Net Ret: {sm['net']*10000:>+5.1f}bps | Raw WR: {sm['raw_hit']:.1%} | Net WR: {sm['hit']:.1%}")

# ── aggregate ──────────────────────────────────────────────────────────────────
l_trades = sum(r['long']['n'] for r in wf)
s_trades = sum(r['short']['n'] for r in wf)

def agg(k, sub):
    trades = l_trades if sub == 'long' else s_trades
    if trades == 0: return 0.0
    return np.average([r[sub][k] for r in wf if r[sub]['n'] > 0], weights=[r[sub]['n'] for r in wf if r[sub]['n'] > 0])

print("\n" + "=" * 64)
print("WALK-FORWARD AGGREGATE (Weighted by Trade Count)")
print("LONG STATS:")
print(f"  Total Trades : {l_trades}")
print(f"  Avg AUC      : {np.mean([r['long']['auc'] for r in wf]):.3f}")
print(f"  Avg Rho      : {np.mean([r['long']['rho'] for r in wf]):.3f}")
print(f"  Raw Return   : {agg('raw', 'long')*10000:+.2f} bps")
print(f"  Net Return   : {agg('net', 'long')*10000:+.2f} bps")
print(f"  Raw Winrate  : {agg('raw_hit', 'long'):.2%}")
print(f"  Net Winrate  : {agg('hit', 'long'):.2%}")
print(f"  Precision    : {agg('prec', 'long'):.2%}")
print(f"  Recall       : {agg('rec', 'long'):.2%}")
print(f"  Accuracy     : {agg('acc', 'long'):.2%}")
print(f"  Totals       -> TP: {sum(r['long']['tp'] for r in wf)} | TN: {sum(r['long']['tn'] for r in wf)} | FP: {sum(r['long']['fp'] for r in wf)} | FN: {sum(r['long']['fn'] for r in wf)}")

print("\nSHORT STATS:")
print(f"  Total Trades : {s_trades}")
print(f"  Avg AUC      : {np.mean([r['short']['auc'] for r in wf]):.3f}")
print(f"  Avg Rho      : {np.mean([r['short']['rho'] for r in wf]):.3f}")
print(f"  Raw Return   : {agg('raw', 'short')*10000:+.2f} bps")
print(f"  Net Return   : {agg('net', 'short')*10000:+.2f} bps")
print(f"  Raw Winrate  : {agg('raw_hit', 'short'):.2%}")
print(f"  Net Winrate  : {agg('hit', 'short'):.2%}")
print(f"  Precision    : {agg('prec', 'short'):.2%}")
print(f"  Recall       : {agg('rec', 'short'):.2%}")
print(f"  Accuracy     : {agg('acc', 'short'):.2%}")
print(f"  Totals       -> TP: {sum(r['short']['tp'] for r in wf)} | TN: {sum(r['short']['tn'] for r in wf)} | FP: {sum(r['short']['fp'] for r in wf)} | FN: {sum(r['short']['fn'] for r in wf)}")

# ── production models ──────────────────────────────────────────────────────────
print("\nTraining production models...")
split_idx = int(len(unique_months) * 0.8)
ptr = df['YearMonth'].isin(unique_months[:split_idx-4]).values
pva = df['YearMonth'].isin(unique_months[split_idx-4:split_idx]).values

prod_long = get_cb_model(pos_weight_long)
prod_long.fit(X[ptr], y_long_binary[ptr], eval_set=(X[pva], y_long_binary[pva]), verbose=False)
prod_long.save_model(LONG_MODEL)

prod_short = get_cb_model(pos_weight_short)
prod_short.fit(X[ptr], y_short_binary[ptr], eval_set=(X[pva], y_short_binary[pva]), verbose=False)
prod_short.save_model(SHORT_MODEL)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(StandardScaler(with_mean=False, with_std=False), f)

def top_features(bst, n=20):
    try:
        sc  = bst.get_feature_importance()
        out = {feature_cols[i]: float(sc[i]) for i in range(len(feature_cols))}
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:n])
    except Exception: return {}

metadata = {
    'description':  'v19 - CatBoost Classifier (Directional, >0bps target, depth 4, GPU)',
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
