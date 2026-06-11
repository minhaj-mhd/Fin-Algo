"""
v14 - LambdaMART without Early Stopping.

This model keeps the rank:ndcg objective and quantile-based graded relevance 
of v13, but completely disables early stopping. It will build exactly 500 trees 
using all available training data in each fold (and in the final production model) 
without holding out a validation set to stop early.

Usage:
    python scripts/training/train_v14_lambdamart_no_es.py

Outputs:
    models/v14_lambdamart_no_es_1h/
"""

import os, pickle, json
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# ── config ─────────────────────────────────────────────────────────────────────
DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL   = 'Next_Hour_Return'
MODEL_DIR = 'models/v14_lambdamart_no_es_1h'
COST      = 0.0010
TOP_K     = 5

os.makedirs(MODEL_DIR, exist_ok=True)
LONG_MODEL  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_MODEL = f'{MODEL_DIR}/xgb_short_model.json'
META_PATH   = f'{MODEL_DIR}/metadata.json'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'


# ── relevance: pure quantile within each query ────────────────────────────────
def quantile_relevance(returns: np.ndarray, qids: np.ndarray,
                       invert: bool = False) -> np.ndarray:
    rel = np.zeros(len(returns), dtype=np.int32)
    for qid in np.unique(qids):
        m   = qids == qid
        r   = -returns[m] if invert else returns[m]
        p   = pd.Series(r).rank(pct=True).values
        q   = np.zeros(m.sum(), dtype=np.int32)
        q[p > 0.50] = 1
        q[p > 0.75] = 2
        q[p > 0.90] = 3
        q[p > 0.95] = 4
        rel[m] = q
    return rel


# ── data ───────────────────────────────────────────────────────────────────────
print("=" * 64)
print("v14 LAMBDAMART NO EARLY STOPPING (500 trees fixed)")
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
print(f"  Features: {len(feature_cols)} | Queries: {df['Query_ID'].nunique():,}")

X         = df[feature_cols].values.astype(np.float64)
y_returns = df[RET_COL].values.astype(np.float64)
query_ids = df['Query_ID'].values

nan_mask = ~np.isfinite(X)
if nan_mask.any():
    for ci in range(X.shape[1]):
        bad = ~np.isfinite(X[:, ci])
        if bad.any():
            good = X[~bad, ci]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

print("  Computing quantile relevance labels ...")
rel_long  = quantile_relevance(y_returns, query_ids, invert=False)
rel_short = quantile_relevance(y_returns, query_ids, invert=True)


# ── XGBoost params ─────────────────────────────────────────────────────────────
device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.zeros(10, dtype=np.int32))
    d.set_group([10])
    xgb.train({'objective': 'rank:ndcg', 'device': 'cuda', 'tree_method': 'hist'},
              d, num_boost_round=1)
    device = 'cuda'
    print("  CUDA GPU detected.")
except Exception:
    print("  CPU training.")

params = {
    'objective':         'rank:ndcg',
    'eval_metric':       'ndcg@5',
    'ndcg_exp_gain':     True,
    'eta':               0.03,
    'max_depth':         5,
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

try:
    _d = xgb.DMatrix(np.random.randn(20, 2), label=np.zeros(20, dtype=np.int32))
    _d.set_group([20])
    xgb.train({**params, 'lambdarank_pair_method': 'topk'}, _d, num_boost_round=1)
    params['lambdarank_pair_method']        = 'topk'
    params['lambdarank_num_pair_per_sample'] = 8
    print("  lambdarank_pair_method='topk' enabled.")
except Exception:
    pass

def make_dmatrix(X_: np.ndarray, rel: np.ndarray, qids: np.ndarray) -> xgb.DMatrix:
    dm     = xgb.DMatrix(X_, label=rel)
    groups = pd.Series(qids).groupby(qids).size().values
    dm.set_group(groups)
    return dm

# ── evaluation ─────────────────────────────────────────────────────────────────
def topk_metrics(df_eval: pd.DataFrame, score_col: str, k: int = TOP_K, cost: float = COST):
    gross, hits_cost, hits_med, n_q = [], 0, 0, 0
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) < k + 1: continue
        ar  = q[RET_COL].values
        sc  = q[score_col].values
        idx = np.argsort(sc)[::-1][:k]
        picked = ar[idx]
        gross.extend(picked.tolist())
        hits_cost += (picked > cost).sum()
        hits_med  += (picked > np.median(ar)).sum()
        n_q       += 1
    if not gross: return 0.0, 0.0, 0.0, 0
    return float(np.mean(gross)) - cost, hits_cost / len(gross), hits_med / len(gross), n_q

def spearman_ic(df_eval: pd.DataFrame, score_col: str, invert: bool = False) -> float:
    rhos = []
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) < 2: continue
        y = -q[RET_COL].values if invert else q[RET_COL].values
        rho, _ = spearmanr(q[score_col].values, y)
        if np.isfinite(rho): rhos.append(rho)
    return float(np.mean(rhos)) if rhos else 0.0

def evaluate_fold(df_eval, long_sc, short_sc):
    d = df_eval.copy()
    d['long_score']  = long_sc
    d['short_score'] = short_sc
    l_net, l_hit, l_wr, n_q = topk_metrics(d, 'long_score')
    s_net, s_hit, s_wr, _   = topk_metrics(d, 'short_score')
    return dict(
        long_net=l_net,   long_hit=l_hit,   long_wr=l_wr,
        short_net=s_net,  short_hit=s_hit,
        long_rho=spearman_ic(d, 'long_score'),
        short_rho=spearman_ic(d, 'short_score', invert=True),
        n_q=n_q,
    )

# ── walk-forward ───────────────────────────────────────────────────────────────
min_train_months, horizon, step = 18, 2, 4
folds = []
for i in range(min_train_months, len(unique_months) - horizon, step):
    folds.append(dict(
        fold=len(folds) + 1,
        train=unique_months[:i+1], # include what was 'val' previously to train fully
        test=unique_months[i + 1:i + horizon + 1],
    ))

print(f"\nWalk-forward folds: {len(folds)}")
print(f"Headline: top-{TOP_K} net return (gross - {int(COST*10000)} bps)\n")

wf = []
for cfg in folds:
    tr_m, te_m = cfg['train'], cfg['test']
    print(f"--- FOLD {cfg['fold']} --- train {tr_m[0]}->{tr_m[-1]} ({len(tr_m)}m) | test {te_m[0]}->{te_m[-1]}")

    trm = df['YearMonth'].isin(tr_m).values
    tem = df['YearMonth'].isin(te_m).values

    Xtr, Xte = X[trm], X[tem]
    dfte = df[tem].copy()

    bl = xgb.train(params, make_dmatrix(Xtr, rel_long[trm], query_ids[trm]), num_boost_round=500, verbose_eval=False)
    bs = xgb.train(params, make_dmatrix(Xtr, rel_short[trm], query_ids[trm]), num_boost_round=500, verbose_eval=False)

    dte = xgb.DMatrix(Xte)
    m   = evaluate_fold(dfte, bl.predict(dte), bs.predict(dte))
    m['fold'] = cfg['fold']
    wf.append(m)

    flag = "Y" if m['long_net'] > 0 else "N"
    print(f"    {flag} net {m['long_net']*10000:+.1f}bps  hit={m['long_hit']:.1%}  wr_med={m['long_wr']:.1%}  rho={m['long_rho']:.4f}")

# ── aggregate ──────────────────────────────────────────────────────────────────
def avg(key): return float(np.mean([r[key] for r in wf]))
profitable = sum(1 for r in wf if r['long_net'] > 0)

print("\n" + "=" * 64)
print("WALK-FORWARD AGGREGATE")
print(f"  Long  net return  : {avg('long_net')*10000:+.2f} bps")
print(f"  Short net return  : {avg('short_net')*10000:+.2f} bps")
print(f"  Long  Spearman IC : {avg('long_rho'):.4f}")
print(f"  Short Spearman IC : {avg('short_rho'):.4f}")

# ── production models ──────────────────────────────────────────────────────────
print("\nTraining production models (Strict 80% Train, NO VALIDATION, 500 trees)...")
split_idx = int(len(unique_months) * 0.8)
# We use the full 80% for training. The final 20% is completely untouched.
ptr = df['YearMonth'].isin(unique_months[:split_idx]).values

prod_long = xgb.train(params, make_dmatrix(X[ptr], rel_long[ptr], query_ids[ptr]), num_boost_round=500)
prod_long.save_model(LONG_MODEL)

prod_short = xgb.train(params, make_dmatrix(X[ptr], rel_short[ptr], query_ids[ptr]), num_boost_round=500)
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
    'description':  'v14 - LambdaMART, NO EARLY STOPPING (fixed 500 rounds)',
    'type':         'lambdamart_no_es',
    'features':     feature_cols,
    'num_features': len(feature_cols),
    'top_features_long':  top_features(prod_long),
    'top_features_short': top_features(prod_short)
}
with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"DONE -> {MODEL_DIR}")
