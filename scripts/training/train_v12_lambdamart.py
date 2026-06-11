"""
v12 — LambdaMART top-5 ranker with asymmetric graded relevance.

The audit of v11 correctly identified that utility regression (reg:squarederror)
is a pointwise loss — it penalises every sample in the universe equally and
never sees the top-5 selection that produces real P&L. This version fixes that
by using a proper listwise ranking loss (rank:ndcg) where the asymmetric
economic penalty is encoded in the relevance labels, and LambdaMART concentrates
gradient on getting the top-k ordering right.

Key differences vs v11 (utility regression):
  1. rank:ndcg objective  →  gradient is position-discounted; errors at the top
                             cost more than errors at the bottom.
  2. Graded relevance labels  →  asymmetric bucketing around the 10 bps cost
                                 threshold.  Sub-cost names get rel=0 or rel=1;
                                 above-cost names get rel=2..4.  With
                                 ndcg_exp_gain=True gains are 2^rel-1:
                                   rel=4 → gain 15   (>20bps gross, strong win)
                                   rel=3 → gain  7   (15-20bps gross)
                                   rel=2 → gain  3   (10-15bps, just clears cost)
                                   rel=1 → gain  1   (5-10bps, just below cost)
                                   rel=0 → gain  0   (<5bps or negative — no gain)
                                 Ranking a rel=0 item into top-5 when a rel=4 item
                                 was available creates a large negative ΔNDCG and
                                 therefore a large penalising lambda gradient.
  3. lambdarank_pair_method='topk'  →  pair sampling focuses on the top of the
                                       list (falls back gracefully if XGBoost
                                       version is too old).
  4. eval_metric ndcg@5             →  early stopping aligned with the top-5
                                       goal (not ndcg@3 as in v10).
  5. Headline metric unchanged       →  top-5 mean net return per fold so the
                                       result is directly comparable to v11.

Usage:
    python scripts/training/train_v12_lambdamart.py

Outputs:
    models/v12_lambdamart_1h/
        xgb_long_model.json
        xgb_short_model.json
        metadata.json
        scaler.pkl
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
MODEL_DIR = 'models/v12_lambdamart_1h'
COST      = 0.0010   # 10 bps

# Relevance bucket boundaries (net return = gross − cost)
# Tune these if you want a wider or narrower "just below cost" band.
#   net >= B3  → rel 4  (strong win)
#   net >= B2  → rel 3
#   net >= B1  → rel 2  (crossed cost threshold)
#   net >= B0  → rel 1  (just below cost)
#   net <  B0  → rel 0  (clear loss — zero gain)
B3 = 0.0010   # gross >= 20 bps
B2 = 0.0005   # gross >= 15 bps
B1 = 0.0000   # gross >= 10 bps  ← cost threshold
B0 = -0.0005  # gross >=  5 bps

TOP_K = 5
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
LONG_MODEL  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_MODEL = f'{MODEL_DIR}/xgb_short_model.json'
META_PATH   = f'{MODEL_DIR}/metadata.json'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'


# ── relevance labels ──────────────────────────────────────────────────────────
def graded_relevance(r: np.ndarray, cost: float = COST) -> np.ndarray:
    """
    Non-negative integer relevance for LambdaMART / NDCG.
    With ndcg_exp_gain=True, gain = 2^rel - 1:
      rel 4 → 15,  rel 3 → 7,  rel 2 → 3,  rel 1 → 1,  rel 0 → 0
    The large jump from 3→7→15 means surfacing a genuine winner into the
    top-5 is rewarded far more than surfacing a borderline name, and the
    LambdaMART delta-NDCG term ensures this asymmetry drives the gradient.
    """
    net = r - cost
    rel = np.zeros(len(r), dtype=np.int32)
    rel[net >= B0] = 1
    rel[net >= B1] = 2
    rel[net >= B2] = 3
    rel[net >= B3] = 4
    return rel


def show_relevance_table():
    print("\nRelevance table (cost=10bps, ndcg_exp_gain=True → gain=2^rel-1):")
    print(f"  {'Gross bps':>10}  {'Net bps':>8}  {'Rel':>5}  {'NDCG gain':>10}")
    for b in [25, 20, 15, 12, 10, 8, 5, 0, -5, -10, -20]:
        r   = b / 10000
        rel = int(graded_relevance(np.array([r]))[0])
        gain = 2**rel - 1
        print(f"  {b:>10}  {b-10:>8}  {rel:>5}  {gain:>10}")
    print()


# ── data ───────────────────────────────────────────────────────────────────────
print("=" * 64)
print("v12 LAMBDAMART TOP-5 RANKER — 1h native")
print("=" * 64)
show_relevance_table()

print(f"Loading {DATA_FILE} ...")
df = pd.read_csv(DATA_FILE)
print(f"  {df.shape[0]:,} rows")

df['YearMonth'] = df['DateTime'].str[:7]
unique_months   = sorted(df['YearMonth'].unique())
print(f"  {len(unique_months)} months: {unique_months[0]} → {unique_months[-1]}")

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

# precompute relevance labels
rel_long  = graded_relevance(y_returns)       # long: high return = high rel
rel_short = graded_relevance(-y_returns)      # short: low return = high rel


# ── XGBoost params ─────────────────────────────────────────────────────────────
device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10, dtype=np.int32))
    d.set_group([10])
    xgb.train({'objective': 'rank:ndcg', 'device': 'cuda', 'tree_method': 'hist'},
              d, num_boost_round=1)
    device = 'cuda'
    print("  CUDA GPU detected.")
except Exception:
    print("  CPU training.")

base_params = {
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

# topk pair method focuses pair sampling on the top — graceful fallback for
# older XGBoost versions that don't support this parameter
topk_params = {
    **base_params,
    'lambdarank_pair_method':        'topk',
    'lambdarank_num_pair_per_sample': 8,
}
params = topk_params


def make_dmatrix(X_: np.ndarray, rel: np.ndarray, qids: np.ndarray) -> xgb.DMatrix:
    dm = xgb.DMatrix(X_, label=rel)
    groups = pd.Series(qids).groupby(qids).size().values
    dm.set_group(groups)
    return dm


# ── evaluation helpers (same as v11 for direct comparability) ─────────────────
def topk_net_return(df_eval: pd.DataFrame, score_col: str,
                    k: int = TOP_K, cost: float = COST):
    gross, hits, n_q = [], 0, 0
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) < k + 1:
            continue
        ar  = q[RET_COL].values
        sc  = q[score_col].values
        idx = np.argsort(sc)[::-1][:k]
        picked = ar[idx]
        gross.extend(picked.tolist())
        hits += (picked > cost).sum()
        n_q  += 1
    if not gross:
        return 0.0, 0.0, 0
    return float(np.mean(gross)) - cost, hits / len(gross), n_q


def winrate_vs_median(df_eval: pd.DataFrame, score_col: str, k: int = TOP_K):
    hits = total = 0
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) < k + 1:
            continue
        ar  = q[RET_COL].values
        sc  = q[score_col].values
        idx = np.argsort(sc)[::-1][:k]
        med = np.median(ar)
        hits  += (ar[idx] > med).sum()
        total += k
    return hits / total if total else 0.0


def spearman_ic(df_eval: pd.DataFrame, score_col: str, invert: bool = False):
    rhos = []
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) < 2:
            continue
        y = -q[RET_COL].values if invert else q[RET_COL].values
        rho, _ = spearmanr(q[score_col].values, y)
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else 0.0


def evaluate_fold(df_eval, long_sc, short_sc):
    d = df_eval.copy()
    d['long_score']  = long_sc
    d['short_score'] = short_sc
    l_net, l_hit, n_q = topk_net_return(d, 'long_score')
    s_net, s_hit, _   = topk_net_return(d, 'short_score')
    l_wr  = winrate_vs_median(d, 'long_score')
    l_rho = spearman_ic(d, 'long_score', invert=False)
    s_rho = spearman_ic(d, 'short_score', invert=True)
    return dict(long_net=l_net, long_hit=l_hit, long_wr=l_wr,
                short_net=s_net, short_hit=s_hit,
                long_rho=l_rho, short_rho=s_rho, n_q=n_q)


# ── walk-forward ───────────────────────────────────────────────────────────────
min_train_months, horizon, step = 18, 2, 4
folds = []
for i in range(min_train_months, len(unique_months) - horizon, step):
    folds.append(dict(
        fold=len(folds) + 1,
        train=unique_months[:i],
        val=[unique_months[i]],
        test=unique_months[i + 1:i + horizon + 1],
    ))

print(f"\nWalk-forward folds: {len(folds)}")
print(f"Objective: rank:ndcg  eval: ndcg@5  pair_method: topk")
print(f"Headline : top-{TOP_K} net return (gross − {int(COST*10000)} bps)")
print()

# probe for topk param support
try:
    _d = xgb.DMatrix(np.random.randn(20, 2), label=np.zeros(20, dtype=np.int32))
    _d.set_group([20])
    xgb.train(topk_params, _d, num_boost_round=1)
    print("  lambdarank_pair_method='topk' supported.")
except Exception:
    params = base_params
    print("  lambdarank_pair_method not supported in this XGBoost version — "
        "falling back to default pair sampling.")

wf = []
for cfg in folds:
    tr_m, val_m, te_m = cfg['train'], cfg['val'], cfg['test']
    print(f"--- FOLD {cfg['fold']} --- train {tr_m[0]}→{tr_m[-1]} "
          f"({len(tr_m)}m) | val {val_m[0]} | test {te_m[0]}→{te_m[-1]}")

    trm = df['YearMonth'].isin(tr_m).values
    vam = df['YearMonth'].isin(val_m).values
    tem = df['YearMonth'].isin(te_m).values

    Xtr, Xva, Xte = X[trm], X[vam], X[tem]
    dfte = df[tem].copy()

    # long
    dtl = make_dmatrix(Xtr, rel_long[trm],  query_ids[trm])
    dvl = make_dmatrix(Xva, rel_long[vam],  query_ids[vam])
    bl  = xgb.train(params, dtl, num_boost_round=500,
                    evals=[(dvl, 'val')], early_stopping_rounds=50,
                    verbose_eval=False)

    # short
    dts = make_dmatrix(Xtr, rel_short[trm], query_ids[trm])
    dvs = make_dmatrix(Xva, rel_short[vam], query_ids[vam])
    bs  = xgb.train(params, dts, num_boost_round=500,
                    evals=[(dvs, 'val')], early_stopping_rounds=50,
                    verbose_eval=False)

    dte = xgb.DMatrix(Xte)
    m   = evaluate_fold(dfte, bl.predict(dte), bs.predict(dte))
    m['fold'] = cfg['fold']
    wf.append(m)

    flag = "✓" if m['long_net'] > 0 else "✗"
    print(f"    {flag} net {m['long_net']*10000:+.1f}bps  "
          f"hit={m['long_hit']:.1%}  wr_vs_med={m['long_wr']:.1%}  "
          f"rho={m['long_rho']:.4f}  q={m['n_q']}")


# ── aggregate ──────────────────────────────────────────────────────────────────
def avg(key): return float(np.mean([r[key] for r in wf]))

profitable = sum(1 for r in wf if r['long_net'] > 0)

print("\n" + "=" * 64)
print("WALK-FORWARD AGGREGATE")
print(f"  Long  net return  : {avg('long_net')*10000:+.2f} bps  "
      f"(hit={avg('long_hit'):.1%}  wr_vs_med={avg('long_wr'):.1%})")
print(f"  Short net return  : {avg('short_net')*10000:+.2f} bps  "
      f"(hit={avg('short_hit'):.1%})")
print(f"  Long  Spearman IC : {avg('long_rho'):.4f}")
print(f"  Short Spearman IC : {avg('short_rho'):.4f}")
print(f"  Profitable folds  : {profitable}/{len(wf)}")

print()
print("Comparison:")
print(f"  {'Model':<20} {'Net bps':>8}  {'Hit rate':>9}  {'WR vs med':>10}  {'Spearman':>9}  {'Prof folds':>11}")
print(f"  {'v10 pairwise':<20} {'−7.7':>8}  {'—':>9}  {'53.2%':>10}  {'0.0261':>9}  {'—':>11}")
print(f"  {'v11 utility reg':<20} {avg('long_net')*10000-0.0:.1f} was-8.53 → simplified below")
print(f"  {'v12 lambdamart':<20} {avg('long_net')*10000:>8.2f}  "
      f"{avg('long_hit'):>9.1%}  {avg('long_wr'):>10.1%}  "
      f"{avg('long_rho'):>9.4f}  {profitable:>2}/{len(wf)}")


# ── production models ──────────────────────────────────────────────────────────
print("\nTraining production models (all-but-last month, val=last)...")
ptr = df['YearMonth'].isin(unique_months[:-1]).values
pva = df['YearMonth'].isin([unique_months[-1]]).values

Xptr, Xpva = X[ptr], X[pva]

dptl = make_dmatrix(Xptr, rel_long[ptr],  query_ids[ptr])
dpvl = make_dmatrix(Xpva, rel_long[pva],  query_ids[pva])
prod_long = xgb.train(params, dptl, num_boost_round=500,
                      evals=[(dpvl, 'val')], early_stopping_rounds=50,
                      verbose_eval=50)
prod_long.save_model(LONG_MODEL)

dpts = make_dmatrix(Xptr, rel_short[ptr], query_ids[ptr])
dpvs = make_dmatrix(Xpva, rel_short[pva], query_ids[pva])
prod_short = xgb.train(params, dpts, num_boost_round=500,
                       evals=[(dpvl, 'val')], early_stopping_rounds=50,
                       verbose_eval=50)
prod_short.save_model(SHORT_MODEL)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(StandardScaler(with_mean=False, with_std=False), f)


def top_features(bst, n=20):
    try:
        sc = bst.get_score(importance_type='gain')
        out = {}
        for k, v in sc.items():
            i = int(k.replace('f', ''))
            if i < len(feature_cols):
                out[feature_cols[i]] = float(v)
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:n])
    except Exception:
        return {}


metadata = {
    'description': 'v12 — LambdaMART top-5 ranker, asymmetric graded relevance',
    'type': 'lambdamart_ndcg5',
    'features': feature_cols,
    'num_features': len(feature_cols),
    'data_file': DATA_FILE,
    'total_rows': int(df.shape[0]),
    'relevance_buckets': dict(B0=B0, B1=B1, B2=B2, B3=B3, cost=COST),
    'top_k': TOP_K,
    'walk_forward_summary': {
        'avg_long_net_bps':    round(avg('long_net')  * 10000, 3),
        'avg_short_net_bps':   round(avg('short_net') * 10000, 3),
        'avg_long_hit_rate':   round(avg('long_hit'),  4),
        'avg_long_wr_vs_med':  round(avg('long_wr'),   4),
        'avg_long_spearman':   round(avg('long_rho'),  4),
        'avg_short_spearman':  round(avg('short_rho'), 4),
        'profitable_folds':    profitable,
        'total_folds':         len(wf),
    },
    'walk_forward_folds': [
        {'fold': r['fold'],
         'long_net_bps':  round(r['long_net']  * 10000, 2),
         'long_hit_rate': round(r['long_hit'],  4),
         'long_wr_vs_med':round(r['long_wr'],   4),
         'long_rho':      round(r['long_rho'],  4),
         'short_net_bps': round(r['short_net'] * 10000, 2),
         'n_queries':     r['n_q']}
        for r in wf
    ],
    'top_features_long':  top_features(prod_long),
    'top_features_short': top_features(prod_short),
    'params': {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))},
    'trained_at': datetime.now().isoformat(),
    'baselines': {
        'v10_pairwise_net_bps':   -7.7,
        'v10_long_spearman':       0.0261,
        'v11_utility_net_bps':    -8.53,
        'v11_long_spearman':       0.0231,
    },
}
with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 64)
print(f"DONE → {MODEL_DIR}")
print(f"  Long net top-{TOP_K}: {avg('long_net')*10000:+.2f} bps  "
      f"({profitable}/{len(wf)} folds profitable)")
print(f"  Long Spearman:       {avg('long_rho'):.4f}")
print("=" * 64)
