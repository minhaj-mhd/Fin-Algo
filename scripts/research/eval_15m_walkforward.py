"""
15m walk-forward evaluation — apples-to-apples vs v10/v11/v12/v13.

Identical methodology to the 1h walk-forward experiments:
  - rank:pairwise objective (matches v10 baseline)
  - same fold structure: 18m min train, 2m test, 4m step
  - same headline metric: top-5 mean net return (gross − 10 bps)
  - same evaluation: hit rate, win rate vs median, Spearman IC

This answers the question: does the 15m model's higher Spearman (0.0586 vs
0.026) actually translate into positive top-5 net return after cost?
If it gets profitable folds where every 1h variant got 0/9, the 15m stack
is worth building. If it also gets 0/9, the problem is the cost floor.

Usage:
    python scripts/research/eval_15m_walkforward.py
"""

import os, json
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.utils import shuffle

DATA_FILE = 'data/ranking_data_upstox_15min_3y_clean.csv'
RET_COL   = 'Next_15Min_Return'
COST      = 0.0010
TOP_K     = 5

print("=" * 64)
print("15m WALK-FORWARD — rank:pairwise, top-5 net return")
print("Same methodology as v10/v11/v12/v13 for direct comparison")
print("=" * 64)

print(f"\nLoading {DATA_FILE} ...")
df = pd.read_csv(DATA_FILE)
print(f"  {df.shape[0]:,} rows")

df['YearMonth'] = df['DateTime'].str[:7]
unique_months   = sorted(df['YearMonth'].unique())
print(f"  {len(unique_months)} months: {unique_months[0]} → {unique_months[-1]}")

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth',
                'Market_Mean_Return', 'Market_Mean_Volatility',
                'Relative_Return', 'Relative_Volatility']
feature_cols = [c for c in df.columns if c not in exclude_cols
                and c not in ('Next_Hour_Return',)]
print(f"  Features: {len(feature_cols)} | Queries: {df['Query_ID'].nunique():,}")
print(f"  Return std: {df[RET_COL].std()*10000:.1f} bps  "
      f"(cost = {COST*10000:.0f} bps = "
      f"{COST/df[RET_COL].std()*100:.1f}% of 1-std move)")

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


# ── rank labels (same as v10) ──────────────────────────────────────────────────
def get_integer_ranks(y_vals, qids, invert=False):
    from scipy.stats import rankdata
    y_int = np.zeros_like(y_vals, dtype=int)
    for qid in np.unique(qids):
        m = qids == qid
        if m.sum() == 0: continue
        vals = -y_vals[m] if invert else y_vals[m]
        y_int[m] = rankdata(vals, method='ordinal') - 1
    return y_int


def make_dmatrix(X_, y_, qids_):
    dm     = xgb.DMatrix(X_, label=y_)
    groups = pd.Series(qids_).groupby(qids_).size().values
    dm.set_group(groups)
    return dm


# ── GPU ────────────────────────────────────────────────────────────────────────
device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10))
    d.set_group([10])
    xgb.train({'objective': 'rank:pairwise', 'device': 'cuda',
               'tree_method': 'hist'}, d, num_boost_round=1)
    device = 'cuda'
    print("  CUDA GPU detected.")
except Exception:
    print("  CPU training.")

params = {
    'objective':        'rank:pairwise',
    'eval_metric':      'ndcg@5',
    'ndcg_exp_gain':    False,
    'eta':              0.03,
    'max_depth':        4,           # v3_15min_clean uses depth 4
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'alpha':            1.0,
    'lambda':           2.0,
    'min_child_weight': 15,          # v3_15min_clean uses 15
    'random_state':     42,
    'verbosity':        0,
    'tree_method':      'hist',
    'device':           device,
}


# ── evaluation ─────────────────────────────────────────────────────────────────
def topk_metrics(df_eval, score_col, k=TOP_K, cost=COST):
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
        n_q += 1
    if not gross:
        return 0.0, 0.0, 0.0, 0
    g = float(np.mean(gross))
    return g - cost, hits_cost / len(gross), hits_med / len(gross), n_q


def spearman_ic(df_eval, score_col, invert=False):
    rhos = []
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) < 2: continue
        y = -q[RET_COL].values if invert else q[RET_COL].values
        rho, _ = spearmanr(q[score_col].values, y)
        if np.isfinite(rho): rhos.append(rho)
    return float(np.mean(rhos)) if rhos else 0.0


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
print(f"Headline: top-{TOP_K} net return (gross − {int(COST*10000)} bps)\n")

wf = []
for cfg in folds:
    tr_m, val_m, te_m = cfg['train'], cfg['val'], cfg['test']
    print(f"--- FOLD {cfg['fold']} --- train {tr_m[0]}→{tr_m[-1]} "
          f"({len(tr_m)}m) | val {val_m[0]} | test {te_m[0]}→{te_m[-1]}")

    trm = df['YearMonth'].isin(tr_m).values
    vam = df['YearMonth'].isin(val_m).values
    tem = df['YearMonth'].isin(te_m).values

    Xtr, Xva, Xte = X[trm], X[vam], X[tem]
    qtr, qva      = query_ids[trm], query_ids[vam]
    ytr, yva      = y_returns[trm], y_returns[vam]
    dfte          = df[tem].copy()

    # long
    bl = xgb.train(params,
                   make_dmatrix(Xtr, get_integer_ranks(ytr, qtr), qtr),
                   num_boost_round=500,
                   evals=[(make_dmatrix(Xva, get_integer_ranks(yva, qva), qva), 'val')],
                   early_stopping_rounds=50, verbose_eval=False)

    # short
    bs = xgb.train(params,
                   make_dmatrix(Xtr, get_integer_ranks(ytr, qtr, invert=True), qtr),
                   num_boost_round=500,
                   evals=[(make_dmatrix(Xva, get_integer_ranks(yva, qva, invert=True), qva), 'val')],
                   early_stopping_rounds=50, verbose_eval=False)

    dte = xgb.DMatrix(Xte)
    dfte['long_score']  = bl.predict(dte)
    dfte['short_score'] = bs.predict(dte)

    l_net, l_hit, l_wr, n_q = topk_metrics(dfte, 'long_score')
    s_net, s_hit, s_wr, _   = topk_metrics(dfte, 'short_score')
    l_rho = spearman_ic(dfte, 'long_score')
    s_rho = spearman_ic(dfte, 'short_score', invert=True)

    m = dict(long_net=l_net, long_hit=l_hit, long_wr=l_wr,
             short_net=s_net, short_hit=s_hit,
             long_rho=l_rho, short_rho=s_rho, n_q=n_q,
             fold=cfg['fold'])
    wf.append(m)

    flag = "✓" if l_net > 0 else "✗"
    print(f"    {flag} net {l_net*10000:+.1f}bps  "
          f"hit={l_hit:.1%}  wr_med={l_wr:.1%}  "
          f"rho={l_rho:.4f}  q={n_q}")


# ── aggregate ──────────────────────────────────────────────────────────────────
def avg(key): return float(np.mean([r[key] for r in wf]))

profitable = sum(1 for r in wf if r['long_net'] > 0)

print("\n" + "=" * 64)
print("WALK-FORWARD AGGREGATE — 15m")
print(f"  Long  net return  : {avg('long_net')*10000:+.2f} bps  "
      f"(hit={avg('long_hit'):.1%}  wr_med={avg('long_wr'):.1%})")
print(f"  Short net return  : {avg('short_net')*10000:+.2f} bps")
print(f"  Long  Spearman IC : {avg('long_rho'):.4f}")
print(f"  Short Spearman IC : {avg('short_rho'):.4f}")
print(f"  Profitable folds  : {profitable}/{len(wf)}")

print()
print(f"  {'Model':<24} {'Net bps':>8}  {'Hit':>6}  {'WR med':>7}  "
      f"{'Spearman':>9}  {'Prof':>5}  {'Horizon'}")
print(f"  {'-'*75}")
print(f"  {'v10 pairwise (1h)':<24} {'−7.7':>8}  {'—':>6}  {'53.2%':>7}  "
      f"{'0.0261':>9}  {'0/9':>5}  1h")
print(f"  {'v11 utility (1h)':<24} {'−8.53':>8}  {'36.2%':>6}  {'—':>7}  "
      f"{'0.0231':>9}  {'0/9':>5}  1h")
print(f"  {'v12 lambdamart (1h)':<24} {'−8.20':>8}  {'42.3%':>6}  {'50.0%':>7}  "
      f"{'−0.007':>9}  {'0/9':>5}  1h")
print(f"  {'v13 ndcg raw (1h)':<24} {'−9.18':>8}  {'42.1%':>6}  {'49.1%':>7}  "
      f"{'−0.014':>9}  {'0/9':>5}  1h")
print(f"  {'15m pairwise':<24} {avg('long_net')*10000:>8.2f}  "
      f"{avg('long_hit'):>6.1%}  {avg('long_wr'):>7.1%}  "
      f"{avg('long_rho'):>9.4f}  {profitable:>2}/{len(wf)}  15m")

results = {
    'model': '15m_pairwise_walkforward',
    'data': DATA_FILE,
    'cost_bps': int(COST * 10000),
    'top_k': TOP_K,
    'trained_at': datetime.now().isoformat(),
    'summary': {
        'avg_long_net_bps':   round(avg('long_net')  * 10000, 3),
        'avg_short_net_bps':  round(avg('short_net') * 10000, 3),
        'avg_long_hit_rate':  round(avg('long_hit'),  4),
        'avg_long_wr_vs_med': round(avg('long_wr'),   4),
        'avg_long_spearman':  round(avg('long_rho'),  4),
        'profitable_folds':   profitable,
        'total_folds':        len(wf),
    },
    'folds': [
        {'fold': r['fold'],
         'long_net_bps':   round(r['long_net']  * 10000, 2),
         'long_hit_rate':  round(r['long_hit'],  4),
         'long_wr_vs_med': round(r['long_wr'],   4),
         'long_rho':       round(r['long_rho'],  4),
         'short_net_bps':  round(r['short_net'] * 10000, 2),
         'n_queries':      r['n_q']}
        for r in wf
    ],
}
out = 'data/eval_15m_walkforward_results.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out}")
print("=" * 64)
