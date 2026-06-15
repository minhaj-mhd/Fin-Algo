"""
v20 isolation test: is the rolling-1h ranking lift over v10 due to the candle CONSTRUCTION,
or just to sampling 18 intraday moments instead of v10's 5?

Re-runs v20's exact purged walk-forward (same folds/params as train_ranking_clean --tf 1h_roll)
but reports per-fold ρ TWICE:
  (a) ALL entry-times (18/day) — should reproduce the headline 0.0323 / 0.0327
  (b) ONLY v10's 5 shared :15 decision moments {10:15,11:15,12:15,13:15,14:15}, each
      predicting the same [T, T+1h] target v10 predicts.

Interpretation:
  - v20@:15 ≈ v20@all and BEATS v10 (0.0261L / 0.0245S)  -> construction itself is better.
  - v20@:15 << v20@all                                   -> lift is mostly the extra off-:15 moments.
  - v20@:15 ≈ v10                                         -> no construction edge at v10's moments.

RESEARCH ONLY. No Gauntlet, no verdict. Does not touch production models.
"""
import os, sys
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, rankdata
sys.path.append(os.getcwd())

DATA = 'data/research/v20_rolling_1h/panel.parquet'
RET_COL = 'Next_Hour_Return'
SHARED_15 = {'10:15', '11:15', '12:15', '13:15', '14:15'}   # v10's decision moments
V10_LONG, V10_SHORT = 0.0261, 0.0245                        # v10 WF avg (metadata b10b37fc)

print(f"Loading {DATA} ...")
df = pd.read_parquet(DATA)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['YearMonth'] = df['DateTime'].dt.strftime('%Y-%m')
df['HHMM'] = df['DateTime'].dt.strftime('%H:%M')
print(f"  {len(df):,} rows | {df['Query_ID'].nunique():,} queries")

exclude = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
           'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth', 'HHMM']
feat_cols = [c for c in df.columns if c not in exclude]
print(f"  features: {len(feat_cols)}")

X = df[feat_cols].values.astype(np.float64)
y = df[RET_COL].values
q = df['Query_ID'].values
bad = np.isnan(X) | np.isinf(X)
if bad.any():
    for ci in range(X.shape[1]):
        col = X[:, ci]; m = np.isnan(col) | np.isinf(col)
        if m.any():
            good = col[~m]; X[m, ci] = float(good.mean()) if len(good) else 0.0

months = sorted(df['YearMonth'].unique())


def int_ranks(yv, qids, invert=False):
    out = np.zeros_like(yv, dtype=int)
    for qid in np.unique(qids):
        mm = qids == qid
        vals = -yv[mm] if invert else yv[mm]
        out[mm] = rankdata(vals, method='ordinal') - 1
    return out


def rho_on(dfe, score_col, invert):
    rhos = []
    for qid in dfe['Query_ID'].unique():
        sub = dfe[dfe['Query_ID'] == qid]
        if len(sub) > 1:
            yy = -sub[RET_COL].values if invert else sub[RET_COL].values
            r, _ = spearmanr(sub[score_col].values, yy)
            if not np.isnan(r):
                rhos.append(r)
    return float(np.mean(rhos)) if rhos else 0.0


device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
    xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, d, num_boost_round=1)
    device = 'cuda'
except Exception:
    pass
print(f"  device: {device}")

params = {'objective': 'rank:pairwise', 'eta': 0.03, 'max_depth': 5, 'subsample': 0.8,
          'colsample_bytree': 0.8, 'alpha': 1.0, 'lambda': 2.0, 'min_child_weight': 10,
          'random_state': 42, 'verbosity': 0, 'eval_metric': 'ndcg@3', 'ndcg_exp_gain': False,
          'tree_method': 'hist', 'device': device}

min_train, horizon = 18, 2
folds = [dict(train=months[:i], val=[months[i]], test=months[i+1:i+horizon+1])
         for i in range(min_train, len(months) - horizon, 4)]
print(f"\nWalk-forward folds: {len(folds)}\n")

agg = {'all_L': [], 'all_S': [], '15_L': [], '15_S': []}
for fi, cfg in enumerate(folds, 1):
    trm = df['YearMonth'].isin(cfg['train']).values
    vam = df['YearMonth'].isin(cfg['val']).values
    tem = df['YearMonth'].isin(cfg['test']).values
    Xtr, ytr, qtr = X[trm], y[trm], q[trm]
    Xva, yva, qva = X[vam], y[vam], q[vam]
    dfte = df[tem].copy()
    gtr = pd.Series(qtr).groupby(qtr).size().values
    gva = pd.Series(qva).groupby(qva).size().values

    dtl = xgb.DMatrix(Xtr, label=int_ranks(ytr, qtr, False)); dtl.set_group(gtr)
    dvl = xgb.DMatrix(Xva, label=int_ranks(yva, qva, False)); dvl.set_group(gva)
    bl = xgb.train(params, dtl, 500, evals=[(dvl, 'val')], early_stopping_rounds=50, verbose_eval=False)
    dts = xgb.DMatrix(Xtr, label=int_ranks(ytr, qtr, True)); dts.set_group(gtr)
    dvs = xgb.DMatrix(Xva, label=int_ranks(yva, qva, True)); dvs.set_group(gva)
    bs = xgb.train(params, dts, 500, evals=[(dvs, 'val')], early_stopping_rounds=50, verbose_eval=False)

    dte = xgb.DMatrix(X[tem])
    dfte['long_score'] = bl.predict(dte)
    dfte['short_score'] = bs.predict(dte)
    sub15 = dfte[dfte['HHMM'].isin(SHARED_15)]

    aL = rho_on(dfte, 'long_score', False); aS = rho_on(dfte, 'short_score', True)
    fL = rho_on(sub15, 'long_score', False); fS = rho_on(sub15, 'short_score', True)
    agg['all_L'].append(aL); agg['all_S'].append(aS); agg['15_L'].append(fL); agg['15_S'].append(fS)
    print(f"FOLD {fi} test {cfg['test'][0]}..{cfg['test'][-1]} | "
          f"ALL L {aL:.4f} S {aS:.4f} | :15 L {fL:.4f} S {fS:.4f} "
          f"(all q={dfte['Query_ID'].nunique()}, :15 q={sub15['Query_ID'].nunique()})")

m = {k: float(np.mean(v)) for k, v in agg.items()}
print("\n" + "=" * 64)
print("ISOLATION RESULT (WF avg)")
print(f"  v20 ALL-18 entry-times : Long {m['all_L']:.4f} | Short {m['all_S']:.4f}")
print(f"  v20 :15-ONLY (5/day)   : Long {m['15_L']:.4f} | Short {m['15_S']:.4f}")
print(f"  v10 native (same :15)  : Long {V10_LONG:.4f} | Short {V10_SHORT:.4f}")
print("-" * 64)
print(f"  v20@:15 - v10  ->  Long {m['15_L']-V10_LONG:+.4f} | Short {m['15_S']-V10_SHORT:+.4f}")
print("=" * 64)
