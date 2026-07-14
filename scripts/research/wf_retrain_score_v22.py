"""
RETRAINED walk-forward for v22_rolling_1h_dynamic.

Retrains the short & long rank:pairwise rankers on an EXPANDING window,
steps forward in 3-month OOS blocks, scores each block with the model that
was trained ONLY on prior data, and saves the concatenated out-of-sample
scored panel.

Features are auto-discovered from the parquet schema (all except meta/target).

Output:
  scratch/wf_v22/oos_scored.parquet
  scratch/wf_v22/fold_meta.json
"""
import os, json, time
import pandas as pd, numpy as np, xgboost as xgb
import pyarrow.parquet as pq

OUT = 'scratch/wf_v22'
os.makedirs(OUT, exist_ok=True)
PANEL = 'data/research/v22_rolling_1h_dynamic/panel.parquet'
SS_QUANTILE = 0.9992  # top 0.08%

META_COLS = {'DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker', 
             'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Return', 'YearMonth'}

PARAMS = {
    'objective': 'rank:pairwise', 'eta': 0.03, 'max_depth': 5,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'alpha': 1.0, 'lambda': 2.0,
    'min_child_weight': 10, 'random_state': 42, 'verbosity': 0,
    'eval_metric': 'ndcg@3', 'ndcg_exp_gain': False, 'tree_method': 'hist', 'device': 'cuda',
}

print("Auto-discovering features from schema...")
schema = pq.read_schema(PANEL)
all_cols = schema.names
feats = [c for c in all_cols if c not in META_COLS]
print(f"Found {len(feats)} features.")

print("Loading panel ...")
cols = ['DateTime', 'Ticker', 'Query_ID', 'Next_Hour_Return'] + feats
df = pd.read_parquet(PANEL, columns=cols)
df['DateTime'] = pd.to_datetime(df['DateTime']).dt.tz_localize(None)
df = df.dropna(subset=['Next_Hour_Return']).reset_index(drop=True)
df['ym'] = df['DateTime'].dt.strftime('%Y-%m')
df = df.sort_values(['Query_ID']).reset_index(drop=True)  # contiguous groups
months = sorted(df['ym'].unique())
print(f"  {len(df):,} rows | {len(months)} months {months[0]}..{months[-1]}")

X = df[feats].values.astype(np.float32)
ret = df['Next_Hour_Return'].values
qid = df['Query_ID'].values

print("Computing rank labels ...")
long_lab = df.groupby('Query_ID')['Next_Hour_Return'].rank(method='first', ascending=True).values - 1
short_lab = df.groupby('Query_ID')['Next_Hour_Return'].rank(method='first', ascending=False).values - 1

tmin, tmax = pd.to_datetime('10:15').time(), pd.to_datetime('14:15').time()
tw = (df['DateTime'].dt.time >= tmin) & (df['DateTime'].dt.time <= tmax)
tw = tw.values

from scipy.stats import spearmanr
def rank_ic(scores, y, groups_idx, invert=False):
    rhos = []
    for idx in groups_idx:
        if len(idx) < 2: continue
        yy = -y[idx] if invert else y[idx]
        r, _ = spearmanr(scores[idx], yy)
        if not np.isnan(r): rhos.append(r)
    return float(np.mean(rhos)) if rhos else 0.0

def group_sizes(mask):
    q = qid[mask]
    _, first_idx, counts = np.unique(q, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    return counts[order]

first_test = months.index('2024-01')
folds = []
s = first_test
while s < len(months):
    tr = months[:s-1]; val = [months[s-1]]; te = months[s:s+3]
    if len(te) == 0: break
    folds.append((len(folds)+1, tr, val, te))
    s += 3
print(f"{len(folds)} folds")

out_parts, fold_meta = [], []
for fold, tr_m, val_m, te_m in folds:
    t0 = time.time()
    trm = df['ym'].isin(tr_m).values
    vam = df['ym'].isin(val_m).values
    tem = df['ym'].isin(te_m).values
    
    fill = np.nanmean(np.where(np.isinf(X[trm]), np.nan, X[trm]), axis=0)
    fill = np.nan_to_num(fill)
    def prep(mask):
        Xm = X[mask].copy(); bad = ~np.isfinite(Xm)
        Xm[bad] = np.take(fill, np.where(bad)[1])
        return Xm
    Xtr, Xva, Xte = prep(trm), prep(vam), prep(tem)
    gtr, gva = group_sizes(trm), group_sizes(vam)

    dtl = xgb.DMatrix(Xtr, label=long_lab[trm]); dtl.set_group(gtr)
    dvl = xgb.DMatrix(Xva, label=long_lab[vam]); dvl.set_group(gva)
    bl = xgb.train(PARAMS, dtl, num_boost_round=500, evals=[(dvl, 'v')],
                   early_stopping_rounds=50, verbose_eval=False)
    
    dts = xgb.DMatrix(Xtr, label=short_lab[trm]); dts.set_group(gtr)
    dvs = xgb.DMatrix(Xva, label=short_lab[vam]); dvs.set_group(gva)
    bs = xgb.train(PARAMS, dts, num_boost_round=500, evals=[(dvs, 'v')],
                   early_stopping_rounds=50, verbose_eval=False)

    dte = xgb.DMatrix(Xte)
    ls_te = bl.predict(dte); ss_te = bs.predict(dte)
    
    ss_tr = bs.predict(xgb.DMatrix(prep(trm & tw)))
    ss_thr = float(np.quantile(ss_tr, SS_QUANTILE))

    sub = df.loc[tem, ['DateTime', 'Ticker', 'Query_ID', 'Next_Hour_Return']].copy()
    sub['ss'] = ss_te; sub['ls'] = ls_te; sub['fold'] = fold; sub['ss_thr'] = ss_thr
    out_parts.append(sub)

    te_idx = np.where(tem & tw)[0]
    gmap = {}
    for i in te_idx: gmap.setdefault(qid[i], []).append(i)
    gl = [np.array(v) for v in gmap.values()]
    
    ss_full = np.full(len(df), np.nan); ls_full = np.full(len(df), np.nan)
    ss_full[np.where(tem)[0]] = ss_te; ls_full[np.where(tem)[0]] = ls_te
    l_ic = rank_ic(ls_full, ret, gl, invert=False)
    s_ic = rank_ic(ss_full, ret, gl, invert=True)
    
    fold_meta.append(dict(fold=fold, train=f"{tr_m[0]}..{tr_m[-1]}", test=f"{te_m[0]}..{te_m[-1]}",
                          n_test=int(tem.sum()), long_ic=l_ic, short_ic=s_ic, ss_thr=ss_thr,
                          best_iter_long=bl.best_iteration, best_iter_short=bs.best_iteration))
    print(f"F{fold} {tr_m[0]}..{tr_m[-1]}->{te_m[0]}..{te_m[-1]} | "
          f"L-IC {l_ic:+.4f} S-IC {s_ic:+.4f} | ss_thr {ss_thr:.4f} | {time.time()-t0:.0f}s")

oos = pd.concat(out_parts, ignore_index=True)
oos.to_parquet(f'{OUT}/oos_scored.parquet')
json.dump(fold_meta, open(f'{OUT}/fold_meta.json', 'w'), indent=2)
print(f"\nSaved {len(oos):,} OOS scored rows -> {OUT}/oos_scored.parquet")
print(f"OOS span {oos['DateTime'].min()} .. {oos['DateTime'].max()}")
