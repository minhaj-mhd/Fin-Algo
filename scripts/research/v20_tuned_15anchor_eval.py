"""
:15 non-overlapping re-check of the Optuna-tuned v20 recipe vs the baseline v20 recipe.

Filters the rolling panel to the 5 NON-overlapping v10-cadence moments {10:15..14:15} (the
authoritative Gauntlet grid — honest significance, no overlap inflation), then runs the SAME
8-fold walk-forward for BOTH param sets and reports, side by side:
  * walk-forward rank-IC (long / short)
  * net-of-cost Top-1/3/5 (long & short) @ 6 and 10 bps  ← the question that matters

Exploratory only — no Gauntlet verdict, no registry stamp, certified v20 untouched.
"""
import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DATA = 'data/research/v20_rolling_1h/panel.parquet'
RET_COL = 'Next_Hour_Return'
ANCHORS = {'10:15', '11:15', '12:15', '13:15', '14:15'}
SEED = 42
NUM_ROUNDS, ES_ROUNDS = 500, 50
COSTS, KS = [6.0, 10.0], [1, 3, 5]

BASELINE = dict(objective='rank:pairwise', eta=0.03, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, alpha=1.0, reg_lambda=2.0, min_child_weight=10, gamma=0.0)
TUNED = json.load(open('artifacts/optuna_v20_best.json'))['best_params']


def device_ok():
    try:
        d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


def group_blocks(qids):
    _, first = np.unique(qids, return_index=True)
    starts = np.sort(first); ends = np.append(starts[1:], len(qids))
    sizes = (ends - starts).astype(int)
    return sizes, [np.arange(s, e) for s, e in zip(starts, ends)]


def integer_ranks(y, qids, invert):
    out = np.zeros(len(y), dtype=int)
    _, first = np.unique(qids, return_index=True)
    starts = np.sort(first); ends = np.append(starts[1:], len(qids))
    for s, e in zip(starts, ends):
        vals = -y[s:e] if invert else y[s:e]
        out[s:e] = rankdata(vals, method='ordinal') - 1
    return out


def mean_spearman(scores, ret, qgroups):
    rhos = []
    for idx in qgroups:
        if len(idx) < 2:
            continue
        a = rankdata(scores[idx]); b = rankdata(ret[idx])
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            rhos.append(float((a * b).sum() / den))
    return float(np.mean(rhos)) if rhos else 0.0


def fixed(params, device):
    p = dict(params)
    p.update(eval_metric='ndcg@3', ndcg_exp_gain=False, tree_method='hist',
             device=device, verbosity=0, random_state=SEED)
    return p


# ── load + filter to :15 anchors ─────────────────────────────────────────────
print("Loading panel, filtering to :15 anchors …")
df = pd.read_parquet(DATA)
dt = pd.to_datetime(df['DateTime'])
df = df[dt.dt.strftime('%H:%M').isin(ANCHORS)].copy()
df = df.sort_values('DateTime').reset_index(drop=True)
df['Query_ID'] = df.groupby('DateTime').ngroup()          # contiguous re-number
df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
months = sorted(df['YearMonth'].unique())
exclude = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
           'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']
feat_cols = [c for c in df.columns if c not in exclude]
X = df[feat_cols].to_numpy(np.float32); np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
y = df[RET_COL].to_numpy(np.float64); qids = df['Query_ID'].to_numpy(); ym = df['YearMonth'].to_numpy()
y_long = integer_ranks(y, qids, False); y_short = integer_ranks(y, qids, True)
qpd = df.groupby(pd.to_datetime(df['DateTime']).dt.date)['Query_ID'].nunique().mean()
print(f"  :15 subset: {len(df):,} rows, {df['Query_ID'].nunique():,} queries, {len(months)} months, {qpd:.1f} entries/day")

MIN_TRAIN, HORIZON = 18, 2
folds = []
for i in range(MIN_TRAIN, len(months) - HORIZON, 4):
    tr_m, va_m, te_m = months[:i], [months[i]], months[i + 1:i + HORIZON + 1]
    trm = np.isin(ym, tr_m); vam = np.isin(ym, va_m); tem = np.isin(ym, te_m)
    gtr, _ = group_blocks(qids[trm]); gva, _ = group_blocks(qids[vam]); _, teg = group_blocks(qids[tem])
    folds.append(dict(Xtr=np.ascontiguousarray(X[trm]), ytr_l=y_long[trm], ytr_s=y_short[trm], gtr=gtr,
                      Xva=np.ascontiguousarray(X[vam]), yva_l=y_long[vam], yva_s=y_short[vam], gva=gva,
                      Xte=np.ascontiguousarray(X[tem]), te_ret=y[tem], teg=teg))
DEVICE = device_ok()
print(f"  folds={len(folds)} device={DEVICE}")


def eval_fold(ls, ss, ret, qgroups):
    out = {'long_rho': mean_spearman(ls, ret, qgroups), 'short_rho': mean_spearman(ss, -ret, qgroups)}
    g = {k: {'l': [], 's': []} for k in KS}
    for idx in qgroups:
        if len(idx) < max(KS) + 1:
            continue
        r = ret[idx]; lo = np.argsort(-ls[idx]); so = np.argsort(-ss[idx])
        for k in KS:
            g[k]['l'].append(r[lo[:k]].mean()); g[k]['s'].append((-r[so[:k]]).mean())
    for k in KS:
        out[f'l_g{k}'] = float(np.mean(g[k]['l'])); out[f's_g{k}'] = float(np.mean(g[k]['s']))
    return out


def run(params, tag):
    p = fixed(params, DEVICE)
    rows = []
    for F in folds:
        dtl = xgb.DMatrix(F['Xtr'], label=F['ytr_l']); dtl.set_group(F['gtr'])
        dvl = xgb.DMatrix(F['Xva'], label=F['yva_l']); dvl.set_group(F['gva'])
        bl = xgb.train(p, dtl, NUM_ROUNDS, evals=[(dvl, 'v')], early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
        dts = xgb.DMatrix(F['Xtr'], label=F['ytr_s']); dts.set_group(F['gtr'])
        dvs = xgb.DMatrix(F['Xva'], label=F['yva_s']); dvs.set_group(F['gva'])
        bs = xgb.train(p, dts, NUM_ROUNDS, evals=[(dvs, 'v')], early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
        dte = xgb.DMatrix(F['Xte'])
        ls = bl.predict(dte, iteration_range=(0, bl.best_iteration + 1))
        ss = bs.predict(dte, iteration_range=(0, bs.best_iteration + 1))
        rows.append(eval_fold(ls, ss, F['te_ret'], F['teg']))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"\n[{tag}] WF rank-IC: long={agg['long_rho']:+.4f}  short={agg['short_rho']:+.4f}")
    for k in KS:
        lg, sg = agg[f'l_g{k}'] * 1e4, agg[f's_g{k}'] * 1e4
        for c in COSTS:
            print(f"   K={k} @{int(c)}bps: LONG net={lg - c:+.2f}bps (gross {lg:+.2f})  "
                  f"SHORT net={sg - c:+.2f}bps (gross {sg:+.2f})")
    return agg


print("\n" + "=" * 74 + f"\n:15 NON-OVERLAPPING RE-CHECK  (baseline vs tuned, {len(folds)} folds)\n" + "=" * 74)
print(f"TUNED params: {TUNED}")
base = run(BASELINE, 'BASELINE')
tune = run(TUNED, 'TUNED')

print("\n" + "=" * 74 + "\nLIFT (tuned − baseline)\n" + "=" * 74)
print(f"  rank-IC: long {tune['long_rho']-base['long_rho']:+.4f}  short {tune['short_rho']-base['short_rho']:+.4f}")
for k in KS:
    for c in COSTS:
        dl = (tune[f'l_g{k}'] - base[f'l_g{k}']) * 1e4
        ds = (tune[f's_g{k}'] - base[f's_g{k}']) * 1e4
        print(f"  K={k} @{int(c)}bps net Δ: LONG {dl:+.2f}bps  SHORT {ds:+.2f}bps  "
              f"(tuned LONG net {tune[f'l_g{k}']*1e4-c:+.2f}, SHORT net {tune[f's_g{k}']*1e4-c:+.2f})")

out = {'baseline': base, 'tuned': tune, 'tuned_params': TUNED, 'n_folds': len(folds),
       'subset': ':15 anchors', 'costs': COSTS}
json.dump(out, open('artifacts/v20_tuned_15anchor_eval.json', 'w'), indent=2, default=float)
print("\nsaved -> artifacts/v20_tuned_15anchor_eval.json")
