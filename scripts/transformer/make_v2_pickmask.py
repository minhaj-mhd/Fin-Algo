"""
Generate daily_macro_v2's GENUINE out-of-sample (purged walk-forward) rank scores, aligned to the
daily transformer panel grid, and the Top-K pick masks the veto gate will operate on.

Reproduces v2's EXACT 4-fold walk-forward (scripts/training/train_daily_xgboost_v2.py): expanding
train, 6-month val, 6-month test; rank:pairwise; same params + integer-rank labels + global mean fill.
For each fold we PREDICT on the test months only -> those scores are honestly OOS (the model never saw
the test days; train+val precede test by >=6 months). The 4 test windows tile the last ~24 months.

CRITICAL (lookahead guard, cf. v8 leakage / G6): we NEVER score the production artifact in-sample.
Every stored score comes from a fold whose train+val months strictly precede the scored day; asserted.

Outputs (data/daily_transformer_panel/):
  v2_long_score.npy (T,N)  v2_short_score.npy (T,N)  v2_oos_mask.npy (T,)  bool days that got OOS scores
  v2_pickmask_long.npy (T,N) bool  v2_pickmask_short.npy (T,N) bool   (Top-K per OOS day)
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

DATA_FILE = 'data/ranking_data_daily_macro_v2.csv'
P = 'data/daily_transformer_panel'
TOPK = 5                      # candidate set per side the gate operates on (v2 targets top-5)
EXCLUDE = ['DateTime', 'Query_ID', 'Ticker', 'Open', 'High', 'Low', 'Close',
           'Volume', 'Label_3D', 'Sector', 'YearMonth']

PARAMS = {  # verbatim from train_daily_xgboost_v2.py
    'objective': 'rank:pairwise', 'eta': 0.01, 'max_depth': 5, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'alpha': 2.0, 'lambda': 4.0, 'min_child_weight': 40,
    'random_state': 42, 'verbosity': 0, 'eval_metric': 'ndcg@5', 'ndcg_exp_gain': False,
    'tree_method': 'hist',
}


def get_integer_ranks(y_vals, qids, invert=False):
    y_int = np.zeros_like(y_vals, dtype=int)
    for qid in np.unique(qids):
        mask = qids == qid
        vals = -y_vals[mask] if invert else y_vals[mask]
        y_int[mask] = rankdata(vals, method='ordinal') - 1
    return y_int


def detect_device():
    try:
        dd = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); dd.set_group([10])
        xgb.train({**PARAMS, 'device': 'cuda'}, dd, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


def main():
    meta = json.load(open(f'{P}/meta.json'))
    tickers = meta['tickers']; tmap = {t: i for i, t in enumerate(tickers)}
    ts_days = np.load(f'{P}/ts_days.npy')
    daymap = {int(d): i for i, d in enumerate(ts_days)}
    T, N = meta['n_days'], meta['n_tickers']

    df = pd.read_csv(DATA_FILE)
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.normalize()
    df['YearMonth'] = df['DateTime'].dt.strftime('%Y-%m')
    df['day_idx'] = pd.Series(df['DateTime'].values.astype('datetime64[ns]').astype('int64')).map(daymap)
    df['tk_idx'] = df['Ticker'].map(tmap)
    df = df.dropna(subset=['day_idx', 'tk_idx'])
    df['day_idx'] = df['day_idx'].astype(int); df['tk_idx'] = df['tk_idx'].astype(int)

    feature_cols = [c for c in df.columns if c not in EXCLUDE + ['day_idx', 'tk_idx']]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    # global mean fill of NaN/Inf -- verbatim with v2 (benign; reproduces certified picks)
    bad = ~np.isfinite(X)
    if bad.any():
        for j in range(X.shape[1]):
            col = X[:, j]; ok = col[np.isfinite(col)]
            if len(ok):
                col[~np.isfinite(col)] = ok.mean(); X[:, j] = col
    y = df['Label_3D'].to_numpy()
    qid = df['Query_ID'].to_numpy()
    ym = df['YearMonth'].to_numpy()

    unique_months = sorted(df['YearMonth'].unique())
    val_size = test_size = 6
    folds = []
    for k in range(1, 5):
        te_end = len(unique_months) - (4 - k) * test_size
        te_start = te_end - test_size
        va_start = te_start - val_size
        folds.append({'fold': k, 'train': unique_months[:va_start],
                      'val': unique_months[va_start:te_start], 'test': unique_months[te_start:te_end]})

    device = detect_device()
    print(f"device={device}  folds={[(f['test'][0], f['test'][-1]) for f in folds]}")

    long_score = np.full((T, N), np.nan, dtype=np.float32)
    short_score = np.full((T, N), np.nan, dtype=np.float32)
    oos_mask = np.zeros(T, dtype=bool)

    for f in folds:
        tr = np.isin(ym, f['train']); va = np.isin(ym, f['val']); te = np.isin(ym, f['test'])
        # lookahead guard: latest train+val day strictly precedes earliest test day
        assert df['DateTime'][tr | va].max() < df['DateTime'][te].min(), "OOS VIOLATION"
        grp_tr = pd.Series(qid[tr]).groupby(qid[tr]).size().values
        grp_va = pd.Series(qid[va]).groupby(qid[va]).size().values
        dte = xgb.DMatrix(X[te])
        scores = {}
        for side, inv in [('long', False), ('short', True)]:
            dtr = xgb.DMatrix(X[tr], label=get_integer_ranks(y[tr], qid[tr], invert=inv)); dtr.set_group(grp_tr)
            dva = xgb.DMatrix(X[va], label=get_integer_ranks(y[va], qid[va], invert=inv)); dva.set_group(grp_va)
            bst = xgb.train({**PARAMS, 'device': device}, dtr, num_boost_round=1500,
                            evals=[(dtr, 'train'), (dva, 'val')], early_stopping_rounds=150,
                            verbose_eval=False)
            scores[side] = bst.predict(dte)
            print(f"  fold{f['fold']} {side}: best_iter={bst.best_iteration} test_rows={te.sum():,}")
        di = df['day_idx'].to_numpy()[te]; ni = df['tk_idx'].to_numpy()[te]
        long_score[di, ni] = scores['long'].astype(np.float32)
        short_score[di, ni] = scores['short'].astype(np.float32)
        oos_mask[np.unique(di)] = True

    # Top-K pick masks per OOS day (high score = strong pick on that side)
    pml = np.zeros((T, N), dtype=bool); pms = np.zeros((T, N), dtype=bool)
    for t in np.where(oos_mask)[0]:
        for sc, pm in [(long_score[t], pml), (short_score[t], pms)]:
            present = np.isfinite(sc)
            idx = np.where(present)[0]
            if len(idx) < TOPK + 1:
                continue
            top = idx[np.argsort(-sc[idx])[:TOPK]]
            pm[t, top] = True

    for name, arr in [('v2_long_score', long_score), ('v2_short_score', short_score),
                      ('v2_oos_mask', oos_mask), ('v2_pickmask_long', pml), ('v2_pickmask_short', pms)]:
        np.save(f'{P}/{name}.npy', arr)

    nd = int(oos_mask.sum())
    span = (pd.Timestamp(int(ts_days[np.where(oos_mask)[0][0]])).date(),
            pd.Timestamp(int(ts_days[np.where(oos_mask)[0][-1]])).date())
    print(f"OOS days={nd}  span {span[0]}..{span[1]}  "
          f"long picks={int(pml.sum()):,} short picks={int(pms.sum()):,}")
    # sanity: v2 OOS long top-5 raw 3d edge vs day mean (gross, no cost), in bps
    Y = np.load(f'{P}/Y_3d.npy')
    le, se = [], []
    for t in np.where(oos_mask)[0]:
        lp = np.where(pml[t])[0]; sp = np.where(pms[t])[0]
        if len(lp): le.append(np.nanmean(Y[t, lp]))
        if len(sp): se.append(-np.nanmean(Y[t, sp]))
    print(f"v2 OOS top-{TOPK} gross 3d edge: long {np.nanmean(le)*1e4:+.1f}bps  "
          f"short {np.nanmean(se)*1e4:+.1f}bps  (per 3-day trade, pre-cost)")


if __name__ == '__main__':
    main()
