"""
Phase 5 eval — did v21's cleaning + tweaks add REAL signal, or just noise?

CONTROLLED, SAME-UNIVERSE, SAME-FOLDS purged monthly walk-forward (rank:pairwise ranker,
per-query Spearman rho + Top-3 net edge @10bps), so the comparison isolates the RECIPE
(not the universe change). All variants run on the v21 liquidity universe:

  v20@110     : v20 recipe (mean/std z, 0.0/0.5 fills, bar-count lookback, no graph/gap) on the
                v21 universe — the honest baseline.
  v21         : full v21 panel from disk (clean + tweaks + graph + gap features).
  v21_nograph : v21 minus the nb_* neighbor features (isolates the sector-graph tweak).
  v21_shuffle : v21 with Next_Hour_Return shuffled WITHIN each query — NEGATIVE CONTROL; rho
                must collapse to ~0 or the whole panel is leaking.

RESEARCH ONLY (AGENTS.md): overlapping windows -> effective N ~1/4 rows; point estimates, NO
t-tests / significance. No Gauntlet, no verdict authority.

Run: python scripts/research/eval_v21_vs_v20.py
"""
import os, sys, json, glob, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

import scripts.research.build_rolling_1h_panel as V20
from scripts.research.build_v21_rolling_1h_panel import _load_raw, build_ticker as build_ticker_v21

V21_PANEL = 'data/research/v21_rolling_1h/panel.parquet'
UNIV_JSON = 'data/research/v21_rolling_1h/universe.json'
SRC_DIR   = 'data/raw_upstox_cache_15min_3y'
RET = 'Next_Hour_Return'
COST = 0.001
SEED = 42
META_EXCLUDE = {'DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET, 'YearMonth'}

PARAMS = {'objective': 'rank:pairwise', 'eta': 0.03, 'max_depth': 5, 'subsample': 0.8,
          'colsample_bytree': 0.8, 'alpha': 1.0, 'lambda': 2.0, 'min_child_weight': 10,
          'random_state': SEED, 'verbosity': 0, 'eval_metric': 'ndcg@3', 'ndcg_exp_gain': False,
          'tree_method': 'hist', 'device': 'cpu'}


def _gpu():
    try:
        d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


def int_ranks(y, q, invert=False):
    out = np.zeros_like(y, dtype=int)
    for qi in np.unique(q):
        m = q == qi
        out[m] = rankdata(-y[m] if invert else y[m], method='ordinal') - 1
    return out


def folds_of(df):
    months = sorted(pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m').unique())
    ym = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m').values
    F = []
    for i in range(18, len(months) - 2, 4):
        F.append((np.isin(ym, months[:i]), np.isin(ym, [months[i]]),
                  np.isin(ym, months[i + 1:i + 3])))
    return F


def top3_net(df_te, col, short=False):
    vals = []
    for qi in df_te['Query_ID'].unique():
        q = df_te[df_te['Query_ID'] == qi]
        if len(q) < 4:
            continue
        ar = q[RET].values
        pick = ar[np.argsort(q[col].values)[::-1][:3]]
        vals.append((-pick.mean() if short else pick.mean()))
    g = float(np.mean(vals)) if vals else 0.0
    return (g - COST) * 1e4   # net bps


def wf_eval(df, feat_cols, shuffle=False, label='?'):
    df = df.reset_index(drop=True).copy()
    X = df[feat_cols].values.astype(np.float64)
    if not np.isfinite(X).all():                       # column-mean fill (mirrors train_ranking_clean)
        col_means = np.nan_to_num(np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0))
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(col_means, idx[1])
    y = df[RET].values.copy()
    q = df['Query_ID'].values
    if shuffle:
        rng = np.random.default_rng(SEED)
        for qi in np.unique(q):                       # shuffle labels within each query
            m = np.where(q == qi)[0]
            y[m] = y[m][rng.permutation(len(m))]
    Lr, Sr, Le, Se = [], [], [], []
    for tr, va, te in folds_of(df):
        gtr = pd.Series(q[tr]).groupby(q[tr]).size().values
        gva = pd.Series(q[va]).groupby(q[va]).size().values
        dl = xgb.DMatrix(X[tr], label=int_ranks(y[tr], q[tr], False)); dl.set_group(gtr)
        dvl = xgb.DMatrix(X[va], label=int_ranks(y[va], q[va], False)); dvl.set_group(gva)
        bl = xgb.train(PARAMS, dl, 500, evals=[(dvl, 'v')], early_stopping_rounds=50, verbose_eval=False)
        ds = xgb.DMatrix(X[tr], label=int_ranks(y[tr], q[tr], True)); ds.set_group(gtr)
        dvs = xgb.DMatrix(X[va], label=int_ranks(y[va], q[va], True)); dvs.set_group(gva)
        bs = xgb.train(PARAMS, ds, 500, evals=[(dvs, 'v')], early_stopping_rounds=50, verbose_eval=False)
        dte = df[te].copy()
        dte['_score'] = bl.predict(xgb.DMatrix(X[te]))
        dte['_sscore'] = bs.predict(xgb.DMatrix(X[te]))
        # rho
        lr = [spearmanr(g['_score'], g[RET])[0] for _, g in dte.groupby('Query_ID') if len(g) > 1]
        sr = [spearmanr(g['_sscore'], -g[RET])[0] for _, g in dte.groupby('Query_ID') if len(g) > 1]
        Lr.append(np.nanmean(lr)); Sr.append(np.nanmean(sr))
        Le.append(top3_net(dte, '_score', short=False)); Se.append(top3_net(dte, '_sscore', short=True))
    print(f"  {label:14s} L_rho {np.mean(Lr):+.4f}  S_rho {np.mean(Sr):+.4f}  "
          f"L_top3net {np.mean(Le):+.2f}bps  S_top3net {np.mean(Se):+.2f}bps  (folds={len(Lr)})")
    return dict(L_rho=np.mean(Lr), S_rho=np.mean(Sr), L_net=np.mean(Le), S_net=np.mean(Se))


def build_v20_at_universe(universe):
    """v20 recipe on the v21 universe (raw cache -> v20 build_ticker -> v20 mean/std build_ranking)."""
    frames = []
    for tk in universe:
        fp = os.path.join(SRC_DIR, tk + '.csv')
        if not os.path.exists(fp):
            continue
        f = V20.build_ticker(tk, pd.read_csv(fp))
        if f is not None and len(f):
            frames.append(f)
    final, fc = V20.build_ranking(pd.concat(frames, ignore_index=True))
    return final, fc


def main():
    PARAMS['device'] = _gpu()
    print(f"device={PARAMS['device']}")
    universe = json.load(open(UNIV_JSON))['tickers']
    print(f"universe: {len(universe)} tickers")

    v21 = pd.read_parquet(V21_PANEL)
    v21_feats = [c for c in v21.columns if c not in META_EXCLUDE]
    nb = [c for c in v21_feats if c.startswith('nb_')]
    print(f"v21 rows={len(v21):,} queries={v21['Query_ID'].nunique():,} feats={len(v21_feats)} (nb_={len(nb)})")

    print("\nBuilding v20@110 (same universe, v20 recipe)...")
    v20, v20_feats = build_v20_at_universe(universe)
    print(f"v20@110 rows={len(v20):,} queries={v20['Query_ID'].nunique():,} feats={len(v20_feats)}")

    print("\n=== Purged monthly walk-forward (research; overlapping => point estimates only) ===")
    res = {}
    res['v20@110']     = wf_eval(v20, v20_feats, label='v20@110')
    res['v21']         = wf_eval(v21, v21_feats, label='v21')
    res['v21_nograph'] = wf_eval(v21, [c for c in v21_feats if c not in nb], label='v21_nograph')
    res['v21_shuffle'] = wf_eval(v21, v21_feats, shuffle=True, label='v21_shuffle(NEG)')

    print("\n=== READ ===")
    print(f"  cleaning effect (v21 - v20@110): L_rho {res['v21']['L_rho']-res['v20@110']['L_rho']:+.4f}  "
          f"S_rho {res['v21']['S_rho']-res['v20@110']['S_rho']:+.4f}")
    print(f"  graph effect   (v21 - v21_nograph): L_rho {res['v21']['L_rho']-res['v21_nograph']['L_rho']:+.4f}  "
          f"S_rho {res['v21']['S_rho']-res['v21_nograph']['S_rho']:+.4f}")
    print(f"  NEG control v21_shuffle L_rho {res['v21_shuffle']['L_rho']:+.4f} S_rho {res['v21_shuffle']['S_rho']:+.4f} "
          "(must be ~0; else leak)")
    json.dump(res, open('data/research/v21_rolling_1h/eval_summary.json', 'w'), indent=2, default=float)
    print("\nsaved data/research/v21_rolling_1h/eval_summary.json")


if __name__ == '__main__':
    main()
