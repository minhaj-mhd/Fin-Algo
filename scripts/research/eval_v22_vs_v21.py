"""
eval_v22_vs_v21.py

Leave-one-group-out ablation for V22:
- v21-dynamic : v21 dynamic panel baseline
- v22-full    : full v22 dynamic panel (all feature groups)
- v22-M       : v22 minus Macro (M_) features
- v22-N       : v22 minus News (N_) features
- v22-T       : v22 minus Technical/Time (T_) features
- v22-R       : v22 minus Regime (R_) features
- v22-V       : v22 minus Volatility (V_) features
- v22_shuffle : v22 with Next_Hour_Return shuffled WITHIN each query (Negative Control)

Outputs:
  per-query spearman rho and Top-3 net bps at 10bps cost (COST=0.001)
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.parquet as pq
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

V21_PANEL = 'data/research/v21_rolling_1h_dynamic/panel.parquet'
V22_PANEL = 'data/research/v22_rolling_1h_dynamic/panel.parquet'
RET = 'Next_Hour_Return'
COST = 0.001
SEED = 42
META_EXCLUDE = {'DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET, 'YearMonth'}

PARAMS = {'objective': 'rank:pairwise', 'eta': 0.03, 'max_depth': 5, 'subsample': 0.8,
          'colsample_bytree': 0.8, 'alpha': 1.0, 'lambda': 2.0, 'min_child_weight': 10,
          'random_state': SEED, 'verbosity': 0, 'eval_metric': 'ndcg@3', 'ndcg_exp_gain': False,
          'tree_method': 'hist', 'device': 'cuda'}

def _gpu():
    try:
        d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'

PARAMS['device'] = _gpu()

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

import gc

def wf_eval(df, feat_cols, shuffle=False, label='?'):
    # Avoid deep copy to save memory, just work with the views where possible
    X = df[feat_cols].values.astype(np.float32) # float32 instead of float64!
    if not np.isfinite(X).all():
        col_means = np.nan_to_num(np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0))
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(col_means, idx[1])
    y = df[RET].values.copy()
    q = df['Query_ID'].values
    if shuffle:
        rng = np.random.default_rng(SEED)
        for qi in np.unique(q):
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
        
        lr = [spearmanr(g['_score'], g[RET])[0] for _, g in dte.groupby('Query_ID') if len(g) > 1]
        sr = [spearmanr(g['_sscore'], -g[RET])[0] for _, g in dte.groupby('Query_ID') if len(g) > 1]
        Lr.append(np.nanmean(lr)); Sr.append(np.nanmean(sr))
        Le.append(top3_net(dte, '_score', short=False)); Se.append(top3_net(dte, '_sscore', short=True))
        
        # Cleanup Memory
        del dl, dvl, bl, ds, dvs, bs, dte
        gc.collect()
        
    print(f"  {label:18s} L_rho {np.mean(Lr):+.4f}  S_rho {np.mean(Sr):+.4f}  "
          f"L_top3net {np.mean(Le):+.2f}bps  S_top3net {np.mean(Se):+.2f}bps  (folds={len(Lr)})")
    del X, y, q
    gc.collect()
    return dict(L_rho=np.mean(Lr), S_rho=np.mean(Sr), L_net=np.mean(Le), S_net=np.mean(Se))

def main():
    print(f"device={PARAMS['device']}")
    
    # Load V21
    if os.path.exists(V21_PANEL):
        v21 = pd.read_parquet(V21_PANEL)
        v21_feats = [c for c in v21.columns if c not in META_EXCLUDE]
        print(f"v21-dynamic rows={len(v21):,} feats={len(v21_feats)}")
    else:
        v21, v21_feats = None, []
        print(f"v21-dynamic not found at {V21_PANEL}")

    # Load V22
    if os.path.exists(V22_PANEL):
        v22 = pd.read_parquet(V22_PANEL)
        v22_feats = [c for c in v22.columns if c not in META_EXCLUDE]
        print(f"v22-full rows={len(v22):,} feats={len(v22_feats)}")
    else:
        print(f"v22-full not found at {V22_PANEL}. Agent 1 might still be building it. Exiting.")
        return

    # Define ablation groups
    # Assuming standard prefixes M_, N_, T_, R_, V_
    feat_groups = {
        'M': ['Breadth_Pct_Positive', 'Breadth_Pct_Above_VWAP', 'Breadth_AD_Ratio', 'Breadth_Pct_NewHigh', 'Breadth_Median_Return', 'Disp_CrossSec_Std', 'Disp_CrossSec_MAD'],
        'N': ['Macro_Nifty_1H', 'Macro_Nifty_RealVol', 'Macro_Nifty_ATR_Pct', 'Macro_Nifty_Gap', 'Macro_India_VIX'],
        'T': ['Time_Sin', 'Time_Cos'],
        'R': ['RelStr_Nifty_1H', 'RelStr_Nifty_2H', 'Beta_Nifty', 'Resid_Return_Nifty'],
        'V': ['VWAP_Slope', 'VWAP_Zscore', 'Price_VWAP_Ratio'],
    }
    
    res = {}
    print("\n=== Purged monthly walk-forward ablation (v21 vs v22) ===")
    
    if v21 is not None:
        res['v21-dynamic'] = wf_eval(v21, v21_feats, label='v21-dynamic')
    
    res['v22-full'] = wf_eval(v22, v22_feats, label='v22-full')
    
    for grp, f_list in feat_groups.items():
        if len(f_list) > 0:
            f_ablated = [c for c in v22_feats if c not in f_list]
            res[f'v22-{grp}'] = wf_eval(v22, f_ablated, label=f'v22-{grp}')
        else:
            print(f"  v22-{grp:14s} Skipped (no features with prefix {grp}_ found)")
            
    res['v22_shuffle'] = wf_eval(v22, v22_feats, shuffle=True, label='v22_shuffle(NEG)')
    
    out_json = 'data/research/v22_rolling_1h_dynamic/eval_summary.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(res, open(out_json, 'w'), indent=2, default=float)
    print(f"\nsaved {out_json}")

if __name__ == '__main__':
    main()
