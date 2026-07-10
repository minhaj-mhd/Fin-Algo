"""
2-HOUR head-to-head: v20/v21 XGBoost ranker vs DualRes transformer, SAME chrono test window +
SAME 110 universe + SAME 2h horizon label. Mirror of benchmark_v21_xgb_vs_transformer.py but on the
2h label (Y_ret_2h / Next_2Hour_Return), so it answers "is a TRANSFORMER 2h model worth it vs a tree
2h model" on identical ground.

Pre-registered bar: the transformer must beat XGBoost test long rho MATERIALLY (and short), AND
survive the shuffle neg-control (train.py --shuffle_labels). Miss either => the transformer doesn't
help; stop. RESEARCH ONLY (no Gauntlet, no verdict authority).

Run: TRANSFORMER_PANEL=data/transformer_panel_v21 TRANSFORMER_LABEL=Y_ret_2h \
     python scripts/transformer/benchmark_2h.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb                         # pandas/numpy/xgb imported BEFORE torch (Windows MKL clash)
from scipy.stats import spearmanr, rankdata
sys.path.append(os.getcwd())
os.environ.setdefault('TRANSFORMER_PANEL', 'data/transformer_panel_v21')
os.environ.setdefault('TRANSFORMER_LABEL', 'Y_ret_2h')
from scripts.transformer.train import valid_decision_timestamps, chrono_split, EMBARGO, LABEL  # torch import here

H = LABEL.replace('Y_ret_', '').replace('h', '')   # 'Y_ret_3h' -> '3'
TPANEL    = os.environ['TRANSFORMER_PANEL']
XGB_PANEL = f'data/research/v21_rolling_1h/panel_{H}h.parquet'
RET = f'Next_{H}Hour_Return'
META_EXCLUDE = {'DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker', 'Open', 'High',
                'Low', 'Close', 'Volume', 'Next_Hour_Return', RET, 'YearMonth'}
PARAMS = {'objective': 'rank:pairwise', 'eta': 0.03, 'max_depth': 5, 'subsample': 0.8,
          'colsample_bytree': 0.8, 'alpha': 1.0, 'lambda': 2.0, 'min_child_weight': 10,
          'random_state': 42, 'verbosity': 0, 'eval_metric': 'ndcg@3', 'ndcg_exp_gain': False,
          'tree_method': 'hist', 'device': 'cpu'}


def _gpu():
    try:
        m = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); m.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, m, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


def int_ranks(y, q, invert=False):
    out = np.zeros_like(y, dtype=int)
    for qi in np.unique(q):
        m = q == qi
        out[m] = rankdata(-y[m] if invert else y[m], method='ordinal') - 1
    return out


def transformer_test_window():
    """Replicate train.py's test split on the tensor panel (using the 2h label) -> (lo, hi)."""
    d = {'Y_ret': np.load(f'{TPANEL}/{LABEL}.npy'),
         'end15': np.load(f'{TPANEL}/end15.npy'),
         'date_idx': np.load(f'{TPANEL}/date_idx.npy'),
         'X_1h': np.load(f'{TPANEL}/X_1h.npy', mmap_mode='r')}
    ts1 = np.load(f'{TPANEL}/ts_1h.npy')
    ts = valid_decision_timestamps(d)
    _, _, te = chrono_split(ts, EMBARGO)
    test_ts = ts1[te]
    return pd.Timestamp(int(test_ts.min())), pd.Timestamp(int(test_ts.max()))


def train_xgb_side(tr, va, te, feats, invert):
    def dm(df, label=True):
        df = df.sort_values('DateTime')
        g = df.groupby('Query_ID').size().values
        d = xgb.DMatrix(df[feats].values.astype(np.float64))
        if label:
            d.set_group(g)
            d.set_label(int_ranks(df[RET].values, df['Query_ID'].values, invert))
        return d
    dtr, dva = dm(tr), dm(va)
    bst = xgb.train(PARAMS, dtr, num_boost_round=500, evals=[(dva, 'val')],
                    early_stopping_rounds=50, verbose_eval=False)
    dte = xgb.DMatrix(te[feats].values.astype(np.float64))
    score = bst.predict(dte)
    sgn = -1.0 if invert else 1.0
    rhos = []
    tte = te.copy(); tte['_s'] = score
    for _, q in tte.groupby('Query_ID'):
        if len(q) > 1 and q['_s'].std() > 0:
            r = spearmanr(q['_s'].values, sgn * q[RET].values).correlation
            if np.isfinite(r):
                rhos.append(r)
    return float(np.mean(rhos)), len(rhos)


def main():
    PARAMS['device'] = _gpu()
    lo, hi = transformer_test_window()
    print(f"label={LABEL}  device={PARAMS['device']}  |  chrono test window: {lo}  ..  {hi}")

    df = pd.read_parquet(XGB_PANEL)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df[df[RET].notna()].copy()                  # 2h-labeled rows only
    feats = [c for c in df.columns if c not in META_EXCLUDE]
    pre = df[df['DateTime'] < lo].sort_values('DateTime')
    te = df[(df['DateTime'] >= lo) & (df['DateTime'] <= hi)].copy()
    for part in (pre, te):
        part['Query_ID'] = part.groupby('DateTime').ngroup()
    cut = pre['DateTime'].quantile(0.90)
    tr, va = pre[pre['DateTime'] <= cut].copy(), pre[pre['DateTime'] > cut].copy()
    for part in (tr, va):
        part['Query_ID'] = part.groupby('DateTime').ngroup()
    print(f"XGBoost {H}h: train rows={len(tr):,} val={len(va):,} test={len(te):,} (feats={len(feats)})")

    xl, nl = train_xgb_side(tr, va, te, feats, invert=False)
    xs, ns = train_xgb_side(tr, va, te, feats, invert=True)

    def tf_rho(side):
        p = f'artifacts/dualres_{side}_{H}h_metrics.json'
        if not os.path.exists(p):
            return None
        return json.load(open(p))['test'].get('rho')
    tl, tsh = tf_rho('long'), tf_rho('short')

    print(f"\n=== {H}-HOUR HEAD-TO-HEAD: test rank-IC (rho), same window + 110 universe ===")
    print(f"  {'model':24s} {'long_rho':>10s} {'short_rho':>10s}")
    print(f"  {f'XGBoost {H}h':24s} {xl:>10.4f} {xs:>10.4f}   (test queries L={nl} S={ns})")
    print(f"  {f'DualRes transformer {H}h':24s} "
          f"{(f'{tl:.4f}' if tl is not None else 'n/a'):>10s} "
          f"{(f'{tsh:.4f}' if tsh is not None else 'n/a'):>10s}")
    if tl is not None:
        print("\n=== BAR (transformer must beat XGBoost materially AND pass shuffle neg-control) ===")
        print(f"  long:  transformer {tl:+.4f} vs xgb {xl:+.4f}  -> delta {tl-xl:+.4f}  "
              f"{'BEATS' if tl > xl else 'does NOT beat'}")
        print(f"  short: transformer {tsh:+.4f} vs xgb {xs:+.4f}  -> delta {tsh-xs:+.4f}  "
              f"{'BEATS' if tsh > xs else 'does NOT beat'}")
    json.dump({'label': LABEL, 'window': [str(lo), str(hi)], 'xgb_long': xl, 'xgb_short': xs,
               'tf_long': tl, 'tf_short': tsh},
              open(f'{TPANEL}/benchmark_{H}h_summary.json', 'w'), indent=2, default=float)
    print(f"\nsaved {TPANEL}/benchmark_{H}h_summary.json")


if __name__ == '__main__':
    main()
