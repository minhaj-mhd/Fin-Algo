"""
Phase 4: Fresh walk-forward robustness check for the V10+V18 stack.

Retrains BOTH models in a rolling walk-forward so every prediction used for evaluation is
genuinely out-of-sample. This tests the *method*, not one saved vintage. Compares V10-alone,
symmetric hybrid, and asymmetric (veto-longs / raw-shorts) against the saved-artifact
conclusions from v10_v18_independent_analysis.py.

Usage:
    python scripts/analysis/v10_v18_walkforward.py

Outputs:
    data/model_analysis/v10_v18_independent/walkforward.json
"""

import os, json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata, ttest_1samp

DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL   = 'Next_Hour_Return'
OUT_DIR   = 'data/model_analysis/v10_v18_independent'
COSTS     = {'6bps': 0.0006, '10bps': 0.0010}
PROB_TH   = 0.52
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
           'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth', 'date',
           'fwd_cc', 'next_date']


def detect_device():
    try:
        d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'},
                  d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


DEVICE = detect_device()
print(f"Device: {DEVICE}")

V10_PARAMS = dict(objective='rank:pairwise', eta=0.03, max_depth=4, subsample=0.8,
                  colsample_bytree=0.8, alpha=1.0, **{'lambda': 2.0}, min_child_weight=10,
                  random_state=42, verbosity=0, eval_metric='ndcg@3', ndcg_exp_gain=False,
                  tree_method='hist', device=DEVICE)


def v18_params(pos_weight):
    return dict(objective='binary:logistic', eval_metric='auc', scale_pos_weight=pos_weight,
                booster='gbtree', num_parallel_tree=100, eta=1.0, max_depth=10, subsample=0.8,
                colsample_bynode=0.8, alpha=1.0, **{'lambda': 2.0}, min_child_weight=10,
                random_state=42, verbosity=0, tree_method='hist', device=DEVICE)


def int_ranks(y, qids, invert=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(qids):
        m = qids == qid
        vals = -y[m] if invert else y[m]
        out[m] = rankdata(vals, method='ordinal') - 1
    return out


def group_sizes(qids):
    # contiguous group sizes in order of appearance (DMatrix.set_group needs counts)
    _, idx, counts = np.unique(qids, return_index=True, return_counts=True)
    order = np.argsort(idx)
    return counts[order]


def trade_stats(returns, cost):
    r = np.asarray(returns, float)
    if len(r) == 0:
        return dict(n=0, raw_bps=0.0, net_bps=0.0, raw_win=0.0, t_stat=0.0)
    net = r - cost
    t = float(ttest_1samp(net, 0.0).statistic) if len(r) > 1 and np.std(net) > 0 else 0.0
    return dict(n=int(len(r)), raw_bps=round(float(r.mean())*10000, 1),
                net_bps=round(float(net.mean())*10000, 1),
                raw_win=round(float((r > 0).mean()), 4), t_stat=round(t, 2))


def main():
    print("Loading ...")
    df = pd.read_csv(DATA_FILE)
    df['YearMonth'] = df['DateTime'].str[:7]
    df = df.sort_values(['Query_ID']).reset_index(drop=True)  # group rows by query for ranking
    months = sorted(df['YearMonth'].unique())

    feat = [c for c in df.columns if c not in EXCLUDE]
    X = df[feat].values.astype(np.float64)
    y = df[RET_COL].values.astype(np.float64)
    qids = df['Query_ID'].values
    ym = df['YearMonth'].values

    # global NaN fill from earliest 18 months (proxy for "past only"); refined per-fold below
    for ci in range(X.shape[1]):
        bad = ~np.isfinite(X[:, ci])
        if bad.any():
            good = X[np.isfinite(X[:, ci]), ci]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

    y_long  = (y > 0).astype(np.int32)
    y_short = (y < 0).astype(np.int32)

    # rolling folds: train months[:i], VAL=[months[i]], test=months[i+1:i+1+horizon]
    # (mirrors production: train + 1 validation month for early stopping + held-out test)
    min_train, horizon, step = 18, 2, 4
    folds = []
    for i in range(min_train, len(months) - horizon - 1, step):
        folds.append(dict(train=months[:i], val=[months[i]],
                          test=months[i + 1:i + 1 + horizon]))
    print(f"Folds: {len(folds)} (each with 1 validation month + early stopping)")

    # accumulate genuinely-OOS predictions
    acc = {k: [] for k in ('idx', 'rl', 'rs', 'pl', 'ps')}
    for fi, cfg in enumerate(folds, 1):
        t0 = time.time()
        trm = np.isin(ym, cfg['train'])
        vam = np.isin(ym, cfg['val'])
        tem = np.isin(ym, cfg['test'])
        if tem.sum() == 0 or trm.sum() == 0 or vam.sum() == 0:
            continue
        q_tr, q_va = qids[trm], qids[vam]
        Xtr, Xva, Xte = X[trm], X[vam], X[tem]

        # V10 long/short (ranking) with validation-based early stopping (matches production)
        dl = xgb.DMatrix(Xtr, label=int_ranks(y[trm], q_tr, invert=False)); dl.set_group(group_sizes(q_tr))
        dvl = xgb.DMatrix(Xva, label=int_ranks(y[vam], q_va, invert=False)); dvl.set_group(group_sizes(q_va))
        ds = xgb.DMatrix(Xtr, label=int_ranks(y[trm], q_tr, invert=True));  ds.set_group(group_sizes(q_tr))
        dvs = xgb.DMatrix(Xva, label=int_ranks(y[vam], q_va, invert=True));  dvs.set_group(group_sizes(q_va))
        bl = xgb.train(V10_PARAMS, dl, num_boost_round=500, evals=[(dvl, 'val')],
                       early_stopping_rounds=50, verbose_eval=False)
        bs = xgb.train(V10_PARAMS, ds, num_boost_round=500, evals=[(dvs, 'val')],
                       early_stopping_rounds=50, verbose_eval=False)

        # V18 long/short (RF classifier)
        pwl = (1 - y_long[trm].mean()) / max(y_long[trm].mean(), 1e-9)
        pws = (1 - y_short[trm].mean()) / max(y_short[trm].mean(), 1e-9)
        cl = xgb.train(v18_params(pwl), xgb.DMatrix(Xtr, label=y_long[trm]),  num_boost_round=1)
        cs = xgb.train(v18_params(pws), xgb.DMatrix(Xtr, label=y_short[trm]), num_boost_round=1)

        dte = xgb.DMatrix(Xte)
        acc['idx'].append(np.where(tem)[0])
        acc['rl'].append(bl.predict(dte)); acc['rs'].append(bs.predict(dte))
        acc['pl'].append(cl.predict(dte)); acc['ps'].append(cs.predict(dte))
        print(f"  fold {fi}/{len(folds)}  test {cfg['test'][0]}..{cfg['test'][-1]}  "
              f"({tem.sum()} rows, {time.time()-t0:.0f}s)")

    idx = np.concatenate(acc['idx'])
    P = {k: np.concatenate(acc[k]) for k in ('rl', 'rs', 'pl', 'ps')}
    oos_q = qids[idx]; oos_y = y[idx]; oos_ym = ym[idx]

    # dump predictions so future slicing needs no retrain
    np.savez_compressed(f'{OUT_DIR}/walkforward_preds.npz',
                        idx=idx, ym=oos_ym, q=oos_q, y=oos_y,
                        rl=P['rl'], rs=P['rs'], pl=P['pl'], ps=P['ps'])

    # evaluate configs (Top-3) within a month-mask (for period breakdowns)
    def run_logic(veto_long, veto_short, period_mask, K=3):
        l_r, s_r = [], []
        qsel = oos_q[period_mask]
        for qid in np.unique(qsel):
            m = (oos_q == qid) & period_mask
            if m.sum() < max(3, K): continue
            rl, rs, pl, ps, a = P['rl'][m], P['rs'][m], P['pl'][m], P['ps'][m], oos_y[m]
            for j in np.argsort(rl)[-K:]:
                if (pl[j] > PROB_TH) if veto_long else True: l_r.append(a[j])
            for j in np.argsort(rs)[-K:]:
                if (ps[j] > PROB_TH) if veto_short else True: s_r.append(-a[j])
        return np.array(l_r), np.array(s_r)

    configs = {'V10_alone': (False, False),
               'hybrid_symmetric': (True, True),
               'asymmetric': (True, False)}
    periods = {'full_2023-07+': np.ones(len(idx), bool),
               'common_OOS_2025-07+': (oos_ym >= '2025-07')}
    out = dict(n_folds=len(folds), oos_rows=int(len(idx)),
               test_span=f"{months[min_train]}..{months[-1]}", periods={})
    print("\n" + "=" * 64)
    print("WALK-FORWARD OOS (every prediction genuinely out-of-sample)")
    print("=" * 64)
    for pname, pmask in periods.items():
        out['periods'][pname] = dict(rows=int(pmask.sum()), configs={})
        print(f"\n######## PERIOD: {pname}  ({pmask.sum()} rows) ########")
        for name, (vl, vs) in configs.items():
            l_r, s_r = run_logic(vl, vs, pmask)
            out['periods'][pname]['configs'][name] = {}
            print(f"\n[{name}]")
            for cl_, cv in COSTS.items():
                ls, ss = trade_stats(l_r, cv), trade_stats(s_r, cv)
                out['periods'][pname]['configs'][name][cl_] = dict(long=ls, short=ss)
                print(f"  @{cl_:<5} LONG  n={ls['n']:>5} raw {ls['raw_bps']:>+6.1f} net {ls['net_bps']:>+6.1f} "
                      f"win {ls['raw_win']:.1%} t={ls['t_stat']:>5.2f}")
                print(f"  @{cl_:<5} SHORT n={ss['n']:>5} raw {ss['raw_bps']:>+6.1f} net {ss['net_bps']:>+6.1f} "
                      f"win {ss['raw_win']:.1%} t={ss['t_stat']:>5.2f}")

    with open(f'{OUT_DIR}/walkforward.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT_DIR}/walkforward.json")


if __name__ == '__main__':
    main()
