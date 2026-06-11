"""
v8 reanalysis, static chronological 70/10/20 split (Train/Val/Test).

Sanity check requested after the rolling walk-forward (v8_walkforward.py) showed
decaying, net-negative results: replicate v8's original "one big split" training
regime (large historical train set, single held-out test period close to the
original test window) but FIX the original's leakage (early stopping on a
dedicated validation slice, never on the test slice). Same architecture,
features, and hyperparameters as the saved v8 artifact.

Usage:
    python scripts/analysis/v8_static_70_10_20.py

Output:
    data/model_analysis/v8_walkforward/static_70_10_20.json
"""

import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata, spearmanr, ttest_1samp

DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL = 'Next_Hour_Return'
META_FILE = 'models/v8_upstox_3y/metadata.json'
OUT_DIR = 'data/model_analysis/v8_walkforward'
COSTS = {'6bps': 0.0006, '10bps': 0.0010}
os.makedirs(OUT_DIR, exist_ok=True)


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

V8_PARAMS = dict(objective='rank:pairwise', eta=0.03, max_depth=5, subsample=0.8,
                  colsample_bytree=0.8, alpha=1.0, **{'lambda': 2.0}, min_child_weight=10,
                  random_state=42, verbosity=0, eval_metric='ndcg@3', ndcg_exp_gain=False,
                  tree_method='hist', device=DEVICE)


def int_ranks(y, qids, invert=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(qids):
        m = qids == qid
        vals = -y[m] if invert else y[m]
        out[m] = rankdata(vals, method='ordinal') - 1
    return out


def group_sizes(qids):
    _, idx, counts = np.unique(qids, return_index=True, return_counts=True)
    order = np.argsort(idx)
    return counts[order]


def trade_stats(returns, cost):
    r = np.asarray(returns, float)
    if len(r) == 0:
        return dict(n=0, raw_bps=0.0, net_bps=0.0, raw_win=0.0, net_win=0.0, t_stat=0.0)
    net = r - cost
    t = float(ttest_1samp(net, 0.0).statistic) if len(r) > 1 and np.std(net) > 0 else 0.0
    return dict(n=int(len(r)), raw_bps=round(float(r.mean()) * 10000, 2),
                net_bps=round(float(net.mean()) * 10000, 2),
                raw_win=round(float((r > 0).mean()), 4),
                net_win=round(float((net > 0).mean()), 4),
                t_stat=round(t, 2))


def main():
    print("Loading ...")
    meta = json.load(open(META_FILE))
    feat = meta['features']

    df = pd.read_csv(DATA_FILE)
    df['YearMonth'] = df['DateTime'].str[:7]
    df['Time'] = df['DateTime'].str[11:16]
    df = df.sort_values(['Query_ID']).reset_index(drop=True)
    months = sorted(df['YearMonth'].unique())
    n = len(months)

    train_end = int(round(0.70 * n))
    val_end = int(round(0.80 * n))
    train_m, val_m, test_m = months[:train_end], months[train_end:val_end], months[val_end:]
    print(f"Months: {n} total")
    print(f"  Train ({len(train_m)}): {train_m[0]}..{train_m[-1]}")
    print(f"  Val   ({len(val_m)}): {val_m[0]}..{val_m[-1]}")
    print(f"  Test  ({len(test_m)}): {test_m[0]}..{test_m[-1]}")

    X = df[feat].values.astype(np.float64)
    y = df[RET_COL].values.astype(np.float64)
    qids = df['Query_ID'].values
    ym = df['YearMonth'].values
    times = df['Time'].values

    trm = np.isin(ym, train_m)
    vam = np.isin(ym, val_m)
    tem = np.isin(ym, test_m)
    print(f"  rows: train={trm.sum()}, val={vam.sum()}, test={tem.sum()}")

    Xtr, Xva, Xte = X[trm].copy(), X[vam].copy(), X[tem].copy()
    col_means = np.nan_to_num(np.nanmean(Xtr, axis=0))
    for arr in (Xtr, Xva, Xte):
        inds = np.where(~np.isfinite(arr))
        if len(inds[0]):
            arr[inds] = np.take(col_means, inds[1])

    q_tr, q_va, q_te = qids[trm], qids[vam], qids[tem]

    dl = xgb.DMatrix(Xtr, label=int_ranks(y[trm], q_tr, invert=False)); dl.set_group(group_sizes(q_tr))
    dvl = xgb.DMatrix(Xva, label=int_ranks(y[vam], q_va, invert=False)); dvl.set_group(group_sizes(q_va))
    ds = xgb.DMatrix(Xtr, label=int_ranks(y[trm], q_tr, invert=True)); ds.set_group(group_sizes(q_tr))
    dvs = xgb.DMatrix(Xva, label=int_ranks(y[vam], q_va, invert=True)); dvs.set_group(group_sizes(q_va))

    print("\nTraining LONG ranker...")
    bl = xgb.train(V8_PARAMS, dl, num_boost_round=500, evals=[(dl, 'train'), (dvl, 'val')],
                   early_stopping_rounds=50, verbose_eval=50)
    print("\nTraining SHORT ranker...")
    bs = xgb.train(V8_PARAMS, ds, num_boost_round=500, evals=[(ds, 'train'), (dvs, 'val')],
                   early_stopping_rounds=50, verbose_eval=50)

    print(f"\nbest_iteration: long={bl.best_iteration}, short={bs.best_iteration}")

    dte = xgb.DMatrix(Xte)
    rl_pred, rs_pred = bl.predict(dte), bs.predict(dte)
    y_te, t_te = y[tem], times[tem]

    # per-query Spearman
    long_corrs, short_corrs = [], []
    for qid in np.unique(q_te):
        m = q_te == qid
        if m.sum() < 2:
            continue
        lc, _ = spearmanr(rl_pred[m], y_te[m])
        sc, _ = spearmanr(rs_pred[m], -y_te[m])
        if not np.isnan(lc):
            long_corrs.append(lc)
        if not np.isnan(sc):
            short_corrs.append(sc)
    long_rho, short_rho = float(np.mean(long_corrs)), float(np.mean(short_corrs))
    print(f"\nTest Spearman: long={long_rho:+.4f}  short={short_rho:+.4f}  "
          f"(static metadata claim: 0.0461 / 0.0490)")

    # Top-1 / Top-3
    out = dict(train_span=f"{train_m[0]}..{train_m[-1]}", val_span=f"{val_m[0]}..{val_m[-1]}",
                test_span=f"{test_m[0]}..{test_m[-1]}",
                rows=dict(train=int(trm.sum()), val=int(vam.sum()), test=int(tem.sum())),
                best_iteration=dict(long=int(bl.best_iteration), short=int(bs.best_iteration)),
                test_spearman=dict(long=round(long_rho, 4), short=round(short_rho, 4)),
                topk={}, time_of_day={})

    print("\n" + "=" * 70)
    print(f"TEST PERIOD TOP-K RESULTS ({test_m[0]}..{test_m[-1]}, {tem.sum()} rows)")
    print("=" * 70)
    for K in (1, 3):
        l_r, s_r, l_t, s_t = [], [], [], []
        for qid in np.unique(q_te):
            m = q_te == qid
            if m.sum() < max(3, K):
                continue
            rl, rs, a, t = rl_pred[m], rs_pred[m], y_te[m], t_te[m]
            for j in np.argsort(rl)[-K:]:
                l_r.append(a[j]); l_t.append(t[j])
            for j in np.argsort(rs)[-K:]:
                s_r.append(-a[j]); s_t.append(t[j])
        l_r, s_r = np.array(l_r), np.array(s_r)
        out['topk'][K] = {}
        print(f"\n[Top-{K}]")
        for cl_, cv in COSTS.items():
            ls, ss = trade_stats(l_r, cv), trade_stats(s_r, cv)
            out['topk'][K][cl_] = dict(long=ls, short=ss)
            print(f"  @{cl_:<5} LONG  n={ls['n']:>5} raw {ls['raw_bps']:>+6.2f} net {ls['net_bps']:>+6.2f} "
                  f"raw_win {ls['raw_win']:.1%} net_win {ls['net_win']:.1%} t={ls['t_stat']:>5.2f}")
            print(f"  @{cl_:<5} SHORT n={ss['n']:>5} raw {ss['raw_bps']:>+6.2f} net {ss['net_bps']:>+6.2f} "
                  f"raw_win {ss['raw_win']:.1%} net_win {ss['net_win']:.1%} t={ss['t_stat']:>5.2f}")

        if K == 3:
            df_l = pd.DataFrame({'ret': l_r, 'time': l_t})
            df_s = pd.DataFrame({'ret': s_r, 'time': s_t})
            cv = COSTS['6bps']
            print(f"\nTime-of-day (Top-3, @6bps):")
            print(f"{'Time':>6} | {'L_n':>5} {'L_raw_win':>9} {'L_net_bps':>9} || {'S_n':>5} {'S_raw_win':>9} {'S_net_bps':>9}")
            for t in sorted(set(df_l['time']) | set(df_s['time'])):
                gl = df_l[df_l['time'] == t]
                gs = df_s[df_s['time'] == t]
                l_n, s_n = len(gl), len(gs)
                l_wr = (gl['ret'] > 0).mean() * 100 if l_n else float('nan')
                s_wr = (gs['ret'] > 0).mean() * 100 if s_n else float('nan')
                l_net = (gl['ret'].mean() - cv) * 10000 if l_n else float('nan')
                s_net = (gs['ret'].mean() - cv) * 10000 if s_n else float('nan')
                out['time_of_day'][t] = dict(long_n=l_n, long_raw_win=round(l_wr, 1) if l_n else None,
                                               long_net_bps=round(l_net, 2) if l_n else None,
                                               short_n=s_n, short_raw_win=round(s_wr, 1) if s_n else None,
                                               short_net_bps=round(s_net, 2) if s_n else None)
                print(f"{t:>6} | {l_n:>5} {l_wr:>8.1f}% {l_net:>+8.2f} || {s_n:>5} {s_wr:>8.1f}% {s_net:>+8.2f}")

    with open(f'{OUT_DIR}/static_70_10_20.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT_DIR}/static_70_10_20.json")


if __name__ == '__main__':
    main()
