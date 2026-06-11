"""
Full empirical walk-forward reanalysis of v8_upstox_3y (1H dual-stage ranker).

The saved v8 artifact was evaluated with a single 80/20 chronological split where
the "test" set doubled as the early-stopping validation set (see
train_ranking_upstox.py). This script re-runs v8's exact architecture
(rank:pairwise, depth=5, same hyperparams, same 86 features) in a rolling
walk-forward with a dedicated validation month for early stopping, so every
evaluated prediction is genuinely out-of-sample -- mirroring the rigor used in
the v10/v18/TBM audits (purged folds, raw-vs-net win rates, explicit cost-sign
check).

Usage:
    python scripts/analysis/v8_walkforward.py

Output:
    data/model_analysis/v8_walkforward/walkforward.json
"""

import os, json, time
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

# Exact v8 hyperparameters (models/v8_upstox_3y/metadata.json)
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
    print(f"v8 feature count: {len(feat)}")

    df = pd.read_csv(DATA_FILE)
    df['YearMonth'] = df['DateTime'].str[:7]
    df['Time'] = df['DateTime'].str[11:16]
    df = df.sort_values(['Query_ID']).reset_index(drop=True)
    months = sorted(df['YearMonth'].unique())

    X = df[feat].values.astype(np.float64)
    y = df[RET_COL].values.astype(np.float64)
    qids = df['Query_ID'].values
    ym = df['YearMonth'].values
    times = df['Time'].values

    # rolling folds: train months[:i], VAL=[months[i]] (early stopping), test=months[i+1:i+1+horizon]
    min_train, horizon, step = 18, 2, 4
    folds = []
    for i in range(min_train, len(months) - horizon - 1, step):
        folds.append(dict(train=months[:i], val=[months[i]], test=months[i + 1:i + 1 + horizon]))
    print(f"Folds: {len(folds)}")

    acc = {k: [] for k in ('idx', 'rl', 'rs')}
    fold_spearman = []
    for fi, cfg in enumerate(folds, 1):
        t0 = time.time()
        trm = np.isin(ym, cfg['train'])
        vam = np.isin(ym, cfg['val'])
        tem = np.isin(ym, cfg['test'])
        if tem.sum() == 0 or trm.sum() == 0 or vam.sum() == 0:
            continue

        Xtr, Xva, Xte = X[trm].copy(), X[vam].copy(), X[tem].copy()

        # per-fold NaN fill using TRAIN-ONLY column means (no leakage)
        col_means = np.nanmean(Xtr, axis=0)
        col_means = np.nan_to_num(col_means)
        for arr in (Xtr, Xva, Xte):
            inds = np.where(~np.isfinite(arr))
            if len(inds[0]):
                arr[inds] = np.take(col_means, inds[1])

        q_tr, q_va = qids[trm], qids[vam]

        dl = xgb.DMatrix(Xtr, label=int_ranks(y[trm], q_tr, invert=False)); dl.set_group(group_sizes(q_tr))
        dvl = xgb.DMatrix(Xva, label=int_ranks(y[vam], q_va, invert=False)); dvl.set_group(group_sizes(q_va))
        ds = xgb.DMatrix(Xtr, label=int_ranks(y[trm], q_tr, invert=True)); ds.set_group(group_sizes(q_tr))
        dvs = xgb.DMatrix(Xva, label=int_ranks(y[vam], q_va, invert=True)); dvs.set_group(group_sizes(q_va))

        bl = xgb.train(V8_PARAMS, dl, num_boost_round=500, evals=[(dvl, 'val')],
                       early_stopping_rounds=50, verbose_eval=False)
        bs = xgb.train(V8_PARAMS, ds, num_boost_round=500, evals=[(dvs, 'val')],
                       early_stopping_rounds=50, verbose_eval=False)

        dte = xgb.DMatrix(Xte)
        rl_pred, rs_pred = bl.predict(dte), bs.predict(dte)

        # per-fold Spearman (query-wise, then averaged) -- comparable to metadata's 0.0461/0.0490
        q_te, y_te = qids[tem], y[tem]
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

        fold_spearman.append(dict(fold=fi, test=cfg['test'],
                                   long_rho=round(float(np.mean(long_corrs)), 4),
                                   short_rho=round(float(np.mean(short_corrs)), 4),
                                   best_iter_long=int(bl.best_iteration),
                                   best_iter_short=int(bs.best_iteration)))

        acc['idx'].append(np.where(tem)[0])
        acc['rl'].append(rl_pred)
        acc['rs'].append(rs_pred)
        print(f"  fold {fi}/{len(folds)}  test {cfg['test'][0]}..{cfg['test'][-1]}  "
              f"({tem.sum()} rows, {time.time()-t0:.0f}s)  "
              f"long_rho={fold_spearman[-1]['long_rho']:+.4f}  short_rho={fold_spearman[-1]['short_rho']:+.4f}")

    idx = np.concatenate(acc['idx'])
    P = {k: np.concatenate(acc[k]) for k in ('rl', 'rs')}
    oos_q, oos_y, oos_ym, oos_time = qids[idx], y[idx], ym[idx], times[idx]

    np.savez_compressed(f'{OUT_DIR}/walkforward_preds.npz',
                        idx=idx, ym=oos_ym, q=oos_q, y=oos_y, time=oos_time,
                        rl=P['rl'], rs=P['rs'])

    print(f"\nTotal genuinely-OOS rows: {len(idx)}  span: {min(oos_ym)}..{max(oos_ym)}")

    # ---- Top-1 / Top-3 evaluation ----
    def topk_returns(K, period_mask):
        l_r, s_r = [], []
        l_meta, s_meta = [], []  # (time bucket) per selected trade
        qsel = oos_q[period_mask]
        for qid in np.unique(qsel):
            m = (oos_q == qid) & period_mask
            if m.sum() < max(3, K):
                continue
            rl, rs, a, t = P['rl'][m], P['rs'][m], oos_y[m], oos_time[m]
            for j in np.argsort(rl)[-K:]:
                l_r.append(a[j]); l_meta.append(t[j])
            for j in np.argsort(rs)[-K:]:
                s_r.append(-a[j]); s_meta.append(t[j])
        return np.array(l_r), np.array(s_r), l_meta, s_meta

    out = dict(n_folds=len(fold_spearman), oos_rows=int(len(idx)),
                oos_span=f"{min(oos_ym)}..{max(oos_ym)}",
                fold_spearman=fold_spearman, topk={}, time_of_day={})

    periods = {'full_OOS': np.ones(len(idx), bool),
               'last_12mo_2025-07+': (oos_ym >= '2025-07')}

    print("\n" + "=" * 70)
    print("WALK-FORWARD OOS TOP-K RESULTS (genuinely out-of-sample)")
    print("=" * 70)
    for pname, pmask in periods.items():
        out['topk'][pname] = dict(rows=int(pmask.sum()), K={})
        print(f"\n######## PERIOD: {pname} ({pmask.sum()} rows) ########")
        for K in (1, 3):
            l_r, s_r, _, _ = topk_returns(K, pmask)
            out['topk'][pname]['K'][K] = {}
            print(f"\n[Top-{K}]")
            for cl_, cv in COSTS.items():
                ls, ss = trade_stats(l_r, cv), trade_stats(s_r, cv)
                # explicit cost-sign check
                if ls['n']:
                    assert abs((ls['raw_bps'] - ls['net_bps']) - cv * 10000) < 1e-6
                if ss['n']:
                    assert abs((ss['raw_bps'] - ss['net_bps']) - cv * 10000) < 1e-6
                out['topk'][pname]['K'][K][cl_] = dict(long=ls, short=ss)
                print(f"  @{cl_:<5} LONG  n={ls['n']:>5} raw {ls['raw_bps']:>+6.2f} net {ls['net_bps']:>+6.2f} "
                      f"raw_win {ls['raw_win']:.1%} net_win {ls['net_win']:.1%} t={ls['t_stat']:>5.2f}")
                print(f"  @{cl_:<5} SHORT n={ss['n']:>5} raw {ss['raw_bps']:>+6.2f} net {ss['net_bps']:>+6.2f} "
                      f"raw_win {ss['raw_win']:.1%} net_win {ss['net_win']:.1%} t={ss['t_stat']:>5.2f}")

    # ---- Time-of-day breakdown (Top-3, full OOS, @6bps) ----
    print("\n" + "=" * 70)
    print("TIME-OF-DAY BREAKDOWN (Top-3, full OOS span)")
    print("=" * 70)
    full_mask = np.ones(len(idx), bool)
    l_r, s_r, l_t, s_t = topk_returns(3, full_mask)
    df_l = pd.DataFrame({'ret': l_r, 'time': l_t})
    df_s = pd.DataFrame({'ret': s_r, 'time': s_t})
    cv = COSTS['6bps']
    print(f"\n{'Time':>6} | {'L_n':>5} {'L_raw_win':>9} {'L_net_bps':>9} || {'S_n':>5} {'S_raw_win':>9} {'S_net_bps':>9}")
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

    with open(f'{OUT_DIR}/walkforward.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT_DIR}/walkforward.json")


if __name__ == '__main__':
    main()
