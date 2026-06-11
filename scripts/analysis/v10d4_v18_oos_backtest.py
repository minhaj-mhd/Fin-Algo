"""
OOS backtest of the SAVED v10_depth4 + v18 production artifacts.

Both models' saved vintages were fit on data ending before 2025-07, so 2025-07+ is a
genuinely out-of-sample window for both:
    v10_depth4 : trained 2022-01..2025-06, validation (early-stop) 2025-07
    v18 RF     : trained 2022-01..2025-03  (val slice computed but unused in prod fit)
Common clean OOS = 2025-07+ (matches the common_OOS slice in v10_v18_independent_analysis.py).

This loads the saved boosters (NO retrain) and reports V10-alone, symmetric hybrid, and
asymmetric (veto-longs / raw-shorts) Top-3 edges at 6 and 10 bps, RAW and clean.

Usage:
    python scripts/analysis/v10d4_v18_oos_backtest.py
Outputs:
    data/model_analysis/v10_v18_independent/v10d4_oos_backtest.json
"""

import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata, ttest_1samp

DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL   = 'Next_Hour_Return'
V10_DIR   = 'models/v10_depth4_1h'        # <-- depth-4, not native/depth-5
V18_DIR   = 'models/v18_random_forest_1h'
OUT_DIR   = 'data/model_analysis/v10_v18_independent'

V10_TRAIN_END = '2025-06'
OOS_START     = '2025-07'
OOS_MID       = '2026-01'
COSTS   = {'6bps': 0.0006, '10bps': 0.0010}
PROB_TH = 0.52
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
           'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth', 'date',
           'fwd_cc', 'next_date']


def trade_stats(returns, cost):
    r = np.asarray(returns, float)
    if len(r) == 0:
        return dict(n=0, raw_bps=0.0, net_bps=0.0, raw_win=0.0, net_hit=0.0, t_stat=0.0)
    net = r - cost
    t = float(ttest_1samp(net, 0.0).statistic) if len(r) > 1 and np.std(net) > 0 else 0.0
    return dict(n=int(len(r)), raw_bps=round(float(r.mean())*10000, 1),
                net_bps=round(float(net.mean())*10000, 1),
                raw_win=round(float((r > 0).mean()), 4),
                net_hit=round(float((r > cost).mean()), 4), t_stat=round(t, 2))


def fmt(s):
    return (f"n={s['n']:>5} | raw {s['raw_bps']:>+6.1f} | net {s['net_bps']:>+6.1f} "
            f"| rawwin {s['raw_win']:.1%} | t={s['t_stat']:>5.2f}")


def logic_returns(qids, P, rets, msk, veto_long, veto_short, K=3):
    l_r, s_r = [], []
    for qid in np.unique(qids[msk]):
        qm = (qids == qid) & msk
        if qm.sum() < max(3, K):
            continue
        rl, rs, pl, ps, a = P['rl'][qm], P['rs'][qm], P['pl'][qm], P['ps'][qm], rets[qm]
        for idx in np.argsort(rl)[-K:]:
            if (pl[idx] > PROB_TH) if veto_long else True:
                l_r.append(a[idx])
        for idx in np.argsort(rs)[-K:]:
            if (ps[idx] > PROB_TH) if veto_short else True:
                s_r.append(-a[idx])
    return np.array(l_r), np.array(s_r)


def clean_mask_of(df):
    g = df.groupby('Ticker', group_keys=False)
    df['fwd_cc']    = g['Close'].apply(lambda s: s.shift(-1) / s - 1)
    df['next_date'] = g['date'].shift(-1)
    next_spans_day  = (df['date'] != df['next_date']).values
    last_hour       = int(df['Hour'].max())
    overnight       = next_spans_day & (df['Hour'].values < last_hour)
    return ~overnight & np.isfinite(df[RET_COL].values)


def main():
    print("Loading ...")
    df = pd.read_csv(DATA_FILE)
    df['YearMonth'] = df['DateTime'].str[:7]
    df['date']      = df['DateTime'].str[:10]
    df = df.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    clean = clean_mask_of(df)

    feat = [c for c in df.columns if c not in EXCLUDE]
    X = df[feat].values.astype(np.float64)
    train_mask = (df['YearMonth'] <= V10_TRAIN_END).values
    for ci in range(X.shape[1]):              # train-only NaN fill (no test leakage)
        bad = ~np.isfinite(X[:, ci])
        if bad.any():
            good = X[np.isfinite(X[:, ci]) & train_mask, ci]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

    def L(p):
        b = xgb.Booster(); b.load_model(p); return b
    m = dict(rl=L(f'{V10_DIR}/xgb_long_model.json'), rs=L(f'{V10_DIR}/xgb_short_model.json'),
             pl=L(f'{V18_DIR}/xgb_long_model.json'), ps=L(f'{V18_DIR}/xgb_short_model.json'))
    d = xgb.DMatrix(X)
    P = {k: m[k].predict(d) for k in m}

    qids = df['Query_ID'].values
    rets = df[RET_COL].values
    ym   = df['YearMonth'].values
    oos       = (ym >= OOS_START)
    oos_clean = oos & clean
    h2_25     = oos_clean & (ym < OOS_MID)
    h1_26     = oos_clean & (ym >= OOS_MID)

    configs = {'V10d4_alone':      (False, False),
               'hybrid_symmetric': (True, True),
               'asymmetric':       (True, False)}
    masks = {'OOS_raw (>=2025-07)': oos, 'OOS_clean': oos_clean,
             'H2_2025': h2_25, 'H1_2026': h1_26}

    out = dict(model_v10=V10_DIR, model_v18=V18_DIR, oos_start=OOS_START,
               oos_rows=int(oos.sum()), oos_clean_rows=int(oos_clean.sum()), periods={})
    print("\n" + "=" * 72)
    print("SAVED-ARTIFACT OOS BACKTEST  v10_depth4 + v18  (Top-3)")
    print("=" * 72)
    for pname, pmask in masks.items():
        out['periods'][pname] = dict(rows=int(pmask.sum()), configs={})
        print(f"\n######## {pname}  ({pmask.sum()} rows) ########")
        for name, (vl, vs) in configs.items():
            l_r, s_r = logic_returns(qids, P, rets, pmask, vl, vs)
            out['periods'][pname]['configs'][name] = {}
            print(f"\n  [{name}]")
            for cl, cv in COSTS.items():
                ls, ss = trade_stats(l_r, cv), trade_stats(s_r, cv)
                out['periods'][pname]['configs'][name][cl] = dict(long=ls, short=ss)
                print(f"    @{cl:<5} LONG  {fmt(ls)}")
                print(f"    @{cl:<5} SHORT {fmt(ss)}")

    with open(f'{OUT_DIR}/v10d4_oos_backtest.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT_DIR}/v10d4_oos_backtest.json")


if __name__ == '__main__':
    main()
