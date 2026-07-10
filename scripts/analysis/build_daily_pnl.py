"""
Build a CONTIGUOUS out-of-sample DAILY P&L series for the v21 1h ranker book — the raw material
for "positive vs negative day" regime research.

Expanding-window walk-forward: train long+short rank:pairwise (same recipe as eval_2h_v20) on all
months strictly before each contiguous 3-month TEST block, score every decision time in the block,
take Top-K LONG and Top-K SHORT per time, realize the 1h net (10bps). Every trading day after the
initial 18-month train is scored OOS exactly once -> a clean, gap-free daily series (unlike the
sparse stop panel). Daily book net = mean over the day's long+short legs.

Output: data/research/regime_days/daily_pnl.csv
  cols: date, net_bps, long_bps, short_bps, n_legs, n_long, n_short, frac_pos_legs
RESEARCH ONLY (AGENTS.md). Run: python scripts/analysis/build_daily_pnl.py [--k 10]
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
from scripts.research.eval_2h_v20 import int_ranks, _gpu, PARAMS, BASE_EXCLUDE, RET1

PANEL = 'data/research/v21_rolling_1h/panel.parquet'
OUTDIR = 'data/research/regime_days'
OUT = os.path.join(OUTDIR, 'daily_pnl.csv')
COST = 10 / 1e4
INIT_TRAIN_M = 18         # months of initial training before first OOS block
BLOCK_M = 3               # contiguous test block length (months)
os.makedirs(OUTDIR, exist_ok=True)


def train_side(X, y, q, tr, va, invert):
    Xtr, ytr, qtr = X[tr], y[tr], q[tr]; Xva, yva, qva = X[va], y[va], q[va]
    gtr = pd.Series(qtr).groupby(qtr).size().values; gva = pd.Series(qva).groupby(qva).size().values
    dtr = xgb.DMatrix(Xtr, label=int_ranks(ytr, qtr, invert)); dtr.set_group(gtr)
    dva = xgb.DMatrix(Xva, label=int_ranks(yva, qva, invert)); dva.set_group(gva)
    return xgb.train(PARAMS, dtr, 500, evals=[(dva, 'v')], early_stopping_rounds=50, verbose_eval=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int, default=10); args = ap.parse_args()
    K = args.k
    PARAMS['device'] = _gpu(); print(f"device={PARAMS['device']}  K={K}")
    df = pd.read_parquet(PANEL); df['DateTime'] = pd.to_datetime(df['DateTime'])
    feats = [c for c in df.columns if c not in BASE_EXCLUDE]
    X = df[feats].values.astype(np.float64)
    if not np.isfinite(X).all():
        cm = np.nan_to_num(np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0))
        bad = np.where(~np.isfinite(X)); X[bad] = np.take(cm, bad[1])
    y = df[RET1].values.astype(np.float64); q = df['Query_ID'].values
    ym = df['DateTime'].dt.strftime('%Y-%m').values
    months = sorted(np.unique(ym))
    print(f"panel rows={len(df):,} feats={len(feats)} months={len(months)} ({months[0]}..{months[-1]})")

    rows = []
    b = INIT_TRAIN_M
    while b < len(months):
        block = months[b:b + BLOCK_M]
        tr_m = months[:b]
        va_m = tr_m[-1:]                                   # last train month = early-stop val
        tr = np.isin(ym, tr_m[:-1]); va = np.isin(ym, va_m); te = np.isin(ym, block)
        if te.sum() == 0:
            break
        bL = train_side(X, y, q, tr, va, invert=False)
        bS = train_side(X, y, q, tr, va, invert=True)
        dte = xgb.DMatrix(X[te])
        sL = bL.predict(dte); sS = bS.predict(dte)
        sub = df.iloc[np.where(te)[0]][['DateTime', 'Query_ID']].copy()
        sub['retL'] = y[te]; sub['_sL'] = sL; sub['_sS'] = sS
        # per decision time: Top-K long & short legs
        for _, g in sub.groupby('Query_ID'):
            if len(g) < K + 1:
                continue
            gi = g.reset_index(drop=True)
            date = pd.Timestamp(gi['DateTime'].iloc[0]).normalize()
            r = gi['retL'].values
            li = np.argsort(-gi['_sL'].values)[:K]; si = np.argsort(-gi['_sS'].values)[:K]
            long_net = r[li] - COST                         # long pnl
            short_net = -r[si] - COST                       # short pnl
            rows.append((date, long_net, short_net))
        print(f"  block {block[0]}..{block[-1]} (train {tr_m[0]}..{tr_m[-1]}, {te.sum():,} rows)")
        b += BLOCK_M

    # aggregate to daily
    daily = {}
    for date, ln, sn in rows:
        d = daily.setdefault(date, {'long': [], 'short': []})
        d['long'].append(ln); d['short'].append(sn)
    recs = []
    for date in sorted(daily):
        ln = np.concatenate(daily[date]['long']); sn = np.concatenate(daily[date]['short'])
        allnet = np.concatenate([ln, sn])
        recs.append({'date': date.date(), 'net_bps': allnet.mean() * 1e4,
                     'long_bps': ln.mean() * 1e4, 'short_bps': sn.mean() * 1e4,
                     'n_legs': len(allnet), 'n_long': len(ln), 'n_short': len(sn),
                     'frac_pos_legs': (allnet > 0).mean()})
    out = pd.DataFrame(recs)
    out.to_csv(OUT, index=False)
    pos = (out['net_bps'] > 0).sum(); neg = (out['net_bps'] < 0).sum()
    print(f"\nsaved {OUT}  days={len(out)}  span {out['date'].iloc[0]}..{out['date'].iloc[-1]}")
    print(f"  positive days={pos} ({pos/len(out)*100:.1f}%)  negative={neg} ({neg/len(out)*100:.1f}%)")
    print(f"  mean daily net={out['net_bps'].mean():+.2f}bps  median={out['net_bps'].median():+.2f}bps")


if __name__ == '__main__':
    main()
