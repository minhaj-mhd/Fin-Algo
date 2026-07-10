"""
Model-signal HOLD-TO-CLOSE (intraday, flat by 15:15) — strongest signal x longest intraday horizon.

Uses the v3_15min ranker's OOS scores (data/v3_15min_oos_scores.parquet: long_pct, short_rank;
true walk-forward OOS, no leak). At an intraday slot T, pick the top-k longs (high long_pct) and
top-k shorts (high short_rank) and HOLD TO THE 15:15 CLOSE (single round trip). Reports NET-of-cost
per side @6/@10 bps with t-stat + win-rate, market-neutral L/S, a TIME-SPLIT (OOS robustness), and
a negative control. Lookahead-safe: score at T, return strictly T->close.

  python scripts/analysis/model_signal_to_close.py
Exploratory — NO Gauntlet verdict authority.
"""
import os, sys, glob, warnings
from datetime import time as dtime
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CACHE = 'data/raw_upstox_cache_15min_3y'
OOS = 'data/v3_15min_oos_scores.parquet'
OPEN_T, CLOSE_T = dtime(9, 15), dtime(15, 15)
SLOTS = ['10:15', '11:15', '12:15', '13:15', '14:15', '14:45']
KS = [3, 5, 10, 20]
COSTS = [6.0, 10.0]
MIN_NAMES = 30


def build_fwd_to_close():
    """Per 15m bar: fwd_close = close_1515/close_bar - 1, keyed by (ts, Ticker.NS)."""
    out = []
    for p in tqdm(glob.glob(f'{CACHE}/*.csv'), desc='fwd'):
        tk = os.path.splitext(os.path.basename(p))[0] + '.NS'
        try:
            raw = pd.read_csv(p, usecols=['timestamp', 'close'])
        except Exception:
            continue
        ts = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        df = pd.DataFrame({'ts': ts, 'close': raw['close'].astype(float)}).dropna().sort_values('ts')
        t = df['ts'].dt.time
        df = df[(t >= OPEN_T) & (t <= CLOSE_T)]
        df['date'] = df['ts'].dt.normalize()
        eod = df[df['ts'].dt.strftime('%H:%M') == '15:15'].set_index('date')['close']
        df['close_eod'] = df['date'].map(eod)
        df['fwd_close'] = df['close_eod'] / df['close'] - 1.0
        df['Ticker'] = tk
        out.append(df[['ts', 'Ticker', 'fwd_close']].dropna())
    return pd.concat(out, ignore_index=True)


def stat(arr):
    a = np.array(arr) * 1e4
    n = len(a); m = a.mean(); t = m / (a.std(ddof=1) / np.sqrt(n)) if n > 2 else 0
    wr = (a > 0).mean()
    return m, t, n, wr


def evaluate(df, k, neg=False, half=None):
    if half == 1:
        df = df[df['ts'] < df['ts'].quantile(0.5)]
    elif half == 2:
        df = df[df['ts'] >= df['ts'].quantile(0.5)]
    rng = np.random.default_rng(0)
    long_n = {c: [] for c in COSTS}; short_n = {c: [] for c in COSTS}; ls_n = {c: [] for c in COSTS}
    for ts, g in df.groupby('ts'):
        if len(g) < MIN_NAMES:
            continue
        fwd = g['fwd_close'].values; mkt = fwd.mean()
        if neg:
            li = rng.permutation(len(g))[:k]; si = rng.permutation(len(g))[:k]
        else:
            li = np.argsort(-g['long_pct'].values)[:k]      # most long-favored
            si = np.argsort(-g['short_rank'].values)[:k]    # most short-favored
        lr = fwd[li].mean(); sr = -fwd[si].mean()
        for c in COSTS:
            long_n[c].append(lr - c / 1e4)
            short_n[c].append(sr - c / 1e4)
            ls_n[c].append((fwd[li].mean() - mkt) + (mkt - fwd[si].mean()) - 2 * c / 1e4)
    return {f'{s}_{int(c)}': stat(d[c]) for c in COSTS for s, d in [('long', long_n), ('short', short_n), ('ls', ls_n)]}


def main():
    print("Building fwd-to-close table from 15m cache...")
    fwd = build_fwd_to_close()
    oos = pd.read_parquet(OOS); oos['ts'] = pd.to_datetime(oos['ts'])
    df = oos.merge(fwd, on=['ts', 'Ticker'], how='inner')
    df['hm'] = df['ts'].dt.strftime('%H:%M')
    print(f"merged {len(df):,} rows  ({df['ts'].dt.normalize().nunique()} days)\n")
    print(f"{'slot':>6} {'k':>3} | {'LONG net@6/@10(t,wr)':>28} | {'SHORT net@6/@10(t,wr)':>28} | {'L/S MN@6/@10(t)':>20}")
    print('-' * 100)
    pos = []
    for T in SLOTS:
        dT = df[df['hm'] == T]
        for k in KS:
            o = evaluate(dT, k)
            l6, l10 = o['long_6'], o['long_10']; s6, s10 = o['short_6'], o['short_10']; m6, m10 = o['ls_6'], o['ls_10']
            print(f"{T:>6} {k:>3} | {l6[0]:+6.2f}/{l10[0]:+6.2f}(t{l6[1]:+4.1f},{l6[3]:.0%}) | "
                  f"{s6[0]:+6.2f}/{s10[0]:+6.2f}(t{s6[1]:+4.1f},{s6[3]:.0%}) | {m6[0]:+6.2f}/{m10[0]:+6.2f}(t{m6[1]:+4.1f})")
            for side, v6, v10 in [('long', l6, l10), ('short', s6, s10), ('ls', m6, m10)]:
                if v6[0] > 0 and v6[1] > 2:
                    pos.append((T, k, side, v6, v10))
    print('-' * 100)
    if pos:
        print("\n*** NET-POSITIVE @6bps with t>2 (candidates): ***")
        for T, k, side, v6, v10 in pos:
            print(f"  T={T} k={k} {side}: @6={v6[0]:+.2f}bps t={v6[1]:+.1f} wr={v6[3]:.0%} | @10={v10[0]:+.2f} t={v10[1]:+.1f}")
        print("\n=== robustness (time-split) + neg-control on each candidate ===")
        for T, k, side, _, _ in pos:
            dT = df[df['hm'] == T]
            r = {tag: evaluate(dT, k, **kw)[f'{side}_6']
                 for tag, kw in [('full', {}), ('half1', {'half': 1}), ('half2', {'half': 2}), ('NEG', {'neg': True})]}
            print(f"  T={T} k={k} {side}: full {r['full'][0]:+.2f}(t{r['full'][1]:+.1f}) "
                  f"h1 {r['half1'][0]:+.2f}(t{r['half1'][1]:+.1f}) h2 {r['half2'][0]:+.2f}(t{r['half2'][1]:+.1f}) "
                  f"NEG {r['NEG'][0]:+.2f}(t{r['NEG'][1]:+.1f})")
    else:
        print("\nNo cell net-positive @6bps with t>2.")


if __name__ == '__main__':
    main()
