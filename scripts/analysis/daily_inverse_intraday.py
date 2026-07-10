"""
Daily INVERSE (reversion) trade, executed INTRADAY (open->close, flat by 3:15) — unites the
user's threads: inverse trade + intraday-only + daily signal.

Signal known at day-t OPEN = yesterday's close-to-close return r[t-1] (strength). Inverse trade:
LONG bottom-k (yesterday's losers, expect bounce), SHORT top-k (yesterday's winners, expect fade).
Decompose the realized move into:
  intraday  = close[t]/open[t]   - 1   (TRADABLE in 9:15-3:15, flat by close)
  overnight = open[t]/close[t-1] - 1   (gap; NOT intraday-capturable)
  c2c       = close[t]/close[t-1]- 1
Reports NET-of-cost @6/@10 per side for each leg-timing, with t-stat, time-split, neg-control.
Fully causal (signal uses only data <= close[t-1]). Exploratory — NO Gauntlet authority.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())
CSV = 'data/ranking_data_daily_macro_v2.csv'
KS = [3, 5, 10]
COSTS = [6.0, 10.0]
MIN_NAMES = 30


def stat(a):
    a = np.array(a) * 1e4; n = len(a)
    return (a.mean(), a.mean() / (a.std(ddof=1) / np.sqrt(n)) if n > 2 else 0, n, (a > 0).mean())


def main():
    print("loading daily OHLC...")
    df = pd.read_csv(CSV, usecols=['DateTime', 'Ticker', 'Open', 'Close'])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(['Ticker', 'DateTime'])
    g = df.groupby('Ticker', group_keys=False)
    df['prev_close'] = g['Close'].shift(1)
    df['sig'] = df['prev_close'] / g['Close'].shift(2) - 1.0       # r[t-1], known at open[t]
    df['intraday'] = df['Close'] / df['Open'] - 1.0
    df['overnight'] = df['Open'] / df['prev_close'] - 1.0
    df['c2c'] = df['Close'] / df['prev_close'] - 1.0
    df = df.dropna(subset=['sig', 'intraday', 'overnight', 'c2c'])
    print(f"rows={len(df):,}  days={df['DateTime'].nunique()}  range {df['DateTime'].min().date()}..{df['DateTime'].max().date()}\n")

    def run(leg, k, neg=False, half=None):
        d = df
        if half == 1:
            d = d[d['DateTime'] < d['DateTime'].quantile(0.5)]
        elif half == 2:
            d = d[d['DateTime'] >= d['DateTime'].quantile(0.5)]
        rng = np.random.default_rng(0)
        long_n = {c: [] for c in COSTS}; short_n = {c: [] for c in COSTS}
        for dt, q in d.groupby('DateTime'):
            if len(q) < MIN_NAMES:
                continue
            ret = q[leg].values
            if neg:
                lo = rng.permutation(len(q))[:k]; hi = rng.permutation(len(q))[:k]
            else:
                order = np.argsort(q['sig'].values)        # ascending: losers first
                lo = order[:k]; hi = order[-k:]
            # INVERSE: long losers (lo), short winners (hi)
            for c in COSTS:
                long_n[c].append(ret[lo].mean() - c / 1e4)
                short_n[c].append(-ret[hi].mean() - c / 1e4)
        return {f'long_{int(c)}': stat(long_n[c]) for c in COSTS} | {f'short_{int(c)}': stat(short_n[c]) for c in COSTS}

    print(f"{'leg':>9} {'k':>3} | {'LONG-loser net@6/@10(t)':>26} | {'SHORT-winner net@6/@10(t)':>27}")
    print('-' * 80)
    pos = []
    for leg in ['intraday', 'overnight', 'c2c']:
        for k in KS:
            o = run(leg, k)
            l6, l10 = o['long_6'], o['long_10']; s6, s10 = o['short_6'], o['short_10']
            print(f"{leg:>9} {k:>3} | {l6[0]:+6.2f}/{l10[0]:+6.2f}(t{l6[1]:+4.1f},{l6[3]:.0%}) | "
                  f"{s6[0]:+6.2f}/{s10[0]:+6.2f}(t{s6[1]:+4.1f},{s6[3]:.0%})")
            for side, v6, v10 in [('long', l6, l10), ('short', s6, s10)]:
                if v6[0] > 0 and v6[1] > 2:
                    pos.append((leg, k, side, v6, v10))
    print('-' * 80)
    print("INVERSE trade: long yesterday's losers / short yesterday's winners. net=bps/trade after cost.")
    if pos:
        print("\n*** NET-POSITIVE @6bps t>2 — robustness (time-split + neg-control): ***")
        for leg, k, side, v6, v10 in pos:
            full = run(leg, k); h1 = run(leg, k, half=1); h2 = run(leg, k, half=2); ng = run(leg, k, neg=True)
            key = f'{side}_6'
            print(f"  {leg} k={k} {side}: @6={v6[0]:+.2f}(t{v6[1]:+.1f}) @10={v10[0]:+.2f}(t{v10[1]:+.1f}) | "
                  f"h1 {h1[key][0]:+.2f}(t{h1[key][1]:+.1f}) h2 {h2[key][0]:+.2f}(t{h2[key][1]:+.1f}) NEG {ng[key][0]:+.2f}(t{ng[key][1]:+.1f})")
    else:
        print("\nNo intraday/overnight/c2c cell net-positive @6bps with t>2.")


if __name__ == '__main__':
    main()
