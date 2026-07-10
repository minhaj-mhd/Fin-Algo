"""
EOD cross-sectional mean-reversion — intraday, flat by 15:15 (user constraint 9:15-3:15).

Hypothesis (from dualtf entry/exit research, EOD reversion ramp p<0.01 but on the restricted
1h-top-3 universe): fade the day's intraday move into the close. At decision time T, rank the
FULL universe by intraday strength (close_T / open_0915 - 1); LONG the weakest (oversold bounce),
SHORT the strongest (overbought fade); hold to the 15:15 close. Single round trip.

Reports, per (T, k, side), NET-of-cost @6 and @10 bps with t-stat + win-rate, a market-neutral
(cross-sectional, demeaned) variant, a negative control (random picks), and a TIME-SPLIT
(first vs second half of dates) for OOS robustness. Lookahead-safe: decision at T uses only
bars <= T; outcome is strictly T->close.

  python scripts/analysis/eod_reversion.py
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
OPEN_T = dtime(9, 15)
CLOSE_T = dtime(15, 15)
DECISIONS = ['12:15', '13:15', '14:00', '14:15', '14:45']   # entry times to sweep
KS = [3, 5, 10]
COSTS = [6.0, 10.0]
MIN_NAMES = 30


def load_one(path):
    tk = os.path.splitext(os.path.basename(path))[0]
    raw = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = pd.DataFrame({'dt': dt, 'open': raw['open'].astype(float), 'close': raw['close'].astype(float)})
    df = df.dropna().sort_values('dt')
    t = df['dt'].dt.time
    df = df[(t >= OPEN_T) & (t <= CLOSE_T)]
    df['date'] = df['dt'].dt.normalize()
    df['hm'] = df['dt'].dt.strftime('%H:%M')
    df['Ticker'] = tk
    return df


def build_panel():
    """Per (date, ticker): open_0915, close at each decision T, close_1515."""
    rows = []
    files = glob.glob(f'{CACHE}/*.csv')
    for p in tqdm(files, desc='load'):
        try:
            df = load_one(p)
        except Exception:
            continue
        # open at 09:15
        op = df[df['hm'] == '09:15'].set_index('date')['open']
        cl = df[df['hm'] == '15:15'].set_index('date')['close']
        rec = {'date': op.index, 'Ticker': df['Ticker'].iloc[0], 'open0': op.values}
        base = pd.DataFrame(rec).set_index('date')
        base['close_eod'] = cl.reindex(base.index).values
        for T in DECISIONS:
            cT = df[df['hm'] == T].set_index('date')['close']
            base[f'cT_{T}'] = cT.reindex(base.index).values
        rows.append(base.reset_index())
    panel = pd.concat(rows, ignore_index=True)
    return panel.dropna(subset=['open0', 'close_eod'])


def evaluate(panel, T, k, neg=False, half=None):
    col = f'cT_{T}'
    df = panel.dropna(subset=[col]).copy()
    if half == 1:
        df = df[df['date'] < df['date'].quantile(0.5)]
    elif half == 2:
        df = df[df['date'] >= df['date'].quantile(0.5)]
    df['strength'] = df[col] / df['open0'] - 1.0        # intraday move to T (decision-time info)
    df['fwd'] = df['close_eod'] / df[col] - 1.0          # T -> close outcome
    rng = np.random.default_rng(0)
    long_net = {c: [] for c in COSTS}; short_net = {c: [] for c in COSTS}
    ls_net = {c: [] for c in COSTS}                       # market-neutral long_loser+short_winner
    for date, g in df.groupby('date'):
        if len(g) < MIN_NAMES:
            continue
        if neg:
            order = rng.permutation(len(g))
        else:
            order = np.argsort(g['strength'].values)      # ascending: losers first
        fwd = g['fwd'].values
        losers = fwd[order[:k]]                            # weakest -> expect bounce (long)
        winners = fwd[order[-k:]]                          # strongest -> expect fade (short)
        mkt = fwd.mean()
        for c in COSTS:
            long_net[c].append(losers.mean() - c / 1e4)
            short_net[c].append(-winners.mean() - c / 1e4)
            # market-neutral: demean both legs by market move, two legs => 2x cost
            ls_net[c].append((losers.mean() - mkt) + (mkt - winners.mean()) - 2 * c / 1e4)
    def stat(arr):
        a = np.array(arr) * 1e4
        n = len(a); m = a.mean(); t = m / (a.std(ddof=1) / np.sqrt(n)) if n > 2 else 0
        return m, t, n
    out = {}
    for c in COSTS:
        out[f'long_{int(c)}'] = stat(long_net[c])
        out[f'short_{int(c)}'] = stat(short_net[c])
        out[f'ls_{int(c)}'] = stat(ls_net[c])
    return out


def main():
    print("Building EOD reversion panel from 15m cache...")
    panel = build_panel()
    nd = panel['date'].nunique(); nt = panel['Ticker'].nunique()
    print(f"panel: {len(panel):,} (date,ticker) rows  days={nd}  tickers={nt}\n")
    print(f"{'T':>6} {'k':>3} | {'LONG net@6/@10 (t)':>26} | {'SHORT net@6/@10 (t)':>26} | {'L/S MN net@6/@10 (t)':>26}")
    print('-' * 100)
    best = []
    for T in DECISIONS:
        for k in KS:
            o = evaluate(panel, T, k)
            l6, l10 = o['long_6'], o['long_10']; s6, s10 = o['short_6'], o['short_10']; m6, m10 = o['ls_6'], o['ls_10']
            print(f"{T:>6} {k:>3} | {l6[0]:+6.2f}/{l10[0]:+6.2f} (t{l6[1]:+4.1f}/{l10[1]:+4.1f}) | "
                  f"{s6[0]:+6.2f}/{s10[0]:+6.2f} (t{s6[1]:+4.1f}/{s10[1]:+4.1f}) | "
                  f"{m6[0]:+6.2f}/{m10[0]:+6.2f} (t{m6[1]:+4.1f}/{m10[1]:+4.1f})")
            for side, v6, v10 in [('long', l6, l10), ('short', s6, s10), ('ls', m6, m10)]:
                if v10[0] > 0 and v10[1] > 2:
                    best.append((T, k, side, v10))
    print('-' * 100)
    print("net = bps per trade after cost; t = t-stat across days. L/S MN = market-neutral long-loser+short-winner (2x cost).")
    if best:
        print("\n*** NET-POSITIVE @10bps with t>2: ***")
        for T, k, side, v in best:
            print(f"  T={T} k={k} {side}: {v[0]:+.2f}bps t={v[1]:+.1f} n={v[2]}")
    else:
        print("\nNo cell net-positive @10bps with t>2.")
    # robustness + neg-control on the best @6 cell (long side, T=14:45 k=5 as a default probe)
    print("\n=== robustness (time-split) + neg-control @6bps, T=14:45 k=5 ===")
    for tag, kw in [('full', {}), ('half1', {'half': 1}), ('half2', {'half': 2}), ('NEG-control', {'neg': True})]:
        o = evaluate(panel, '14:45', 5, **kw)
        print(f"  {tag:12s} long@6 {o['long_6'][0]:+.2f} (t{o['long_6'][1]:+.1f})  "
              f"short@6 {o['short_6'][0]:+.2f} (t{o['short_6'][1]:+.1f})  "
              f"LS_MN@6 {o['ls_6'][0]:+.2f} (t{o['ls_6'][1]:+.1f})")


if __name__ == '__main__':
    main()
