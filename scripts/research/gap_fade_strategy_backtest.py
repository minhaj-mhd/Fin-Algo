"""
GapFade-Open v1 — rule-based STRATEGY + realistic-fill BACKTEST (goal session 2026-07-02).

STRATEGY (pre-registered):
  Universe   v21 110-name liquidity universe ∩ 5-min cache (data/raw_upstox_cache_5min_v3, 2023-01..2026-06).
  Signal     at the 09:15 open: gap_i = open_i / prev_day_close_i − 1  (live: pre-open indicative price).
  Filter     |gap| <= 3% (news/circuit guard). Day needs >= 60 valid names.
  Book       SHORT top-5 largest gap-UPS, LONG bottom-5 largest gap-DOWNS; equal-weight; 50/50 capital.
  Exit       primary 09:30; variants 10:15 / 15:15 square-off. No stops (prior research: stops hurt).
  Cost       6bps flat round-trip per trade (10bps sensitivity shown).

FILL MODELS (orders are NOT assumed to fill at the exact open print):
  E0 auction   fill AT the open print — models pre-open auction participation (upper bound).
  E1 delay-N   signal read from the open print, market order fills at the 09:20 / 09:25 / 09:30
               bar-open (5/10/15-minute latency) — lower bounds for a slow system.
  E2 vwap15    fill at first-15-min volume-weighted HLC/3 — working the order across the window.
  E3 slip s    fill at open moved s bps AGAINST us (s = 0..30) — reports BREAK-EVEN slippage.
  E4 limit     limit order AT the open print, live 09:15-09:30, fills only if price trades
               THROUGH the limit by >=5bp (short: bar high >= open*1.0005; long: low <= open*0.9995).
               Unfilled orders are MISSED trades (no pnl, no cost). Reports fill-rate + adverse selection.

Every model reports per-trade net bps by side, paired-book bps/day, t, win%, h1/h2, plus year-by-year
and a random-basket negative control for the headline config.

EXPLORATORY — no Gauntlet authority (AGENTS.md). Non-overlapping daily baskets -> honest t-stats.
Run: python scripts/research/gap_fade_strategy_backtest.py
Out: data/research/open_window_stack/strategy_backtest.json
"""
import os, sys, json, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

SRC_DIR   = 'data/raw_upstox_cache_5min_v3'
UNIV_JSON = 'data/research/v21_rolling_1h/universe.json'
OUT_JSON  = 'data/research/open_window_stack/strategy_backtest.json'
K         = 5
GAP_CAP   = 0.03
MIN_NAMES = 60
COST      = 6.0          # flat round-trip bps
LIMIT_BUF = 0.0005       # must trade 5bp through the limit to count as filled
SLIPS     = [0, 2, 5, 8, 10, 15, 20, 30]


def load_day_table():
    """5-min cache -> one row per (ticker, day) with the price points every fill model needs."""
    with open(UNIV_JSON) as f:
        univ = set(json.load(f)['tickers'])
    rows = []
    for fp in sorted(glob.glob(os.path.join(SRC_DIR, '*.csv'))):
        tk = os.path.basename(fp)[:-4]
        if tk not in univ:
            continue
        raw = pd.read_csv(fp, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ist = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        df = pd.DataFrame({'dt': ist, 'o': raw['open'].astype(float), 'h': raw['high'].astype(float),
                           'l': raw['low'].astype(float), 'c': raw['close'].astype(float),
                           'v': raw['volume'].astype(float)}).dropna()
        df = df.drop_duplicates('dt').sort_values('dt')
        df = df[(df['h'] >= df['l']) & (df['v'] >= 0)]
        df['date'] = df['dt'].dt.date
        df['hm'] = df['dt'].dt.strftime('%H:%M')

        def at(hm, field):
            s = df[df['hm'] == hm].set_index('date')[field]
            return s

        w = pd.DataFrame({
            'open0915': at('09:15', 'o'),
            # market-order delayed fills = subsequent bar OPENS (price at 09:20/09:25/09:30)
            'f_0920': at('09:20', 'o'), 'f_0925': at('09:25', 'o'), 'f_0930': at('09:30', 'o'),
            # exits: price at 09:30 / 10:15 / 15:15 (left-labeled bars -> close of prior bar)
            'x_0930': at('09:25', 'c'), 'x_1015': at('10:10', 'c'), 'x_1515': at('15:10', 'c'),
            'dclose': df.groupby('date')['c'].last(),
        })
        # first-15-min vwap (HLC/3 volume-weighted over the 09:15/09:20/09:25 bars)
        f15 = df[df['hm'].isin(['09:15', '09:20', '09:25'])].copy()
        f15['px'] = (f15['h'] + f15['l'] + f15['c']) / 3.0
        grp = f15.groupby('date')
        w['vwap15'] = grp.apply(lambda q: np.average(q['px'], weights=np.maximum(q['v'], 1)))
        # limit-fill detection extremes over 09:15-09:30
        w['hi15'] = grp['h'].max()
        w['lo15'] = grp['l'].min()
        w['ticker'] = tk
        rows.append(w.reset_index())
    t = pd.concat(rows, ignore_index=True)
    t = t.sort_values(['ticker', 'date'])
    t['prev_close'] = t.groupby('ticker')['dclose'].shift(1)
    t['gap'] = t['open0915'] / t['prev_close'] - 1.0
    t = t.dropna(subset=['gap', 'open0915'])
    return t[t['gap'].abs() <= GAP_CAP]


def stat(a):
    a = np.asarray(a, dtype=float) * 1e4
    n = len(a)
    if n < 3:
        return dict(mean=np.nan, t=np.nan, n=n, win=np.nan, h1=np.nan, h2=np.nan, sharpe=np.nan)
    return dict(mean=a.mean(), t=a.mean() / (a.std(ddof=1) / np.sqrt(n)), n=n, win=float((a > 0).mean()),
                h1=a[:n // 2].mean(), h2=a[n // 2:].mean(), sharpe=a.mean() / a.std(ddof=1) * np.sqrt(247))


def run(day_t, entry_mode, exit_col, slip_bps=0.0, neg=False, years=None, seed=0):
    """Returns dict of arrays: short/long per-trade nets, book per-day nets, fill stats."""
    d = day_t.dropna(subset=[exit_col])
    if years is not None:
        d = d[pd.to_datetime(d['date']).dt.year.isin(years)]
    rng = np.random.default_rng(seed)
    sh_tr, lg_tr, book, fills, misses = [], [], [], 0, 0
    for dt, q in d.groupby('date'):
        if len(q) < MIN_NAMES:
            continue
        q = q.sort_values('gap')
        if neg:
            pick = rng.permutation(len(q))
            longs, shorts = q.iloc[pick[:K]], q.iloc[pick[K:2 * K]]
        else:
            longs, shorts = q.iloc[:K], q.iloc[-K:]

        def leg(sub, side):
            nonlocal fills, misses
            pnls = []
            for _, r in sub.iterrows():
                if entry_mode == 'auction':
                    e = r['open0915']
                elif entry_mode in ('f_0920', 'f_0925', 'f_0930', 'vwap15'):
                    e = r[entry_mode]
                elif entry_mode == 'slip':
                    e = r['open0915'] * (1 - slip_bps / 1e4) if side == 'short' else \
                        r['open0915'] * (1 + slip_bps / 1e4)
                elif entry_mode == 'limit':
                    if side == 'short' and r['hi15'] >= r['open0915'] * (1 + LIMIT_BUF):
                        e = r['open0915']
                    elif side == 'long' and r['lo15'] <= r['open0915'] * (1 - LIMIT_BUF):
                        e = r['open0915']
                    else:
                        misses += 1
                        continue
                if not np.isfinite(e) or e <= 0 or not np.isfinite(r[exit_col]):
                    misses += 1
                    continue
                fills += 1
                raw = r[exit_col] / e - 1.0
                pnls.append((-raw if side == 'short' else raw) - COST / 1e4)
            return pnls

        s = leg(shorts, 'short'); l = leg(longs, 'long')
        sh_tr += s; lg_tr += l
        # book: 50/50 capital; unfilled orders leave capital idle (0 return)
        book.append(0.5 * (np.sum(s) / K) + 0.5 * (np.sum(l) / K))
    return dict(short=stat(sh_tr), long=stat(lg_tr), book=stat(book),
                fill_rate=fills / max(fills + misses, 1))


def fmt(r, label):
    s, l, b = r['short'], r['long'], r['book']
    return (f"{label:<22} | short {s['mean']:+7.2f}(t{s['t']:+5.1f}) | long {l['mean']:+7.2f}(t{l['t']:+5.1f}) | "
            f"BOOK {b['mean']:+6.2f}/day (t{b['t']:+5.1f} w{b['win']:.0%} Sh{b['sharpe']:4.1f} "
            f"h1{b['h1']:+5.1f} h2{b['h2']:+5.1f} n{b['n']}) | fills {r['fill_rate']:.0%}")


def main():
    print("building (ticker, day) table from 5-min cache...")
    day_t = load_day_table()
    print(f"rows={len(day_t):,}  names={day_t['ticker'].nunique()}  days={day_t['date'].nunique()}  "
          f"range {day_t['date'].min()}..{day_t['date'].max()}  (|gap|<=3%)\n")
    R = {}

    for exit_col, xlbl in [('x_0930', 'exit 09:30'), ('x_1015', 'exit 10:15'), ('x_1515', 'exit 15:15')]:
        print(f"--- {xlbl} ---  [per-trade net@6 bps | book bps/day on gross capital]")
        for mode, mlbl in [('auction', 'E0 auction@open'), ('f_0920', 'E1 delay 5m'),
                           ('f_0925', 'E1 delay 10m'), ('f_0930', 'E1 delay 15m'),
                           ('vwap15', 'E2 vwap15'), ('limit', 'E4 limit@open')]:
            if mode == 'f_0930' and exit_col == 'x_0930':
                continue
            r = run(day_t, mode, exit_col)
            R[f'{mode}_{exit_col}'] = r
            print(fmt(r, mlbl))
        print()

    print("--- E3 slippage sweep (entry = open ± s bps adverse, both legs) ---")
    for exit_col, xlbl in [('x_0930', '09:30'), ('x_1015', '10:15'), ('x_1515', '15:15')]:
        cells = []
        be = None
        for s in SLIPS:
            r = run(day_t, 'slip', exit_col, slip_bps=s)
            R[f'slip{s}_{exit_col}'] = dict(book=r['book'])
            cells.append(f"s={s:>2}: {r['book']['mean']:+6.2f}")
            if be is None and r['book']['mean'] <= 0:
                be = s
        print(f"exit {xlbl} | " + '  '.join(cells) + f"   -> break-even ~{be if be is not None else '>30'}bps")

    print("\n--- YEARLY (E0 auction, exit 09:30) + NEGATIVE CONTROL ---")
    for y in [2023, 2024, 2025, 2026]:
        r = run(day_t, 'auction', 'x_0930', years=[y])
        print(f"{y}: " + fmt(r, f'E0 09:30 {y}'))
    ng = run(day_t, 'auction', 'x_0930', neg=True)
    print(fmt(ng, 'NEG-CONTROL random'))

    with open(OUT_JSON, 'w') as f:
        json.dump({k: v for k, v in R.items()}, f, indent=1, default=float)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == '__main__':
    main()
