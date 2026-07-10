"""
Open-window trade STACKING research — can we get >=2-3 net-positive trade events/day @6bps
from the 15-min cache alone?  (Goal session 2026-07-02; extends the certified-exploratory
open gap-reversal edge in scripts/analysis/daily_inverse_intraday.py.)

Tests (all cross-sectional top/bottom-k on the v21 110-name liquid universe, flat cost/round-trip):
  T1  Open reversal (signal = yesterday's c2c, causal pre-open): SHORT top-k winners at the
      09:15 open — EXIT SWEEP 09:30/10:15/11:15/12:15/13:15/14:15/15:15. Answers the
      pre-registered "cover@09:30 vs hold-to-close" question. Long-losers shown for reference.
  T1b Same trade, signal = overnight GAP (open/prevclose-1; known only at the auction print —
      slightly optimistic for an at-open fill) and combined rank (c2c + gap).
  T2  Second-window trade: signal = realized 09:15->09:30 return; enter 09:30,
      exits 10:15/11:15/15:15. Both directions read from gross.
  T3  Close-window trade: signal = intraday-so-far return at 14:15; enter 14:15, exit 15:15.
  T4  Model-free time-of-day CROSS-SECTIONAL DISPERSION profile of fwd-15m/fwd-1h/fwd-to-close
      returns — where does a fixed IC~0.03 buy the most gross bps?

Bars are LEFT-labeled (09:15 bar covers 09:15->09:30; 15:15 bar = close stub w/ auction).
Day close (for c2c signal) = close of 15:15 stub bar. Intraday exit "15:15" = close of 15:00 bar.

EXPLORATORY ONLY — no Gauntlet authority, no verdict (AGENTS.md). Point estimates + t-stats on
non-overlapping daily baskets (1 trade/day/strategy -> t-stats are honest here, unlike rolling panels).
Robustness (half-split + random-k negative control) printed for every net@6>0, t>2 cell.

Run: python scripts/research/open_window_stack.py
Out: data/research/open_window_stack/results.json (+ stdout tables)
"""
import os, sys, json, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

SRC_DIR   = 'data/raw_upstox_cache_15min_3y'
UNIV_JSON = 'data/research/v21_rolling_1h/universe.json'
OUT_DIR   = 'data/research/open_window_stack'
KS        = [3, 5, 10]
COSTS     = [6.0, 10.0]
MIN_NAMES = 60
os.makedirs(OUT_DIR, exist_ok=True)

# wall-clock price points we need per (ticker, day):  label -> (bar_start_time, field)
PRICE_POINTS = {
    'open':  ('09:15', 'open'),    # 09:15 auction/open price
    'p0930': ('09:15', 'close'),   # price at 09:30
    'p1015': ('10:00', 'close'),
    'p1115': ('11:00', 'close'),
    'p1215': ('12:00', 'close'),
    'p1315': ('13:00', 'close'),
    'p1415': ('14:00', 'close'),
    'p1515': ('15:00', 'close'),   # intraday exit price at 15:15 ("flat by 3:15")
    'dclose': ('15:15', 'close'),  # official day close (stub bar w/ closing auction)
}
EXITS = ['p0930', 'p1015', 'p1115', 'p1215', 'p1315', 'p1415', 'p1515']
EXIT_LBL = {'p0930': '09:30', 'p1015': '10:15', 'p1115': '11:15', 'p1215': '12:15',
            'p1315': '13:15', 'p1415': '14:15', 'p1515': '15:15'}


def load_panel():
    with open(UNIV_JSON) as f:
        univ = set(json.load(f)['tickers'])
    rows = []
    for fp in sorted(glob.glob(os.path.join(SRC_DIR, '*.csv'))):
        tk = os.path.basename(fp)[:-4]
        if tk not in univ:
            continue
        raw = pd.read_csv(fp, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        df = pd.DataFrame({'dt': dt, 'open': raw['open'].astype(float),
                           'high': raw['high'].astype(float), 'low': raw['low'].astype(float),
                           'close': raw['close'].astype(float), 'vol': raw['volume'].astype(float)})
        df = df.dropna().drop_duplicates('dt').sort_values('dt')
        # bar hygiene (v21): drop frozen / zero-volume bars
        df = df[(df['high'] > df['low']) & (df['vol'] > 0)]
        df['date'] = df['dt'].dt.date
        df['hm'] = df['dt'].dt.strftime('%H:%M')
        piv = {}
        for lbl, (hm, field) in PRICE_POINTS.items():
            s = df[df['hm'] == hm].set_index('date')[field]
            piv[lbl] = s
        w = pd.DataFrame(piv)
        w['ticker'] = tk
        rows.append(w.reset_index())
    panel = pd.concat(rows, ignore_index=True).rename(columns={'index': 'date'})
    panel = panel.sort_values(['ticker', 'date'])
    g = panel.groupby('ticker', group_keys=False)
    panel['prev_close']  = g['dclose'].shift(1)
    panel['prev_close2'] = g['dclose'].shift(2)
    panel['sig_c2c'] = panel['prev_close'] / panel['prev_close2'] - 1.0   # causal pre-open
    panel['sig_gap'] = panel['open'] / panel['prev_close'] - 1.0          # known at auction print
    panel['sig_first15'] = panel['p0930'] / panel['open'] - 1.0           # known at 09:30
    panel['sig_sofar1415'] = panel['p1415'] / panel['open'] - 1.0         # known at 14:15
    return panel


def stat(a):
    a = np.asarray(a, dtype=float) * 1e4
    n = len(a)
    if n < 3:
        return (np.nan, np.nan, n, np.nan)
    return (a.mean(), a.mean() / (a.std(ddof=1) / np.sqrt(n)), n, float((a > 0).mean()))


def basket(panel, sig_col, entry_col, exit_col, k, side, neg=False, half=None, seed=0):
    """One trade/day: enter entry_col price, exit exit_col price, names picked by sig_col rank.
    side='short_top' | 'long_bot' | 'long_top' | 'short_bot'. Returns list of per-day net-gross
    (gross only; cost subtracted by caller)."""
    d = panel.dropna(subset=[sig_col, entry_col, exit_col])
    if half is not None:
        med = d['date'].astype('datetime64[ns]').quantile(0.5)
        d = d[d['date'].astype('datetime64[ns]') < med] if half == 1 else d[d['date'].astype('datetime64[ns]') >= med]
    rng = np.random.default_rng(seed)
    out = []
    for dt, q in d.groupby('date'):
        if len(q) < MIN_NAMES:
            continue
        fwd = (q[exit_col].values / q[entry_col].values) - 1.0
        if neg:
            idx = rng.permutation(len(q))[:k]
        else:
            order = np.argsort(q[sig_col].values)
            idx = order[-k:] if 'top' in side else order[:k]
        r = fwd[idx].mean()
        out.append(-r if side.startswith('short') else r)
    return out


def show(title, rows, header):
    print(f"\n=== {title} ===")
    print(header); print('-' * len(header))
    for r in rows:
        print(r)


def main():
    print("loading 15-min cache -> wall-clock price panel (110-name universe)...")
    panel = load_panel()
    ndays = panel['date'].nunique()
    print(f"rows={len(panel):,}  names={panel['ticker'].nunique()}  days={ndays}  "
          f"range {panel['date'].min()}..{panel['date'].max()}")
    results = {}

    # ---------- T1: open reversal exit sweep (signal = prev c2c) ----------
    rows = []
    for k in KS:
        for ex in EXITS:
            g_s = basket(panel, 'sig_c2c', 'open', ex, k, 'short_top')
            g_l = basket(panel, 'sig_c2c', 'open', ex, k, 'long_bot')
            ms, ts, n, ws = stat(g_s); ml, tl, _, wl = stat(g_l)
            results[f'T1_short_k{k}_{ex}'] = dict(gross=ms, t=ts, n=n, win=ws,
                                                  net6=ms - 6, net10=ms - 10)
            rows.append(f"k={k:>2} exit {EXIT_LBL[ex]} | SHORT-win net@6 {ms-6:+7.2f} @10 {ms-10:+7.2f} "
                        f"(t{ts:+5.1f} w{ws:.0%} n{n}) | LONG-los net@6 {ml-6:+7.2f} (t{tl:+5.1f})")
    show("T1: open reversal (sig=prev c2c) — entry 09:15 open, exit sweep [bps/trade]", rows,
         "   k  exit  | SHORT yesterday-winners           | LONG yesterday-losers")

    # ---------- T1b: gap signal + combined ----------
    panel['sig_comb'] = panel.groupby('date')['sig_c2c'].rank(pct=True) + \
                        panel.groupby('date')['sig_gap'].rank(pct=True)
    rows = []
    for sig in ['sig_gap', 'sig_comb']:
        for k in KS:
            for ex in ['p0930', 'p1015', 'p1515']:
                m, t, n, w = stat(basket(panel, sig, 'open', ex, k, 'short_top'))
                results[f'T1b_{sig}_short_k{k}_{ex}'] = dict(gross=m, t=t, n=n, net6=m - 6)
                rows.append(f"{sig:>9} k={k:>2} exit {EXIT_LBL[ex]} | SHORT-top net@6 {m-6:+7.2f} @10 {m-10:+7.2f} (t{t:+5.1f} w{w:.0%})")
    show("T1b: gap / combined signal shorts at open (gap known only at auction print!)", rows,
         "   signal  k  exit  | SHORT top")

    # ---------- T2: second-window trade at 09:30 (sig = first-15m return) ----------
    rows = []
    for k in KS:
        for ex in ['p1015', 'p1115', 'p1515']:
            gt, tt, n, wt = stat(basket(panel, 'sig_first15', 'p0930', ex, k, 'long_top'))
            gb, tb, _, wb = stat(basket(panel, 'sig_first15', 'p0930', ex, k, 'long_bot'))
            results[f'T2_k{k}_{ex}'] = dict(gross_top=gt, t_top=tt, gross_bot=gb, t_bot=tb, n=n)
            rows.append(f"k={k:>2} exit {EXIT_LBL[ex]} | TOP(15m-winners) gross {gt:+7.2f} (t{tt:+5.1f}) | "
                        f"BOT(15m-losers) gross {gb:+7.2f} (t{tb:+5.1f})  [fade=short-top/long-bot]")
    show("T2: 09:30 second-window trade (sig = realized 09:15->09:30 ret) — GROSS long-basket bps", rows,
         "   k  exit  | top basket (follow=+, fade=-)      | bottom basket")

    # ---------- T3: close-window trade at 14:15 (sig = intraday-so-far) ----------
    rows = []
    for k in KS:
        gt, tt, n, wt = stat(basket(panel, 'sig_sofar1415', 'p1415', 'p1515', k, 'long_top'))
        gb, tb, _, wb = stat(basket(panel, 'sig_sofar1415', 'p1415', 'p1515', k, 'long_bot'))
        results[f'T3_k{k}'] = dict(gross_top=gt, t_top=tt, gross_bot=gb, t_bot=tb, n=n)
        rows.append(f"k={k:>2} 14:15->15:15 | TOP(day-winners) gross {gt:+7.2f} (t{tt:+5.1f}) | "
                    f"BOT(day-losers) gross {gb:+7.2f} (t{tb:+5.1f})")
    show("T3: close-window trade (sig = 09:15->14:15 ret) — GROSS long-basket bps", rows,
         "   k             | top basket                        | bottom basket")

    # ---------- T4: time-of-day cross-sectional dispersion ----------
    fwd_pairs = [('open', 'p1015', '09:15 fwd-1h'), ('p1015', 'p1115', '10:15 fwd-1h'),
                 ('p1115', 'p1215', '11:15 fwd-1h'), ('p1215', 'p1315', '12:15 fwd-1h'),
                 ('p1315', 'p1415', '13:15 fwd-1h'), ('p1415', 'p1515', '14:15 fwd-1h'),
                 ('open', 'p0930', '09:15 fwd-15m'),
                 ('open', 'p1515', '09:15 fwd-to-close'), ('p1015', 'p1515', '10:15 fwd-to-close')]
    rows = []
    for a, b, lbl in fwd_pairs:
        d = panel.dropna(subset=[a, b])
        sig = d.groupby('date').apply(
            lambda q: np.std((q[b].values / q[a].values) - 1.0) * 1e4 if len(q) >= MIN_NAMES else np.nan)
        sig = sig.dropna()
        m = sig.mean()
        results[f'T4_{lbl}'] = dict(cs_sigma_bps=m)
        # what IC=0.03 buys at top-10 of ~110 (E[mean z of top decile] ~ 1.75): gross ~ IC*sigma*1.75
        rows.append(f"{lbl:>20} | cs-sigma {m:7.1f} bps | IC=0.03 top-decile gross ~ {0.03*m*1.75:5.1f} bps")
    show("T4: cross-sectional dispersion by window (model-free)", rows,
         "               window |  dispersion  |  what a fixed IC~0.03 buys")

    # ---------- robustness for every net@6>0 & t>2 cell in T1/T1b ----------
    print("\n=== ROBUSTNESS (half-split + random-k neg-control) for net@6>0, t>2 cells ===")
    checked = 0
    for key, v in list(results.items()):
        if not key.startswith(('T1_', 'T1b_')) or v.get('net6') is None:
            continue
        if v['net6'] > 0 and abs(v.get('t', 0)) > 2:
            sig = 'sig_c2c' if key.startswith('T1_') else ('sig_gap' if 'sig_gap' in key else 'sig_comb')
            k = int(key.split('_k')[1].split('_')[0]); ex = 'p' + key.split('_p')[-1]
            h1 = stat(basket(panel, sig, 'open', ex, k, 'short_top', half=1))
            h2 = stat(basket(panel, sig, 'open', ex, k, 'short_top', half=2))
            ng = stat(basket(panel, sig, 'open', ex, k, 'short_top', neg=True))
            results[key]['h1'] = h1[0] - 6; results[key]['h2'] = h2[0] - 6; results[key]['neg'] = ng[0] - 6
            print(f"{key:>28}: net@6 {v['net6']:+6.2f}(t{v['t']:+4.1f}) | h1 {h1[0]-6:+6.2f}(t{h1[1]:+4.1f}) "
                  f"h2 {h2[0]-6:+6.2f}(t{h2[1]:+4.1f}) | NEG {ng[0]-6:+6.2f}(t{ng[1]:+4.1f})")
            checked += 1
    if not checked:
        print("(no qualifying cells)")

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print(f"\nsaved -> {OUT_DIR}/results.json")


if __name__ == '__main__':
    main()
