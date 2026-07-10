"""
KILL-TEST battery for the open gap-fade short (found by open_window_stack.py T1b:
short top-k overnight gap-ups at 09:15 open -> net@6 +18..+31bps/trade, t 6-14).

A gap-sorted trade ENTERED AT THE OPEN IT SORTS ON is exposed to the noisy-print artifact:
if a name's recorded open is idiosyncratically high (thin auction print), it ranks top AND we
"sell" at that unattainable price; reversion of print noise fakes an edge. These checks try to
kill the result:

  C1  DELAYED ENTRY: same gap signal, enter at 09:30 price (p0930) — fully causal & executable
      (open is known by 09:16). If edge dies -> it lived in the first-print noise.
  C2  CONSERVATIVE FILL: enter at the 09:15 bar's (H+L+C)/3 instead of Open — you get an
      average continuous-market fill in the first 15 min, not the auction print.
  C3  LIQUIDITY: split universe into top-40 ADV vs rest. Thin names -> noisier prints; a real
      edge should survive (attenuated) in the most-liquid names.
  C4  GAP-MAGNITUDE CAP: exclude |gap|>3% (news/circuit-band names); does a moderate-gap
      version survive? Also report avg |gap| of the shorted basket.
  C5  LONG side (bottom-k gap-downs) for reference.
  C6  Year-by-year net@6 for headline cells (k=5 and k=10, exits 09:30/10:15/15:15).
  C7  Cross-dataset: same trade on the DAILY macro CSV (2016+, open->close leg only) —
      independent vendor path; catches 15m-cache-specific artifacts.

EXPLORATORY — no Gauntlet authority. Cost = flat round-trip bps subtracted once.
Run: python scripts/research/gap_fade_artifact_checks.py
"""
import os, sys, json, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts', 'research'))
from open_window_stack import load_panel, stat, MIN_NAMES  # reuse identical construction

UNIV_JSON = 'data/research/v21_rolling_1h/universe.json'
DAILY_CSV = 'data/ranking_data_daily_macro_v2.csv'
OUT_DIR   = 'data/research/open_window_stack'
KS = [3, 5, 10]


def run_cell(panel, sig_col, entry_col, exit_col, k, side='short_top', gap_cap=None,
             sub=None, years=None, seed=0, neg=False):
    d = panel.dropna(subset=[sig_col, entry_col, exit_col])
    if gap_cap is not None:
        d = d[d['sig_gap'].abs() <= gap_cap]
    if sub is not None:
        d = d[d['ticker'].isin(sub)]
    if years is not None:
        yr = pd.to_datetime(d['date']).dt.year
        d = d[yr.isin(years)]
    rng = np.random.default_rng(seed)
    out, gaps = [], []
    for dt, q in d.groupby('date'):
        if len(q) < max(20, MIN_NAMES if sub is None else 25):
            continue
        fwd = (q[exit_col].values / q[entry_col].values) - 1.0
        if neg:
            idx = rng.permutation(len(q))[:k]
        else:
            order = np.argsort(q[sig_col].values)
            idx = order[-k:] if 'top' in side else order[:k]
        r = fwd[idx].mean()
        out.append(-r if side.startswith('short') else r)
        gaps.append(np.abs(q['sig_gap'].values[idx]).mean())
    return out, (np.mean(gaps) * 1e4 if gaps else np.nan)


def fmt(o, cost=6.0):
    m, t, n, w = stat(o)
    return f"net@6 {m-cost:+7.2f} (t{t:+5.1f} w{w:.0%} n{n})"


def main():
    print("loading panel (same construction as open_window_stack)...")
    panel = load_panel()
    # (H+L+C)/3 conservative fill for the 09:15 bar needs H/L of that bar — rebuild quickly
    with open(UNIV_JSON) as f:
        univ = json.load(f)
    adv = pd.Series(univ['adv_inr'])
    top40 = set(adv.sort_values(ascending=False).head(40).index)
    rest = set(univ['tickers']) - top40

    hlc = []
    for fp in sorted(glob.glob('data/raw_upstox_cache_15min_3y/*.csv')):
        tk = os.path.basename(fp)[:-4]
        if tk not in set(univ['tickers']):
            continue
        raw = pd.read_csv(fp, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        raw['ist'] = dt
        first = raw[(raw['ist'].dt.strftime('%H:%M') == '09:15') & (raw['high'] > raw['low']) & (raw['volume'] > 0)]
        h = first.copy()
        h['date'] = h['ist'].dt.date
        h['fill_hlc3'] = (h['high'] + h['low'] + h['close']) / 3.0
        hlc.append(h[['date', 'fill_hlc3']].assign(ticker=tk))
    panel = panel.merge(pd.concat(hlc, ignore_index=True), on=['date', 'ticker'], how='left')

    print(f"panel: {len(panel):,} rows, {panel['date'].nunique()} days\n")
    R = {}

    print("=== C1: DELAYED ENTRY 09:30 (gap signal, causal & executable) vs baseline open entry ===")
    for k in KS:
        for ex, lbl in [('p1015', '10:15'), ('p1115', '11:15'), ('p1515', '15:15')]:
            base, _ = run_cell(panel, 'sig_gap', 'open', ex, k)
            dele, _ = run_cell(panel, 'sig_gap', 'p0930', ex, k)
            R[f'C1_k{k}_{ex}'] = dict(base=stat(base)[0] - 6, delayed=stat(dele)[0] - 6)
            print(f"k={k:>2} exit {lbl} | entry@open {fmt(base)} | entry@09:30 {fmt(dele)}")

    print("\n=== C2: CONSERVATIVE FILL (HLC/3 of 09:15 bar) vs auction-open fill ===")
    for k in KS:
        for ex, lbl in [('p0930', '09:30'), ('p1015', '10:15'), ('p1515', '15:15')]:
            base, _ = run_cell(panel, 'sig_gap', 'open', ex, k)
            cons, _ = run_cell(panel, 'sig_gap', 'fill_hlc3', ex, k)
            R[f'C2_k{k}_{ex}'] = dict(open_fill=stat(base)[0] - 6, hlc3_fill=stat(cons)[0] - 6)
            print(f"k={k:>2} exit {lbl} | fill@open {fmt(base)} | fill@HLC/3 {fmt(cons)}")

    print("\n=== C3: LIQUIDITY SPLIT (top-40 ADV vs rest-70) — gap signal, open entry ===")
    for k in [3, 5]:
        for ex, lbl in [('p0930', '09:30'), ('p1515', '15:15')]:
            liq, gl = run_cell(panel, 'sig_gap', 'open', ex, k, sub=top40)
            ill, gi = run_cell(panel, 'sig_gap', 'open', ex, k, sub=rest)
            R[f'C3_k{k}_{ex}'] = dict(top40=stat(liq)[0] - 6, rest=stat(ill)[0] - 6)
            print(f"k={k:>2} exit {lbl} | TOP-40 {fmt(liq)} avg|gap|{gl:4.0f}bp | REST-70 {fmt(ill)} avg|gap|{gi:4.0f}bp")

    print("\n=== C4: GAP-MAGNITUDE CAP |gap|<=3% (drop news/circuit names) ===")
    for k in KS:
        for ex, lbl in [('p0930', '09:30'), ('p1515', '15:15')]:
            cap, gc = run_cell(panel, 'sig_gap', 'open', ex, k, gap_cap=0.03)
            unc, gu = run_cell(panel, 'sig_gap', 'open', ex, k)
            R[f'C4_k{k}_{ex}'] = dict(capped=stat(cap)[0] - 6, uncapped=stat(unc)[0] - 6)
            print(f"k={k:>2} exit {lbl} | capped {fmt(cap)} avg|gap|{gc:4.0f}bp | uncapped {fmt(unc)} avg|gap|{gu:4.0f}bp")

    print("\n=== C5: LONG bottom-k gap-DOWNS (reference) ===")
    for k in [5, 10]:
        for ex, lbl in [('p0930', '09:30'), ('p1515', '15:15')]:
            lng, _ = run_cell(panel, 'sig_gap', 'open', ex, k, side='long_bot')
            print(f"k={k:>2} exit {lbl} | LONG gap-downs {fmt(lng)}")

    print("\n=== C6: YEAR-BY-YEAR net@6 (gap signal, open entry) ===")
    for k in [5, 10]:
        for ex, lbl in [('p0930', '09:30'), ('p1515', '15:15')]:
            cells = []
            for y in [2022, 2023, 2024, 2025, 2026]:
                o, _ = run_cell(panel, 'sig_gap', 'open', ex, k, years=[y])
                m, t, n, w = stat(o)
                cells.append(f"{y}:{m-6:+6.1f}(t{t:+4.1f})")
            print(f"k={k:>2} exit {lbl} | " + '  '.join(cells))

    print("\n=== C7: CROSS-DATASET — daily CSV (independent vendor), open->close leg ===")
    df = pd.read_csv(DAILY_CSV, usecols=['DateTime', 'Ticker', 'Open', 'Close'])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(['Ticker', 'DateTime'])
    g = df.groupby('Ticker', group_keys=False)
    df['prev_close'] = g['Close'].shift(1)
    df['sig_gap'] = df['Open'] / df['prev_close'] - 1.0
    df['intraday'] = df['Close'] / df['Open'] - 1.0
    df = df.dropna(subset=['sig_gap', 'intraday'])
    for half, lbl in [(None, 'FULL 2016+'), (1, 'H1'), (2, 'H2')]:
        d = df
        if half == 1:
            d = d[d['DateTime'] < d['DateTime'].quantile(0.5)]
        elif half == 2:
            d = d[d['DateTime'] >= d['DateTime'].quantile(0.5)]
        for k in [5, 10]:
            out = []
            for dt, q in d.groupby('DateTime'):
                if len(q) < 30:
                    continue
                order = np.argsort(q['sig_gap'].values)
                out.append(-q['intraday'].values[order[-k:]].mean())
            m, t, n, w = stat(out)
            print(f"{lbl:>10} k={k:>2} open->close | net@6 {m-6:+7.2f} (t{t:+5.1f} w{w:.0%} n{n})")

    with open(os.path.join(OUT_DIR, 'artifact_checks.json'), 'w') as f:
        json.dump(R, f, indent=1, default=float)
    print(f"\nsaved -> {OUT_DIR}/artifact_checks.json")


if __name__ == '__main__':
    main()
