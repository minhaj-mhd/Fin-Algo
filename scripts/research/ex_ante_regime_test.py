"""
Ex ante regime test.

For each fold in the hybrid v10+v16 backtest, compute market regime indicators
using ONLY data available BEFORE the fold's test period starts.
Then check if those indicators predict which folds will be good or bad.

If a clear relationship exists -> regime filter is legitimate.
If no relationship -> adding a filter is hindsight bias.

Indicators computed at the last trading day before each fold's test month:
  - ADX_14      : trend strength (low = choppy, high = trending)
  - ATR_pct     : volatility normalised by price (ATR/close)
  - SMA50_dist  : close vs 50-day SMA (negative = below = bearish)
  - Mom_20      : 20-day price return (recent momentum)
  - RealVol_20  : 20-day annualised realised volatility
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── fold definitions (from train_hybrid_v10_v16.py) ─────────────────────────────
FOLD_TEST_STARTS = [
    '2023-11', '2024-03', '2024-07', '2024-11',
    '2025-03', '2025-07', '2025-11', '2026-03',
]

# A1_S (top-1 short) results per fold — the only strategy with positive avg
A1_S_NET = [1.4, 40.0, 4.2, 1.7, 14.8, -2.6, 7.5, -4.6]
A1_S_N   = [55,  12,   58,  44,  15,   50,   10,  25]

# Also track longs for completeness
A1_L_NET = [92.3, -47.0, -12.5, -1.4, -6.1, 12.1, -28.1, -12.4]
A1_L_N   = [2,    11,    38,    27,   19,   13,   9,     26]

# ── load Nifty 500 daily ─────────────────────────────────────────────────────────
print("Loading Nifty 500 daily data...")
idx = pd.read_csv('data/raw_index_cache/nifty500_1d.csv')
idx['date'] = pd.to_datetime(idx['timestamp']).dt.tz_localize(None).dt.normalize()
idx = idx.sort_values('date').reset_index(drop=True)

close = idx['close'].values
high  = idx['high'].values
low   = idx['low'].values
dates = idx['date'].values

# ── indicator helpers ────────────────────────────────────────────────────────────
def compute_atr(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    atr = np.full(len(c), np.nan)
    atr[n] = tr[:n].mean()
    for i in range(n + 1, len(c)):
        atr[i] = (atr[i-1] * (n-1) + tr[i-1]) / n
    return atr

def compute_adx(h, l, c, n=14):
    tr  = compute_atr(h, l, c, n)
    dm_plus  = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                         np.maximum(h[1:] - h[:-1], 0), 0)
    dm_minus = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                         np.maximum(l[:-1] - l[1:], 0), 0)

    sm_tr    = np.full(len(c), np.nan)
    sm_plus  = np.full(len(c), np.nan)
    sm_minus = np.full(len(c), np.nan)
    sm_tr[n]    = tr[1:n+1].sum()
    sm_plus[n]  = dm_plus[:n].sum()
    sm_minus[n] = dm_minus[:n].sum()
    for i in range(n+1, len(c)):
        sm_tr[i]    = sm_tr[i-1]    - sm_tr[i-1]/n    + tr[i]
        sm_plus[i]  = sm_plus[i-1]  - sm_plus[i-1]/n  + dm_plus[i-1]
        sm_minus[i] = sm_minus[i-1] - sm_minus[i-1]/n + dm_minus[i-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus  = 100 * sm_plus  / sm_tr
        di_minus = 100 * sm_minus / sm_tr
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

    adx = np.full(len(c), np.nan)
    adx[2*n] = np.nanmean(dx[n:2*n+1])
    for i in range(2*n+1, len(c)):
        adx[i] = (adx[i-1] * (n-1) + dx[i]) / n
    return adx

print("Computing indicators...")

c = pd.Series(close, index=idx['date'])
h = pd.Series(high,  index=idx['date'])
l = pd.Series(low,   index=idx['date'])

# ATR (14)
hl  = h - l
hc  = (h - c.shift(1)).abs()
lc  = (l - c.shift(1)).abs()
tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
atr_s = tr.ewm(alpha=1/14, adjust=False).mean()

# ADX (14) via pandas
dm_plus  = (h - h.shift(1)).clip(lower=0)
dm_minus = (l.shift(1) - l).clip(lower=0)
dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
dm_minus = dm_minus.where(dm_minus > dm_plus.shift(1).fillna(0), 0)  # recompute after masking

# smoother
sm_tr    = tr.ewm(alpha=1/14, adjust=False).mean()
sm_plus  = dm_plus.ewm(alpha=1/14, adjust=False).mean()
sm_minus = dm_minus.ewm(alpha=1/14, adjust=False).mean()

di_plus  = 100 * sm_plus  / sm_tr
di_minus = 100 * sm_minus / sm_tr
dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
adx_s = dx.ewm(alpha=1/14, adjust=False).mean()

atr_arr = atr_s.values
adx_arr = adx_s.values

# SMA 50
sma50 = c.rolling(50).mean().values
# 20-day momentum
mom20 = c.pct_change(20).values * 100
# 20-day realised vol (annualised %)
rvol20 = c.pct_change().rolling(20).std().values * np.sqrt(252) * 100

# ── extract ex ante snapshot for each fold ──────────────────────────────────────
print("\nEx ante regime snapshot at start of each fold test period:")
print(f"{'Fold':<5} {'Date':<12} {'ADX':>6} {'ATR%':>6} {'SMA50%':>7} {'Mom20%':>7} {'RVol20%':>8}  "
      f"{'A1_S bps':>9} {'N':>4}")
print("-" * 80)

rows = []
for fi, (test_start, a1s_net, a1s_n, a1l_net, a1l_n) in enumerate(
        zip(FOLD_TEST_STARTS, A1_S_NET, A1_S_N, A1_L_NET, A1_L_N), 1):

    # last trading day STRICTLY BEFORE the test month
    cutoff = pd.Timestamp(test_start + '-01') - pd.Timedelta(days=1)
    mask   = dates <= np.datetime64(cutoff)
    if not mask.any():
        continue
    i = int(np.where(mask)[0][-1])

    adx_val  = float(adx_arr[i]) if np.isfinite(adx_arr[i]) else np.nan
    atr_pct  = float(atr_arr[i] / close[i] * 100) if np.isfinite(atr_arr[i]) else np.nan
    sma_dist = float((close[i] / sma50[i] - 1) * 100) if np.isfinite(sma50[i]) else np.nan
    mom_val  = float(mom20[i])  if np.isfinite(mom20[i])  else np.nan
    rvol_val = float(rvol20[i]) if np.isfinite(rvol20[i]) else np.nan
    snap_date = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')

    print(f"  {fi:<3}  {snap_date:<12} {adx_val:>6.1f} {atr_pct:>6.2f} {sma_dist:>+7.2f} "
          f"{mom_val:>+7.2f} {rvol_val:>8.1f}   {a1s_net:>+8.1f}  {a1s_n:>4}")

    rows.append(dict(fold=fi, date=snap_date,
                     adx=adx_val, atr_pct=atr_pct, sma50_dist=sma_dist,
                     mom20=mom_val, rvol20=rvol_val,
                     a1s_net=a1s_net, a1s_n=a1s_n,
                     a1l_net=a1l_net, a1l_n=a1l_n))

df = pd.DataFrame(rows)

# ── Spearman correlation: each indicator vs fold net return ──────────────────────
print("\n" + "=" * 60)
print("SPEARMAN CORRELATION: regime indicator vs A1 Short net bps")
print(f"  (N={len(df)} folds — only clean ex ante info)\n")
print(f"  {'Indicator':<15} {'rho':>6}  {'p':>6}  interpretation")
print(f"  {'-'*55}")

indicators = {
    'ADX_14':    ('adx',      'high ADX = trending = better?'),
    'ATR_pct':   ('atr_pct',  'high ATR = volatile = better?'),
    'SMA50_dist':('sma50_dist','above SMA = bull = better?'),
    'Mom_20':    ('mom20',    'positive momentum = better?'),
    'RVol_20':   ('rvol20',   'high realised vol = better?'),
}

for name, (col, interp) in indicators.items():
    rho, p = spearmanr(df[col], df['a1s_net'])
    flag = '<< SIGNAL' if p < 0.10 else ''
    print(f"  {name:<15} {rho:>+6.3f}  {p:>6.3f}  {interp}  {flag}")

# ── threshold test: ADX > 20? ────────────────────────────────────────────────────
if 'adx' in df.columns:
    print("\nADX threshold split:")
    for th in [15, 20, 25]:
        above = df[df['adx'] >= th]
        below = df[df['adx'] <  th]
        if len(above) > 0 and len(below) > 0:
            print(f"  ADX >= {th}: folds={len(above)}  avg A1_S = {above['a1s_net'].mean():+.1f} bps  "
                  f"| ADX < {th}: folds={len(below)}  avg = {below['a1s_net'].mean():+.1f} bps")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("  If any rho has p < 0.10 AND the direction makes economic sense,")
print("  a regime filter on that indicator MAY be legitimate (ex ante).")
print("  If all p > 0.10, there is no ex ante predictability in 8 folds —")
print("  any regime filter you build would be hindsight bias.")

with open('data/ex_ante_regime_results.json', 'w') as f:
    json.dump({'folds': rows, 'n_folds': len(df)}, f, indent=2)
print("\nResults saved -> data/ex_ante_regime_results.json")
