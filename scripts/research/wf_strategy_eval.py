"""
Evaluate the retrained-WF OOS scored panel (scratch/wf/oos_scored.parquet).

TEST 1  Core engine (no gates): top-1 / top-3 short & long per anchor, per year,
        net bps + day-clustered t-stat + win-rate. Pure retrained-model edge.
TEST 2  Full gated strategy (user's exact rules) over 2025-08+, RETRAINED WF:
        nifty gates + lunch veto + ss-quantile floor + conviction top-1 + 1-slot
        queue + 5x compounding. Directly comparable to the +1258% / +288% claim.
TEST 2b Same gated strategy extended to full 2024-2026 using nifty500-1h approx gates.
"""
import json, numpy as np, pandas as pd

COST = 6.0
oos = pd.read_parquet('scratch/wf/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime'])
oos['date'] = oos['DateTime'].dt.date
oos['year'] = oos['DateTime'].dt.year
# per-scan mean-centred convictions
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean')
oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss'] - oos['ss_m']) - (oos['ls'] - oos['ls_m'])
oos['long_conv'] = (oos['ls'] - oos['ls_m']) - (oos['ss'] - oos['ss_m'])
tmin, tmax = pd.to_datetime('10:15').time(), pd.to_datetime('14:15').time()
win = oos[(oos['DateTime'].dt.time >= tmin) & (oos['DateTime'].dt.time <= tmax)].copy()
print(f"OOS scored: {len(oos):,} rows | trade-window {len(win):,} | span {oos['date'].min()}..{oos['date'].max()}")

def day_t(df, bps_col='net_bps'):
    """day-clustered t-stat: each day's mean net-bps is one obs."""
    dm = df.groupby('date')[bps_col].mean()
    if len(dm) < 2: return np.nan, len(dm)
    return dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))), len(dm)

def book_stats(trades, label):
    if len(trades) == 0:
        return f"{label:28s} | 0 trades"
    t, nd = day_t(trades)
    net = trades['net_bps']
    return (f"{label:28s} | n {len(trades):4d} | WR {(net>0).mean():5.1%} | "
            f"net {net.mean():+6.2f}bps | t_day {t:+5.2f} | sum {net.sum():+8.0f}bps")

# ---------- TEST 1: core engine, no gates ----------
print("\n" + "="*92)
print("TEST 1  CORE ENGINE (retrained WF, NO gates) — top-1 conviction pick per anchor")
print("="*92)
def topk_book(g_df, conv_col, side, k):
    rows = []
    for ts, g in g_df.groupby('DateTime'):
        gg = g.nlargest(k, conv_col)
        r = gg['Next_Hour_Return'].values
        bps = (-r if side == 'S' else r) * 10000 - COST
        for b in bps: rows.append({'date': gg['date'].iloc[0], 'net_bps': b})
    return pd.DataFrame(rows)

for side, conv in [('S', 'short_conv'), ('L', 'long_conv')]:
    name = 'SHORT' if side == 'S' else 'LONG'
    for k in (1, 3):
        bk = topk_book(win, conv, side, k)
        print(book_stats(bk, f"{name} top-{k} (all yrs)"))
        for yr in sorted(win['year'].unique()):
            byk = topk_book(win[win['year'] == yr], conv, side, k)
            if len(byk): print("   " + book_stats(byk, f"  {yr}"))
    print()

# ---------- gates ----------
def build_gates_15m():
    n = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
    n['ts'] = pd.to_datetime(n['ts'])
    n = n.sort_values('ts')
    n['n2h'] = n['close'] / n['close'].shift(8) - 1
    n['d'] = n['ts'].dt.date
    dop = n.groupby('d')['open'].first().rename('dop')
    n = n.merge(dop, on='d', how='left')
    n['nin'] = n['close'] / n['dop'] - 1
    return dict(zip(n['ts'], n['n2h'])), dict(zip(n['ts'], n['nin']))

def build_gates_1h_asof(anchors):
    h = pd.read_csv('data/raw_index_cache/nifty500_1h.csv')
    h['ts'] = pd.to_datetime(h['timestamp'])
    h = h.sort_values('ts')
    h['n2h'] = h['close'] / h['close'].shift(2) - 1
    h['d'] = h['ts'].dt.date
    dop = h.groupby('d')['open'].first().rename('dop')
    h = h.merge(dop, on='d', how='left')
    h['nin'] = h['close'] / h['dop'] - 1
    a = pd.DataFrame({'ts': sorted(anchors)}).sort_values('ts')
    m = pd.merge_asof(a, h[['ts', 'n2h', 'nin']], on='ts', direction='backward',
                      tolerance=pd.Timedelta('90min'))
    return dict(zip(m['ts'], m['n2h'])), dict(zip(m['ts'], m['nin']))

def run_gated(df, n2h_map, nin_map, label, start=None):
    d = df.copy()
    if start: d = d[d['date'] >= start]
    d['n2h'] = d['DateTime'].map(n2h_map)
    d['nin'] = d['DateTime'].map(nin_map)
    d = d.dropna(subset=['n2h', 'nin'])
    trades = []
    for ts, g in d.groupby('DateTime'):
        n2h = g['n2h'].iloc[0]; nin = g['nin'].iloc[0]; t = ts.time()
        thr = g['ss_thr'].iloc[0]
        # SHORT
        if (n2h <= 0.0025 or nin > 0.0036) and (t < pd.to_datetime('11:30').time() or t > pd.to_datetime('13:00').time()):
            c = g[g['ss'] > thr].nlargest(1, 'short_conv')
            if len(c):
                r = c['Next_Hour_Return'].iloc[0]
                trades.append({'ts': ts, 'date': c['date'].iloc[0], 'side': 'S', 'so': 0,
                               'net_bps': -r*10000 - COST})
        # LONG
        if n2h > 0.0025 and nin > 0.0020:
            c = g.nlargest(1, 'long_conv')
            if len(c):
                r = c['Next_Hour_Return'].iloc[0]
                trades.append({'ts': ts, 'date': c['date'].iloc[0], 'side': 'L', 'so': 1,
                               'net_bps': r*10000 - COST})
    td = pd.DataFrame(trades)
    if len(td) == 0:
        print(f"\n{label}: 0 trades"); return None
    td = td.sort_values(['ts', 'so']).reset_index(drop=True)
    # 1-slot queue
    keep, active = [], pd.Timestamp('2000-01-01')
    for _, r in td.iterrows():
        if r['ts'] >= active:
            keep.append(r); active = r['ts'] + pd.Timedelta(hours=1)
    ex = pd.DataFrame(keep).reset_index(drop=True)
    # 5x compounding
    cap = 100000.0; caps = [cap]; pnls = []
    for b in ex['net_bps']:
        p = cap * 5 * (b/10000); cap += p; pnls.append(p); caps.append(cap)
    ex['pnl'] = pnls; ex['cap'] = caps[1:]
    # drawdown on realised sequence
    eq = np.array([100000.0] + list(ex['cap'])); peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()

    print(f"\n{'='*92}\n{label}\n{'='*92}")
    print(f"  Trades (post-queue): {len(ex)}  | span {ex['ts'].min().date()}..{ex['ts'].max().date()}")
    print(f"  Final capital (5x compounded from 1L): Rs {cap:,.0f}   (ROI {cap/100000-1:+.1%})")
    print(f"  Flat sum net bps (order-independent):  {ex['net_bps'].sum():+.0f} bps  | avg {ex['net_bps'].mean():+.2f}/trade")
    print(f"  Realised-sequence MDD: {mdd:.1%}")
    for side, nm in [('S', 'SHORT'), ('L', 'LONG')]:
        sd = ex[ex['side'] == side]
        print("   " + book_stats(sd, f"  {nm}"))
    # reshuffle MDD distribution
    rng = np.random.default_rng(42); mdds = []
    b = ex['net_bps'].values
    for _ in range(2000):
        seq = rng.permutation(b); cap = 100000.0; e = [cap]
        for x in seq:
            cap += cap*5*(x/10000); e.append(cap)
        e = np.array(e); pk = np.maximum.accumulate(e); mdds.append(((e-pk)/pk).min())
    mdds = np.array(mdds)
    print(f"  Reshuffle MDD: median {np.median(mdds):.1%} | worst {mdds.min():.1%} | best {mdds.max():.1%}")
    # yearly
    for yr in sorted(ex['ts'].dt.year.unique()):
        ey = ex[ex['ts'].dt.year == yr]
        print(f"   {yr}: n {len(ey):3d} | net {ey['net_bps'].mean():+6.2f}bps | sum {ey['net_bps'].sum():+7.0f} | flat-pnl Rs {(ey['net_bps'].sum()*50):,.0f}")
    return ex

# ---------- TEST 2: exact-fidelity gated (nifty50 15m), 2025-08+ ----------
n2h15, nin15 = build_gates_15m()
import datetime as dt
run_gated(win, n2h15, nin15, "TEST 2  FULL GATED STRATEGY — retrained WF, exact NIFTY50-15m gates, 2025-08+",
          start=dt.date(2025, 8, 1))

# ---------- TEST 2b: extended multi-year, approx 1h gates ----------
n2h1h, nin1h = build_gates_1h_asof(win['DateTime'].unique())
run_gated(win, n2h1h, nin1h, "TEST 2b  FULL GATED STRATEGY — retrained WF, approx NIFTY500-1h gates, full 2024-2026")

print("\n" + json.dumps(json.load(open('scratch/wf/fold_meta.json')), indent=1))
