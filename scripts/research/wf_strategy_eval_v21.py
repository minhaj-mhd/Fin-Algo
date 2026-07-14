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
oos = pd.read_parquet('scratch/wf_v21/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime'])
oos['date'] = oos['DateTime'].dt.date
oos['year'] = oos['DateTime'].dt.year
# per-scan mean-centred convictions
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean')
oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss'] - oos['ss_m']) - (oos['ls'] - oos['ls_m'])
oos['long_conv'] = (oos['ls'] - oos['ls_m']) - (oos['ss'] - oos['ss_m'])

# V21: NO EXOGENOUS TIME MASK. The model must learn to avoid bad hours organically.
win = oos.copy()
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

# In V21, there is no TEST 2 because there are no exogenous gates to simulate.
print("\n" + json.dumps(json.load(open('scratch/wf_v21/fold_meta.json')), indent=1))
