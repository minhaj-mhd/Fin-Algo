"""
PANEL EDGE PROBES (exploratory, no verdict authority) — fresh pass over
data/research/entry_exit/dualtf_trade_panel.csv for structure not yet tested.
Small, pre-stated probe set (~12 cells); demands per-fold consistency; any hit is
a HYPOTHESIS for one pre-registered Gauntlet run, not a result.

A) DEPTH-CONDITIONED REVERSION: at the +90 checkpoint, does the FAV- bounce scale
   with dip depth? Quintiles of signed move-so-far (fav<0 only) -> mean remaining
   return (+90..+120). Trade implication if deep quintile > 10bps: "late dip entry"
   = enter host names at +90 only when dipped, pay one round-trip, hold 30 min.
B) JOINT ENTRY GATE: pre-entry slope (the orthogonal ~2bps signal) x pre-entry
   rank LEVEL (rk_45) -> full-hold net. 3x2 cells per side.
C) FOLD STABILITY + HOUR: for the best cell of A and B, per-fold means; and
   remaining-return reversion by entry hour.
All returns signed by direction; cost 10 bps where a real trade is implied.
"""
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp

PANEL = 'data/research/entry_exit/dualtf_trade_panel.csv'
COST = 10/1e4
P = pd.read_csv(PANEL)
P = P.dropna(subset=['sub_60','sub_75','sub_90','sub_105']).reset_index(drop=True)
P['sgn'] = np.where(P['dir'] == 'long', 1, -1)
P['hour'] = pd.to_datetime(P['dt1']).dt.hour

pre = ['0','15','30','45']
own = np.where(P['dir'].to_numpy(dtype=object)[:, None] == 'long',
               P[[f'rkL_{m}' for m in pre]].values,
               P[[f'rkS_{m}' for m in pre]].values)
P['pre_lvl'] = own[:, 3]                                   # rank at T-1 (rk_45)
ok = ~np.isnan(own).any(axis=1)
P['pre_slope'] = np.nan
P.loc[ok, 'pre_slope'] = np.polyfit(np.arange(4), own[ok].T, 1)[0]

P['fav'] = (np.prod(1 + P[['sub_60','sub_75']].values, axis=1) - 1) * P['sgn']
P['rem'] = (np.prod(1 + P[['sub_90','sub_105']].values, axis=1) - 1) * P['sgn']
P['fh_net'] = (np.prod(1 + P[['sub_60','sub_75','sub_90','sub_105']].values, axis=1) - 1) * P['sgn'] - COST

def line(label, r, cost=0.0):
    r = r - cost
    t, p = ttest_1samp(r, 0) if len(r) > 1 else (np.nan, np.nan)
    return f"  {label:<34} N={len(r):>5}  mean={r.mean()*1e4:>+6.1f}bps  t={t:>+5.2f}  p={p:.2g}"

print("="*96)
print("A) DEPTH-CONDITIONED REVERSION — remaining(+90..+120) by depth of adverse move so far")
print("   (deepest quintile would need > ~10bps to support a 'late dip entry' trade)")
print("="*96)
for d in ('long', 'short'):
    s = P[(P['dir'] == d) & (P['fav'] < 0)].copy()
    s['q'] = pd.qcut(s['fav'], 5, labels=False)            # q0 = deepest dip
    print(f"\n  -- {d.upper()} (FAV- only, N={len(s)}) --")
    for q in range(5):
        g = s[s['q'] == q]
        tag = 'DEEPEST' if q == 0 else ('shallowest' if q == 4 else '')
        print(line(f"depth Q{q+1} {tag} (avg dip {g['fav'].mean()*1e4:+.0f}bps)", g['rem'].values))
    deep = s[s['q'] == 0]
    print(line("  -> DEEPEST as late-entry trade, net@10", deep['rem'].values, cost=COST))
    pf = deep.groupby('fold')['rem'].mean() * 1e4
    print(f"     per-fold rem (bps): {dict(pf.round(1))}  positive {int((pf>0).sum())}/6 folds")

print()
print("="*96)
print("B) JOINT ENTRY GATE — pre-entry slope tercile x level (rk_45 >=0.8 hi / <0.8 lo) -> full-hold net")
print("="*96)
for d in ('long', 'short'):
    s = P[(P['dir'] == d) & P['pre_slope'].notna()].copy()
    s['slope_t'] = pd.qcut(s['pre_slope'], 3, labels=['falling','flat','rising'])
    s['lvl'] = np.where(s['pre_lvl'] >= 0.8, 'hi', 'lo')
    print(f"\n  -- {d.upper()} (N={len(s)}) --")
    for st in ['rising','flat','falling']:
        for lv in ['hi','lo']:
            g = s[(s['slope_t'] == st) & (s['lvl'] == lv)]
            print(line(f"slope={st:<8} lvl={lv}", g['fh_net'].values))
    best = s[(s['slope_t'] == 'rising') & (s['lvl'] == 'hi')]
    pf = best.groupby('fold')['fh_net'].mean() * 1e4
    print(f"     rising+hi per-fold net (bps): {dict(pf.round(1))}  positive {int((pf>0).sum())}/6 folds")

print()
print("="*96)
print("C) REVERSION BY HOUR — mean remaining(+90..+120) for FAV- trades, by 1h-bar entry hour")
print("="*96)
for d in ('long', 'short'):
    s = P[(P['dir'] == d) & (P['fav'] < 0)]
    print(f"\n  -- {d.upper()} --")
    for h, g in s.groupby('hour'):
        if len(g) < 100: continue
        print(line(f"hour {h:02d}", g['rem'].values))
