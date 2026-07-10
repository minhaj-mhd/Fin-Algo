"""
CONVICTION-MOMENTUM x PRICE-DIRECTION -> REMAINING RETURN (exploratory, no verdict).

Tests the hypothesis that the *trajectory* (slope) of 15m conviction during the
hold, crossed with the price move so far, predicts the REMAINING (forward) return
of the trade -- the correct selectivity test for an exit signal.

Reads data/research/entry_exit/dualtf_trade_panel.csv (conviction is re-scored
every 15m bar: rk*_{60,75,90,105}). Checkpoint = +90 min into the hold:
  KNOWN at +90 (no look-ahead): conviction rk at +60,+75,+90 (3-pt slope) and the
    realized move so far = compound(sub_60, sub_75).
  REMAINING (what we predict)  : compound(sub_90, sub_105)  [+90..+120], signed.

Axes (in the trade's frame; for shorts everything is already signed by sgn):
  FAV move : signed realized return so far > 0  (price moved our way)
  CONV mom : slope of own-side rank over the 3 readings > 0  (model gaining belief)

Reports mean remaining return (bps) + t-stat per bucket, then evaluates exiting
the bearish buckets vs full-hold. WIN = buckets separate on REMAINING return and
exiting the weak bucket(s) beats full_hold net of cost.
"""
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp, ttest_ind
PANEL = 'data/research/entry_exit/dualtf_trade_panel.csv'
COST = 10/1e4

P = pd.read_csv(PANEL)
need = ['sub_60','sub_75','sub_90','sub_105']
P = P.dropna(subset=need).reset_index(drop=True)

def own(row, m):
    return row[('rkL_' if row['dir']=='long' else 'rkS_')+str(m)]

for d, sgn in [('long', 1), ('short', -1)]:
    s = P[P['dir'] == d].reset_index(drop=True)
    s = s.dropna(subset=['rkL_60','rkL_75','rkL_90','rkS_60','rkS_75','rkS_90']).reset_index(drop=True)
    conv = np.array([[own(r, m) for m in (60,75,90)] for _, r in s.iterrows()])
    slope = np.polyfit([0,1,2], conv.T, 1)[0]                       # conviction momentum
    sub = s[['sub_60','sub_75','sub_90','sub_105']].values
    fav  = (np.prod(1+sub[:, :2], axis=1) - 1) * sgn               # signed move SO FAR (+60..+90)
    rem  = (np.prod(1+sub[:, 2:], axis=1) - 1) * sgn               # signed REMAINING (+90..+120)
    full = (np.prod(1+sub, axis=1) - 1) * sgn - COST               # full-hold net

    print(f"\n===== {d.upper()} (N={len(s)}) — REMAINING return (+90..+120) by bucket =====")
    print(f"  {'bucket':<22}{'N':>6}{'mean_rem(bps)':>15}{'t':>8}")
    buckets = {
        'FAV+ / CONV+ (hold)':  (fav > 0) & (slope > 0),
        'FAV+ / CONV- (weaken)':(fav > 0) & (slope <= 0),
        'FAV- / CONV+ (dip)':   (fav <= 0) & (slope > 0),
        'FAV- / CONV- (exit?)': (fav <= 0) & (slope <= 0),
    }
    for name, mask in buckets.items():
        r = rem[mask]
        t, _ = ttest_1samp(r, 0) if len(r) > 1 else (np.nan, np.nan)
        print(f"  {name:<22}{mask.sum():>6}{r.mean()*1e4:>+15.1f}{t:>+8.2f}")
    # does the worst bucket actually have worse remaining return than the rest? (selectivity)
    worst = buckets['FAV- / CONV- (exit?)']
    t, p = ttest_ind(rem[worst], rem[~worst], equal_var=False)
    print(f"  selectivity: FAV-/CONV- remaining vs rest  diff={ (rem[worst].mean()-rem[~worst].mean())*1e4:+.1f}bps  t={t:+.2f} p={p:.1e}")

    # exit-rule evaluation: realize move-so-far for exited buckets, full-hold otherwise
    def policy_net(exit_mask):
        held = (np.prod(1+sub, axis=1) - 1) * sgn                # signed full-hold return
        realized = np.where(exit_mask, fav, held) - COST          # exit -> keep signed move-so-far
        return realized
    print(f"  full_hold mean_net = {full.mean()*1e4:+.1f} bps")
    for label, em in [('exit FAV-/CONV-', buckets['FAV- / CONV- (exit?)']),
                      ('exit both CONV- (FAV+/- )', slope <= 0),
                      ('exit FAV-/CONV- and FAV+/CONV-', buckets['FAV- / CONV- (exit?)'] | buckets['FAV+ / CONV- (weaken)'])]:
        net = policy_net(em)
        print(f"    {label:<32} mean_net={net.mean()*1e4:>+6.1f}  (exits {em.mean()*100:>4.0f}% of trades)")

print("\n" + "="*88)
print("  WIN: FAV-/CONV- remaining return is sharply negative AND significantly below the")
print("  rest (selectivity p<0.05) AND exiting it beats full_hold net. Else momentum adds no edge.")
print("="*88)
