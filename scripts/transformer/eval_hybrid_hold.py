import os, sys, json
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())

from scripts.transformer.daily_veto_walkforward import gather, cluster_boot, SIDES, KS, COST

P = 'data/daily_transformer_panel'

Y_1d = np.load(f'{P}/Y_1d.npy')
Y_3d = np.load(f'{P}/Y_3d.npy')
oos_mask = np.load(f'{P}/v2_oos_mask.npy')
oos_days = np.where(oos_mask)[0]

print("==============================================================================")
print("HYBRID HOLD EVALUATION (Cut losers at Day 1, Hold winners to Day 3)")
print(f"OOS days={len(oos_days)}  cost={COST*1e4:.0f}bps")
print("==============================================================================\n")

for side, scname, gname, sgn in SIDES:
    v2s = np.load(f'{P}/{scname}.npy')
    gate = np.load(f'{P}/{gname}.npy')
    print(f"\n################  {side.upper()}  ################")
    for k in KS:
        days, r1, _ = gather(Y_1d, v2s, gate, oos_days, sgn, k)
        _, r3, _ = gather(Y_3d, v2s, gate, oos_days, sgn, k)
        if len(r1) == 0: continue
        
        # Hybrid logic: if r1 < 0 (losing at day 1), take r1. else take r3.
        # Wait, if we exit at day 1, we still pay the same roundtrip cost (10bps).
        # So we just construct a new return array `r_hyb`.
        
        r_hyb = np.where(r1 < 0, r1, r3)
        
        v2net_3d, t_3d, _, _ = cluster_boot(days, r3 - COST)
        v2net_hyb, t_hyb, _, _ = cluster_boot(days, r_hyb - COST)
        
        print(f"  K={k}: picks={len(r1)}")
        print(f"     Standard 3-Day Hold net = {v2net_3d*1e4:>+6.2f} bps (t={t_3d:>+5.2f})")
        print(f"     Hybrid 1D/3D Hold   net = {v2net_hyb*1e4:>+6.2f} bps (t={t_hyb:>+5.2f})")
        
        diff = (r_hyb - COST) - (r3 - COST)
        d_net, d_t, _, _ = cluster_boot(days, diff)
        print(f"     Uplift from Hybrid rule = {d_net*1e4:>+6.2f} bps (t={d_t:>+5.2f})\n")

