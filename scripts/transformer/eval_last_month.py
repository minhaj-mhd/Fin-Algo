import os, sys, json
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())

from scripts.transformer.daily_veto_walkforward import gather, cluster_boot, uplift_boot, SIDES, KS, COST

P = 'data/daily_transformer_panel'
ts = np.load(f'{P}/ts_days.npy')
oos_mask = np.load(f'{P}/v2_oos_mask.npy')
ts_dates = pd.to_datetime(ts).normalize()

# Last month filter
last_month_mask = (ts_dates >= '2026-05-01')
oos_days = np.where(oos_mask & last_month_mask)[0]

print('OOS days in last month:', len(oos_days))
if len(oos_days) > 0:
    print('From', ts_dates[oos_days[0]].date(), 'to', ts_dates[oos_days[-1]].date())

Y_3d = np.load(f'{P}/Y_3d.npy')
Y_1d = np.load(f'{P}/Y_1d.npy')

def run_eval(Y_arr, label):
    print(f'\n========== {label} RETURNS ==========')
    for side, scname, gname, sgn in SIDES:
        v2s = np.load(f'{P}/{scname}.npy')
        gate = np.load(f'{P}/{gname}.npy')
        print(f'\n--- {side.upper()} ---')
        for k in KS:
            days, r, g = gather(Y_arr, v2s, gate, oos_days, sgn, k)
            if len(r) == 0:
                continue
            keep = g > 0.0
            v2net, v2t, _, _ = cluster_boot(days, (r - COST))
            kpnet, kpt, _, _ = cluster_boot(days, (r - COST), mask=keep)
            dnet, dt, _, _ = uplift_boot(days, r, keep)
            print(f' K={k}: picks={len(r)} | v2 alone net={v2net*1e4:+.2f}bps (t={v2t:+.2f}) | v2+veto net={kpnet*1e4:+.2f}bps (t={kpt:+.2f}) | uplift={dnet*1e4:+.2f}bps (t={dt:+.2f})')

run_eval(Y_1d, '1-DAY')
run_eval(Y_3d, '3-DAY')
