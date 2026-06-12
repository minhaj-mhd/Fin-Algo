"""
Faithful test of the cost-aware GATE as a veto on v10's picks (the actual objective).

v10 1h XGBoost picks Top-K per side each timestamp; the per-side gate vetoes the weakest
keep_rate-complement of those picks (keep a {side} pick iff its {side}-gate score is in the
top keep_rate of that timestamp's cross-section). We compare v10-alone vs v10+gate net bps.

Restricted to the GATE's own OOS test window (dates >= the 85% split start) so the veto is
honest. v10 OOS preds: data/model_analysis/v10_v18_independent/walkforward_preds.npz.
Exploratory only — no Gauntlet verdict. Survivorship caveat applies (fixed 172-name panel).
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_1samp

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from scripts.transformer.train import load_panel, valid_decision_timestamps, EMBARGO, KS, L1, L2
from scripts.transformer.model import DualResCSTransformer

V3_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
NPZ = 'data/model_analysis/v10_v18_independent/walkforward_preds.npz'
COSTS = {'6bps': 0.0006, '10bps': 0.0010, '20bps': 0.0020}
KEEP_RATE = 0.70                       # keep top 70% by gate score -> veto weakest 30%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d = load_panel()
F = d['meta']['n_features']; M = d['macro'].shape[1]; n_sec = len(d['meta']['sectors'])
tickers = d['meta']['tickers']; ts1 = d['ts_1h']
sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)
ts_all = valid_decision_timestamps(d)
n = len(ts_all); i_va = int(n * 0.85)
te = ts_all[i_va + EMBARGO:]                       # gate OOS test timestamps
cutoff = int(ts1[te[0]])
print(f"gate OOS window starts {pd.Timestamp(cutoff)}  ({len(te)} timestamps)")


@torch.no_grad()
def infer_gate(path):
    m = DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                             n_slots_15m=d['meta']['n_slots_15m'], d_model=64).to(device)
    m.load_state_dict(torch.load(path, map_location=device)); m.eval()
    X1, X15, S1, S15, MA, DI, E = d['X_1h'], d['X_15m'], d['slot_1h'], d['slot_15m'], d['macro'], d['date_idx'], d['end15']
    table = {}
    for t in te:
        t = int(t); e = int(E[t])
        x1 = np.transpose(np.nan_to_num(X1[t - L1 + 1:t + 1]), (1, 0, 2))[None]
        x15 = np.transpose(np.nan_to_num(X15[e - L2 + 1:e + 1]), (1, 0, 2))[None]
        s1 = S1[t - L1 + 1:t + 1].astype(np.int64)[None]
        s15 = S15[e - L2 + 1:e + 1].astype(np.int64)[None]
        macro = np.nan_to_num(MA[int(DI[t])]).astype(np.float32)[None]
        present = np.isfinite(X1[t, :, 0])
        tt = lambda a: torch.from_numpy(a).to(device)
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = m(tt(x1.astype(np.float32)), tt(x15.astype(np.float32)), tt(s1), tt(s15),
                      tt(macro), sector_ids, ~tt(present[None].astype(np.bool_)))
        sc = logit.float()[0].cpu().numpy()
        tns = int(ts1[t])
        for j in np.where(present)[0]:
            table[(tns, tickers[j])] = float(sc[j])
    return table


import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--short_ckpt', default='artifacts/dualres_gate_short.pt')
_ap.add_argument('--long_ckpt', default='artifacts/dualres_gate_long.pt')
_a = _ap.parse_args()
print(f"inferring short gate ({_a.short_ckpt}) ..."); tab_s = infer_gate(_a.short_ckpt)
print(f"inferring long gate  ({_a.long_ckpt}) ..."); tab_l = infer_gate(_a.long_ckpt)

z = np.load(NPZ, allow_pickle=True)
idx, q, y, rl, rs = z['idx'], z['q'], z['y'], z['rl'], z['rs']
v3 = pd.read_csv(V3_FILE, usecols=['DateTime', 'Ticker'])
dt_ns = pd.to_datetime(v3['DateTime']).values.astype('datetime64[ns]').astype('int64')[idx]
tk = v3['Ticker'].str.replace('.NS', '', regex=False).values[idx]
gs = np.array([tab_s.get((int(dt_ns[k]), tk[k]), np.nan) for k in range(len(idx))])
gl = np.array([tab_l.get((int(dt_ns[k]), tk[k]), np.nan) for k in range(len(idx))])
base = (dt_ns >= cutoff) & np.isfinite(gs) & np.isfinite(gl)
print(f"v10 OOS rows in gate window: {base.sum():,}\n")


def tstat(r, cost):
    r = np.asarray(r, float)
    if len(r) == 0:
        return dict(n=0, net=0.0, t=0.0, win=0.0)
    net = r - cost
    t = float(ttest_1samp(net, 0).statistic) if len(r) > 1 and np.std(net) > 0 else 0.0
    return dict(n=len(r), net=round(net.mean() * 1e4, 2), t=round(t, 2), win=round((r > 0).mean(), 3))


def run(K):
    L0, S0, Lg, Sg = [], [], [], []        # v10-alone long/short, v10+gate long/short
    for qid in np.unique(q[base]):
        m = (q == qid) & base
        if m.sum() < K:
            continue
        rl_, rs_, y_, gl_, gs_ = rl[m], rs[m], y[m], gl[m], gs[m]
        thr_l = np.quantile(gl_, 1 - KEEP_RATE)
        thr_s = np.quantile(gs_, 1 - KEEP_RATE)
        for j in np.argsort(rl_)[-K:]:
            L0.append(y_[j])
            if gl_[j] >= thr_l:
                Lg.append(y_[j])
        for j in np.argsort(rs_)[-K:]:
            S0.append(-y_[j])
            if gs_[j] >= thr_s:
                Sg.append(-y_[j])
    return L0, S0, Lg, Sg


out = {'keep_rate': KEEP_RATE, 'cutoff': str(pd.Timestamp(cutoff)), 'results': {}}
print("=" * 88)
print(f"GATE-as-veto on v10  (keep {KEEP_RATE:.0%} by gate score, OOS only)")
print("=" * 88)
for K in KS:
    L0, S0, Lg, Sg = run(K)
    print(f"\n#### Top-{K} ####   (long kept {len(Lg)}/{len(L0)}, short kept {len(Sg)}/{len(S0)})")
    for cname, cv in COSTS.items():
        a_l, a_s = tstat(L0, cv), tstat(S0, cv)
        g_l, g_s = tstat(Lg, cv), tstat(Sg, cv)
        out['results'].setdefault(f'K{K}', {})[cname] = dict(
            v10_long=a_l, v10_short=a_s, gate_long=g_l, gate_short=g_s)
        print(f"  @{cname:<5} LONG  v10 {a_l['net']:+6.2f}(t{a_l['t']:+.2f}) -> gate {g_l['net']:+6.2f}"
              f"(t{g_l['t']:+.2f})  Δ{g_l['net']-a_l['net']:+.2f} | "
              f"SHORT v10 {a_s['net']:+6.2f}(t{a_s['t']:+.2f}) -> gate {g_s['net']:+6.2f}"
              f"(t{g_s['t']:+.2f})  Δ{g_s['net']-a_s['net']:+.2f}")

os.makedirs('artifacts', exist_ok=True)
json.dump(out, open('artifacts/gate_veto_v10.json', 'w'), indent=2, default=float)
print("\nsaved -> artifacts/gate_veto_v10.json")
