"""
Walk-forward: v10 1h XGBoost ranker picks Top-1/3/5 per side; the transformer VETOES picks
whose direction it disagrees with. Tests BOTH transformers (BCE + cost-aware net-PnL).

v10 OOS rank predictions are the cached genuine walk-forward (every pred OOS, retrained per fold):
  data/model_analysis/v10_v18_independent/walkforward_preds.npz  (idx into v3 csv, rl/rs ranks, y, q)
Transformer P(up) is joined by (DateTime, Ticker). Evaluation is restricted to the period that is
ALSO out-of-sample for the transformer (date > its train cutoff), so the veto is honest.

Veto rule (threshold th): take a LONG pick only if P(up) > th ; take a SHORT pick only if P(up) < 1-th.
Exploratory only — no Gauntlet verdict.
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
from scripts.transformer.train import load_panel, DecisionDataset, valid_decision_timestamps
from scripts.transformer.model import DualResCSTransformer

V3_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
NPZ = 'data/model_analysis/v10_v18_independent/walkforward_preds.npz'
COSTS = {'6bps': 0.0006, '10bps': 0.0010, '20bps': 0.0020}
KS = [1, 3, 5]
TH = 0.50            # transformer must agree on direction
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── transformer inference over ALL valid decision timestamps → {(ts_ns,ticker): P(up)} ──
d = load_panel()
F = d['meta']['n_features']; M = d['macro'].shape[1]; n_sec = len(d['meta']['sectors'])
tickers = d['meta']['tickers']; ts1 = d['ts_1h']
sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)
ts_all = valid_decision_timestamps(d)
cutoff = int(ts1[ts_all[int(len(ts_all) * 0.70)]])     # transformer train cutoff (ns)
print(f"transformer train cutoff: {pd.Timestamp(cutoff)}  (veto eval uses dates AFTER this)")


def infer(path):
    m = DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                             n_slots_15m=d['meta']['n_slots_15m'], d_model=96).to(device)
    m.load_state_dict(torch.load(path)); m.eval()
    ds = DecisionDataset(d, ts_all)
    table = {}
    with torch.no_grad():
        for i in range(len(ds)):
            x1, x15, s1, s15, macro, ybin, y, present, valid = ds[i]
            tt = lambda a: torch.from_numpy(a[None]).to(device)
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                logit = m(tt(x1), tt(x15), tt(s1), tt(s15), tt(macro), sector_ids,
                          ~tt(present.astype(np.bool_)))
            p = torch.sigmoid(logit.float())[0].cpu().numpy()
            tns = int(ts1[ts_all[i]])
            for j in np.where(present)[0]:
                table[(tns, tickers[j])] = float(p[j])
    return table


print("inferring BCE transformer ..."); tab_bce = infer('artifacts/dualres_transformer.pt')
print("inferring netpnl transformer ..."); tab_net = infer('artifacts/dualres_transformer_netpnl.pt')

# ── v10 OOS preds + recover (DateTime,Ticker) ────────────────────────────────
z = np.load(NPZ, allow_pickle=True)
idx, q, y, rl, rs = z['idx'], z['q'], z['y'], z['rl'], z['rs']
v3 = pd.read_csv(V3_FILE, usecols=['DateTime', 'Ticker'])
dt_ns = pd.to_datetime(v3['DateTime']).values.astype('datetime64[ns]').astype('int64')[idx]
tk = v3['Ticker'].str.replace('.NS', '', regex=False).values[idx]

p_bce = np.array([tab_bce.get((int(dt_ns[k]), tk[k]), np.nan) for k in range(len(idx))])
p_net = np.array([tab_net.get((int(dt_ns[k]), tk[k]), np.nan) for k in range(len(idx))])

base = (dt_ns > cutoff) & np.isfinite(p_bce) & np.isfinite(p_net)   # common OOS universe
print(f"v10 OOS rows: {len(idx):,}  | transformer-OOS & joined: {base.sum():,} "
      f"({base.sum()/len(idx)*100:.1f}%)  span {pd.Timestamp(dt_ns[base].min())}..{pd.Timestamp(dt_ns[base].max())}")


def tstats(r, cost):
    r = np.asarray(r, float)
    if len(r) == 0:
        return dict(n=0, raw=0.0, net=0.0, win=0.0, t=0.0)
    net = r - cost
    t = float(ttest_1samp(net, 0).statistic) if len(r) > 1 and np.std(net) > 0 else 0.0
    return dict(n=len(r), raw=round(r.mean()*1e4, 2), net=round(net.mean()*1e4, 2),
                win=round((r > 0).mean(), 3), t=round(t, 2))


def run(K, veto, mask, th=TH):
    """veto: None | array of P(up). returns long_returns, short_returns, (kept,total) per side."""
    L, S = [], []
    kept = [0, 0]; tot = [0, 0]
    for qid in np.unique(q[mask]):
        m = (q == qid) & mask
        if m.sum() < K:
            continue
        rl_, rs_, y_ = rl[m], rs[m], y[m]
        p_ = veto[m] if veto is not None else None
        for j in np.argsort(rl_)[-K:]:
            tot[0] += 1
            if veto is None or p_[j] > th:
                L.append(y_[j]); kept[0] += 1
        for j in np.argsort(rs_)[-K:]:
            tot[1] += 1
            if veto is None or p_[j] < 1 - th:
                S.append(-y_[j]); kept[1] += 1
    return np.array(L), np.array(S), kept, tot


configs = [('v10_alone', None), ('v10 + BCE-veto', p_bce), ('v10 + netPnL-veto', p_net)]
out = {'cutoff': str(pd.Timestamp(cutoff)), 'n_oos': int(base.sum()), 'th': TH, 'results': {}}
print("\n" + "=" * 78)
print(f"WALK-FORWARD VETO TEST  (transformer-OOS only, n={base.sum():,} rows, veto th={TH})")
print("=" * 78)
for K in KS:
    print(f"\n################  Top-{K}  ################")
    for name, veto in configs:
        L, S, kept, tot = run(K, veto, base)
        keep_l = f"{kept[0]}/{tot[0]}" if veto is not None else f"{tot[0]}"
        keep_s = f"{kept[1]}/{tot[1]}" if veto is not None else f"{tot[1]}"
        print(f"\n[{name}]  (kept long {keep_l}, short {keep_s})")
        for cname, cv in COSTS.items():
            ls, ss = tstats(L, cv), tstats(S, cv)
            out['results'].setdefault(f'K{K}', {}).setdefault(name, {})[cname] = dict(long=ls, short=ss)
            print(f"  @{cname:<5} LONG  n={ls['n']:>4} raw{ls['raw']:>+6.1f} net{ls['net']:>+6.1f} "
                  f"win{ls['win']:.0%} t={ls['t']:>5.2f}  | SHORT n={ss['n']:>4} raw{ss['raw']:>+6.1f} "
                  f"net{ss['net']:>+6.1f} win{ss['win']:.0%} t={ss['t']:>5.2f}")

# ── veto-aggressiveness sweep on the netPnL transformer (the short side is the live one) ──
print("\n" + "=" * 78)
print("VETO-AGGRESSIVENESS SWEEP (netPnL transformer, SHORT side — keep only highest conviction)")
print("harder threshold => keep fewer, higher-conviction shorts. net@10 and net@20 bps")
print("=" * 78)
print(f"{'th':>5} {'K':>2} {'kept':>10} {'rawbps':>7} {'net@10':>7} {'net@20':>7} {'win':>5} {'t@10':>6}")
for th in [0.50, 0.52, 0.55, 0.58, 0.60]:
    for K in [1, 3]:
        _, S, kept, tot = run(K, p_net, base, th=th)
        if len(S) == 0:
            continue
        s10, s20 = tstats(S, 0.0010), tstats(S, 0.0020)
        print(f"{1-th:>5.2f} {K:>2} {f'{kept[1]}/{tot[1]}':>10} {s10['raw']:>+7.1f} "
              f"{s10['net']:>+7.1f} {s20['net']:>+7.1f} {s10['win']:>5.0%} {s10['t']:>6.2f}")

# ── fragility check: are the few "great" tight-threshold short trades clustered in time? ──
print("\n" + "=" * 78)
print("FRAGILITY CHECK: the kept SHORT trades at th=0.55 (P_up<0.45), Top-3 — date & return")
print("=" * 78)
th = 0.55; K = 3; rows = []
for qid in np.unique(q[base]):
    m = (q == qid) & base
    if m.sum() < K:
        continue
    rs_, y_, p_, dtn_ = rs[m], y[m], p_net[m], dt_ns[m]
    for j in np.argsort(rs_)[-K:]:
        if p_[j] < 1 - th:
            rows.append((pd.Timestamp(int(dtn_[j])), -y_[j] * 1e4))
rows.sort()
days = sorted(set(r[0].normalize() for r in rows))
print(f"  n_trades={len(rows)}  distinct_days={len(days)}  "
      f"span {rows[0][0].date() if rows else '-'}..{rows[-1][0].date() if rows else '-'}")
for ts_, ret in rows:
    print(f"    {ts_}  short_ret={ret:+.1f}bps")

os.makedirs('artifacts', exist_ok=True)
json.dump(out, open('artifacts/veto_walkforward.json', 'w'), indent=2, default=float)
print("\nsaved -> artifacts/veto_walkforward.json")
