"""In-depth analysis of the trained DualRes transformer on the held-out TEST split:
confidence calibration, accuracy/return by confidence bucket, and selective-trading
net-of-cost per side. Exploratory only (no Gauntlet verdict)."""
import os, sys, json
import numpy as np
import torch

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from scripts.transformer.train import (load_panel, DecisionDataset, collate,
                                       valid_decision_timestamps, EMBARGO)
from scripts.transformer.model import DualResCSTransformer
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = load_panel()
F = d['meta']['n_features']; M = d['macro'].shape[1]; n_sec = len(d['meta']['sectors'])
sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)

ts = valid_decision_timestamps(d); n = len(ts)
te = ts[int(n * 0.85) + EMBARGO:]
dl = DataLoader(DecisionDataset(d, te), batch_size=8, shuffle=False, collate_fn=collate)

model = DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                             n_slots_15m=d['meta']['n_slots_15m'], d_model=96).to(device)
model.load_state_dict(torch.load('artifacts/dualres_transformer.pt'))
model.eval()

P, R = [], []
with torch.no_grad():
    for batch in dl:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
        prob = torch.sigmoid(logit.float())
        for b in range(prob.shape[0]):
            m = valid[b]
            P.append(prob[b][m].cpu().numpy()); R.append(y[b][m].cpu().numpy())
p = np.concatenate(P); r = np.concatenate(R); up = (r > 0).astype(int)
print(f"TEST samples: {len(p):,}   base up-rate: {up.mean():.4f}\n")

print("=== CALIBRATION (does P(up) match realized up-rate?) ===")
print(f"{'prob bucket':>12} {'n':>8} {'pred_up':>8} {'actual_up':>10} {'mean_ret_bps':>13}")
edges = np.arange(0, 1.01, 0.1)
for lo, hi in zip(edges[:-1], edges[1:]):
    msk = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
    if msk.sum() == 0:
        continue
    print(f"{lo:.1f}-{hi:.1f}    {msk.sum():>8,} {p[msk].mean():>8.3f} "
          f"{up[msk].mean():>10.3f} {r[msk].mean()*1e4:>13.2f}")

print("\n=== SELECTIVE TRADING by confidence threshold (per side, NET of cost) ===")
print("conf = |P(up)-0.5|; LONG if P>0.5+thr, SHORT if P<0.5-thr")
print(f"{'thr':>5} | {'LONG n':>8} {'rawWR':>6} {'gross':>7} {'net@6':>7} {'net@10':>7} "
      f"| {'SHORT n':>8} {'rawWR':>6} {'gross':>7} {'net@6':>7} {'net@10':>7}")
for thr in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
    lm = p > 0.5 + thr
    sm = p < 0.5 - thr
    lg = r[lm]; sg = -r[sm]
    def stats(g):
        if len(g) == 0:
            return (0, 0, 0, 0, 0)
        return (len(g), (g > 0).mean(), g.mean()*1e4, g.mean()*1e4 - 6, g.mean()*1e4 - 10)
    ln, lwr, lgr, l6, l10 = stats(lg)
    sn, swr, sgr, s6, s10 = stats(sg)
    print(f"{thr:>5.2f} | {ln:>8,} {lwr:>6.3f} {lgr:>7.2f} {l6:>7.2f} {l10:>7.2f} "
          f"| {sn:>8,} {swr:>6.3f} {sgr:>7.2f} {s6:>7.2f} {s10:>7.2f}")

# overall AUC
order = np.argsort(p); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(p))
n1 = up.sum(); n0 = len(up) - n1
auc = (ranks[up == 1].sum() - n1*(n1-1)/2) / (n1*n0)
print(f"\noverall AUC={auc:.4f}  acc@0.5={((p>0.5).astype(int)==up).mean():.4f}")
