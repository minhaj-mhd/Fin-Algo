"""
SKEPTICAL AUDIT of the DualRes transformer test results. Hunts for: (a) statistical
non-significance under cross-sectional clustering, (b) metric inflation/leakage via a
negative control, (c) the model being beaten by trivial baselines (i.e., adding nothing).
"""
import os, sys, json
import numpy as np
import torch

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from scripts.transformer.train import load_panel, DecisionDataset, valid_decision_timestamps, EMBARGO
from scripts.transformer.model import DualResCSTransformer

COST = 10.0 / 1e4
RET_IDX, RELRET_IDX = 21, 51
rng = np.random.default_rng(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
d = load_panel()
F = d['meta']['n_features']; M = d['macro'].shape[1]; n_sec = len(d['meta']['sectors'])
sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)
ts = valid_decision_timestamps(d); n = len(ts)
te = ts[int(n * 0.85) + EMBARGO:]
print(f"TEST timestamps: {len(te)}  (each = up to 172 tickers)")

model = DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                             n_slots_15m=d['meta']['n_slots_15m'], d_model=96).to(device)
model.load_state_dict(torch.load('artifacts/dualres_transformer.pt')); model.eval()
ds = DecisionDataset(d, te)

# ── per-timestamp inference: store prob, return, and baseline scores ──────────
PT = []   # list of (p, r, ret_score, relret_score)
with torch.no_grad():
    for i in range(len(ds)):
        x1, x15, s1, s15, macro, ybin, y, present, valid = ds[i]
        if valid.sum() < 5:
            continue
        t = lambda a: torch.from_numpy(a[None]).to(device)
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(t(x1), t(x15), t(s1), t(s15), t(macro), sector_ids,
                          ~t(present.astype(np.bool_)))
        p = torch.sigmoid(logit.float())[0].cpu().numpy()[valid]
        r = y[valid]
        PT.append((p, r, x1[:, -1, RET_IDX][valid], x1[:, -1, RELRET_IDX][valid]))
print(f"usable test timestamps: {len(PT)}")

def auc(p, y):
    yb = (y > 0).astype(int); o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(len(p))
    n1 = yb.sum(); n0 = len(yb) - n1
    return (rk[yb == 1].sum() - n1*(n1-1)/2) / (n1*n0 + 1e-9)

def k1_net(score, r):
    """per-ts: long = pick max(score), short = pick min(score). returns (long_net, short_net) bps."""
    L = np.array([r[np.argmax(s)] for s, r in zip(score, r_)]) if False else None
    # (kept explicit below)
    return None

# pooled
P = np.concatenate([x[0] for x in PT]); R = np.concatenate([x[1] for x in PT])
print(f"\npooled n={len(P):,}  base_up={(R>0).mean():.4f}  AUC={auc(P,R):.4f}  acc={((P>0.5)==(R>0)).mean():.4f}")

def topk_series(score_list, r_list, k=1):
    longs = np.array([r[np.argsort(-s)[:k]].mean() for s, r in zip(score_list, r_list)])
    shorts = np.array([-r[np.argsort(s)[:k]].mean() for s, r in zip(score_list, r_list)])
    return longs, shorts

p_list = [x[0] for x in PT]; r_list = [x[1] for x in PT]
ret_list = [x[2] for x in PT]; rel_list = [x[3] for x in PT]

print("\n=== 1) TRANSFORMER Top-1 per-side (gross / net@10bps) + block-bootstrap 95% CI by timestamp ===")
for name, sl in [('transformer', p_list)]:
    lg, sg = topk_series(sl, r_list, 1)
    # bootstrap over timestamps
    B = 3000; nL = []; nS = []; aucs = []
    idxs = np.arange(len(PT))
    for _ in range(B):
        bs = rng.choice(idxs, len(idxs), replace=True)
        nL.append(lg[bs].mean()); nS.append(sg[bs].mean())
    ci = lambda a: (np.percentile(a, 2.5)*1e4, np.percentile(a, 97.5)*1e4)
    print(f"  LONG  gross={lg.mean()*1e4:+.2f} net={lg.mean()*1e4-10:+.2f}bps  "
          f"net95%CI=[{ci(nL)[0]-10:+.2f},{ci(nL)[1]-10:+.2f}]")
    print(f"  SHORT gross={sg.mean()*1e4:+.2f} net={sg.mean()*1e4-10:+.2f}bps  "
          f"net95%CI=[{ci(nS)[0]-10:+.2f},{ci(nS)[1]-10:+.2f}]")

# AUC bootstrap by timestamp
aucs = []
for _ in range(2000):
    bs = rng.choice(len(PT), len(PT), replace=True)
    pp = np.concatenate([p_list[j] for j in bs]); rr = np.concatenate([r_list[j] for j in bs])
    aucs.append(auc(pp, rr))
print(f"  AUC 95% CI by timestamp: [{np.percentile(aucs,2.5):.4f}, {np.percentile(aucs,97.5):.4f}]  "
      f"(0.5 inside CI => NOT significant)")

print("\n=== 2) NEGATIVE CONTROL: shuffle returns within each timestamp (breaks real signal) ===")
shuf_r = [rng.permutation(r) for r in r_list]
Ps = np.concatenate(p_list); Rs = np.concatenate(shuf_r)
lg, sg = topk_series(p_list, shuf_r, 1)
print(f"  shuffled AUC={auc(Ps,Rs):.4f} (expect ~0.5)  "
      f"Top1 LONG gross={lg.mean()*1e4:+.2f} SHORT gross={sg.mean()*1e4:+.2f} (expect ~0)")

print("\n=== 3) TRIVIAL BASELINES (no transformer) Top-1 net@10bps ===")
for name, sl in [('rank by Return (momentum)', ret_list),
                 ('rank by Relative_Return (momentum)', rel_list)]:
    lg, sg = topk_series(sl, r_list, 1)
    print(f"  {name:36s} LONG net={lg.mean()*1e4-10:+.2f}  SHORT net={sg.mean()*1e4-10:+.2f}")
    # reversal = flip the score
    lg2, sg2 = topk_series([-s for s in sl], r_list, 1)
    print(f"  {name:36s} (reversal) LONG net={lg2.mean()*1e4-10:+.2f}  SHORT net={sg2.mean()*1e4-10:+.2f}")
# always-short baseline
allshort = np.concatenate([-r for r in r_list])
print(f"  always-SHORT every ticker: gross={allshort.mean()*1e4:+.2f} net={allshort.mean()*1e4-10:+.2f}bps")
