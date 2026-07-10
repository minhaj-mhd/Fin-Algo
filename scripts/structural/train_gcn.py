"""
Train the Level-Graph Gated GCN with the listwise objective and report the SAME
per-side metrics as the transformer baseline (rank-IC + Top-K net @6/10bps with the
median(net-gross)==-cost sanity check). Same decision universe (graph panel is aligned
1:1 to transformer_panel_smc), same split/seed -> direct architecture comparison vs the
listwise transformer (baseline TEST rank-IC: long +0.0014, short +0.0066 = DEAD).

  python scripts/structural/train_gcn.py --target short --epochs 15
  python scripts/structural/train_gcn.py --target short --epochs 15 --neg_control

--neg_control mismatches each timestamp's structural graphs to a RANDOM other timestamp's
graphs (label kept) — destroys the structure<->outcome link. Real signal must vanish here.
"""
import os, sys, json, time, argparse
import pandas as pd                      # import pandas before torch (Windows MKL/OpenMP)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from scripts.structural.gated_gcn import LevelGraphGCN
from scripts.transformer.train import listnet_loss, COSTS_BPS, KS, EMBARGO, SEED

GP = 'data/graph_panel_smc'
RP = 'data/transformer_panel_smc'


def load():
    nodes = np.load(f'{GP}/nodes.npy')               # (T,N,K,D)
    mask = np.load(f'{GP}/node_mask.npy')            # (T,N,K)
    Y = np.load(f'{RP}/Y_ret.npy')
    macro = np.load(f'{RP}/macro.npy')
    date_idx = np.load(f'{RP}/date_idx.npy')
    sector_ids = np.load(f'{RP}/sector_ids.npy')
    meta = json.load(open(f'{RP}/meta.json'))
    return nodes, mask, Y, macro, date_idx, sector_ids, meta


class GraphDS(Dataset):
    def __init__(self, nodes, mask, Y, macro, date_idx, t_idx, struct_idx=None):
        self.nodes, self.mask, self.Y = nodes, mask, Y
        self.macro, self.date_idx = macro, date_idx
        self.t_idx = t_idx
        # struct_idx[i] = which timestamp's GRAPHS to use for item i (neg-control = shuffled)
        self.struct_idx = t_idx if struct_idx is None else struct_idx

    def __len__(self):
        return len(self.t_idx)

    def __getitem__(self, i):
        t = int(self.t_idx[i]); ts = int(self.struct_idx[i])
        nd = np.nan_to_num(self.nodes[ts])           # (N,K,D) graphs (real t, or mismatched)
        mk = self.mask[ts]                           # (N,K)
        macro = np.nan_to_num(self.macro[int(self.date_idx[t])])
        y = self.Y[t]                                # label from the REAL t
        present = mk[:, 0]                            # NOW present == ticker present
        valid = present & np.isfinite(y)
        return (nd.astype(np.float32), mk.astype(np.bool_), macro.astype(np.float32),
                np.nan_to_num(y).astype(np.float32), present.astype(np.bool_),
                valid.astype(np.bool_))


def collate(batch):
    nd, mk, macro, y, present, valid = zip(*batch)
    f = lambda a: torch.from_numpy(np.stack(a))
    return f(nd), f(mk), f(macro), f(y), f(present), f(valid)


@torch.no_grad()
def evaluate(model, loader, device, sector_ids, side_sign, tag=''):
    from scipy.stats import spearmanr
    model.eval()
    rhos, gross = [], {k: [] for k in KS}
    net = {k: {c: [] for c in COSTS_BPS} for k in KS}
    wr = {k: [] for k in KS}
    for nd, mk, macro, y, present, valid in loader:
        nd, mk, macro = nd.to(device), mk.to(device), macro.to(device)
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(nd, mk, macro, sector_ids, ~present.to(device))
        score = logit.float().cpu().numpy()
        yv = y.numpy(); vv = valid.numpy()
        for b in range(score.shape[0]):
            m = vv[b]
            if m.sum() < max(KS) + 1:
                continue
            sc = score[b][m]; r = side_sign * yv[b][m]
            if np.std(sc) == 0:
                continue
            rho = spearmanr(sc, r).correlation
            if np.isfinite(rho):
                rhos.append(rho)
            order = np.argsort(-sc)
            for k in KS:
                picks = r[order[:k]]
                gross[k].append(picks.mean()); wr[k].append((picks > 0).mean())
                for c in COSTS_BPS:
                    net[k][c].append(picks.mean() - c / 1e4)
    out = {'rho': float(np.mean(rhos)) if rhos else 0.0, 'n_ts': len(rhos)}
    rep = [f"[{tag}] n_ts={len(rhos)} rank-IC(rho)={out['rho']:+.4f}"]
    for k in KS:
        g = np.mean(gross[k]) * 1e4 if gross[k] else 0.0
        for c in COSTS_BPS:
            nv = np.mean(net[k][c]) * 1e4 if net[k][c] else 0.0
            chk = (np.median(np.array(net[k][c]) - np.array(gross[k])) * 1e4) if gross[k] else 0.0
            out[f'K{k}_net_{int(c)}'] = nv
            rep.append(f"  K={k} @cost{int(c)}: net={nv:+.2f}bps (gross {g:+.2f}) "
                       f"rawWR={np.mean(wr[k]):.0%} chk(net-gross)={chk:+.2f}")
    print('\n'.join(rep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['long', 'short'], default='short')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--gcn_layers', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--neg_control', action='store_true')
    args = ap.parse_args()
    side_sign = 1.0 if args.target == 'long' else -1.0

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  target={args.target}  neg_control={args.neg_control}")

    nodes, mask, Y, macro, date_idx, sector_ids, meta = load()
    T, N, K, D = nodes.shape
    sec = torch.from_numpy(sector_ids.astype(np.int64)).to(device)
    n_sec = len(meta['sectors'])

    finite = np.isfinite(Y).sum(1) > 0
    ok = finite & (date_idx >= 0) & (mask[:, :, 0].any(1))
    ts_all = np.where(ok)[0]
    n = len(ts_all)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    tr, va, te = ts_all[:i_tr], ts_all[i_tr + EMBARGO:i_va], ts_all[i_va + EMBARGO:]
    print(f"decision timestamps: {n}  train={len(tr)} val={len(va)} test={len(te)}  K={K} D={D}")

    # neg-control: shuffle which timestamp's graphs each train item sees (label kept)
    rng = np.random.default_rng(SEED)
    str_tr = rng.permutation(tr) if args.neg_control else None
    str_va = rng.permutation(va) if args.neg_control else None
    str_te = rng.permutation(te) if args.neg_control else None

    mk = lambda idx, sh, sidx: DataLoader(
        GraphDS(nodes, mask, Y, macro, date_idx, idx, sidx),
        batch_size=args.batch, shuffle=sh, collate_fn=collate, num_workers=0)
    dl_tr, dl_va, dl_te = mk(tr, True, str_tr), mk(va, False, str_va), mk(te, False, str_te)

    model = LevelGraphGCN(D, macro.shape[1], n_sec, d_model=args.d_model,
                          gcn_layers=args.gcn_layers, dropout=args.dropout).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    best, best_state, bad, patience = -1e9, None, 0, 6
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for nd, mkb, macrob, y, present, valid in dl_tr:
            nd, mkb, macrob = nd.to(device), mkb.to(device), macrob.to(device)
            y, present, valid = y.to(device), present.to(device), valid.to(device)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                logit = model(nd, mkb, macrob, sec, ~present)
                loss = listnet_loss(logit, side_sign * y, valid)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)")
        vm = evaluate(model, dl_va, device, sec, side_sign, f'val e{ep+1}')
        if vm['rho'] > best:
            best = vm['rho']; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop (best val rho {best:.4f})"); break
    if best_state:
        model.load_state_dict(best_state)
    print("\n================ FINAL TEST (best-val checkpoint) ================")
    tag = f'TEST gcn {args.target}{" NEG" if args.neg_control else ""}'
    test = evaluate(model, dl_te, device, sec, side_sign, tag)
    os.makedirs('artifacts', exist_ok=True)
    suf = f'_{args.target}{"_neg" if args.neg_control else ""}'
    json.dump({'target': args.target, 'neg_control': args.neg_control,
               'val_best_rho': float(best), 'test': test},
              open(f'artifacts/levelgcn{suf}_metrics.json', 'w'), indent=2, default=float)
    print(f"saved -> artifacts/levelgcn{suf}_metrics.json")


if __name__ == '__main__':
    main()
