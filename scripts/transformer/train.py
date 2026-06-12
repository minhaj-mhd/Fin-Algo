"""
Train the Dual-Resolution Cross-Sectional Transformer on the GPU.

Splits chronologically (train / val / test) with an embargo gap. Per-query z-scored
features mean NaN -> 0 imputation == cross-sectional mean. Reports, honestly and PER SIDE:
  * directional AUC / accuracy (raw skill)
  * Top-K long & short NET bps @ 6 and 10 bps round-trip cost
  * RAW win-rate vs NET win-rate, and median(net - gross) == -cost sanity check
    (guards the one-sided cost-sign bug — see [[feedback_validate_cost_accounting]]).
Exploratory only: NO verdict authority (only the Validation Gauntlet grades models).
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from scripts.transformer.model import DualResCSTransformer

P = 'data/transformer_panel'
L1, L2 = 30, 60                       # 1h / 15m sequence lengths
COSTS_BPS = [6.0, 10.0]
KS = [1, 3, 5]
EMBARGO = 30                          # decision-timestamp gap between splits
SEED = 42


def load_panel():
    d = {}
    for k in ['X_1h', 'Y_ret', 'slot_1h', 'end15', 'ts_1h', 'date_idx', 'macro',
              'slot_15m', 'sector_ids']:
        d[k] = np.load(f'{P}/{k}.npy')
    d['X_15m'] = np.load(f'{P}/X_15m.npy')          # ~1GB, fits in RAM
    d['meta'] = json.load(open(f'{P}/meta.json'))
    return d


class DecisionDataset(Dataset):
    """One item = one 1h decision timestamp -> all N tickers (cross-sectional).
    Optional `restrict` ([T,N] bool) ANDs into `valid`, so the gate's loss/coverage operate
    only on a chosen subset (e.g. v10's actual picks) without changing the tuple shape."""
    def __init__(self, d, t_indices, restrict=None):
        self.X1, self.X15, self.Y = d['X_1h'], d['X_15m'], d['Y_ret']
        self.s1, self.s15, self.end15 = d['slot_1h'], d['slot_15m'], d['end15']
        self.macro, self.date_idx = d['macro'], d['date_idx']
        self.t_idx = t_indices
        self.restrict = restrict

    def __len__(self):
        return len(self.t_idx)

    def __getitem__(self, i):
        t = int(self.t_idx[i])
        e = int(self.end15[t])
        x1 = np.nan_to_num(self.X1[t - L1 + 1:t + 1])                 # (L1,N,F)
        x15 = np.nan_to_num(self.X15[e - L2 + 1:e + 1])              # (L2,N,F)
        x1 = np.transpose(x1, (1, 0, 2))                             # (N,L1,F)
        x15 = np.transpose(x15, (1, 0, 2))                          # (N,L2,F)
        s1 = self.s1[t - L1 + 1:t + 1].astype(np.int64)             # (L1,)
        s15 = self.s15[e - L2 + 1:e + 1].astype(np.int64)          # (L2,)
        macro = np.nan_to_num(self.macro[int(self.date_idx[t])])    # (M,)
        y = self.Y[t]                                               # (N,) returns
        present = np.isfinite(self.X1[t, :, 0])                     # ticker present at t
        valid = present & np.isfinite(y)                           # usable for loss/eval
        if self.restrict is not None:
            valid = valid & self.restrict[t]                       # focus on a subset (e.g. v10 picks)
        ybin = np.where(y > 0, 1.0, 0.0).astype(np.float32)
        return (x1.astype(np.float32), x15.astype(np.float32), s1, s15,
                macro.astype(np.float32), ybin, np.nan_to_num(y).astype(np.float32),
                present.astype(np.bool_), valid.astype(np.bool_))


def collate(batch):
    x1, x15, s1, s15, macro, ybin, y, present, valid = zip(*batch)
    f = lambda a: torch.from_numpy(np.stack(a))
    return (f(x1), f(x15), f(s1), f(s15), f(macro), f(ybin), f(y),
            f(present), f(valid))


def listnet_loss(score, sret, valid):
    """ListNet top-1 cross-sectional ranking loss.
    score (B,N) logits; sret (B,N) side-target returns (y for long, -y for short);
    valid (B,N) bool. Target distribution = softmax(zscore(sret)) over valid names per
    timestamp; predicted = log_softmax(score). Masked so absent/NaN names are ignored."""
    neg = torch.finfo(score.dtype).min
    s = score.masked_fill(~valid, neg)
    logq = torch.log_softmax(s, dim=1)                       # (B,N) over names
    vf = valid.float()
    cnt = vf.sum(1, keepdim=True).clamp(min=1)
    mean = (sret * vf).sum(1, keepdim=True) / cnt
    var = (((sret - mean) * vf) ** 2).sum(1, keepdim=True) / cnt
    z = (sret - mean) / (var.sqrt() + 1e-6)
    z = z.masked_fill(~valid, neg)
    P = torch.softmax(z, dim=1)                              # target top-1 distribution
    per_ts = -(P * logq).sum(1)                              # (B,)
    enough = (valid.sum(1) >= 2).float()
    return (per_ts * enough).sum() / enough.sum().clamp(min=1)


@torch.no_grad()
def evaluate_sided(model, loader, device, sector_ids, side_sign, tag=''):
    """Side-specialist eval: rank-IC + Top-K net of the model's own Top-K basket taken in
    its intended direction. side_sign=+1 long (PnL=+y), -1 short (PnL=-y)."""
    from scipy.stats import spearmanr
    model.eval()
    rhos = []
    gross = {k: [] for k in KS}
    net = {k: {c: [] for c in COSTS_BPS} for k in KS}
    wr = {k: [] for k in KS}
    for batch in loader:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
        score = logit.float()
        for b in range(score.shape[0]):
            m = valid[b]
            if m.sum() < max(KS) + 1:
                continue
            sc = score[b][m].cpu().numpy()
            r = (side_sign * y[b][m]).cpu().numpy()           # realized side return
            if np.std(sc) == 0:
                continue
            rho = spearmanr(sc, r).correlation
            if np.isfinite(rho):
                rhos.append(rho)
            order = np.argsort(-sc)                            # high score = strong side pick
            for k in KS:
                picks = r[order[:k]]
                gross[k].append(picks.mean())
                wr[k].append((picks > 0).mean())
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


def valid_decision_timestamps(d):
    T = d['X_1h'].shape[0]
    finite_label = np.isfinite(d['Y_ret']).sum(axis=1) > 0
    ok = np.zeros(T, dtype=bool)
    for t in range(T):
        if t < L1 - 1 or not finite_label[t]:
            continue
        e = int(d['end15'][t])
        if e < L2 - 1 or d['date_idx'][t] < 0:
            continue
        ok[t] = True
    return np.where(ok)[0]


@torch.no_grad()
def evaluate(model, loader, device, sector_ids, tag=''):
    model.eval()
    P_all, Y_all = [], []          # prob, return for every valid (t,n)
    topk = {k: {'long_g': [], 'long_n': {c: [] for c in COSTS_BPS},
                'short_g': [], 'short_n': {c: [] for c in COSTS_BPS}} for k in KS}
    for batch in loader:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
        prob = torch.sigmoid(logit.float())
        for b in range(prob.shape[0]):
            m = valid[b]
            if m.sum() < max(KS) + 1:
                continue
            p = prob[b][m].cpu().numpy()
            r = y[b][m].cpu().numpy()
            P_all.append(p); Y_all.append(r)
            order = np.argsort(-p)               # high prob -> long candidates
            for k in KS:
                lr = r[order[:k]]                # long picks realized return
                sr = -r[order[-k:]]              # short picks realized (low prob)
                topk[k]['long_g'].append(lr.mean())
                topk[k]['short_g'].append(sr.mean())
                for c in COSTS_BPS:
                    topk[k]['long_n'][c].append(lr.mean() - c / 1e4)
                    topk[k]['short_n'][c].append(sr.mean() - c / 1e4)
    p = np.concatenate(P_all); r = np.concatenate(Y_all)
    ybin = (r > 0).astype(int)
    # AUC (rank-based)
    order = np.argsort(p); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(p))
    n1 = ybin.sum(); n0 = len(ybin) - n1
    auc = (ranks[ybin == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0 + 1e-9)
    acc = ((p > 0.5).astype(int) == ybin).mean()
    # position-based net-of-cost PnL (cost paid proportional to position size)
    pos = 2 * p - 1
    netpnl10 = float(np.mean(pos * r) * 1e4 - 10.0 * np.mean(np.abs(pos)))
    deploy = float(np.mean(np.abs(pos)))
    out = {'auc': float(auc), 'acc': float(acc), 'n': int(len(p)), 'base_up': float(ybin.mean()),
           'netpnl10': netpnl10, 'deploy': deploy}
    rep = [f"[{tag}] n={len(p):,} AUC={auc:.4f} acc={acc:.3f} base_up={ybin.mean():.3f} "
           f"| netPnL@10={netpnl10:+.3f}bps deploy={deploy:.3f}"]
    for k in KS:
        lg = np.mean(topk[k]['long_g']) * 1e4
        sg = np.mean(topk[k]['short_g']) * 1e4
        for c in COSTS_BPS:
            ln = np.mean(topk[k]['long_n'][c]) * 1e4
            sn = np.mean(topk[k]['short_n'][c]) * 1e4
            # cost-accounting sanity: median(net-gross) must equal -c
            chk_l = np.median(np.array(topk[k]['long_n'][c]) - np.array(topk[k]['long_g'])) * 1e4
            out[f'K{k}_long_net_{int(c)}'] = ln
            out[f'K{k}_short_net_{int(c)}'] = sn
            rep.append(f"  K={k} @cost{int(c)}: LONG net={ln:+.2f}bps (gross {lg:+.2f}) | "
                       f"SHORT net={sn:+.2f}bps (gross {sg:+.2f}) | chk(net-gross)={chk_l:+.2f}")
    print('\n'.join(rep))
    return out


def gate_loss(score, sret, valid, cost, keep_rate, lam):
    """Cost-aware coverage-budgeted veto-gate loss (per side).
    g = sigmoid(score) is the keep-weight; n = sret - cost is the net per-position outcome
    (return units). Maximize captured net PnL  Σ g·n  while pinning the soft keep-rate
    mean(g) near keep_rate (else the trivial optimum in a sub-cost universe is g->0 = veto
    everything). Net is scaled to bps so the capture and coverage terms are comparable."""
    g = torch.sigmoid(score)
    n = sret - cost
    vf = valid.float()
    cnt = vf.sum().clamp(min=1)
    capture_bps = (g * n * vf).sum() / cnt * 1e4        # mean captured net, bps
    cov = (g * vf).sum() / cnt                           # realized soft keep-rate
    return -capture_bps + lam * (cov - keep_rate) ** 2


@torch.no_grad()
def evaluate_gate(model, loader, device, sector_ids, side_sign, keep_rate, tag=''):
    """At the fixed keep budget, hard-keep the top-keep_rate fraction by gate score and report
    net@cost of KEPT vs VETOED vs ALL (in the side direction). Early-stop on kept net@10;
    kept should beat all, vetoed should be worse (that gap == veto value)."""
    model.eval()
    kept = {c: [] for c in COSTS_BPS}
    vetoed = {c: [] for c in COSTS_BPS}
    alln = {c: [] for c in COSTS_BPS}
    cov = []
    for batch in loader:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
        score = logit.float()
        for b in range(score.shape[0]):
            m = valid[b]
            nv = int(m.sum())
            if nv < 2:                                            # restricted (v10 picks) -> few names/ts
                continue
            sc = score[b][m].cpu().numpy()
            r = (side_sign * y[b][m]).cpu().numpy()
            k = max(1, int(round(keep_rate * nv)))
            order = np.argsort(-sc)                       # high score = keep
            kp, vt = order[:k], order[k:]
            cov.append(k / nv)
            for c in COSTS_BPS:
                kept[c].append(r[kp].mean() - c / 1e4)
                alln[c].append(r.mean() - c / 1e4)
                if len(vt):
                    vetoed[c].append(r[vt].mean() - c / 1e4)
    out = {'keepnet': float(np.mean(kept[10.0]) * 1e4), 'coverage': float(np.mean(cov)), 'n_ts': len(cov)}
    rep = [f"[{tag}] n_ts={len(cov)} keep~{np.mean(cov):.0%}"]
    for c in COSTS_BPS:
        kp = np.mean(kept[c]) * 1e4
        al = np.mean(alln[c]) * 1e4
        vt = np.mean(vetoed[c]) * 1e4 if vetoed[c] else float('nan')
        out[f'kept_net_{int(c)}'] = kp
        out[f'all_net_{int(c)}'] = al
        rep.append(f"  @cost{int(c)}: KEPT {kp:+.2f}  ALL {al:+.2f}  VETOED {vt:+.2f}  "
                   f"uplift(kept-all)={kp-al:+.2f}bps")
    print('\n'.join(rep))
    return out


def build_v10_pickmask(d, side):
    """[T,N] bool mask of v10's Top-5 {side} picks per timestamp, aligned to the panel grid.
    Precomputed torch-free by scripts/transformer/make_v10_pickmask.py (building it in-process
    after the torch import segfaults on Windows via the OpenMP/MKL clash)."""
    path = f'{P}/v10_pickmask_{side}.npy'
    mask = np.load(path)
    print(f"[v10-restrict] loaded {path}: {int(mask.sum()):,} pick-cells over {int(mask.any(1).sum()):,} timestamps")
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--objective', choices=['bce', 'netpnl', 'listwise', 'gate'], default='bce')
    ap.add_argument('--target', choices=['long', 'short'], default='long',
                    help='side specialist (listwise/gate): long uses +y, short uses -y')
    ap.add_argument('--cost_bps', type=float, default=10.0)
    ap.add_argument('--keep_rate', type=float, default=0.70, help='gate: target soft keep budget rho')
    ap.add_argument('--gate_lambda', type=float, default=100.0, help='gate: coverage-budget penalty weight')
    ap.add_argument('--v10_restrict', action='store_true',
                    help='gate: restrict loss/eval to v10 Top-5 {target} picks (decision-boundary focus)')
    args = ap.parse_args()
    side_sign = 1.0 if args.target == 'long' else -1.0

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    d = load_panel()
    F = d['meta']['n_features']; M = d['macro'].shape[1]
    n_sec = len(d['meta']['sectors'])
    sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)

    ts = valid_decision_timestamps(d)
    n = len(ts)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    tr = ts[:i_tr]
    va = ts[i_tr + EMBARGO:i_va]
    te = ts[i_va + EMBARGO:]
    print(f"decision timestamps: {n}  train={len(tr)} val={len(va)} test={len(te)}")
    import pandas as pd
    print(f"  date span: {pd.Timestamp(int(d['ts_1h'][ts[0]]))} .. {pd.Timestamp(int(d['ts_1h'][ts[-1]]))}")

    restrict = build_v10_pickmask(d, args.target) if args.v10_restrict else None
    mk = lambda idx, sh: DataLoader(DecisionDataset(d, idx, restrict=restrict), batch_size=args.batch,
                                    shuffle=sh, collate_fn=collate, num_workers=0)
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    model = DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                                 n_slots_15m=d['meta']['n_slots_15m'], d_model=args.d_model,
                                 dropout=args.dropout).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    bce = nn.BCEWithLogitsLoss(reduction='none')

    cost = args.cost_bps / 1e4
    sel = {'netpnl': 'netpnl10', 'listwise': 'rho', 'gate': 'keepnet'}.get(args.objective, 'auc')
    side_tag = f" target={args.target}" if args.objective in ('listwise', 'gate') else ''
    if args.objective == 'gate':
        val_fn = lambda tag: evaluate_gate(model, dl_va, device, sector_ids, side_sign, args.keep_rate, tag)
    elif args.objective == 'listwise':
        val_fn = lambda tag: evaluate_sided(model, dl_va, device, sector_ids, side_sign, tag)
    else:
        val_fn = lambda tag: evaluate(model, dl_va, device, sector_ids, tag)
    print(f"objective={args.objective}{side_tag}  early-stop on val {sel}  (cost={args.cost_bps}bps)")
    best_metric, best_state, patience, bad = -1e9, None, 6, 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for batch in dl_tr:
            x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
                if args.objective == 'gate':
                    loss = gate_loss(logit, side_sign * y, valid, cost, args.keep_rate, args.gate_lambda)
                elif args.objective == 'listwise':
                    loss = listnet_loss(logit, side_sign * y, valid)
                elif args.objective == 'netpnl':
                    pos = 2 * torch.sigmoid(logit) - 1           # [-1,1], 0 = no trade
                    pnl = pos * y - cost * pos.abs()             # net-of-cost PnL per name
                    loss = -(pnl * valid).sum() / valid.sum().clamp(min=1)
                else:
                    loss = (bce(logit, ybin) * valid).sum() / valid.sum().clamp(min=1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)")
        vm = val_fn(f'val e{ep+1}')
        if vm[sel] > best_metric:
            best_metric = vm[sel]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop (best val {sel} {best_metric:.4f})"); break

    if best_state:
        model.load_state_dict(best_state)
    print("\n================ FINAL TEST (best-val checkpoint) ================")
    if args.objective == 'gate':
        test_metrics = evaluate_gate(model, dl_te, device, sector_ids, side_sign, args.keep_rate,
                                     tag=f'TEST gate {args.target}')
        ckpt = f'artifacts/dualres_gate_{args.target}{"_v10" if args.v10_restrict else ""}.pt'
    elif args.objective == 'listwise':
        test_metrics = evaluate_sided(model, dl_te, device, sector_ids, side_sign, tag=f'TEST {args.target}')
        ckpt = f'artifacts/dualres_{args.target}.pt'
    else:
        test_metrics = evaluate(model, dl_te, device, sector_ids, tag='TEST')
        suffix = '' if args.objective == 'bce' else f'_{args.objective}{int(args.cost_bps)}'
        ckpt = f'artifacts/dualres_transformer{suffix}.pt'
    os.makedirs('artifacts', exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    meta_path = ckpt.replace('.pt', '_metrics.json')
    json.dump({'objective': args.objective, 'target': args.target,
               'val_best_metric': float(best_metric), 'test': test_metrics},
              open(meta_path, 'w'), indent=2, default=float)
    print(f"saved -> {ckpt} + {meta_path}")


if __name__ == '__main__':
    main()
