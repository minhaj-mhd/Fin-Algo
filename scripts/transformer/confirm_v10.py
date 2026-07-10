"""
HONEST BEST-SHOT of the co-sign decision layer the user actually wants:
v10 ranks Top-K per side; an INDEPENDENT directional transformer predicts P(up) per
ticker; a v10 pick is taken iff the transformer AGREES with its side (long iff P_up>th,
short iff P_up<1-th).

Unlike the earlier veto test, the direction model here is trained FOR THE JOB:
  * SIGNED single direction logit (one opinion per ticker; high => up, low => down) so it
    cannot "keep" a name for both sides — it is a genuine direction, not two side-gates.
  * COST-AWARE capture loss, RESTRICTED to v10's actual Top-5 picks (the decision set):
        L = -mean[ sigmoid(side*logit) * (side*y - cost) ]*1e4 + lam*(mean(agree)-keep_rate)^2
    i.e. confirm v10's winners (drive agree->1) and veto its losers (agree->0), net of cost,
    with a soft coverage anchor so it does not trivially veto everything in a sub-cost universe.
  * MODEL-SELECTED on val "confirmed net@10" of v10 picks (deployment metric), NOT global AUC.

Final verdict mirrors the trusted honest eval (scripts/transformer/veto_walkforward.py /
gate_veto_v10.py): NPZ-join to v10's cached WF-OOS ranks, AND-gate over the transformer's
TEST window only (dates >= 85% split), Top-1/3/5, net@6/10/20 with t-stats, raw-vs-net WR,
plus a within-timestamp shuffle negative control. Exploratory only -- NO Gauntlet verdict.
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd          # import BEFORE torch/CUDA init: lazy pandas-after-CUDA segfaults on Windows (MKL/OpenMP double-load)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from scripts.transformer.train import load_panel, valid_decision_timestamps, EMBARGO, L1, L2, SEED
from scripts.transformer.model import DualResCSTransformer

P = 'data/transformer_panel'
V3_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
NPZ = 'data/model_analysis/v10_v18_independent/walkforward_preds.npz'
KS = [1, 3, 5]


def build_side_panel():
    """[T,N] int8: +1 where v10 picks the name long, -1 short, 0 otherwise.
    Cells that are BOTH (19 total, 0.2%) are zeroed (ambiguous -> excluded)."""
    ml = np.load(f'{P}/v10_pickmask_long.npy')
    ms = np.load(f'{P}/v10_pickmask_short.npy')
    side = ml.astype(np.int8) - ms.astype(np.int8)
    side[ml & ms] = 0
    print(f"[v10 picks] long {int((side==1).sum()):,}  short {int((side==-1).sum()):,}  "
          f"over {int((side!=0).any(1).sum()):,} timestamps")
    return side


class ConfirmDataset(Dataset):
    """One item = one 1h decision timestamp. Returns model inputs + realized return + the
    per-ticker v10 SIDE ({-1,0,1}); the loss/eval operate only on side!=0 cells (v10 picks)."""
    def __init__(self, d, t_indices, side):
        self.X1, self.X15, self.Y = d['X_1h'], d['X_15m'], d['Y_ret']
        self.s1, self.s15, self.end15 = d['slot_1h'], d['slot_15m'], d['end15']
        self.macro, self.date_idx = d['macro'], d['date_idx']
        self.t_idx = t_indices
        self.side = side

    def __len__(self):
        return len(self.t_idx)

    def __getitem__(self, i):
        t = int(self.t_idx[i]); e = int(self.end15[t])
        x1 = np.transpose(np.nan_to_num(self.X1[t - L1 + 1:t + 1]), (1, 0, 2))   # (N,L1,F)
        x15 = np.transpose(np.nan_to_num(self.X15[e - L2 + 1:e + 1]), (1, 0, 2))  # (N,L2,F)
        s1 = self.s1[t - L1 + 1:t + 1].astype(np.int64)
        s15 = self.s15[e - L2 + 1:e + 1].astype(np.int64)
        macro = np.nan_to_num(self.macro[int(self.date_idx[t])]).astype(np.float32)
        y = np.nan_to_num(self.Y[t]).astype(np.float32)                          # (N,)
        present = np.isfinite(self.X1[t, :, 0])
        side = self.side[t].astype(np.float32)                                   # (N,) {-1,0,1}
        side = side * np.isfinite(self.Y[t])                                     # need a label to learn
        return (x1.astype(np.float32), x15.astype(np.float32), s1, s15, macro, y,
                present.astype(np.bool_), side)


def collate(batch):
    f = lambda a: torch.from_numpy(np.stack(a))
    x1, x15, s1, s15, macro, y, present, side = zip(*batch)
    return f(x1), f(x15), f(s1), f(s15), f(macro), f(y), f(present), f(side)


def confirm_loss(logit, y, side, cost, keep_rate, lam):
    """Signed cost-aware capture restricted to v10 picks. agree=sigmoid(side*logit) is the
    soft 'take it' weight in the pick's direction; n=side*y-cost its net outcome."""
    mask = (side != 0)
    vf = mask.float()
    cnt = vf.sum().clamp(min=1)
    s = side
    agree = torch.sigmoid(s * logit)
    n = s * y - cost
    capture_bps = (agree * n * vf).sum() / cnt * 1e4
    cov = (agree * vf).sum() / cnt
    return -capture_bps + lam * (cov - keep_rate) ** 2


@torch.no_grad()
def eval_confirm(model, loader, device, sector_ids, keep_rate, tag=''):
    """Per timestamp keep the top keep_rate v10 picks by directional agreement; report net@cost
    of KEPT vs ALL picks (PnL taken in v10's side). Select on kept net@10."""
    model.eval()
    kept = {6.0: [], 10.0: []}; alln = {6.0: [], 10.0: []}; cov = []
    for batch in loader:
        x1, x15, s1, s15, macro, y, present, side = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present).float()
        for b in range(logit.shape[0]):
            m = side[b] != 0
            nv = int(m.sum())
            if nv < 2:
                continue
            s = side[b][m]; yy = y[b][m]; lg = logit[b][m]
            agree = torch.sigmoid(s * lg).cpu().numpy()
            pnl = (s * yy).cpu().numpy()                 # gross PnL of pick in its direction
            k = max(1, int(round(keep_rate * nv)))
            order = np.argsort(-agree)
            cov.append(k / nv)
            for c in (6.0, 10.0):
                kept[c].append(pnl[order[:k]].mean() - c / 1e4)
                alln[c].append(pnl.mean() - c / 1e4)
    out = {'kept_net_10': float(np.mean(kept[10.0]) * 1e4), 'coverage': float(np.mean(cov)),
           'n_ts': len(cov)}
    rep = [f"[{tag}] n_ts={len(cov)} keep~{np.mean(cov):.0%}"]
    for c in (6.0, 10.0):
        kp = np.mean(kept[c]) * 1e4; al = np.mean(alln[c]) * 1e4
        rep.append(f"  @cost{int(c)}: KEPT {kp:+.2f}  ALL {al:+.2f}  uplift={kp-al:+.2f}bps")
    print('\n'.join(rep))
    return out


# ───────────────────────── final honest AND-gate eval (NPZ join) ─────────────────────────
COSTS = {'6bps': 0.0006, '10bps': 0.0010, '20bps': 0.0020}


@torch.no_grad()
def infer_pup(model, d, te, sector_ids, device):
    X1, X15, S1, S15, MA, DI, E = (d['X_1h'], d['X_15m'], d['slot_1h'], d['slot_15m'],
                                   d['macro'], d['date_idx'], d['end15'])
    ts1 = d['ts_1h']; tickers = d['meta']['tickers']
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
            logit = model(tt(x1.astype(np.float32)), tt(x15.astype(np.float32)), tt(s1), tt(s15),
                          tt(macro), sector_ids, ~tt(present[None].astype(np.bool_)))
        p = torch.sigmoid(logit.float())[0].cpu().numpy()
        tns = int(ts1[t])
        for j in np.where(present)[0]:
            table[(tns, tickers[j])] = float(p[j])
    return table


def tstats(r, cost):
    from scipy.stats import ttest_1samp
    r = np.asarray(r, float)
    if len(r) == 0:
        return dict(n=0, raw=0.0, net=0.0, win=0.0, netwin=0.0, t=0.0)
    net = r - cost
    t = float(ttest_1samp(net, 0).statistic) if len(r) > 1 and np.std(net) > 0 else 0.0
    return dict(n=len(r), raw=round(r.mean() * 1e4, 2), net=round(net.mean() * 1e4, 2),
                win=round((r > 0).mean(), 3), netwin=round((net > 0).mean(), 3), t=round(t, 2))


def and_gate(K, q, rl, rs, y, pup, mask, th):
    """v10 Top-K per side; keep long iff pup>th, short iff pup<1-th. Returns (L,S) PnL lists."""
    L, S = [], []; keptL = keptS = totL = totS = 0
    for qid in np.unique(q[mask]):
        m = (q == qid) & mask
        if m.sum() < K:
            continue
        rl_, rs_, y_, p_ = rl[m], rs[m], y[m], pup[m]
        for j in np.argsort(rl_)[-K:]:
            totL += 1
            if p_[j] > th:
                L.append(y_[j]); keptL += 1
        for j in np.argsort(rs_)[-K:]:
            totS += 1
            if p_[j] < 1 - th:
                S.append(-y_[j]); keptS += 1
    return np.array(L), np.array(S), (keptL, totL), (keptS, totS)


def final_eval(model, d, te, sector_ids, device, out_json):
    import pandas as pd
    ts1 = d['ts_1h']; cutoff = int(ts1[te[0]])
    print(f"\ninferring P(up) on TEST window (>= {pd.Timestamp(cutoff)}, {len(te)} ts) ...")
    table = infer_pup(model, d, te, sector_ids, device)

    z = np.load(NPZ, allow_pickle=True)
    idx, q, y, rl, rs = z['idx'], z['q'], z['y'], z['rl'], z['rs']
    v3 = pd.read_csv(V3_FILE, usecols=['DateTime', 'Ticker'])
    dt_ns = pd.to_datetime(v3['DateTime']).values.astype('datetime64[ns]').astype('int64')[idx]
    tk = v3['Ticker'].str.replace('.NS', '', regex=False).values[idx]
    pup = np.array([table.get((int(dt_ns[k]), tk[k]), np.nan) for k in range(len(idx))])
    base = (dt_ns >= cutoff) & np.isfinite(pup)
    print(f"v10 OOS rows in TEST window: {base.sum():,}\n")

    out = {'cutoff': str(pd.Timestamp(cutoff)), 'n_oos': int(base.sum()), 'results': {}}
    print("=" * 92)
    print("CO-SIGN DECISION LAYER  (v10 Top-K, take iff directional transformer AGREES)  TEST/OOS")
    print("=" * 92)
    for K in KS:
        print(f"\n################  Top-{K}  ################")
        # v10 alone (th=0 keeps everything)
        for name, th in [('v10_alone', -1.0), ('v10 + confirm@0.50', 0.50),
                         ('v10 + confirm@0.55', 0.55), ('v10 + confirm@0.60', 0.60)]:
            L, S, kl, ks = and_gate(K, q, rl, rs, y, pup, base, th)
            for cname, cv in COSTS.items():
                ls, ss = tstats(L, cv), tstats(S, cv)
                out['results'].setdefault(f'K{K}', {}).setdefault(name, {})[cname] = dict(long=ls, short=ss)
            ls, ss = tstats(L, 0.0010), tstats(S, 0.0010)
            print(f"  [{name:20s}] keepL {kl[0]:>4}/{kl[1]:<4} keepS {ks[0]:>4}/{ks[1]:<4} | "
                  f"@10bps LONG net{ls['net']:>+6.1f} win{ls['win']:.0%} t{ls['t']:>+5.2f} | "
                  f"SHORT net{ss['net']:>+6.1f} win{ss['win']:.0%} t{ss['t']:>+5.2f}")

    # negative control: shuffle P(up) within each timestamp -> agreement should add ~0
    print("\n--- NEGATIVE CONTROL (shuffle P(up) within timestamp; uplift should vanish) ---")
    rng = np.random.default_rng(0)
    pup_sh = pup.copy()
    for qid in np.unique(q[base]):
        m = (q == qid) & base
        ii = np.where(m)[0]
        pup_sh[ii] = rng.permutation(pup_sh[ii])
    for K in [3]:
        for name, th in [('v10_alone', -1.0), ('shuf confirm@0.55', 0.55)]:
            L, S, kl, ks = and_gate(K, q, rl, rs, y, pup_sh, base, th)
            ls, ss = tstats(L, 0.0010), tstats(S, 0.0010)
            print(f"  [{name:20s}] @10bps LONG net{ls['net']:>+6.1f} t{ls['t']:>+5.2f} | "
                  f"SHORT net{ss['net']:>+6.1f} t{ss['t']:>+5.2f}")

    os.makedirs('artifacts', exist_ok=True)
    json.dump(out, open(out_json, 'w'), indent=2, default=float)
    print(f"\nsaved -> {out_json}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d_model', type=int, default=96)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--cost_bps', type=float, default=10.0)
    ap.add_argument('--keep_rate', type=float, default=0.70)
    ap.add_argument('--lam', type=float, default=100.0)
    ap.add_argument('--smoke', action='store_true', help='1 epoch sanity only')
    ap.add_argument('--eval_only', action='store_true',
                    help='skip training; load saved checkpoint and run the NPZ AND-gate eval '
                         '(separate process to dodge the Windows torch+pandas OpenMP segfault)')
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    d = load_panel()
    F = d['meta']['n_features']; M = d['macro'].shape[1]; n_sec = len(d['meta']['sectors'])
    sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)

    ts = valid_decision_timestamps(d); n = len(ts)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    tr = ts[:i_tr]; va = ts[i_tr + EMBARGO:i_va]; te = ts[i_va + EMBARGO:]
    import pandas as pd
    print(f"decision ts: {n}  train={len(tr)} val={len(va)} test={len(te)}  "
          f"span {pd.Timestamp(int(d['ts_1h'][ts[0]]))}..{pd.Timestamp(int(d['ts_1h'][ts[-1]]))}")

    ckpt = 'artifacts/dualres_confirm_v10.pt'
    model = DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                                 n_slots_15m=d['meta']['n_slots_15m'], d_model=args.d_model,
                                 dropout=args.dropout).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}  (d_model={args.d_model})")

    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
        print(f"loaded {ckpt} -> running final OOS AND-gate eval")
        final_eval(model, d, te, sector_ids, device, 'artifacts/confirm_v10_eval.json')
        return

    side = build_side_panel()
    mk = lambda idx, sh: DataLoader(ConfirmDataset(d, idx, side), batch_size=args.batch,
                                    shuffle=sh, collate_fn=collate, num_workers=0)
    dl_tr, dl_va = mk(tr, True), mk(va, False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    cost = args.cost_bps / 1e4

    print(f"objective=confirm (signed cost-aware, v10-restricted)  early-stop on val kept_net_10")
    best, best_state, patience, bad = -1e9, None, 6, 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for batch in dl_tr:
            x1, x15, s1, s15, macro, y, present, sd = [b.to(device) for b in batch]
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
                loss = confirm_loss(logit, y, sd, cost, args.keep_rate, args.lam)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)")
        vm = eval_confirm(model, dl_va, device, sector_ids, args.keep_rate, tag=f'val e{ep+1}')
        if vm['kept_net_10'] > best:
            best = vm['kept_net_10']; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop (best val kept_net_10 {best:.4f})"); break

    if best_state:
        model.load_state_dict(best_state)
    os.makedirs('artifacts', exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"\nsaved checkpoint -> {ckpt}  (best val kept_net_10={best:+.3f}bps)")
    if not args.smoke:
        print("run the OOS verdict in a fresh process (avoids torch+pandas segfault):")
        print("  python -u scripts/transformer/confirm_v10.py --eval_only")


if __name__ == '__main__':
    main()
