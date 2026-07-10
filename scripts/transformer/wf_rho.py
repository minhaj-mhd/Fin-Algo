"""
Walk-forward rank-IC harness — close the val->test generalization gap by recency-matched
training, and dissect per-fold TRAIN vs TEST rho for any panel/side.

Motivation (Conv-2026-06-15): single 70/15/15 split shows train rho +0.0276 (t=10.5) ->
test +0.0058 (n.s.); the train->val gap is small (not overfit) but val->test drops
significantly (p=0.026) = non-stationarity. Fix to TRY: expanding walk-forward so each test
block is trained on all history up to its start (train distribution tracks test). Reports,
per fold and aggregate: train rho, test rho, test net@10 K5, and test rho's t-stat.

    python scripts/transformer/wf_rho.py --target short --folds 5 --epochs 10
    TRANSFORMER_PANEL=data/transformer_panel_smc python scripts/transformer/wf_rho.py --target long
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from scripts.transformer.train import (load_panel, DecisionDataset, collate, listnet_loss,
                                       valid_decision_timestamps, EMBARGO, SEED, COSTS_BPS)
from scripts.transformer.model import DualResCSTransformer


@torch.no_grad()
def eval_rho(model, d, idx, device, sec, side_sign):
    from scipy.stats import spearmanr
    dl = DataLoader(DecisionDataset(d, idx), batch_size=16, shuffle=False, collate_fn=collate)
    model.eval(); rhos = []; net5 = []
    for batch in dl:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sec, ~present)
        sc_a = logit.float().cpu().numpy(); yv = y.cpu().numpy(); vv = valid.cpu().numpy()
        for b in range(sc_a.shape[0]):
            m = vv[b]
            if m.sum() < 6:
                continue
            sc = sc_a[b][m]; r = side_sign * yv[b][m]
            if np.std(sc) == 0:
                continue
            rho = spearmanr(sc, r).correlation
            if np.isfinite(rho):
                rhos.append(rho)
                order = np.argsort(-sc)
                net5.append(r[order[:5]].mean() - 10.0 / 1e4)
    rhos = np.array(rhos)
    t = rhos.mean() / (rhos.std(ddof=1) / np.sqrt(len(rhos))) if len(rhos) > 2 else 0.0
    return rhos.mean(), t, len(rhos), (np.mean(net5) * 1e4 if net5 else 0.0)


def train_fold(d, tr, device, sec, F, M, n_sec, side_sign, epochs, lr=3e-4):
    torch.manual_seed(SEED); np.random.seed(SEED)
    # hold out last 10% of train as val for early stop (no test peeking)
    cut = int(len(tr) * 0.9)
    tr_in, va = tr[:cut], tr[cut + EMBARGO:]
    model = DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                                 n_slots_15m=d['meta']['n_slots_15m']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    dl = DataLoader(DecisionDataset(d, tr_in), batch_size=16, shuffle=True, collate_fn=collate)
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        model.train()
        for batch in dl:
            x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                logit = model(x1, x15, s1, s15, macro, sec, ~present)
                loss = listnet_loss(logit, side_sign * y, valid)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        vr, _, _, _ = eval_rho(model, d, va, device, sec, side_sign) if len(va) > 5 else (0, 0, 0, 0)
        if vr > best:
            best = vr; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= 4:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='short')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=10)
    args = ap.parse_args()
    side_sign = 1.0 if args.target == 'long' else -1.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    panel = os.environ.get('TRANSFORMER_PANEL', 'data/transformer_panel')
    print(f"panel={panel}  target={args.target}  folds={args.folds}  epochs={args.epochs}")

    d = load_panel()
    F = d['meta']['n_features']; M = d['macro'].shape[1]; n_sec = len(d['meta']['sectors'])
    sec = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)
    ts = valid_decision_timestamps(d); n = len(ts)

    # expanding folds over the last 50%: train ts[:a_i], test ts[a_i+emb : a_{i+1}]
    anchors = np.linspace(0.5, 1.0, args.folds + 1)
    rows = []
    for i in range(args.folds):
        a0, a1 = int(n * anchors[i]), int(n * anchors[i + 1])
        tr = ts[:a0]; te = ts[a0 + EMBARGO:a1]
        if len(te) < 20:
            continue
        t0 = time.time()
        model = train_fold(d, tr, device, sec, F, M, n_sec, side_sign, args.epochs)
        tr_rho, _, _, _ = eval_rho(model, d, tr[-len(te):], device, sec, side_sign)   # recent-train rho
        te_rho, te_t, te_n, te_net = eval_rho(model, d, te, device, sec, side_sign)
        import pandas as pd
        span = f"{pd.Timestamp(int(d['ts_1h'][te[0]])).date()}..{pd.Timestamp(int(d['ts_1h'][te[-1]])).date()}"
        print(f"fold {i+1}: train<={len(tr)} test={len(te)} [{span}]  "
              f"trainRho={tr_rho:+.4f}  TESTrho={te_rho:+.4f} (t={te_t:+.2f}) net@10K5={te_net:+.2f}bps "
              f"({time.time()-t0:.0f}s)")
        rows.append((tr_rho, te_rho, te_t, te_net, te_n))

    R = np.array(rows)
    # pooled test-rho t across folds (weight by n)
    print("\n================ WALK-FORWARD SUMMARY ================")
    print(f"folds={len(R)}  mean trainRho={R[:,0].mean():+.4f}  mean TESTrho={R[:,1].mean():+.4f}  "
          f"mean test net@10K5={R[:,3].mean():+.2f}bps")
    print(f"single-split baseline TEST rho was +0.0058 (short) — compare above.")
    os.makedirs('artifacts', exist_ok=True)
    json.dump({'panel': panel, 'target': args.target, 'folds': R.tolist(),
               'mean_train_rho': float(R[:,0].mean()), 'mean_test_rho': float(R[:,1].mean()),
               'mean_test_net10k5': float(R[:,3].mean())},
              open(f"artifacts/wf_rho_{args.target}_{os.path.basename(panel)}.json", 'w'), indent=2)


if __name__ == '__main__':
    main()
