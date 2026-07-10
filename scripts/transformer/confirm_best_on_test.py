"""
Phase 2 — single honest confirmation of the Optuna-best BCE config on the SEALED TEST set.

Reads the best params from artifacts/optuna_bce_best.json, retrains on FULL train data (no
per-epoch subsampling) across several seeds with early-stop on full val AUC, then evaluates the
veto edge ONCE on the TEST window — measured two ways:
  * coverage-matched (top `cov` by rank)  — the rule we tuned;
  * fixed threshold P>0.50                 — IDENTICAL to the v20_bce_veto_walkforward baseline
                                             (+1.14 bps / t +2.27 at K=3 LONG), for apples-to-apples.
Each cell carries a day-clustered CI and a MULTI-shuffle negative control (mean±sd) plus a
control-subtracted edge. Multiple seeds guard against a lucky init; we report the median.

This is the one-shot look at TEST — no threshold/seed cherry-picking afterwards.
Exploratory only — no Gauntlet verdict, no registry stamp. Does NOT overwrite production artifacts.
"""
import os, sys, json, time, argparse
os.environ.setdefault('TRANSFORMER_PANEL', 'data/transformer_panel_v20')
import numpy as np
import pandas as pd          # MUST precede torch on Windows (OpenMP/MKL segfault)
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from scripts.transformer.train import (
    load_panel, valid_decision_timestamps, chrono_split,
    DecisionDataset, collate, bce_family_loss)
from scripts.transformer.model import DualResCSTransformer
from scripts.transformer import veto_lib as vl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── one-time setup ───────────────────────────────────────────────────────────
best = json.load(open('artifacts/optuna_bce_best.json'))
P = best['best_params']
print(f"device={device}\nbest params: {P}\nbest val score: {best['best_value']:+.3f}")

d = load_panel()
F = d['meta']['n_features']; M = d['macro'].shape[1]
n_sec = len(d['meta']['sectors']); tickers = d['meta']['tickers']
sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)

ts = valid_decision_timestamps(d)
tr, va, te = chrono_split(ts, embargo=30)
test_start = int(d['ts_1h'][te[0]]); test_end = int(d['ts_1h'][te[-1]])
print(f"TRAIN={len(tr)} VAL={len(va)} TEST={len(te)}  "
      f"TEST window {pd.Timestamp(test_start)} .. {pd.Timestamp(test_end)}")

print("Scoring v20 XGB on the TEST window once …")
xgb_long, xgb_short, feat_cols = vl.load_xgb()
test_df = vl.build_scored_window(test_start, test_end, xgb_long, xgb_short, feat_cols)
print(f"  cached test_df: {len(test_df):,} rows, {test_df['DateTime'].nunique():,} timestamps")


@torch.no_grad()
def full_auc(model, loader):
    model.eval()
    Ps, Ys = [], []
    for batch in loader:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
        p = torch.sigmoid(logit.float())
        for b in range(p.shape[0]):
            m = valid[b]
            if m.sum() < 2:
                continue
            Ps.append(p[b][m].cpu().numpy()); Ys.append(y[b][m].cpu().numpy())
    if not Ps:
        return 0.5
    p = np.concatenate(Ps); r = np.concatenate(Ys); yb = (r > 0).astype(int)
    n1 = yb.sum(); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(p); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(p))
    return float((ranks[yb == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def train_full(seed, epochs, patience):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DualResCSTransformer(
        F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'], n_slots_15m=d['meta']['n_slots_15m'],
        d_model=P['d_model'], t_layers=P['t_layers'], c_layers=P['c_layers'],
        nhead=P['nhead'], dropout=P['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=P['lr'], weight_decay=P['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    mk = lambda idx, sh: DataLoader(DecisionDataset(d, idx), batch_size=P['batch'], shuffle=sh,
                                    collate_fn=collate, num_workers=0)
    dl_tr, dl_va = mk(tr, True), mk(va, False)
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
                loss = bce_family_loss(P['loss'], logit, ybin, y, valid,
                                       mag_beta=P.get('mag_beta', 0.0), pos_weight=P.get('pos_weight', 1.0),
                                       gamma=P.get('focal_gamma', 2.0), alpha=P.get('focal_alpha', 0.5),
                                       hybrid_mix=P.get('hybrid_mix', 0.5), hybrid_cost=P.get('hybrid_cost', 10.0))
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        auc = full_auc(model, dl_va)
        flag = ''
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0; flag = ' *'
        else:
            bad += 1
        print(f"    seed{seed} epoch {ep+1}/{epochs} val_auc={auc:.4f}{flag}")
        if bad >= patience:
            print(f"    early stop (best val auc {best_auc:.4f})")
            break
    model.load_state_dict(best_state)
    return model, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,1,7')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--cov', type=float, default=0.65)
    ap.add_argument('--th', type=float, default=0.50)
    ap.add_argument('--nshuffle', type=int, default=40)
    ap.add_argument('--save_ckpt', action='store_true', help='save tuned checkpoints (separate names)')
    ap.add_argument('--eval_only', action='store_true',
                    help='skip training; load artifacts/dualres_tuned_seed{seed}.pt and just evaluate')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    # cells to evaluate: (K, side, n_shuffle)  — heavier control on the primary cell
    cells = [(3, 'LONG', args.nshuffle), (1, 'LONG', 10), (5, 'LONG', 10), (3, 'SHORT', 10)]

    per_seed = []
    for seed in seeds:
        t0 = time.time()
        if args.eval_only:
            ckpt = f'artifacts/dualres_tuned_seed{seed}.pt'
            print(f"\n=== seed {seed}: EVAL-ONLY (loading {ckpt}) ===", flush=True)
            model = DualResCSTransformer(
                F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'], n_slots_15m=d['meta']['n_slots_15m'],
                d_model=P['d_model'], t_layers=P['t_layers'], c_layers=P['c_layers'],
                nhead=P['nhead'], dropout=P['dropout']).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device)); vauc = None
        else:
            print(f"\n=== seed {seed}: training on FULL data ===", flush=True)
            model, vauc = train_full(seed, args.epochs, args.patience)
        tab = vl.transformer_pup_table(model, te, d, sector_ids, tickers, device)
        df = vl.attach_pup(test_df, tab)
        res = {'seed': seed, 'val_auc': vauc, 'cells': {}}
        for K, side, nsh in cells:
            cov_m = vl.veto_metrics(df, K, side, keep_mode='coverage', target_cov=args.cov,
                                    n_shuffle=nsh, seed=seed)
            fix_m = vl.veto_metrics(df, K, side, keep_mode='fixed', th=args.th,
                                    n_shuffle=nsh, seed=seed)
            res['cells'][f'K{K}_{side}'] = {'coverage': cov_m, 'fixed': fix_m}
            print(f"  TEST K={K} {side}: "
                  f"cov Δ={cov_m['uplift_bps']:+.2f}(t{cov_m['uplift_t']:+.2f}) "
                  f"nc={cov_m['neg_ctrl_uplift']:+.2f}±{cov_m['neg_ctrl_sd']:.2f} "
                  f"adj={cov_m['adj_uplift_bps']:+.2f} | "
                  f"fixed Δ={fix_m['uplift_bps']:+.2f}(t{fix_m['uplift_t']:+.2f}) "
                  f"nc={fix_m['neg_ctrl_uplift']:+.2f} adj={fix_m['adj_uplift_bps']:+.2f}")
        per_seed.append(res)
        if args.save_ckpt and not args.eval_only:
            torch.save(model.state_dict(), f'artifacts/dualres_tuned_seed{seed}.pt')
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ── aggregate (median across seeds) for the primary cell, both modes ────────
    def agg(cell_key, mode, field):
        vals = [s['cells'][cell_key][mode][field] for s in per_seed
                if s['cells'][cell_key][mode][field] is not None]
        return float(np.median(vals)) if vals else None

    print("\n" + "=" * 78)
    print("PHASE-2 TEST CONFIRMATION  (median across seeds)")
    print(f"Baseline (fixed P>0.50, K=3 LONG): +1.14 bps / t +2.27")
    print("=" * 78)
    summary = {'params': P, 'seeds': seeds, 'baseline_k3long_fixed_bps': 1.14, 'per_seed': per_seed,
               'median': {}}
    for K, side, _ in cells:
        ck = f'K{K}_{side}'
        row = {}
        for mode in ('coverage', 'fixed'):
            row[mode] = {f: agg(ck, mode, f) for f in
                         ('uplift_bps', 'uplift_t', 'uplift_CI_lo', 'uplift_CI_hi',
                          'neg_ctrl_uplift', 'adj_uplift_bps', 'coverage')}
        summary['median'][ck] = row
        cm, fm = row['coverage'], row['fixed']
        print(f"  {ck:9s} | coverage: Δ={cm['uplift_bps']:+.2f} t={cm['uplift_t']:+.2f} "
              f"adj={cm['adj_uplift_bps']:+.2f}  | fixed P>0.50: Δ={fm['uplift_bps']:+.2f} "
              f"t={fm['uplift_t']:+.2f} adj={fm['adj_uplift_bps']:+.2f}")

    json.dump(summary, open('artifacts/phase2_test_confirmation.json', 'w'), indent=2, default=float)
    print("\nsaved -> artifacts/phase2_test_confirmation.json")
    p3 = summary['median']['K3_LONG']['fixed']
    verdict = ("BEATS baseline" if (p3['adj_uplift_bps'] or -9) > 1.14 and (p3['uplift_t'] or 0) > 2
               else "does NOT clearly beat baseline")
    print(f"VERDICT (K=3 LONG, baseline-identical rule, control-adjusted): {verdict}")


if __name__ == '__main__':
    main()
