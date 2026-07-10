"""
Phase 1 — Optuna hyperparameter search for the BCE veto transformer.

OBJECTIVE (maximize): coverage-matched K=3 LONG veto **t-statistic** on the VALIDATION window,
minus a worst-block penalty (stability floor), with trials rejected if the shuffled negative
control survives. See [[BCE Optuna Tuning — Step-by-Step]] for the plain-language rationale.

DISCIPLINE (non-negotiable):
  * Trains on the TRAIN split, scores the objective on the VAL split. The TEST split is NEVER
    read here — it is reserved for the single Phase-2 confirmation.
  * Panel loaded once; v20 XGB val-window scored once; both reused across all trials.
  * Per-epoch val AUC (on a subsample) is reported to Optuna -> MedianPruner kills weak trials.
  * --no_save semantics: this script writes ONLY the study db + best-params json, never a model
    checkpoint, so production artifacts are untouched.

Exploratory only — no Gauntlet verdict, no registry stamp.
"""
import os, sys, json, time, argparse
os.environ.setdefault('TRANSFORMER_PANEL', 'data/transformer_panel_v20')
import numpy as np
import pandas as pd          # MUST precede torch on Windows (OpenMP/MKL segfault — see [[project_confirm_v10_cosign_deadend]])
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from scripts.transformer.train import (
    load_panel, valid_decision_timestamps, chrono_split,
    DecisionDataset, collate, bce_family_loss)
from scripts.transformer.model import DualResCSTransformer
from scripts.transformer import veto_lib as vl

SEED = 42
K_OBJ = 3                 # objective basket
SIDE_OBJ = 'LONG'         # the side that showed signal
TARGET_COV = 0.65         # coverage-matched keep fraction
PRUNE_VAL_TS = 600        # subsample size for the fast per-epoch AUC pruning signal
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── one-time global setup (panel, splits, cached v20 XGB val scoring) ────────
print(f"device={device}")
d = load_panel()
F = d['meta']['n_features']; M = d['macro'].shape[1]
n_sec = len(d['meta']['sectors']); tickers = d['meta']['tickers']
sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)

ts = valid_decision_timestamps(d)
tr, va, te = chrono_split(ts, embargo=30)
print(f"timestamps: train={len(tr)} val={len(va)} test={len(te)} (TEST untouched)")

val_start = int(d['ts_1h'][va[0]]); val_end = int(d['ts_1h'][va[-1]])
print(f"VAL window: {pd.Timestamp(val_start)} .. {pd.Timestamp(val_end)}")

print("Scoring v20 XGB on the VAL window once (cached for all trials) …")
xgb_long, xgb_short, feat_cols = vl.load_xgb()
val_df = vl.build_scored_window(val_start, val_end, xgb_long, xgb_short, feat_cols)
print(f"  cached val_df: {len(val_df):,} rows, {val_df['DateTime'].nunique():,} timestamps")

# subsample of val timestamps for the cheap per-epoch pruning AUC
rng0 = np.random.default_rng(SEED)
va_sub = np.sort(rng0.choice(va, size=min(PRUNE_VAL_TS, len(va)), replace=False))


@torch.no_grad()
def quick_auc(model, loader):
    """Cheap AUC over a val subsample — pruning signal only (no Top-K baskets)."""
    model.eval()
    P, Y = [], []
    for batch in loader:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
        p = torch.sigmoid(logit.float())
        for b in range(p.shape[0]):
            m = valid[b]
            if m.sum() < 2:
                continue
            P.append(p[b][m].cpu().numpy()); Y.append(y[b][m].cpu().numpy())
    if not P:
        return 0.5
    p = np.concatenate(P); r = np.concatenate(Y); yb = (r > 0).astype(int)
    n1 = yb.sum(); n0 = len(yb) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(p); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(p))
    return float((ranks[yb == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def suggest_params(trial):
    loss_type = trial.suggest_categorical('loss',
                ['plain_bce', 'weighted_bce', 'focal', 'bce_profit_hybrid'])
    d_model = trial.suggest_categorical('d_model', [48, 64, 96, 128])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    if d_model % nhead != 0:
        raise optuna.TrialPruned()          # invalid head/width combo — skip cheaply
    p = dict(
        loss=loss_type, d_model=d_model, nhead=nhead,
        lr=trial.suggest_float('lr', 1e-4, 2e-3, log=True),
        batch=trial.suggest_categorical('batch', [8, 16, 32]),
        t_layers=trial.suggest_int('t_layers', 1, 3),
        c_layers=trial.suggest_int('c_layers', 1, 3),
        dropout=trial.suggest_float('dropout', 0.0, 0.4),
        weight_decay=trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
    )
    if loss_type == 'weighted_bce':
        p['mag_beta'] = trial.suggest_float('mag_beta', 0.0, 5.0)
        p['pos_weight'] = trial.suggest_float('pos_weight', 0.7, 1.5)
    elif loss_type == 'focal':
        p['focal_gamma'] = trial.suggest_float('focal_gamma', 0.5, 3.0)
        p['focal_alpha'] = trial.suggest_float('focal_alpha', 0.25, 0.75)
    elif loss_type == 'bce_profit_hybrid':
        p['hybrid_mix'] = trial.suggest_float('hybrid_mix', 0.1, 0.9)
        p['hybrid_cost'] = trial.suggest_categorical('hybrid_cost', [6.0, 10.0])
    return p


def train_trial(trial, p, epochs):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = DualResCSTransformer(
        F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'], n_slots_15m=d['meta']['n_slots_15m'],
        d_model=p['d_model'], t_layers=p['t_layers'], c_layers=p['c_layers'],
        nhead=p['nhead'], dropout=p['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    mk = lambda idx, sh: DataLoader(DecisionDataset(d, idx), batch_size=p['batch'], shuffle=sh,
                                    collate_fn=collate, num_workers=0)
    dl_sub = mk(va_sub, False)
    # per-epoch resampled train subset (speed lever for the SEARCH only; Phase-2 retrains on full
    # data + full epochs). Same SEED across trials -> fair ranking. frac=1.0 -> use all timestamps.
    sub_rng = np.random.default_rng(SEED)
    n_keep = len(tr) if TRAIN_FRAC >= 1.0 else int(len(tr) * TRAIN_FRAC)
    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        ep_idx = tr if n_keep >= len(tr) else np.sort(sub_rng.choice(tr, n_keep, replace=False))
        dl_tr = mk(ep_idx, True)
        model.train()
        for batch in dl_tr:
            x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
                loss = bce_family_loss(p['loss'], logit, ybin, y, valid,
                                       mag_beta=p.get('mag_beta', 0.0), pos_weight=p.get('pos_weight', 1.0),
                                       gamma=p.get('focal_gamma', 2.0), alpha=p.get('focal_alpha', 0.5),
                                       hybrid_mix=p.get('hybrid_mix', 0.5), hybrid_cost=p.get('hybrid_cost', 10.0))
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        auc = quick_auc(model, dl_sub)
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        trial.report(auc, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()
    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


def objective(trial):
    t0 = time.time()
    p = suggest_params(trial)
    model, best_auc = train_trial(trial, p, EPOCHS)

    # downstream veto objective on VAL (the thing we actually care about)
    tab = vl.transformer_pup_table(model, va, d, sector_ids, tickers, device)
    df = vl.attach_pup(val_df, tab)
    m = vl.veto_metrics(df, K=K_OBJ, side=SIDE_OBJ, target_cov=TARGET_COV, nb=1000, seed=SEED)

    for k, v in m.items():
        if not isinstance(v, list):
            trial.set_user_attr(k, v)
    trial.set_user_attr('val_auc', best_auc)
    trial.set_user_attr('secs', round(time.time() - t0, 1))

    nc = m.get('neg_ctrl_uplift')
    if nc is not None and abs(nc) > 1.0:        # negative control survived → fake edge
        trial.set_user_attr('rejected', 'neg_ctrl')
        return -10.0
    t = m['uplift_t']
    if not np.isfinite(t):
        return -10.0
    bm = m['block_min_uplift']
    penalty = max(0.0, -bm) if np.isfinite(bm) else 0.0    # punish a negative worst-block
    score = t - penalty
    print(f"  trial {trial.number}: loss={p['loss']} auc={best_auc:.4f} "
          f"Δt={t:+.2f} Δbps={m['uplift_bps']:+.2f} blockmin={bm:+.2f} nc={nc:+.3f} "
          f"-> score={score:+.3f} ({trial.user_attrs['secs']}s)")
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--n_trials', type=int, default=50)
    ap.add_argument('--timeout', type=int, default=9000, help='seconds (default 2.5h)')
    ap.add_argument('--train_frac', type=float, default=0.7,
                    help='fraction of train timestamps per epoch (search speed lever; 1.0 = all)')
    ap.add_argument('--study', default='bce_veto_v20')
    args = ap.parse_args()
    global EPOCHS, TRAIN_FRAC
    EPOCHS = args.epochs
    TRAIN_FRAC = args.train_frac

    os.makedirs('artifacts', exist_ok=True)
    storage = 'sqlite:///artifacts/optuna_bce_study.db'
    study = optuna.create_study(
        study_name=args.study, storage=storage, load_if_exists=True, direction='maximize',
        sampler=TPESampler(seed=SEED, n_startup_trials=8),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    print(f"study '{args.study}' @ {storage}  epochs={EPOCHS} n_trials={args.n_trials} timeout={args.timeout}s")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True, catch=(RuntimeError,))

    done = [t for t in study.trials if t.value is not None]
    print(f"\n{'='*70}\nfinished: {len(study.trials)} trials ({len(done)} scored)")
    if study.best_trial:
        bt = study.best_trial
        print(f"BEST score={bt.value:+.3f}  params={bt.params}")
        print(f"  attrs: {bt.user_attrs}")
        out = {'best_value': bt.value, 'best_params': bt.params, 'best_user_attrs': bt.user_attrs,
               'n_trials': len(study.trials), 'n_scored': len(done),
               'config': {'epochs': EPOCHS, 'K': K_OBJ, 'side': SIDE_OBJ, 'target_cov': TARGET_COV}}
        json.dump(out, open('artifacts/optuna_bce_best.json', 'w'), indent=2, default=float)
        print("saved -> artifacts/optuna_bce_best.json")


if __name__ == '__main__':
    main()
