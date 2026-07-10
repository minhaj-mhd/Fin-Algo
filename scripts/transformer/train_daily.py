"""
Train the daily cross-sectional gate transformer (veto overlay on daily_macro_v2), purged walk-forward.

Mirrors v2's 4-fold WF month boundaries (expanding train / 6mo val / 6mo test). For each fold and each
side it trains the cost-aware GATE loss (the tool proven right on the intraday v10 veto) on the FULL
cross-section, early-stops on val, and predicts a per-(day,ticker) gate score on the TEST months. The 4
test windows tile the last ~24 months -> genuine OOS gate scores aligned to v2's OOS picks. The veto is
then APPLIED to v2's top-5 picks in daily_veto_walkforward.py (training on all names, applying to v2's
picks == exactly the intraday veto_walkforward pattern).

Discipline:
  * Purged WF, NOT a single split. Train+val strictly precede test (asserted via month sets).
  * Macro normalized TRAIN-ONLY per fold (panel stores raw macro). Stock feats already per-day z-scored.
  * Training target winsorized per-fold (Label_3D has a +8592% split outlier) so one row can't dominate
    the gate's captured-PnL gradient. EVAL/audit uses RAW returns (fragility-checked separately).
  * cost_bps=10, keep_rate=0.70 fixed (do NOT tune to pass -- pre-registered stop rule).

Exploratory only: NO verdict authority. Outputs gate scores; the Gauntlet alone grades.
Outputs (data/daily_transformer_panel/): daily_gate_long.npy (T,N)  daily_gate_short.npy (T,N)
"""
import os, sys, json, time, argparse
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # Windows OpenMP/MKL clash (numpy<->torch segfault)
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from scripts.transformer.daily_model import DailyCSTransformer
from scripts.transformer.train import gate_loss

P = 'data/daily_transformer_panel'
L = 60                                   # daily lookback window
COST_BPS = 10.0
KEEP_RATE = 0.70
EMBARGO_D = 3                            # 3-day fwd label -> purge 3 trading days at split joins
SEED = 42


def load_panel():
    d = {k: np.load(f'{P}/{k}.npy') for k in
         ['X_daily', 'Y_3d', 'macro_raw', 'dow', 'ts_days', 'sector_ids']}
    d['meta'] = json.load(open(f'{P}/meta.json'))
    return d


class DailyDataset(Dataset):
    """One item = one decision day -> all N tickers (last L daily bars each)."""
    def __init__(self, d, day_idx, macro_norm, y_clip=None):
        self.X, self.Y, self.dow = d['X_daily'], d['Y_3d'], d['dow']
        self.macro = macro_norm
        self.t_idx = day_idx
        self.y_clip = y_clip

    def __len__(self):
        return len(self.t_idx)

    def __getitem__(self, i):
        t = int(self.t_idx[i])
        x = np.nan_to_num(self.X[t - L + 1:t + 1])              # (L,N,F)
        x = np.transpose(x, (1, 0, 2))                          # (N,L,F)
        dow = self.dow[t - L + 1:t + 1].astype(np.int64)       # (L,)
        macro = np.nan_to_num(self.macro[t]).astype(np.float32) # (M,)
        y = self.Y[t]                                          # (N,) raw 3d return
        present = np.isfinite(self.X[t, :, 0])
        valid = present & np.isfinite(y)
        yt = np.nan_to_num(y).astype(np.float32)
        if self.y_clip is not None:                            # winsorize TRAIN target only
            yt = np.clip(yt, self.y_clip[0], self.y_clip[1])
        return (x.astype(np.float32), dow, macro, yt,
                present.astype(np.bool_), valid.astype(np.bool_))


def collate(batch):
    x, dow, macro, y, present, valid = zip(*batch)
    f = lambda a: torch.from_numpy(np.stack(a))
    return f(x), f(dow), f(macro), f(y), f(present), f(valid)


def valid_days(d, day_set):
    """day indices in day_set with a full L-window and finite labels/macro."""
    T = d['X_daily'].shape[0]
    finite = np.isfinite(d['Y_3d']).sum(1) > 0
    macro_ok = np.isfinite(d['macro_raw']).all(1)
    out = [t for t in day_set if t >= L - 1 and finite[t] and macro_ok[t]]
    return np.array(sorted(out), dtype=int)


@torch.no_grad()
def eval_keepnet(model, loader, device, sector_ids, side_sign, keep_rate):
    """Val proxy: hard-keep top keep_rate by gate score per day, report kept net@10 (side dir)."""
    model.eval(); kept = []
    for x, dow, macro, y, present, valid in loader:
        x, dow, macro, y = x.to(device), dow.to(device), macro.to(device), y.to(device)
        present, valid = present.to(device), valid.to(device)
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            score = model(x, dow, macro, sector_ids, ~present).float()
        for b in range(score.shape[0]):
            m = valid[b]
            nv = int(m.sum())
            if nv < 5:
                continue
            sc = score[b][m].cpu().numpy()
            r = (side_sign * y[b][m]).cpu().numpy()
            k = max(1, int(round(keep_rate * nv)))
            keep = np.argsort(-sc)[:k]
            kept.append(r[keep].mean() - COST_BPS / 1e4)
    return float(np.mean(kept) * 1e4) if kept else -1e9


@torch.no_grad()
def predict_days(model, d, days, macro_norm, device, sector_ids):
    """Per-(day,ticker) gate score on `days` for all present tickers -> dict t -> (N,) score."""
    model.eval()
    ds = DailyDataset(d, days, macro_norm)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    out = np.full((d['X_daily'].shape[0], d['X_daily'].shape[1]), np.nan, dtype=np.float32)
    bi = 0
    for x, dow, macro, y, present, valid in dl:
        x, dow, macro = x.to(device), dow.to(device), macro.to(device)
        present = present.to(device)
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            score = model(x, dow, macro, sector_ids, ~present).float().cpu().numpy()
        for j in range(score.shape[0]):
            t = int(days[bi + j])
            pres = np.isfinite(d['X_daily'][t, :, 0])
            out[t, pres] = score[j, pres]
        bi += score.shape[0]
    return out


def run_fold(d, side, train_days, val_days, test_days, device, sector_ids, args):
    side_sign = 1.0 if side == 'long' else -1.0
    # train-only macro normalization
    mtr = d['macro_raw'][train_days]
    mu, sd = np.nanmean(mtr, 0), np.nanstd(mtr, 0) + 1e-6
    macro_norm = (d['macro_raw'] - mu) / sd
    # train-only label winsorization (1/99 pct of finite train labels)
    ytr = d['Y_3d'][train_days]; ytr = ytr[np.isfinite(ytr)]
    y_clip = (float(np.percentile(ytr, 1)), float(np.percentile(ytr, 99)))

    meta = d['meta']
    model = DailyCSTransformer(meta['n_stock_feats'], meta['n_macro'], meta['n_sectors'],
                               n_dow=meta['n_dow'], d_model=args.d_model, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    cost = COST_BPS / 1e4

    dl_tr = DataLoader(DailyDataset(d, train_days, macro_norm, y_clip=y_clip),
                       batch_size=args.batch, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(DailyDataset(d, val_days, macro_norm),
                       batch_size=16, shuffle=False, collate_fn=collate)

    best, best_state, bad, patience = -1e9, None, 0, 5
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for x, dow, macro, y, present, valid in dl_tr:
            x, dow, macro, y = x.to(device), dow.to(device), macro.to(device), y.to(device)
            present, valid = present.to(device), valid.to(device)
            opt.zero_grad()
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                score = model(x, dow, macro, sector_ids, ~present)
                loss = gate_loss(score, side_sign * y, valid, cost, KEEP_RATE, args.gate_lambda)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        vk = eval_keepnet(model, dl_va, device, sector_ids, side_sign, KEEP_RATE)
        if vk > best:
            best, best_state, bad = vk, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep == 0 or (ep + 1) % 5 == 0 or bad >= patience:
            print(f"    {side} ep{ep+1}/{args.epochs} loss={tot/max(nb,1):.3f} "
                  f"val_keepnet@10={vk:+.2f}bps ({time.time()-t0:.0f}s)")
        if bad >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    print(f"    {side} best val_keepnet@10={best:+.2f}bps")
    return predict_days(model, d, test_days, macro_norm, device, sector_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d_model', type=int, default=48)
    ap.add_argument('--dropout', type=float, default=0.4)
    ap.add_argument('--gate_lambda', type=float, default=100.0)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    d = load_panel()
    sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)
    nparam = sum(p.numel() for p in DailyCSTransformer(
        d['meta']['n_stock_feats'], d['meta']['n_macro'], d['meta']['n_sectors'],
        n_dow=d['meta']['n_dow'], d_model=args.d_model, dropout=args.dropout).parameters())
    print(f"model params: {nparam:,}  (small by design -- daily data is sample-starved)")

    # month-based folds identical to v2. Build 'YYYY-MM' labels with numpy (NOT pandas:
    # vectorized pandas datetime ops after the torch import segfault on Windows / OpenMP clash).
    ym = d['ts_days'].astype('datetime64[ns]').astype('datetime64[M]').astype(str)
    months = sorted(set(ym.tolist()))
    folds = []
    for k in range(1, 5):
        te_end = len(months) - (4 - k) * 6; te_start = te_end - 6; va_start = te_start - 6
        folds.append((months[:va_start], months[va_start:te_start], months[te_start:te_end]))

    gate = {'long': np.full((d['X_daily'].shape[0], d['X_daily'].shape[1]), np.nan, dtype=np.float32),
            'short': np.full((d['X_daily'].shape[0], d['X_daily'].shape[1]), np.nan, dtype=np.float32)}
    for fi, (tr_m, va_m, te_m) in enumerate(folds, 1):
        # purge EMBARGO_D trading days off the end of train (labels overlap the val boundary)
        tr_days = valid_days(d, np.where(np.isin(ym, tr_m))[0])[:-EMBARGO_D]
        va_days = valid_days(d, np.where(np.isin(ym, va_m))[0])
        te_days = valid_days(d, np.where(np.isin(ym, te_m))[0])
        assert max(ym[tr_days]) < min(ym[te_days]) and max(ym[va_days]) < min(ym[te_days])
        print(f"\n--- FOLD {fi}  train={len(tr_days)}d ({tr_m[0]}..{tr_m[-1]}) "
              f"val={len(va_days)}d test={len(te_days)}d ({te_m[0]}..{te_m[-1]}) ---")
        for side in ('long', 'short'):
            sc = run_fold(d, side, tr_days, va_days, te_days, device, sector_ids, args)
            m = np.isfinite(sc)
            gate[side][m] = sc[m]

    for side in ('long', 'short'):
        np.save(f'{P}/daily_gate_{side}.npy', gate[side])
        print(f"saved daily_gate_{side}.npy  scored cells={int(np.isfinite(gate[side]).sum()):,}")


if __name__ == '__main__':
    main()
