"""
Diagnose the listwise transformer's val->test rank-IC "collapse" (short: val 0.0222 -> test 0.0066).

Decomposes it into (A) early-stop selection bias + sampling noise vs (B) genuine OOS/regime decay,
by recomputing PER-TIMESTAMP spearman rho on train/val/test for the best-val checkpoint and reporting
mean, SE, t-stat, bootstrap 95% CI, a Welch test of val-vs-test, and rho by test sub-period.

    python scripts/transformer/analyze_rho_collapse.py --target short --ckpt artifacts/dualres_short.pt
"""
import os, sys, argparse
os.environ.setdefault('TRANSFORMER_PANEL', 'data/transformer_panel')   # baseline panel
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, ttest_ind

sys.path.append(os.getcwd())
from scripts.transformer.train import (load_panel, DecisionDataset, collate,
                                       valid_decision_timestamps, EMBARGO, SEED)
from scripts.transformer.model import DualResCSTransformer
from torch.utils.data import DataLoader


@torch.no_grad()
def per_ts_rho(model, d, t_idx, device, sector_ids, side_sign):
    ds = DecisionDataset(d, t_idx)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    model.eval()
    rhos, ts_used = [], []
    base = 0
    for batch in dl:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
        score = logit.float().cpu().numpy()
        yv = y.cpu().numpy(); vv = valid.cpu().numpy()
        for b in range(score.shape[0]):
            m = vv[b]
            if m.sum() < 6:
                base += 1; continue
            sc = score[b][m]; r = side_sign * yv[b][m]
            if np.std(sc) == 0:
                base += 1; continue
            rho = spearmanr(sc, r).correlation
            if np.isfinite(rho):
                rhos.append(rho); ts_used.append(int(t_idx[base]))
            base += 1
    return np.array(rhos), np.array(ts_used)


def describe(name, rhos):
    n = len(rhos); mean = rhos.mean(); sd = rhos.std(ddof=1)
    se = sd / np.sqrt(n); t = mean / se
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(rhos, n, replace=True).mean() for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:5s} n={n:4d}  mean rho={mean:+.4f}  SE={se:.4f}  t={t:+.2f}  "
          f"95%CI=[{lo:+.4f},{hi:+.4f}]  {'SIGNIF' if lo>0 or hi<0 else 'n.s. (CI spans 0)'}")
    return mean, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='short')
    ap.add_argument('--ckpt', default='artifacts/dualres_short.pt')
    args = ap.parse_args()
    side_sign = 1.0 if args.target == 'long' else -1.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = load_panel()
    F = d['meta']['n_features']; M = d['macro'].shape[1]; n_sec = len(d['meta']['sectors'])
    sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)
    ts = valid_decision_timestamps(d); n = len(ts)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    tr = ts[:i_tr]; va = ts[i_tr + EMBARGO:i_va]; te = ts[i_va + EMBARGO:]

    model = DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                                 n_slots_15m=d['meta']['n_slots_15m']).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"loaded {args.ckpt}  panel={os.environ['TRANSFORMER_PANEL']}  target={args.target}")

    print("\nPER-TIMESTAMP rank-IC (rho) by split:")
    r_tr, _ = per_ts_rho(model, d, tr, device, sector_ids, side_sign)
    r_va, _ = per_ts_rho(model, d, va, device, sector_ids, side_sign)
    r_te, ts_te = per_ts_rho(model, d, te, device, sector_ids, side_sign)
    describe('train', r_tr); m_va, se_va = describe('val', r_va); m_te, se_te = describe('test', r_te)

    # is the val->test drop itself significant?
    t_diff, p_diff = ttest_ind(r_va, r_te, equal_var=False)
    print(f"\nval - test gap = {m_va - m_te:+.4f}  Welch t={t_diff:+.2f} p={p_diff:.3f}  "
          f"-> {'SIGNIFICANT drop' if p_diff < 0.05 else 'NOT significant (gap within noise)'}")

    # regime check: rho over test sub-periods
    print("\nTEST rho by sub-period (regime check):")
    dts = pd.to_datetime(d['ts_1h'][ts_te])
    df = pd.DataFrame({'rho': r_te, 'ym': dts.to_period('M').astype(str)})
    halves = np.array_split(np.arange(len(r_te)), 2)
    for i, idx in enumerate(halves):
        print(f"  test half {i+1}: n={len(idx)} mean rho={r_te[idx].mean():+.4f}")
    by_m = df.groupby('ym')['rho'].agg(['mean', 'count'])
    print(by_m.round(4).to_string())


if __name__ == '__main__':
    main()
