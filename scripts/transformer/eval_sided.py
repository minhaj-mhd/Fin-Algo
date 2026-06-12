"""
Sided-transformer evaluation (Phase 3 of the pre-registered sided-ranking experiment).

Loads the two side-specialist rankers (dualres_short.pt = ranks by -y, dualres_long.pt = +y)
and, on the SAME chronological OOS test split as train.py, reports per side:
  * cross-sectional rank-IC (mean per-timestamp Spearman of score vs side target) + t-stat
  * Top-1/3/5 NET bps @6/10/20 with t-stat over the per-timestamp series
  * RAW vs NET win-rate, and the median(net-gross)==-cost cost-sign sanity check
Baseline: the single-head BCE transformer (artifacts/dualres_transformer.pt), scored in both
directional roles (long = top by P(up), short = bottom by P(up)) on the identical split.

Exploratory only — NO Gauntlet verdict. Survivorship caveat applies (see pre-registration).
"""
import os, sys, json
import numpy as np
import torch
from scipy.stats import spearmanr, ttest_1samp

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
from torch.utils.data import DataLoader
from scripts.transformer.train import (load_panel, DecisionDataset, collate,
                                       valid_decision_timestamps, EMBARGO, KS, COSTS_BPS)
from scripts.transformer.model import DualResCSTransformer

COSTS_BPS = [6.0, 10.0, 20.0]   # eval reports an extra 20bps stress column
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(d):
    F = d['meta']['n_features']; M = d['macro'].shape[1]; n_sec = len(d['meta']['sectors'])
    return DualResCSTransformer(F, M, n_sec, n_slots_1h=d['meta']['n_slots_1h'],
                                n_slots_15m=d['meta']['n_slots_15m'], d_model=64).to(device)


@torch.no_grad()
def collect_scores(model, loader, sector_ids):
    """Return list of (score[Nv], y[Nv]) per valid test timestamp."""
    model.eval(); rows = []
    for batch in loader:
        x1, x15, s1, s15, macro, ybin, y, present, valid = [b.to(device) for b in batch]
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(x1, x15, s1, s15, macro, sector_ids, ~present)
        sc = logit.float()
        for b in range(sc.shape[0]):
            m = valid[b]
            if m.sum() < max(KS) + 1:
                continue
            rows.append((sc[b][m].cpu().numpy(), y[b][m].cpu().numpy()))
    return rows


def sided_metrics(rows, side_sign, label):
    """rows: list of (score[Nv], y[Nv]). side_sign=+1 long, -1 short. Top-K by score, PnL=side_sign*y."""
    rhos, topk = [], {k: [] for k in KS}
    raw_wr, net_wr = {k: [] for k in KS}, {k: {c: [] for c in COSTS_BPS} for k in KS}
    for sc, y in rows:
        r = side_sign * y
        if np.std(sc) == 0:
            continue
        rho = spearmanr(sc, r).correlation
        if np.isfinite(rho):
            rhos.append(rho)
        order = np.argsort(-sc)
        for k in KS:
            picks = r[order[:k]]
            topk[k].append(picks.mean())
            raw_wr[k].append((picks > 0).mean())
            for c in COSTS_BPS:
                net_wr[k][c].append((picks.mean() - c / 1e4) > 0)
    out = {'label': label, 'side': 'long' if side_sign > 0 else 'short',
           'n_ts': len(rhos), 'rho': float(np.mean(rhos)),
           'rho_t': float(ttest_1samp(rhos, 0).statistic) if len(rhos) > 1 else 0.0}
    for k in KS:
        g = np.array(topk[k])
        out[f'K{k}'] = {'gross_bps': round(g.mean() * 1e4, 2), 'raw_wr': round(np.mean(raw_wr[k]), 3)}
        for c in COSTS_BPS:
            net = g - c / 1e4
            t = float(ttest_1samp(net, 0).statistic) if len(net) > 1 and net.std() > 0 else 0.0
            chk = float(np.median(net - g) * 1e4)   # must equal -c
            out[f'K{k}'][f'net_{int(c)}'] = {'bps': round(net.mean() * 1e4, 2), 't': round(t, 2),
                                             'net_wr': round(np.mean(net_wr[k][c]), 3),
                                             'cost_chk': round(chk, 2)}
    return out


def print_block(m):
    print(f"\n[{m['label']}] side={m['side']}  n_ts={m['n_ts']}  "
          f"rank-IC={m['rho']:+.4f} (t={m['rho_t']:+.2f})")
    for k in KS:
        b = m[f'K{k}']
        cells = "  ".join(f"@{c}: {b[f'net_{c}']['bps']:+.2f}bps t={b[f'net_{c}']['t']:+.2f}"
                          for c in (6, 10, 20))
        print(f"  K{k}: gross {b['gross_bps']:+.2f}  rawWR {b['raw_wr']:.0%}  | {cells}  "
              f"cost_chk={b['net_10']['cost_chk']:+.2f}")


def main():
    d = load_panel()
    sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)
    ts = valid_decision_timestamps(d)
    n = len(ts); i_va = int(n * 0.85)
    te = ts[i_va + EMBARGO:]
    import pandas as pd
    print(f"OOS test timestamps: {len(te)}  span "
          f"{pd.Timestamp(int(d['ts_1h'][te[0]]))} .. {pd.Timestamp(int(d['ts_1h'][te[-1]]))}")
    dl = DataLoader(DecisionDataset(d, te), batch_size=16, shuffle=False, collate_fn=collate)

    results = {}
    # ── sided specialists ──
    for tgt, sign, ckpt in [('short', -1.0, 'artifacts/dualres_short.pt'),
                            ('long', 1.0, 'artifacts/dualres_long.pt')]:
        if not os.path.exists(ckpt):
            print(f"[skip] {ckpt} not found"); continue
        m = build_model(d); m.load_state_dict(torch.load(ckpt, map_location=device))
        rows = collect_scores(m, dl, sector_ids)
        met = sided_metrics(rows, sign, f'dualres_{tgt} (listwise)')
        results[f'dualres_{tgt}'] = met; print_block(met)

    # ── single-head BCE baseline, both directional roles ──
    base = 'artifacts/dualres_transformer.pt'
    if os.path.exists(base):
        m = build_model(d); m.load_state_dict(torch.load(base, map_location=device))
        rows = collect_scores(m, dl, sector_ids)
        for tgt, sign in [('short', -1.0), ('long', 1.0)]:
            met = sided_metrics(rows, sign, f'BCE single-head (as {tgt})')
            results[f'bce_{tgt}'] = met; print_block(met)

    os.makedirs('artifacts', exist_ok=True)
    json.dump(results, open('artifacts/dualres_sided_eval.json', 'w'), indent=2, default=float)
    print("\nsaved -> artifacts/dualres_sided_eval.json")


if __name__ == '__main__':
    main()
