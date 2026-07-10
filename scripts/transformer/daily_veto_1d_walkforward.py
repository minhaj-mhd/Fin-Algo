"""
Audit: does the daily gate transformer's veto raise daily_macro_v2's NET edge? (Pre-registered.)

Honest OOS walk-forward eval on the common window (v2's 4 test folds == last ~24 months, 478 days).
For each side and K in {1,3,5}: take v2's genuine-OOS Top-K picks, then KEEP a pick iff its transformer
gate score > 0 (the gate's own causal decision boundary -- no pooled-threshold lookahead) and compare:
    v2-alone net   vs   v2+veto (kept) net   vs   vetoed net,  all @10bps round-trip on 3-day returns.

Pre-registered success: Δnet = kept_net - v2alone_net significantly > 0 (day-clustered bootstrap CI
excludes 0) AND the kept side still clears a TRIGGER analog (pooled net >= 2bps, t >= 2). Otherwise the
veto adds nothing -> dead-end (do NOT tune keep_rate/threshold to pass; pre-registered stop rule).

Skepticism battery (cf. audit.py / feedback_validate_cost_accounting):
  * cost-accounting sanity: median(net - gross) == -cost PER SIDE.
  * RAW win-rate vs NET win-rate per side.
  * negative control: shuffle returns within each day -> v2 edge AND veto uplift must collapse to ~0.
  * fragility: share of total kept-net contributed by the 5 biggest trades.
NO verdict authority -- exploratory; only the Gauntlet grades.
"""
import os, sys, json
import numpy as np

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

P = 'data/daily_transformer_panel'
COST = 10.0 / 1e4
KS = [1, 3, 5]
SIDES = [('long', 'v2_long_score', 'daily_gate_long', +1.0),
         ('short', 'v2_short_score', 'daily_gate_short', -1.0)]
RNG = np.random.default_rng(42)
NB = 3000


def topk_picks(score_row, present, k):
    idx = np.where(present)[0]
    if len(idx) < k + 1:
        return np.array([], dtype=int)
    return idx[np.argsort(-score_row[idx])[:k]]


def gather(Y, v2score, gate, oos_days, side_sign, k, shuffle=False):
    """Per-pick arrays over OOS days: day, side-return r, gate score g."""
    days, rs, gs = [], [], []
    for t in oos_days:
        yt = Y[t].copy()
        present = np.isfinite(v2score[t]) & np.isfinite(yt)
        if shuffle:                                  # negative control: permute returns within day
            pi = np.where(present)[0]
            yt[pi] = RNG.permutation(yt[pi])
        picks = topk_picks(v2score[t], present, k)
        for p in picks:
            days.append(t); rs.append(side_sign * yt[p]); gs.append(gate[t, p])
    return np.array(days), np.array(rs), np.array(gs)


def cluster_boot(days, vals, mask=None):
    """Day-clustered bootstrap mean of vals (optionally over mask subset): (mean, t, lo, hi)."""
    if mask is not None:
        days, vals = days[mask], vals[mask]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, np.nan
    uniq = np.unique(days)
    by = {d: vals[days == d] for d in uniq}
    boots = np.empty(NB)
    for b in range(NB):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        boots[b] = np.concatenate([by[d] for d in samp]).mean()
    m = vals.mean()
    se = boots.std() + 1e-12
    return m, m / se, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def uplift_boot(days, r, keep):
    """Day-clustered bootstrap of Δnet = mean(r[keep]) - mean(r[all]) (cost cancels): (Δ, t, lo, hi)."""
    uniq = np.unique(days)
    idx_by = {d: np.where(days == d)[0] for d in uniq}
    boots = np.empty(NB)
    for b in range(NB):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[d] for d in samp])
        kk = keep[ii]
        boots[b] = (r[ii][kk].mean() if kk.any() else 0.0) - r[ii].mean()
    d0 = (r[keep].mean() if keep.any() else 0.0) - r.mean()
    se = boots.std() + 1e-12
    return d0, d0 / se, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    meta = json.load(open(f'{P}/meta.json'))
    Y = np.load(f'{P}/Y_1d.npy')
    oos_mask = np.load(f'{P}/v2_oos_mask.npy')
    oos_days = np.where(oos_mask)[0]
    ts = np.load(f'{P}/ts_days.npy')
    print("=" * 78)
    print(f"DAILY VETO AUDIT  OOS days={len(oos_days)}  "
          f"{str(ts[oos_days[0]].astype('datetime64[ns]'))[:10]}..{str(ts[oos_days[-1]].astype('datetime64[ns]'))[:10]}"
          f"  cost={COST*1e4:.0f}bps  (1-day returns)")
    print("=" * 78)

    for side, scname, gname, sgn in SIDES:
        v2s = np.load(f'{P}/{scname}.npy')
        gate = np.load(f'{P}/{gname}.npy')
        print(f"\n################  {side.upper()}  ################")
        for k in KS:
            days, r, g = gather(Y, v2s, gate, oos_days, sgn, k)
            if len(r) == 0:
                continue
            keep = g > 0.0                                   # causal veto: gate's own boundary
            gross = r.mean() * 1e4
            v2net, v2t, v2lo, v2hi = cluster_boot(days, (r - COST))
            kpnet, kpt, kplo, kphi = cluster_boot(days, (r - COST), mask=keep)
            vtnet = (r[~keep] - COST).mean() * 1e4 if (~keep).any() else float('nan')
            dnet, dt, dlo, dhi = uplift_boot(days, r, keep)
            cov = keep.mean()
            raw_wr = (r > 0).mean(); net_wr = (r > COST).mean()
            chk = np.median((r - COST) - r) * 1e4            # must equal -cost
            print(f"  K={k}: picks={len(r)} cov(kept)={cov:.0%}")
            print(f"     v2-alone net={v2net*1e4:+.2f}bps (t={v2t:+.2f} CI[{v2lo*1e4:+.1f},{v2hi*1e4:+.1f}]) "
                  f"gross={gross:+.2f}  rawWR={raw_wr:.0%} netWR={net_wr:.0%}  chk(net-gross)={chk:+.2f}")
            print(f"     v2+VETO  net={kpnet*1e4:+.2f}bps (t={kpt:+.2f} CI[{kplo*1e4:+.1f},{kphi*1e4:+.1f}]) "
                  f"| vetoed net={vtnet:+.2f}")
            print(f"     UPLIFT Δnet={dnet*1e4:+.2f}bps (t={dt:+.2f} CI[{dlo*1e4:+.1f},{dhi*1e4:+.1f}])  "
                  f"{'<-- sig>0' if dlo>0 else ''}")
            # fragility on kept
            kr = (r[keep] - COST)
            if len(kr) >= 5:
                top5 = np.sort(np.abs(kr))[-5:].sum()
                print(f"     fragility: top-5 kept trades = {top5/ (np.abs(kr).sum()+1e-12):.0%} of |kept net| mass")
            # negative control
            nd, nr, ng = gather(Y, v2s, gate, oos_days, sgn, k, shuffle=True)
            nkeep = ng > 0.0
            nc_v2 = (nr - COST).mean() * 1e4
            nc_up = ((nr[nkeep].mean() if nkeep.any() else 0.0) - nr.mean()) * 1e4
            print(f"     neg-control (shuffled): v2-alone net={nc_v2:+.2f}bps  uplift={nc_up:+.2f}bps "
                  f"{'[OK ~0]' if abs(nc_up) < 1.5 else '[!! leak?]'}")
    print("\n" + "=" * 78)
    print("Pre-registered read: WIN only if some side shows Δnet CI>0 AND v2+VETO net>=2bps@t>=2.")
    print("Else: veto adds nothing -> dead-end. Do NOT sweep to pass (stop rule).")
    print("=" * 78)


if __name__ == '__main__':
    main()
