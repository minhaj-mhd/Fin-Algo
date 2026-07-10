"""Evaluate zero-shot Kronos-base veto scores on the 1h ranker book (net returns).

Research question: does vetoing the 1h book's top-3 picks by Kronos-base's
sampled next-hour forecast direction improve net bps/trade?

Inputs:  data/research/kronos_veto/scores.csv   (from kronos_veto_score.py)
         data/research/entry_exit/dualtf_trade_panel.csv (13,020 OOS WF trades)

Pre-registered decision cell (logged in vault BEFORE scoring):
  keep-70% per side by side-aligned score (long: p_up, short: 1-p_up),
  delta-net(KEPT - ALL) @10bps, day-clustered bootstrap t, POST-cutoff window.
  PASS iff delta-net>0 with t>2 on >=1 side AND within-dt1 score-shuffle
  neg-control shows no comparable uplift AND sign holds at keep-50%.
  Leakage: Kronos HF weights uploaded 2025-09-09; trades before that date may
  sit inside its pretraining corpus -> POST window is the honest primary read.

Conventions (match wf_rank_trajectory.py / gate_veto_v10.py):
  gross = nhr * (+1 long / -1 short); net = gross - COST, one deduction/trade.

Exploratory only - NO Gauntlet verdict authority.

Usage: python scripts/research/kronos_veto_eval.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL = os.path.join(ROOT, "data", "research", "entry_exit", "dualtf_trade_panel.csv")
SCORES = os.path.join(ROOT, "data", "research", "kronos_veto", "scores.csv")
OUT_TXT = os.path.join(ROOT, "data", "research", "kronos_veto", "results_2026-07-03.txt")
OUT_JSON = os.path.join(ROOT, "artifacts", "kronos_veto.json")

CUTOFF = pd.Timestamp("2025-09-09")   # HF weight upload date (pretraining cannot extend past it)
COSTS = {"10bps": 0.0010, "6bps": 0.0006}
COVERAGES = {"keep70": 0.70, "keep50": 0.50}
N_BOOT = 2000
N_PERM = 200
SEED = 7


def day_boot(delta_fn, days_arr, rng, n_boot=N_BOOT):
    """Day-clustered bootstrap of a statistic. delta_fn(day_index_arrays) -> float."""
    uniq = np.unique(days_arr)
    idx_by_day = {d: np.where(days_arr == d)[0] for d in uniq}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in take])
        vals[b] = delta_fn(idx)
    m, s = float(np.mean(vals)), float(np.std(vals))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return m, (m / s if s > 0 else 0.0), float(lo), float(hi)


def cell(df, cov, cost, rng):
    """delta-net(KEPT-ALL) in bps for one side-subset at one coverage/cost."""
    net = (df["gross"] - cost).values
    score = df["aligned"].values
    thr = np.quantile(score, 1.0 - cov)
    kept = score >= thr
    if kept.sum() == 0 or kept.all():
        return None
    delta = net[kept].mean() - net.mean()
    days = df["day"].values

    def stat(idx):
        k = kept[idx]
        return net[idx][k].mean() - net[idx].mean() if k.any() else 0.0

    bm, bt, blo, bhi = day_boot(stat, days, rng)
    return dict(n_all=int(len(net)), n_kept=int(kept.sum()),
                net_all=round(net.mean() * 1e4, 2), net_kept=round(net[kept].mean() * 1e4, 2),
                delta=round(delta * 1e4, 2), boot_t=round(bt, 2),
                ci_lo=round(blo * 1e4, 2), ci_hi=round(bhi * 1e4, 2))


def neg_control(df, cov, cost, rng, n_perm=N_PERM):
    """Permute p_up within each dt1 cross-section; distribution of delta-net."""
    net = (df["gross"] - cost).values
    deltas = np.empty(n_perm)
    dt1_codes = pd.factorize(df["dt1"])[0]
    p_up = df["p_up"].values
    is_long = (df["dir"] == "long").values
    order = np.argsort(dt1_codes, kind="stable")
    bounds = np.searchsorted(dt1_codes[order], np.arange(dt1_codes.max() + 2))
    for p in range(n_perm):
        perm = p_up.copy()
        for g in range(len(bounds) - 1):
            sl = order[bounds[g]:bounds[g + 1]]
            if len(sl) > 1:
                perm[sl] = rng.permutation(perm[sl])
        aligned = np.where(is_long, perm, 1.0 - perm)
        thr = np.quantile(aligned, 1.0 - cov)
        kept = aligned >= thr
        deltas[p] = net[kept].mean() - net.mean() if 0 < kept.sum() < len(net) else 0.0
    return dict(null_mean=round(deltas.mean() * 1e4, 2),
                null_lo=round(np.percentile(deltas, 2.5) * 1e4, 2),
                null_hi=round(np.percentile(deltas, 97.5) * 1e4, 2))


def main():
    rng = np.random.default_rng(SEED)
    panel = pd.read_csv(PANEL, usecols=["dt1", "ticker", "dir", "nhr"], parse_dates=["dt1"])
    scores = pd.read_csv(SCORES, parse_dates=["dt1"])
    df = panel.merge(scores[["ticker", "dt1", "p_up", "mean_ret"]], on=["ticker", "dt1"], how="inner")
    miss = len(panel) - len(df)

    df["sgn"] = np.where(df["dir"] == "long", 1.0, -1.0)
    df["gross"] = df["nhr"] * df["sgn"]
    df["aligned"] = np.where(df["dir"] == "long", df["p_up"], 1.0 - df["p_up"])
    df["day"] = df["dt1"].dt.date.astype(str)

    # cost-accounting sanity: net - gross must equal -cost for every trade
    for cname, c in COSTS.items():
        med = np.median((df["gross"] - c) - df["gross"])
        assert abs(med + c) < 1e-12, f"cost accounting broken for {cname}"

    windows = {"full": df,
               "pre_cutoff": df[df["dt1"] < CUTOFF],
               "post_cutoff": df[df["dt1"] >= CUTOFF]}

    res = {"spec": dict(model="NeoQuasar/Kronos-base", tokenizer="NeoQuasar/Kronos-Tokenizer-base",
                        lookback=480, pred_len=4, T=1.0, top_p=0.9, R=30,
                        cutoff=str(CUTOFF.date()), n_boot=N_BOOT, n_perm=N_PERM, seed=SEED,
                        merged_trades=int(len(df)), unmatched_panel_rows=int(miss)),
           "windows": {}}

    lines = [f"Kronos-base zero-shot veto on 1h book - {pd.Timestamp.now():%Y-%m-%d %H:%M}",
             "Exploratory only - NO Gauntlet verdict authority.",
             f"merged trades: {len(df)} (unmatched panel rows: {miss})",
             f"p_up: mean={df['p_up'].mean():.3f} std={df['p_up'].std():.3f} "
             f"pct(p_up in {{0,1}})={(df['p_up'].isin([0.0,1.0])).mean()*100:.1f}%", ""]

    for wname, wdf in windows.items():
        res["windows"][wname] = {}
        lines.append(f"===== window: {wname}  ({wdf['dt1'].min():%Y-%m-%d} .. {wdf['dt1'].max():%Y-%m-%d}, "
                     f"n={len(wdf)}) =====")
        for side in ["long", "short"]:
            sdf = wdf[wdf["dir"] == side]
            res["windows"][wname][side] = {}
            for cov_name, cov in COVERAGES.items():
                for cost_name, cost in COSTS.items():
                    r = cell(sdf, cov, cost, rng)
                    res["windows"][wname][side][f"{cov_name}_{cost_name}"] = r
                    if r:
                        lines.append(f"  {side:5s} {cov_name} @{cost_name:5s}: ALL n={r['n_all']:5d} "
                                     f"net={r['net_all']:+7.2f} | KEPT n={r['n_kept']:5d} net={r['net_kept']:+7.2f} "
                                     f"| dNET={r['delta']:+6.2f}bps t={r['boot_t']:+5.2f} "
                                     f"CI[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}]")
            # neg-control on the primary coverage/cost only
            nc = neg_control(sdf, COVERAGES["keep70"], COSTS["10bps"], rng)
            res["windows"][wname][side]["neg_control_keep70_10bps"] = nc
            lines.append(f"  {side:5s} NEG-CONTROL keep70@10bps: null dNET mean={nc['null_mean']:+.2f} "
                         f"95%[{nc['null_lo']:+.2f},{nc['null_hi']:+.2f}]")
        # diagnostic: rank-IC of aligned score vs signed gross (not a decision cell)
        for side in ["long", "short"]:
            sdf = wdf[wdf["dir"] == side]
            if len(sdf) > 10:
                rho = sdf["aligned"].corr(sdf["gross"], method="spearman")
                lines.append(f"  [diag] {side} spearman(aligned, gross) = {rho:+.4f}")
        lines.append("")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\n".join(lines))
    print(f"wrote {OUT_TXT}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
