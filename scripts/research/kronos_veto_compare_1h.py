"""Compare zero-shot base vs LoRA-adapted Kronos as a 1h veto on the dualtf book.

Reads the two 1h scores files produced by kronos_veto_score_1h.py
  data/research/kronos_veto/scores_1h_base.csv
  data/research/kronos_veto/scores_1h_lora.csv
merges each with the trade panel, and reports coverage-matched delta-net (KEPT - ALL),
day-clustered bootstrap t, and a within-dt1 score-shuffle neg-control -- the same
pre-registered decision cells as kronos_veto_eval.py -- side by side, focusing on the
POST-cutoff (>=2025-09-09) honest window.

Verdict question: does the LoRA adapter's p_up veto beat zero-shot base's on net bps/trade?

Exploratory only - NO Gauntlet verdict authority.

Usage: python scripts/research/kronos_veto_compare_1h.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL = os.path.join(ROOT, "data", "research", "entry_exit", "dualtf_trade_panel.csv")
SDIR = os.path.join(ROOT, "data", "research", "kronos_veto")
OUT_JSON = os.path.join(ROOT, "artifacts", "kronos_veto_1h_compare.json")

CUTOFF = pd.Timestamp("2025-09-09")
COSTS = {"10bps": 0.0010, "6bps": 0.0006}
COVERAGES = {"keep70": 0.70, "keep50": 0.50}
N_BOOT, N_PERM, SEED = 2000, 200, 7


def day_boot(delta_fn, days_arr, rng, n_boot=N_BOOT):
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
    net = (df["gross"] - cost).values
    score = df["aligned"].values
    thr = np.quantile(score, 1.0 - cov)
    kept = score >= thr
    if kept.sum() == 0 or kept.all():
        return None
    days = df["day"].values

    def stat(idx):
        k = kept[idx]
        return net[idx][k].mean() - net[idx].mean() if k.any() else 0.0

    bm, bt, blo, bhi = day_boot(stat, days, rng)
    return dict(n_all=int(len(net)), n_kept=int(kept.sum()),
                net_all=round(net.mean() * 1e4, 2), net_kept=round(net[kept].mean() * 1e4, 2),
                delta=round((net[kept].mean() - net.mean()) * 1e4, 2), boot_t=round(bt, 2),
                ci_lo=round(blo * 1e4, 2), ci_hi=round(bhi * 1e4, 2))


def neg_control(df, cov, cost, rng, n_perm=N_PERM):
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


def eval_tag(tag, panel, rng, lines):
    fp = os.path.join(SDIR, f"scores_1h_{tag}.csv")
    if not os.path.exists(fp):
        lines.append(f"[{tag}] scores file missing: {fp}")
        return {}
    scores = pd.read_csv(fp, parse_dates=["dt1"])
    df = panel.merge(scores[["ticker", "dt1", "p_up", "mean_ret"]], on=["ticker", "dt1"], how="inner")
    df["sgn"] = np.where(df["dir"] == "long", 1.0, -1.0)
    df["gross"] = df["nhr"] * df["sgn"]
    df["aligned"] = np.where(df["dir"] == "long", df["p_up"], 1.0 - df["p_up"])
    df["day"] = df["dt1"].dt.date.astype(str)
    for cname, c in COSTS.items():
        assert abs(np.median((df["gross"] - c) - df["gross"]) + c) < 1e-12, "cost accounting broken"

    post = df[df["dt1"] >= CUTOFF]
    res = {"merged": int(len(df)), "post_n": int(len(post)),
           "p_up_mean": round(float(df["p_up"].mean()), 4), "windows": {}}
    lines.append(f"\n===== {tag.upper()}  (merged {len(df)}, post-cutoff {len(post)}, "
                 f"p_up mean {df['p_up'].mean():.3f}) =====")
    for side in ["long", "short"]:
        sdf = post[post["dir"] == side]
        res["windows"][side] = {}
        for cov_name, cov in COVERAGES.items():
            for cost_name, cost in COSTS.items():
                r = cell(sdf, cov, cost, rng)
                res["windows"][side][f"{cov_name}_{cost_name}"] = r
                if r:
                    lines.append(f"  {side:5s} {cov_name} @{cost_name:5s}: ALL net={r['net_all']:+7.2f} "
                                 f"| KEPT net={r['net_kept']:+7.2f} | dNET={r['delta']:+6.2f}bps "
                                 f"t={r['boot_t']:+5.2f} CI[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]")
        nc = neg_control(sdf, COVERAGES["keep70"], COSTS["10bps"], rng)
        res["windows"][side]["neg_control_keep70_10bps"] = nc
        lines.append(f"  {side:5s} NEG-CTRL keep70@10bps: null dNET {nc['null_mean']:+.2f} "
                     f"[{nc['null_lo']:+.2f},{nc['null_hi']:+.2f}]")
        if len(sdf) > 10:
            rho = sdf["aligned"].corr(sdf["gross"], method="spearman")
            lines.append(f"  [diag] {side} spearman(aligned,gross) = {rho:+.4f}")
    return res


def main():
    rng = np.random.default_rng(SEED)
    panel = pd.read_csv(PANEL, usecols=["dt1", "ticker", "dir", "nhr"], parse_dates=["dt1"])
    lines = [f"Kronos 1h veto: BASE vs LoRA on dualtf book - {pd.Timestamp.now():%Y-%m-%d %H:%M}",
             "Post-cutoff (>=2025-09-09) honest window. Exploratory - NO Gauntlet authority."]
    out = {"cutoff": str(CUTOFF.date()), "tags": {}}
    for tag in ["base", "lora"]:
        out["tags"][tag] = eval_tag(tag, panel, rng, lines)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
