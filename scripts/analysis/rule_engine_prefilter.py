"""
Rule-based ticker EXCLUSION engine applied BEFORE ranking, measured for uplift
on the current Tier-A top-1 book at 6 bps (2026-07-05 signal-funnel setup).

For each Tier-A (anchor, side) slot the engine removes rule-flagged tickers from
the pool, the live v20 ranker then takes its top-1 from the filtered pool.
Uplift is measured per book against (a) the unfiltered baseline and (b) a
RANDOM-exclusion control that removes the same NUMBER of names — the guard the
2026-07-05 exclusion research proved essential (rule ~= random => deleveraging,
not alpha). Paired t-stats on per-book deltas; H1/H2 date-half split.

Prior context (Conv-2026-07-05-Short-Long-Exclusion-Rule-Engine): on top-5 books
@10bps every intuitive rule was wrong-signed and nothing was learnable. This
harness re-measures on TOP-1 @6bps where a single bad ticker dominates the book.

EXPLORATORY RESEARCH ONLY (scripts/analysis/ has no verdict authority; no
Gauntlet run consumed; production state untouched). Panel features are
cross-sectionally z-scored per hour — rule thresholds are in z-units.

Usage:  python scripts/analysis/rule_engine_prefilter.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

COST_BPS = 6.0
CAP = 0.15
OOS_FRAC = 0.30
SEED = 42

# Tier-A slate (Conv-2026-07-05 Part 5): the slots this engine must improve.
TIER_A = [("10:30", "short"), ("11:00", "short"), ("11:15", "short"),
          ("11:30", "short"), ("13:45", "short"), ("10:30", "long"), ("14:15", "long")]

# ---- the rule book (exclusion masks; True = ticker removed from that side's pool) ----
RULES = {
    "short": {
        "up_thrust (prior: hurts)":   lambda d: (d["Return"] > 1.0) & (d["Volume_Zscore"] > 1.0),
        "fade_guard 52wH+RVOL (live)": lambda d: (d["Dist_52W_High"] > 1.0) & (d["RVOL"] > 1.5),
        "illiquid DV<-1":             lambda d: d["Dollar_Volume"] < -1.0,
        "vol_extreme HLR>2.5":        lambda d: d["HL_Range"] > 2.5,
        "risk_gate (illiq|vol)":      lambda d: (d["Dollar_Volume"] < -1.0) | (d["HL_Range"] > 2.5),
        # INVERTED keep-list (probe4): keep only up-momentum names, exclude the rest
        "INV keep-mom (excl MOM<=.5)": lambda d: ~(d["MOM_12_pct"] > 0.5),
    },
    "long": {
        "down_thrust (prior: hurts)": lambda d: (d["Return"] < -1.0) & (d["Volume_Zscore"] > 1.0),
        # INVERTED keep-list (probe4): keep only oversold/dip names, exclude the rest
        "INV keep-oversold":          lambda d: ~((d["MOM_12_pct"] < -0.5) | (d["RSI_14"] < -0.5)),
        "illiquid DV<-1":             lambda d: d["Dollar_Volume"] < -1.0,
        "vol_extreme HLR>2.5":        lambda d: d["HL_Range"] > 2.5,
        "risk_gate (illiq|vol)":      lambda d: (d["Dollar_Volume"] < -1.0) | (d["HL_Range"] > 2.5),
    },
}


def load_scored_panel():
    mdir = "models/research/v20_rolling_1h"
    feat = json.load(open(os.path.join(mdir, "metadata.json")))["features"]
    bl = xgb.Booster(); bl.load_model(os.path.join(mdir, "xgb_long_model.json"))
    bs = xgb.Booster(); bs.load_model(os.path.join(mdir, "xgb_short_model.json"))
    anchors = sorted({a for a, _ in TIER_A})
    cols = list(dict.fromkeys(feat + ["DateTime", "Ticker", "Next_Hour_Return"]))
    df = pd.read_parquet("data/research/v20_rolling_1h/panel.parquet", columns=cols)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hod"] = df["DateTime"].dt.strftime("%H:%M")
    cut = df["DateTime"].quantile(1 - OOS_FRAC)
    df = df[(df["DateTime"] >= cut) & (df["hod"].isin(anchors))]
    df = df.dropna(subset=["Next_Hour_Return"]).copy()
    df["nhr"] = df["Next_Hour_Return"].clip(-CAP, CAP)
    X = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    dm = xgb.DMatrix(X, feature_names=feat)
    df["rl"] = bl.predict(dm)
    df["rs"] = bs.predict(dm)
    dates = np.sort(df["DateTime"].dt.date.unique())
    df["half"] = np.where(df["DateTime"].dt.date < dates[len(dates) // 2], "H1", "H2")
    return df


def top1_net(g, side, excl=None):
    """net@6 of the top-1 pick on `side` after removing excl-masked rows."""
    gg = g if excl is None else g[~excl]
    if gg.empty:
        return np.nan
    col = "rl" if side == "long" else "rs"
    r = gg.loc[gg[col].idxmax(), "nhr"] * 1e4
    return (r if side == "long" else -r) - COST_BPS


def evaluate(df):
    rng = np.random.default_rng(SEED)
    print(f"pseudo-OOS {df.DateTime.min().date()} -> {df.DateTime.max().date()} | rows {len(df):,}")
    print(f"Tier-A slots: {TIER_A}\n")
    hdr = f"{'rule':30s} {'base':>7s} {'rule':>7s} {'drule':>7s} {'drand':>7s} {'alpha':>7s} {'t(d)':>6s} {'chg%':>5s} {'aH1':>6s} {'aH2':>6s}"
    for side in ("short", "long"):
        slots = [a for a, s in TIER_A if s == side]
        books = [(ts, g) for ts, g in df[df.hod.isin(slots)].groupby("DateTime")]
        base = np.array([top1_net(g, side) for _, g in books])
        halves = np.array([g["half"].iat[0] for _, g in books])
        print(f"===== {side.upper()} slots {slots} | books {len(books)} | baseline net@6 {np.nanmean(base):+.2f} =====")
        print(hdr)
        for name, fn in RULES[side].items():
            d_rule, d_rand = np.zeros(len(books)), np.zeros(len(books))
            changed = 0
            for i, (_, g) in enumerate(books):
                try:
                    m = fn(g).values.astype(bool)
                except KeyError:
                    m = np.zeros(len(g), bool)
                if not m.any():
                    continue
                n1 = top1_net(g, side, m)
                if np.isnan(n1):
                    continue
                d_rule[i] = n1 - base[i]
                if d_rule[i] != 0.0:
                    changed += 1
                rm = np.zeros(len(g), bool)
                rm[rng.choice(len(g), size=int(m.sum()), replace=False)] = True
                nr = top1_net(g, side, rm)
                d_rand[i] = (nr - base[i]) if not np.isnan(nr) else 0.0
            alpha = d_rule - d_rand
            t = alpha.mean() / (alpha.std(ddof=1) / np.sqrt(len(alpha))) if alpha.std(ddof=1) > 0 else 0.0
            a1, a2 = alpha[halves == "H1"].mean(), alpha[halves == "H2"].mean()
            print(f"{name:30s} {np.nanmean(base):+7.2f} {np.nanmean(base + d_rule):+7.2f} "
                  f"{d_rule.mean():+7.2f} {d_rand.mean():+7.2f} {alpha.mean():+7.2f} {t:+6.2f} "
                  f"{100 * changed / len(books):4.0f}% {a1:+6.2f} {a2:+6.2f}")
        print()


if __name__ == "__main__":
    evaluate(load_scored_panel())
