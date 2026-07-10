"""
Gate-1 step 3: PURGED EXPANDING-WINDOW WALK-FORWARD.

The single-split Gate-1 result (rankIC 0.013->0.025 with +DYNAMIC) is exactly the
shape that inflated v8's static Spearman before genuine purged WF exposed decay.
This script is the decisive test: train on an expanding history, test on the next
chronological block, purge a 3-day gap (== label horizon), repeat over folds, and
report rank-IC PER FOLD so we can see whether the +DYNAMIC advantage holds or decays.

Variants: BASE, +DYN_GRP (novel exogenous group edges), +DYNAMIC (group+sector).
EXPLORATORY TIER — no Gauntlet authority. ASCII-only output.
Run: python scripts/structural/gate1_walkforward.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(os.getcwd())
from scripts.structural.business_groups import base                       # noqa: E402
from scripts.structural.gate1_eval import (nan_neighbor_mean,             # noqa: E402
                                           per_day_rank_ic, side_winrate,
                                           AGG_FEATS, COST, PURGE, SEED)

D = "data/daily_transformer_panel/"
G = "data/research/graph/"
N_FOLDS = 5
INIT_FRAC = 0.5                      # initial train = first 50% of days


def build_features():
    meta = json.load(open(D + "meta.json"))
    feats = meta["stock_feats"]
    X = np.load(D + "X_daily.npy"); Y = np.load(D + "Y_3d.npy")
    ts = np.load(D + "ts_days.npy")
    T, N, F = X.shape
    tickers = [base(t) for t in meta["tickers"]]
    idx = {t: i for i, t in enumerate(tickers)}
    edges = pd.read_csv(G + "edges.csv")
    A_grp = np.zeros((N, N)); A_sec = np.zeros((N, N))
    for _, e in edges.iterrows():
        i, j = idx[e["src"]], idx[e["dst"]]
        tgt = A_grp if e["type"] == "group" else A_sec
        tgt[i, j] = tgt[j, i] = 1.0
    grp_cols, sec_cols = [], []
    dyn = []
    for fn in AGG_FEATS:
        f = feats.index(fn)
        grp_cols.append(len(dyn)); dyn.append(nan_neighbor_mean(X[:, :, f], A_grp))
        sec_cols.append(len(dyn)); dyn.append(nan_neighbor_mean(X[:, :, f], A_sec))
    DYN = np.stack(dyn, axis=-1).reshape(T * N, len(dyn))
    base_flat = X.reshape(T * N, F)
    Y_flat = Y.reshape(T * N)
    row_ts = np.repeat(ts, N)
    valid = ~np.isnan(Y_flat) & ~np.isnan(base_flat).any(1)
    variants = {
        "BASE":     base_flat,
        "+DYN_GRP": np.hstack([base_flat, DYN[:, grp_cols]]),
        "+DYNAMIC": np.hstack([base_flat, DYN]),
    }
    return variants, Y_flat, row_ts, valid


def main():
    variants, Y, row_ts, valid = build_features()
    udays = np.unique(row_ts[valid])
    n = len(udays)
    bounds = np.linspace(int(INIT_FRAC * n), n, N_FOLDS + 1).astype(int)
    params = dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                  random_state=SEED, tree_method="hist")

    print(f"days={n}, folds={N_FOLDS}, initial train frac={INIT_FRAC}")
    print("\n%-5s %9s %9s %9s %9s %9s %9s" %
          ("fold", "n_test_d", "BASE_ic", "GRP_ic", "DYN_ic", "DYN_Lbps", "DYN_Sbps"))
    print("-" * 66)
    agg = {k: [] for k in variants}
    dyn_net = {"long": [], "short": []}
    for k in range(N_FOLDS):
        test_vals = udays[bounds[k]:bounds[k + 1]]
        test_start = test_vals.min()
        train_vals = udays[udays < test_start - PURGE]
        tr = valid & np.isin(row_ts, train_vals)
        te = valid & np.isin(row_ts, test_vals)
        ics = {}
        for name, Xall in variants.items():
            m = xgb.XGBRegressor(**params)
            m.fit(Xall[tr], Y[tr])
            pred = m.predict(Xall[te])
            ic, _, _ = per_day_rank_ic(pred, Y[te], row_ts[te])
            ics[name] = ic
            agg[name].append(ic)
            if name == "+DYNAMIC":
                wr = side_winrate(pred, Y[te], row_ts[te])
                dyn_net["long"].append(wr["long"]["net_bps"])
                dyn_net["short"].append(wr["short"]["net_bps"])
        print("%-5d %9d %9.4f %9.4f %9.4f %9.2f %9.2f" %
              (k + 1, len(test_vals), ics["BASE"], ics["+DYN_GRP"], ics["+DYNAMIC"],
               dyn_net["long"][-1], dyn_net["short"][-1]))
    print("-" * 66)
    print("%-5s %9s %9.4f %9.4f %9.4f %9.2f %9.2f" %
          ("MEAN", "", np.mean(agg["BASE"]), np.mean(agg["+DYN_GRP"]),
           np.mean(agg["+DYNAMIC"]), np.mean(dyn_net["long"]), np.mean(dyn_net["short"])))
    d = np.array(agg["+DYNAMIC"]) - np.array(agg["BASE"])
    print(f"\n+DYNAMIC - BASE rankIC delta per fold: {np.round(d, 4).tolist()}")
    print(f"  mean delta={d.mean():.4f}  (decaying/sign-flipping => v8-style mirage)")
    print(f"  folds where +DYNAMIC > BASE: {int((d > 0).sum())}/{N_FOLDS}")


if __name__ == "__main__":
    main()
