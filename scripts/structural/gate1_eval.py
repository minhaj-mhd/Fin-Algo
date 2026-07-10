"""
Gate-1 evaluation: does the EXOGENOUS structural graph add net-of-cost signal to
the daily cross-sectional ranker?

Variants (XGBoost regressor on per-day z-scored features -> 3-day fwd return):
  BASE        : 55 stock features only
  +DYNAMIC    : BASE + group/sector neighbor-mean aggregates (1 GNN message-pass, by hand)
  +BOTH       : BASE + dynamic + static topology (centrality/community/embeddings)
  NEG-CONTROL : +BOTH but with node identities PERMUTED (graph<->ticker link broken).
                If NEG matches +BOTH, any "gain" is a fixed-effect/overfit artifact.

Split: train on days before the OOS region (purged by the 3-day label horizon);
test on v2_oos_mask days. Metrics per variant: mean daily rank-IC (Spearman) with
a t-stat across OOS days, and net-of-cost win-rate per side (top/bottom-k).
Cost discipline: prints median(net-gross) per side, which MUST equal -cost.

EXPLORATORY TIER — no Gauntlet authority. ASCII-only output (Windows cp1252 console).
Run: python scripts/structural/gate1_eval.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(os.getcwd())
from scripts.structural.business_groups import base  # noqa: E402

D = "data/daily_transformer_panel/"
G = "data/research/graph/"
COST = 0.001          # 10 bps round-trip, in return units
TOPK = 10             # picks per side per day
PURGE = 3             # days, == label horizon
SEED = 42
AGG_FEATS = ["Return", "Return_5D", "Return_20D", "RSI_14", "Volume_Zscore", "Dist_SMA_20"]


def nan_neighbor_mean(Xf, A):
    """NaN-aware weighted neighbor mean. Xf:[T,N], A:[N,N] weights -> [T,N]."""
    valid = (~np.isnan(Xf)).astype(np.float64)
    num = np.nan_to_num(Xf) @ A.T
    den = valid @ A.T
    return np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)


def per_day_rank_ic(pred, true, day_idx):
    """Mean & t-stat of per-day Spearman IC (Pearson on within-day ranks)."""
    ics = []
    for d in np.unique(day_idx):
        m = day_idx == d
        if m.sum() < 5:
            continue
        p = pd.Series(pred[m]).rank().values
        t = pd.Series(true[m]).rank().values
        if p.std() == 0 or t.std() == 0:
            continue
        ics.append(np.corrcoef(p, t)[0, 1])
    ics = np.array(ics)
    tstat = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 1 else np.nan
    return ics.mean(), tstat, len(ics)


def side_winrate(pred, true, day_idx, k=TOPK):
    """Net-of-cost WR and mean net return for long (top-k) and short (bottom-k)."""
    longs, shorts = [], []
    for d in np.unique(day_idx):
        m = day_idx == d
        if m.sum() < 2 * k:
            continue
        p, y = pred[m], true[m]
        order = np.argsort(p)
        shorts.extend((-y[order[:k]]).tolist())     # short the lowest-predicted
        longs.extend((y[order[-k:]]).tolist())       # long the highest-predicted
    longs, shorts = np.array(longs), np.array(shorts)
    out = {}
    for name, gross in [("long", longs), ("short", shorts)]:
        net = gross - COST
        out[name] = dict(n=len(gross), wr_net=float((net > 0).mean()),
                         net_bps=float(net.mean() * 1e4),
                         med_net_minus_gross_bps=float(np.median(net - gross) * 1e4))
    return out


def main():
    meta = json.load(open(D + "meta.json"))
    feats = meta["stock_feats"]
    X = np.load(D + "X_daily.npy")                    # [T,N,F]
    Y = np.load(D + "Y_3d.npy")                       # [T,N]
    ts = np.load(D + "ts_days.npy")
    oos = np.load(D + "v2_oos_mask.npy")
    T, N, F = X.shape
    tickers = [base(t) for t in meta["tickers"]]

    # ---- adjacency (group binary, sector binary), panel ticker order ----
    idx = {t: i for i, t in enumerate(tickers)}
    edges = pd.read_csv(G + "edges.csv")
    A_grp = np.zeros((N, N)); A_sec = np.zeros((N, N))
    for _, e in edges.iterrows():
        i, j = idx[e["src"]], idx[e["dst"]]
        tgt = A_grp if e["type"] == "group" else A_sec
        tgt[i, j] = tgt[j, i] = 1.0
    print(f"adjacency: group edges/2={int(A_grp.sum()//2)}, sector edges/2={int(A_sec.sum()//2)}")

    # ---- dynamic neighbor-mean features ----
    dyn_list, dyn_names = [], []
    for fn in AGG_FEATS:
        f = feats.index(fn)
        for tag, A in [("grp", A_grp), ("sec", A_sec)]:
            dyn_list.append(nan_neighbor_mean(X[:, :, f], A))
            dyn_names.append(f"nb_{tag}_{fn}")
    DYN = np.stack(dyn_list, axis=-1)                 # [T,N,n_dyn]

    # ---- static topology features (broadcast per ticker over time) ----
    nf = pd.read_csv(G + "node_features.csv").set_index("ticker")
    stat_cols = [c for c in nf.columns if c not in ("sector", "group")]
    STAT_node = nf.loc[tickers, stat_cols].values.astype(np.float64)   # [N, n_stat]
    STAT = np.broadcast_to(STAT_node[None, :, :], (T, N, STAT_node.shape[1]))

    # ---- assemble row table; drop NaN target/base-feature rows ----
    day_grid = np.repeat(np.arange(T), N)
    base_flat = X.reshape(T * N, F)
    Y_flat = Y.reshape(T * N)
    dyn_flat = DYN.reshape(T * N, DYN.shape[-1])
    stat_flat = STAT.reshape(T * N, STAT.shape[-1])
    valid = ~np.isnan(Y_flat) & ~np.isnan(base_flat).any(1)

    first_oos_ts = ts[oos].min()
    is_train = (ts < first_oos_ts)                    # purge applied below via ts gap
    train_cut_ts = first_oos_ts                       # OOS strictly after this
    row_ts = np.repeat(ts, N)
    train_mask = valid & (row_ts < train_cut_ts - PURGE) & np.repeat(~oos, N) \
        & np.repeat(is_train, N)
    test_mask = valid & np.repeat(oos, N)
    print(f"rows: train={int(train_mask.sum())}, OOS test={int(test_mask.sum())}")

    # negative control: permute node identity for BOTH static + dynamic
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    DYN_perm = DYN[:, perm, :].reshape(T * N, DYN.shape[-1])
    STAT_perm = STAT[:, perm, :].reshape(T * N, STAT.shape[-1])

    # group cols are even (grp emitted first per feature), sector cols odd
    grp_cols = list(range(0, dyn_flat.shape[1], 2))
    sec_cols = list(range(1, dyn_flat.shape[1], 2))
    variants = {
        "BASE":        base_flat,
        "+DYN_GRP":    np.hstack([base_flat, dyn_flat[:, grp_cols]]),   # novel exogenous edges
        "+DYN_SEC":    np.hstack([base_flat, dyn_flat[:, sec_cols]]),   # sector momentum
        "+DYNAMIC":    np.hstack([base_flat, dyn_flat]),
        "+BOTH":       np.hstack([base_flat, dyn_flat, stat_flat]),
        "NEG-CONTROL": np.hstack([base_flat, DYN_perm, STAT_perm]),
    }

    params = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                  random_state=SEED, tree_method="hist")
    ytr = Y_flat[train_mask]
    yte = Y_flat[test_mask]
    day_te = row_ts[test_mask]

    print("\n%-12s %8s %7s %9s %9s %8s %8s" %
          ("variant", "rankIC", "t", "L_wr", "L_bps", "S_wr", "S_bps"))
    print("-" * 70)
    for name, Xall in variants.items():
        model = xgb.XGBRegressor(**params)
        model.fit(Xall[train_mask], ytr)
        pred = model.predict(Xall[test_mask])
        ic, t, nd = per_day_rank_ic(pred, yte, day_te)
        wr = side_winrate(pred, yte, day_te)
        print("%-12s %8.4f %7.2f %9.3f %9.2f %8.3f %8.2f" %
              (name, ic, t, wr["long"]["wr_net"], wr["long"]["net_bps"],
               wr["short"]["wr_net"], wr["short"]["net_bps"]))
    # cost sanity (per side, from last variant) -- must be -10 bps
    print("\ncost sanity (median net-gross, must be -10.00 bps): "
          f"L={wr['long']['med_net_minus_gross_bps']:.2f}  "
          f"S={wr['short']['med_net_minus_gross_bps']:.2f}")
    print(f"OOS days scored: {nd}")
    print("\nREAD: a real graph edge => +BOTH/+DYNAMIC beats BASE on rankIC/net AND")
    print("NEG-CONTROL does NOT. If NEG matches +BOTH, the 'gain' is a fixed-effect artifact.")


if __name__ == "__main__":
    main()
