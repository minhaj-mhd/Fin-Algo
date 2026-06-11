"""
HONEST holdout re-eval of the v10 native-1h ranking model.

The original scripts/research/evaluate_v10_topk.py is IN-SAMPLE: the production
v10 model was trained on all-months-but-last (train_ranking_clean.py:204), yet the
eval tests on the last 20% of queries -> the model already saw ~the whole test block.

This script fixes that:
  * sort queries chronologically (Query_ID is time-ordered by construction)
  * TRAIN on the first 80% of queries only (last 10% of that as val for early stop)
  * EVAL on the untouched last 20% of queries
  * identical v10 hyperparameters (rank:pairwise, depth5, mcw10, ndcg@3, early stop)

Reports, on the genuine holdout:
  * Spearman rho (ranking skill)
  * precision@K vs cross-sectional median (same metric as the original)
  * avg top-3 long return  -- RAW/gross (contains market beta)
  * avg top-3 long DEMEANED return -- skill-only (beta removed)
  * net edge after 10 bps round-trip cost
"""
import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, rankdata
sys.path.append(os.getcwd())

DATA_FILE = "data/ranking_data_upstox_1h_v3_3y.csv"
META_FILE = "models/v10_native_1h/metadata.json"
RET_COL   = "Next_Hour_Return"
COST_BPS  = 10.0   # round-trip cost assumption

print(f"Loading {DATA_FILE} ...")
df = pd.read_csv(DATA_FILE)
df = df.sort_values(["Query_ID"]).reset_index(drop=True)
with open(META_FILE) as f:
    feature_cols = json.load(f)["features"]

unique_qids = np.sort(df["Query_ID"].unique())
n = len(unique_qids)
split_idx = int(n * 0.8)
train_pool = unique_qids[:split_idx]
test_qids  = unique_qids[split_idx:]
# carve last 10% of the train pool as validation for early stopping
val_cut    = int(len(train_pool) * 0.9)
train_qids = train_pool[:val_cut]
val_qids   = train_pool[val_cut:]

months = pd.to_datetime(df["DateTime"])
def span(qids):
    m = months[df["Query_ID"].isin(qids)]
    return f"{m.min():%Y-%m} -> {m.max():%Y-%m}"
print(f"Queries: {n}  | train {len(train_qids)} ({span(train_qids)}) | "
      f"val {len(val_qids)} ({span(val_qids)}) | TEST {len(test_qids)} ({span(test_qids)})")

def slice_xy(qids):
    d = df[df["Query_ID"].isin(qids)]
    X = np.nan_to_num(d[feature_cols].values.astype(np.float64))
    return d, X, d[RET_COL].values, d["Query_ID"].values

d_tr, X_tr, y_tr, q_tr = slice_xy(train_qids)
d_va, X_va, y_va, q_va = slice_xy(val_qids)
d_te, X_te, y_te, q_te = slice_xy(test_qids)

def int_ranks(y, q, invert=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(q):
        m = q == qid
        out[m] = rankdata(-y[m] if invert else y[m], method="ordinal") - 1
    return out

def groups(q):
    return pd.Series(q).groupby(q, sort=False).size().values

# GPU detect
device = "cpu"
try:
    dd = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); dd.set_group([10])
    xgb.train({"objective": "rank:pairwise", "device": "cuda", "tree_method": "hist"}, dd, num_boost_round=1)
    device = "cuda"
except Exception:
    pass
print(f"device={device}")

params = dict(objective="rank:pairwise", eta=0.03, max_depth=5, subsample=0.8,
              colsample_bytree=0.8, alpha=1.0, **{"lambda": 2.0}, min_child_weight=10,
              random_state=42, verbosity=0, eval_metric="ndcg@3", ndcg_exp_gain=False,
              tree_method="hist", device=device)

def train_side(invert):
    dtr = xgb.DMatrix(X_tr, label=int_ranks(y_tr, q_tr, invert)); dtr.set_group(groups(q_tr))
    dva = xgb.DMatrix(X_va, label=int_ranks(y_va, q_va, invert)); dva.set_group(groups(q_va))
    return xgb.train(params, dtr, num_boost_round=500, evals=[(dva, "val")],
                     early_stopping_rounds=50, verbose_eval=False)

print("Training LONG (holdout) ..."); bl = train_side(False)
print("Training SHORT (holdout) ..."); bs = train_side(True)

dte = xgb.DMatrix(X_te)
d_te = d_te.copy()
d_te["long_score"]  = bl.predict(dte)
d_te["short_score"] = bs.predict(dte)

# metrics over test queries
lwr = {1: [0, 0], 3: [0, 0], 5: [0, 0]}
swr = {3: [0, 0]}
long_rhos, short_rhos = [], []
raw_top3, dem_top3, mkt_mean = [], [], []
for qid, q in d_te.groupby("Query_ID"):
    ar = q[RET_COL].values
    if len(q) > 1:
        r, _ = spearmanr(q["long_score"].values, ar)
        if not np.isnan(r): long_rhos.append(r)
        r, _ = spearmanr(q["short_score"].values, -ar)
        if not np.isnan(r): short_rhos.append(r)
    med = np.median(ar)
    for k in lwr:
        if len(q) >= k + 1:
            li = np.argsort(q["long_score"].values)[::-1][:k]
            lwr[k][0] += (ar[li] > med).sum(); lwr[k][1] += k
    if len(q) >= 4:
        si = np.argsort(q["short_score"].values)[::-1][:3]
        swr[3][0] += (ar[si] < med).sum(); swr[3][1] += 3
        li3 = np.argsort(q["long_score"].values)[::-1][:3]
        m = ar.mean()
        raw_top3.append(ar[li3].mean())
        dem_top3.append(ar[li3].mean() - m)   # demeaned = skill above market
        mkt_mean.append(m)

res = {
    "holdout_test_span": span(test_qids),
    "n_test_queries": int(len(test_qids)),
    "long_rho_OOS": float(np.mean(long_rhos)),
    "short_rho_OOS": float(np.mean(short_rhos)),
    "long_prec@1": lwr[1][0] / lwr[1][1],
    "long_prec@3": lwr[3][0] / lwr[3][1],
    "long_prec@5": lwr[5][0] / lwr[5][1],
    "short_prec@3": swr[3][0] / swr[3][1],
    "avg_top3_long_return_RAW_pct":   float(np.mean(raw_top3)) * 100,
    "avg_market_mean_return_pct":     float(np.mean(mkt_mean)) * 100,
    "avg_top3_long_DEMEANED_pct":     float(np.mean(dem_top3)) * 100,
    "demeaned_edge_bps":              float(np.mean(dem_top3)) * 1e4,
    "net_after_%.0fbps_cost_bps" % COST_BPS: float(np.mean(dem_top3)) * 1e4 - COST_BPS,
}
os.makedirs("data", exist_ok=True)
with open("data/v10_holdout_eval_results.json", "w") as f:
    json.dump(res, f, indent=2)
print("\n" + "=" * 60)
print("V10 HONEST HOLDOUT RESULTS")
print("=" * 60)
print(json.dumps(res, indent=2))
