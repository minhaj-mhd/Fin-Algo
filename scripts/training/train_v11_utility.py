"""
v11 — Utility-target ranking baseline for the 1h native model.

Changes vs v10 (train_ranking_clean.py --tf 1h_v3):
  1. Target is expected utility u(r) instead of raw return rank.
     u is asymmetric around the 10 bps cost threshold:
       above cost  → linear reward (slope w_up)
       below cost  → smooth exponential penalty (scale w_down, decay tau)
       floor        → u_max cap to prevent outlier-driven instability
  2. Objective is reg:squarederror on u(r) — model learns E[u | features].
     Ranking at inference = argsort of predicted utility, take top-5.
  3. Walk-forward selection and headline metric = top-5 mean net return
     (gross − cost, in pct) averaged over all queries in the test fold.
     This is the real economic yardstick; Spearman is reported but not used
     for model selection.
  4. All other pipeline choices (cross-sectional z-scored inputs, walk-forward
     splits, GPU, early-stopping on val, prod model = all-but-last) unchanged.

Usage:
    python scripts/training/train_v11_utility.py

Outputs:
    models/v11_utility_1h/
        xgb_long_model.json
        xgb_short_model.json   (short: model learns E[u(-r)] so top-5 = worst longs)
        metadata.json
        scaler.pkl             (identity, kept for pipeline compatibility)
"""

import os, sys, pickle, json
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# ── config ────────────────────────────────────────────────────────────────────
DATA_FILE   = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL     = 'Next_Hour_Return'
MODEL_DIR   = 'models/v11_utility_1h'
COST        = 0.0010          # 10 bps transaction cost threshold
W_UP        = 1.0             # reward slope above cost (per unit net return)
W_DOWN      = 3.0             # penalty multiplier vs reward at the kink
TAU         = 0.0015          # decay scale for penalty below cost (~15 bps)
U_MAX       = 0.030           # floor (cap on penalty magnitude) — prevents
                              # a −30 bps outlier from dominating gradient
TOP_K       = 5               # slots you actually trade
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

LONG_MODEL  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_MODEL = f'{MODEL_DIR}/xgb_short_model.json'
META_PATH   = f'{MODEL_DIR}/metadata.json'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'


# ── utility function ──────────────────────────────────────────────────────────
def utility(r: np.ndarray,
            cost:  float = COST,
            w_up:  float = W_UP,
            w_down:float = W_DOWN,
            tau:   float = TAU,
            u_max: float = U_MAX) -> np.ndarray:
    """
    Asymmetric utility around the cost threshold.

    Net return x = r - cost:
      x >= 0  →  u =  w_up  * x                       (linear reward)
      x <  0  →  u = -w_down * tau * (exp(-x/tau) - 1) (smooth, convex penalty)
    Then clamp to [-u_max, +inf).

    The exponential term guarantees:
      - u is continuous and differentiable at x = 0 (no hard kink)
      - penalty accelerates as x falls below cost (convex in the loss region)
      - w_down/w_up controls the asymmetry ratio at the threshold
      - tau controls how fast the penalty grows; smaller tau → steeper curve
    """
    x = r - cost
    pos = x >= 0
    u = np.where(pos,
                 w_up * x,
                 -w_down * tau * (np.exp(-x / tau) - 1))
    return np.maximum(u, -u_max)


def show_utility_table():
    bps_vals = [20, 15, 12, 10, 8, 5, 0, -5, -10, -20]
    print("\nUtility table (cost=10bps):")
    print(f"  {'Gross bps':>10}  {'Net bps':>8}  {'Utility':>10}")
    for b in bps_vals:
        r = b / 10000
        u = float(utility(np.array([r]))[0])
        print(f"  {b:>10}  {b-10:>8}  {u:>10.5f}")
    print()


# ── data ──────────────────────────────────────────────────────────────────────
print("=" * 64)
print("v11 UTILITY-TARGET 1h TRAINING")
print("=" * 64)
show_utility_table()

print(f"Loading {DATA_FILE} ...")
df = pd.read_csv(DATA_FILE)
print(f"  {df.shape[0]:,} rows")

df['YearMonth'] = df['DateTime'].str[:7]
unique_months   = sorted(df['YearMonth'].unique())
print(f"  {len(unique_months)} months: {unique_months[0]} → {unique_months[-1]}")

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"  Features: {len(feature_cols)} | Queries: {df['Query_ID'].nunique():,}")

X         = df[feature_cols].values.astype(np.float64)
y_returns = df[RET_COL].values.astype(np.float64)
query_ids = df['Query_ID'].values

# replace NaN/Inf with column mean (same as v10)
nan_mask = ~np.isfinite(X)
if nan_mask.any():
    for ci in range(X.shape[1]):
        bad = ~np.isfinite(X[:, ci])
        if bad.any():
            good = X[~bad, ci]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

# precompute utility labels for long (+r) and short (−r)
u_long  = utility(y_returns)
u_short = utility(-y_returns)   # short model: high score = good short = bad long return


# ── XGBoost params ─────────────────────────────────────────────────────────
device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.random.randn(10))
    xgb.train({'objective': 'reg:squarederror', 'device': 'cuda', 'tree_method': 'hist'},
              d, num_boost_round=1)
    device = 'cuda'
    print("  CUDA GPU detected.")
except Exception:
    print("  CPU training.")

params = {
    'objective':     'reg:squarederror',
    'eval_metric':   'rmse',
    'eta':           0.03,
    'max_depth':     5,
    'subsample':     0.8,
    'colsample_bytree': 0.8,
    'alpha':         1.0,
    'lambda':        2.0,
    'min_child_weight': 10,
    'random_state':  42,
    'verbosity':     0,
    'tree_method':   'hist',
    'device':        device,
}


# ── evaluation helpers ────────────────────────────────────────────────────────
def topk_net_return(df_eval: pd.DataFrame, score_col: str, k: int = TOP_K,
                    cost: float = COST) -> tuple[float, float, int]:
    """
    Returns (mean_net_return, hit_rate, n_queries) for top-k picks.
    mean_net_return = mean(gross_return of top-k) − cost
    hit_rate        = fraction of top-k picks whose gross > cost
    """
    gross, hits, n_q = [], 0, 0
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) < k + 1:
            continue
        ar  = q[RET_COL].values
        sc  = q[score_col].values
        idx = np.argsort(sc)[::-1][:k]
        picked = ar[idx]
        gross.extend(picked.tolist())
        hits += (picked > cost).sum()
        n_q  += 1
    if not gross:
        return 0.0, 0.0, 0
    mean_gross = float(np.mean(gross))
    return mean_gross - cost, hits / len(gross), n_q


def spearman_ic(df_eval: pd.DataFrame, score_col: str, invert: bool = False) -> float:
    rhos = []
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) < 2:
            continue
        y = -q[RET_COL].values if invert else q[RET_COL].values
        rho, _ = spearmanr(q[score_col].values, y)
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else 0.0


def evaluate_fold(df_eval: pd.DataFrame,
                  long_scores: np.ndarray,
                  short_scores: np.ndarray) -> dict:
    d = df_eval.copy()
    d['long_score']  = long_scores
    d['short_score'] = short_scores

    l_net, l_hit, n_q  = topk_net_return(d, 'long_score',  TOP_K)
    s_net, s_hit, _    = topk_net_return(d, 'short_score', TOP_K)
    l_rho = spearman_ic(d, 'long_score',  invert=False)
    s_rho = spearman_ic(d, 'short_score', invert=True)
    return dict(
        long_net_return=l_net,   long_hit_rate=l_hit,
        short_net_return=s_net,  short_hit_rate=s_hit,
        long_rho=l_rho,          short_rho=s_rho,
        n_queries=n_q,
    )


# ── walk-forward ──────────────────────────────────────────────────────────────
min_train_months, horizon, step = 18, 2, 4
folds = []
for i in range(min_train_months, len(unique_months) - horizon, step):
    folds.append(dict(
        fold=len(folds) + 1,
        train=unique_months[:i],
        val=[unique_months[i]],
        test=unique_months[i + 1:i + horizon + 1],
    ))
print(f"\nWalk-forward folds: {len(folds)}")
print(f"Headline metric: top-{TOP_K} mean net return (gross − {int(COST*10000)} bps)\n")

wf = []
for cfg in folds:
    tr_m, val_m, te_m = cfg['train'], cfg['val'], cfg['test']
    print(f"--- FOLD {cfg['fold']} --- train {tr_m[0]}→{tr_m[-1]} ({len(tr_m)}m) | "
          f"val {val_m[0]} | test {te_m[0]}→{te_m[-1]}")

    trm = df['YearMonth'].isin(tr_m).values
    vam = df['YearMonth'].isin(val_m).values
    tem = df['YearMonth'].isin(te_m).values

    Xtr, Xva, Xte = X[trm], X[vam], X[tem]
    dfte = df[tem].copy()

    # long model
    dtl = xgb.DMatrix(Xtr, label=u_long[trm])
    dvl = xgb.DMatrix(Xva, label=u_long[vam])
    bl  = xgb.train(params, dtl, num_boost_round=500,
                    evals=[(dvl, 'val')], early_stopping_rounds=50,
                    verbose_eval=False)

    # short model
    dts = xgb.DMatrix(Xtr, label=u_short[trm])
    dvs = xgb.DMatrix(Xva, label=u_short[vam])
    bs  = xgb.train(params, dts, num_boost_round=500,
                    evals=[(dvs, 'val')], early_stopping_rounds=50,
                    verbose_eval=False)

    dte = xgb.DMatrix(Xte)
    m   = evaluate_fold(dfte, bl.predict(dte), bs.predict(dte))
    m['fold'] = cfg['fold']
    wf.append(m)

    flag = "✓" if m['long_net_return'] > 0 else "✗"
    print(f"    {flag} Long net {m['long_net_return']*10000:+.1f}bps  "
          f"hit={m['long_hit_rate']:.1%}  rho={m['long_rho']:.4f}  "
          f"queries={m['n_queries']}")


# ── aggregate ────────────────────────────────────────────────────────────────
def avg(key): return float(np.mean([r[key] for r in wf]))

print("\n" + "=" * 64)
print("WALK-FORWARD AGGREGATE")
print(f"  Long  net return : {avg('long_net_return')*10000:+.2f} bps  "
      f"(hit={avg('long_hit_rate'):.1%})")
print(f"  Short net return : {avg('short_net_return')*10000:+.2f} bps  "
      f"(hit={avg('short_hit_rate'):.1%})")
print(f"  Long  Spearman   : {avg('long_rho'):.4f}")
print(f"  Short Spearman   : {avg('short_rho'):.4f}")
profitable_folds = sum(1 for r in wf if r['long_net_return'] > 0)
print(f"  Profitable folds : {profitable_folds}/{len(wf)}")


# ── production models (all-but-last, val=last) ───────────────────────────────
print("\nTraining production models (all-but-last month, val=last)...")
ptr = df['YearMonth'].isin(unique_months[:-1]).values
pva = df['YearMonth'].isin([unique_months[-1]]).values

Xptr, Xpva = X[ptr], X[pva]

dptl = xgb.DMatrix(Xptr, label=u_long[ptr])
dpvl = xgb.DMatrix(Xpva, label=u_long[pva])
prod_long = xgb.train(params, dptl, num_boost_round=500,
                      evals=[(dpvl, 'val')], early_stopping_rounds=50,
                      verbose_eval=50)
prod_long.save_model(LONG_MODEL)

dpts = xgb.DMatrix(Xptr, label=u_short[ptr])
dpvs = xgb.DMatrix(Xpva, label=u_short[pva])
prod_short = xgb.train(params, dpts, num_boost_round=500,
                       evals=[(dpvs, 'val')], early_stopping_rounds=50,
                       verbose_eval=50)
prod_short.save_model(SHORT_MODEL)

# identity scaler kept for pipeline compatibility
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(StandardScaler(with_mean=False, with_std=False), f)


# ── feature importance ────────────────────────────────────────────────────────
def top_features(bst, n=20):
    try:
        sc = bst.get_score(importance_type='gain')
        out = {}
        for k, v in sc.items():
            i = int(k.replace('f', ''))
            if i < len(feature_cols):
                out[feature_cols[i]] = float(v)
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:n])
    except Exception:
        return {}


# ── metadata ──────────────────────────────────────────────────────────────────
metadata = {
    'description': 'v11 — utility-target 1h ranker (asymmetric cost-aware)',
    'type': 'utility_regression',
    'features': feature_cols,
    'num_features': len(feature_cols),
    'data_file': DATA_FILE,
    'total_rows': int(df.shape[0]),
    'utility_params': dict(cost=COST, w_up=W_UP, w_down=W_DOWN, tau=TAU, u_max=U_MAX),
    'top_k': TOP_K,
    'walk_forward_summary': {
        'avg_long_net_return_bps':  round(avg('long_net_return')  * 10000, 3),
        'avg_short_net_return_bps': round(avg('short_net_return') * 10000, 3),
        'avg_long_hit_rate':        round(avg('long_hit_rate'),  4),
        'avg_long_spearman':        round(avg('long_rho'),       4),
        'avg_short_spearman':       round(avg('short_rho'),      4),
        'profitable_folds':         profitable_folds,
        'total_folds':              len(wf),
    },
    'walk_forward_folds': [
        {k: (round(v * 10000, 3) if 'return' in k else round(v, 4) if isinstance(v, float) else v)
         for k, v in r.items()}
        for r in wf
    ],
    'top_features_long':  top_features(prod_long),
    'top_features_short': top_features(prod_short),
    'params': params,
    'trained_at': datetime.now().isoformat(),
    'v10_baseline': {
        'avg_long_spearman':       0.0261,
        'long_prec@5':             0.5324,
        'holdout_long_net_bps':   -7.677,
    },
}
with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 64)
print(f"DONE → {MODEL_DIR}")
print(f"  Long  net top-{TOP_K}: {avg('long_net_return')*10000:+.2f} bps "
      f"({profitable_folds}/{len(wf)} folds profitable)")
print(f"  Long  Spearman:       {avg('long_rho'):.4f}  "
      f"(v10 baseline: 0.0261)")
print("=" * 64)
