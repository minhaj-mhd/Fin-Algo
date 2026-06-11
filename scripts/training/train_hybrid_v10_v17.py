import os, sys, json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata, ttest_1samp

# ── config ─────────────────────────────────────────────────────────────────────
DATA_FILE   = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL     = 'Next_Hour_Return'
COST        = 0.0010
BREAKOUT_TH = 0.0020  # 20 bps target
TRADE_PROB  = 0.62    # 62% probability threshold for V16

# ── setup ──────────────────────────────────────────────────────────────────────
print("=" * 64)
print(f"HYBRID V10 (Ranking) + V17 (Random Forest Breakout) EVALUATION")
print("=" * 64)

df = pd.read_csv(DATA_FILE)
df['YearMonth'] = df['DateTime'].str[:7]
unique_months   = sorted(df['YearMonth'].unique())

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']
feature_cols = [c for c in df.columns if c not in exclude_cols]

X         = df[feature_cols].values.astype(np.float64)
y_returns = df[RET_COL].values.astype(np.float64)
query_ids = df['Query_ID'].values

# Handle NaNs
nan_mask = ~np.isfinite(X)
if nan_mask.any():
    for ci in range(X.shape[1]):
        bad = ~np.isfinite(X[:, ci])
        if bad.any():
            good = X[~bad, ci]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

# V17 Targets
y_long_binary  = (y_returns >= BREAKOUT_TH).astype(np.int32)
y_short_binary = (y_returns <= -BREAKOUT_TH).astype(np.int32)
pos_weight_long  = (1.0 - y_long_binary.mean()) / y_long_binary.mean()
pos_weight_short = (1.0 - y_short_binary.mean()) / y_short_binary.mean()

# V10 Targets (Integer Ranks)
def get_integer_ranks(y_vals, qids, invert=False):
    y_int = np.zeros_like(y_vals, dtype=int)
    for qid in np.unique(qids):
        m = qids == qid
        if m.sum() == 0: continue
        vals = -y_vals[m] if invert else y_vals[m]
        y_int[m] = rankdata(vals, method='ordinal') - 1
    return y_int

# ── XGBoost params ─────────────────────────────────────────────────────────────
device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.zeros(10, dtype=np.int32))
    xgb.train({'objective': 'binary:logistic', 'device': 'cuda', 'tree_method': 'hist'},
              d, num_boost_round=1)
    device = 'cuda'
except Exception:
    pass

def get_v17_params(pos_weight: float) -> dict:
    return {
        'objective':         'binary:logistic',
        'eval_metric':       'auc',
        'scale_pos_weight':  pos_weight,
        'booster':           'gbtree',
        'num_parallel_tree': 100,
        'eta':               1.0,
        'max_depth':         10,
        'subsample':         0.8,
        'colsample_bynode':  0.8,
        'alpha':             1.0,
        'lambda':            2.0,
        'min_child_weight':  10,
        'random_state':      42,
        'verbosity':         0,
        'tree_method':       'hist',
        'device':            device,
    }

def get_v10_params() -> dict:
    return {
        'objective': 'rank:pairwise', 'eval_metric': 'ndcg@3',
        'eta': 0.03, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'alpha': 1.0, 'lambda': 2.0, 'min_child_weight': 10, 'random_state': 42,
        'verbosity': 0, 'tree_method': 'hist', 'device': device, 'ndcg_exp_gain': False,
    }

# ── walk-forward ───────────────────────────────────────────────────────────────
min_train_months, val_months, horizon, step = 18, 4, 2, 4
folds = []
for i in range(min_train_months, len(unique_months) - val_months - horizon, step):
    folds.append(dict(
        fold=len(folds) + 1,
        train=unique_months[:i],
        val=unique_months[i:i + val_months],
        test=unique_months[i + val_months:i + val_months + horizon],
    ))

print(f"\nWalk-forward folds: {len(folds)}")

results = []
all_trades = {k: [] for k in ['A1_L', 'A1_S', 'A3_L', 'A3_S', 'B_L', 'B_S']}

for cfg in folds:
    tr_m, val_m, te_m = cfg['train'], cfg['val'], cfg['test']
    print(f"\n--- FOLD {cfg['fold']} --- test {te_m[0]}->{te_m[-1]} -------------------------")

    trm = df['YearMonth'].isin(tr_m).values
    vam = df['YearMonth'].isin(val_m).values
    tem = df['YearMonth'].isin(te_m).values

    Xtr, Xva, Xte = X[trm], X[vam], X[tem]
    qtr, qva, qte = query_ids[trm], query_ids[vam], query_ids[tem]
    y_ret_te = y_returns[tem]

    # --- Train V17 ---
    print("  Training V17 Classifiers...")
    dtr_l_v17 = xgb.DMatrix(Xtr, label=y_long_binary[trm])
    bl_v17 = xgb.train(get_v17_params(pos_weight_long), dtr_l_v17, num_boost_round=1, verbose_eval=False)

    dtr_s_v17 = xgb.DMatrix(Xtr, label=y_short_binary[trm])
    bs_v17 = xgb.train(get_v17_params(pos_weight_short), dtr_s_v17, num_boost_round=1, verbose_eval=False)

    # --- Train V10 ---
    print("  Training V10 Rankers...")
    gtr = pd.Series(qtr).groupby(qtr).size().values
    gva = pd.Series(qva).groupby(qva).size().values

    dtr_l_v10 = xgb.DMatrix(Xtr, label=get_integer_ranks(y_returns[trm], qtr, False)); dtr_l_v10.set_group(gtr)
    dva_l_v10 = xgb.DMatrix(Xva, label=get_integer_ranks(y_returns[vam], qva, False)); dva_l_v10.set_group(gva)
    bl_v10 = xgb.train(get_v10_params(), dtr_l_v10, num_boost_round=500,
                       evals=[(dva_l_v10, 'val')], early_stopping_rounds=50, verbose_eval=False)

    dtr_s_v10 = xgb.DMatrix(Xtr, label=get_integer_ranks(y_returns[trm], qtr, True)); dtr_s_v10.set_group(gtr)
    dva_s_v10 = xgb.DMatrix(Xva, label=get_integer_ranks(y_returns[vam], qva, True)); dva_s_v10.set_group(gva)
    bs_v10 = xgb.train(get_v10_params(), dtr_s_v10, num_boost_round=500,
                       evals=[(dva_s_v10, 'val')], early_stopping_rounds=50, verbose_eval=False)

    # --- Predictions ---
    print("  Evaluating Hybrid Logics...")
    dte = xgb.DMatrix(Xte)
    pred_l_v17 = bl_v17.predict(dte)
    pred_s_v17 = bs_v17.predict(dte)
    pred_l_v10 = bl_v10.predict(dte)
    pred_s_v10 = bs_v10.predict(dte)

    # Variables to track (fold-level)
    logicA_k1_l_rets, logicA_k1_s_rets = [], []
    logicA_k3_l_rets, logicA_k3_s_rets = [], []
    logicB_l_rets, logicB_s_rets = [], []

    for qid in np.unique(qte):
        m = qte == qid
        if m.sum() < 3: continue
        
        r_l_v10 = pred_l_v10[m]
        r_s_v10 = pred_s_v10[m]
        p_l_v17 = pred_l_v17[m]
        p_s_v17 = pred_s_v17[m]
        actual  = y_ret_te[m]

        # LOGIC A (Rank then Veto)
        # Top 1
        top1_l_idx = np.argsort(r_l_v10)[-1]
        if p_l_v17[top1_l_idx] > TRADE_PROB:
            logicA_k1_l_rets.append(actual[top1_l_idx])
            
        top1_s_idx = np.argsort(r_s_v10)[-1]
        if p_s_v17[top1_s_idx] > TRADE_PROB:
            logicA_k1_s_rets.append(-actual[top1_s_idx])

        # Top 3
        top3_l_idx = np.argsort(r_l_v10)[-3:]
        for idx in top3_l_idx:
            if p_l_v17[idx] > TRADE_PROB:
                logicA_k3_l_rets.append(actual[idx])
                
        top3_s_idx = np.argsort(r_s_v10)[-3:]
        for idx in top3_s_idx:
            if p_s_v17[idx] > TRADE_PROB:
                logicA_k3_s_rets.append(-actual[idx])

        # LOGIC B (Filter then Rank)
        # Long
        pass_l_idx = np.where(p_l_v17 > TRADE_PROB)[0]
        if len(pass_l_idx) > 0:
            best_pass_l_idx = pass_l_idx[np.argmax(r_l_v10[pass_l_idx])]
            logicB_l_rets.append(actual[best_pass_l_idx])
            
        # Short
        pass_s_idx = np.where(p_s_v17 > TRADE_PROB)[0]
        if len(pass_s_idx) > 0:
            best_pass_s_idx = pass_s_idx[np.argmax(r_s_v10[pass_s_idx])]
            logicB_s_rets.append(-actual[best_pass_s_idx])

    # Summarize fold
    def get_metrics(rets):
        if not rets: return 0, 0.0, 0.0
        r = np.array(rets)
        return len(r), float(r.mean() - COST), float(np.mean(r > COST))

    metrics = {
        'A1_L': get_metrics(logicA_k1_l_rets), 'A1_S': get_metrics(logicA_k1_s_rets),
        'A3_L': get_metrics(logicA_k3_l_rets), 'A3_S': get_metrics(logicA_k3_s_rets),
        'B_L':  get_metrics(logicB_l_rets),  'B_S':  get_metrics(logicB_s_rets)
    }

    results.append(metrics)
    all_trades['A1_L'].extend(logicA_k1_l_rets)
    all_trades['A1_S'].extend(logicA_k1_s_rets)
    all_trades['A3_L'].extend(logicA_k3_l_rets)
    all_trades['A3_S'].extend(logicA_k3_s_rets)
    all_trades['B_L'].extend(logicB_l_rets)
    all_trades['B_S'].extend(logicB_s_rets)

    def print_logic_stats(name, rets):
        if len(rets) == 0:
            print(f"      {name:<6} :    0 trades | raw  +0.0bps | net  +0.0bps | raw win 0.0% | net hit 0.0%")
            return
        r = np.array(rets)
        raw_ret = float(np.mean(r))
        net_ret = raw_ret - COST
        raw_win = float(np.mean(r > 0))
        net_hit = float(np.mean(r > COST))
        print(f"      {name:<6} : {len(rets):>4} trades | raw {raw_ret*10000:>+5.1f}bps | net {net_ret*10000:>+5.1f}bps | raw win {raw_win:.1%} | net hit {net_hit:.1%}")

    print("    Logic A (Top 1) - Rank then Veto")
    print_logic_stats("Longs", logicA_k1_l_rets)
    print_logic_stats("Shorts", logicA_k1_s_rets)
    print("    Logic A (Top 3) - Rank then Veto")
    print_logic_stats("Longs", logicA_k3_l_rets)
    print_logic_stats("Shorts", logicA_k3_s_rets)
    print("    Logic B (Filter then Rank)")
    print_logic_stats("Longs", logicB_l_rets)
    print_logic_stats("Shorts", logicB_s_rets)

# ── Aggregate + t-stats ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("AGGREGATE RESULTS  (t-test: mean net return vs 0)")
print("  Key    Trades |  Raw Bps  Net Bps | Raw Win  Net Hit | t-stat  Sig")
print("  " + "-"*76)

for name in ['A1', 'A3', 'B']:
    for side in ['L', 'S']:
        key = f"{name}_{side}"
        rets = np.array(all_trades[key])
        if len(rets) == 0:
            print(f"  {key:<4}        0 |    +0.00    +0.00 |    0.0%     0.0% |   0.00  ns")
            continue
            
        raw_bps = np.mean(rets) * 10000
        net_bps = raw_bps - (COST * 10000)
        raw_win = np.mean(rets > 0) * 100
        net_hit = np.mean(rets > COST) * 100
        
        net_rets = rets - COST
        t_stat, p_val = ttest_1samp(net_rets, 0.0, alternative='greater')
        sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        print(f"  {key:<4} {len(rets):>8} |  {raw_bps:>7.2f}  {net_bps:>7.2f} |  {raw_win:>5.1f}%   {net_hit:>5.1f}% | {t_stat:>6.2f}  {sig}")

print("=" * 80)

# Per-fold breakdown
print("\nPER-FOLD BREAKDOWN")
print(f"  {'Fold':<5} {'A1_S n':>7} {'A1_S bps':>9} {'B_S n':>6} {'B_S bps':>8}")
for i, m in enumerate(results, 1):
    a1s = m['A1_S']; bs = m['B_S']
    print(f"  {i:<5} {a1s[0]:>7}  {a1s[1]*10000:>+8.1f}  {bs[0]:>6}  {bs[1]*10000:>+7.1f}")

# Save
out = {'cost_bps': int(COST*10000), 'breakout_th_bps': int(BREAKOUT_TH*10000),
       'prob_threshold': TRADE_PROB,
       'folds': [{'fold': i+1, **{k: list(v) for k, v in m.items()}} for i, m in enumerate(results)]}
with open('data/hybrid_v10_v17_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nResults saved -> data/hybrid_v10_v17_results.json")
print("=" * 64)
