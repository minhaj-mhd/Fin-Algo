"""
Train XGBoost ranking models on the CLEAN, aligned datasets (rebuilt from one 1-min/15-min
source, IST-anchored to the 09:15 open, session-masked forward returns — no overnight leak,
perfect 15m<->1h alignment).

Usage:
    python scripts/training/train_ranking_clean.py --tf 15min
    python scripts/training/train_ranking_clean.py --tf 1h

Mirrors the original walk-forward pipeline (rank:pairwise, ndcg@3, early stopping, GPU),
per-timeframe hyperparameters matching each model's original. Saves to NEW model dirs so the
existing v8_upstox_3y / v2_15min_3y stay intact.
"""
import os, sys, pickle, json, argparse
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, rankdata
sys.path.append(os.getcwd())

CFG = {
    '15min': dict(
        data='data/ranking_data_upstox_15min_3y_clean.csv',
        ret_col='Next_15Min_Return',
        model_dir='models/v3_15min_clean',
        desc='CLEAN 3-year 15-minute XGBoost Ranking Model (aligned, session-masked)',
        params=dict(max_depth=4, min_child_weight=15),
    ),
    '1h': dict(
        data='data/ranking_data_upstox_1h_3y_clean.csv',
        ret_col='Next_Hour_Return',
        model_dir='models/v9_clean_1h',
        desc='CLEAN 3-year 1-hour XGBoost Ranking Model (aligned, session-masked)',
        params=dict(max_depth=5, min_child_weight=10),
    ),
    '1h_v3': dict(
        data='data/ranking_data_upstox_1h_v3_3y.csv',
        ret_col='Next_Hour_Return',
        model_dir='models/v10_native_1h',
        desc='NATIVE 1-hour XGBoost Ranking Model (Upstox V3 hours/1, session-masked) — production-faithful',
        params=dict(max_depth=5, min_child_weight=10),
    ),
    '1h_v3_d4': dict(
        data='data/ranking_data_upstox_1h_v3_3y.csv',
        ret_col='Next_Hour_Return',
        model_dir='models/v10_depth4_1h',
        desc='NATIVE 1-hour XGBoost Ranking Model (Upstox V3 hours/1) — Depth 4',
        params=dict(max_depth=4, min_child_weight=10),
    ),
    '1h_roll': dict(
        data='data/research/v20_rolling_1h/panel.parquet',
        ret_col='Next_Hour_Return',
        model_dir='models/research/v20_rolling_1h',
        desc='RESEARCH v20: overlapping rolling 1h candles (15-min step) — anchor-agnostic 1h ranker. '
             'Same recipe/features as v10, only the candle grid differs. NOT Gauntlet-certified; '
             'overlapping windows inflate significance (effective N ~1/4 rows) — point estimates only.',
        params=dict(max_depth=5, min_child_weight=10),
    ),
    '1h_roll_v21': dict(
        data='data/research/v21_rolling_1h/panel.parquet',
        ret_col='Next_Hour_Return',
        model_dir='models/research/v21_rolling_1h',
        desc='RESEARCH v21 (LEAN): cleanest rolling-1h ranker — liquidity universe + bar hygiene + '
             'mask-not-fill + WALL-CLOCK LOOKBACK FIX (the only ablation-positive lever, +0.0048 long rho) '
             '+ causal session-boundary gap representation. Mean/std scoring and the sector-graph feature '
             'were DROPPED (ablation: neutral-to-negative). Same XGB recipe as v20/v10. NOT Gauntlet-certified; '
             'overlapping windows inflate significance (effective N ~1/4 rows) — point estimates only.',
        params=dict(max_depth=5, min_child_weight=10),
    ),
    '1h_roll_v23': dict(
        data='data/research/v20_rolling_1h/panel.parquet',
        ret_col='Next_Hour_Return',
        model_dir='models/research/v23_rolling_1h',
        desc='RESEARCH v23: V20 but strictly limited to the top 20 features by SHAP / Permutation Drop, split by side.',
        params=dict(max_depth=5, min_child_weight=10),
        selected_features_long=[
            'Return', 'IBS', 'Dist_Donchian_Upper', 'Log_Return', 'Dist_Keltner_Lower', 
            'CMF_20', 'Relative_Return', 'Lower_Shadow', 'Hour', 'Dist_52W_Low', 
            'Dollar_Volume', 'PercentB', 'RVOL', 'Keltner_Width', 'Alpha_3H', 
            'Time_To_Close', 'Volume_Zscore', 'Is_Close_Hour', 'Dist_SMA_6', 'Intraday_Return'
        ],
        selected_features_short=[
            'Dist_Keltner_Lower', 'Relative_Return', 'Keltner_Width', 'Return', 
            'CMF_20', 'Log_Return', 'PPO_Signal', 'Dist_52W_Low', 'RVOL', 
            'Lower_Shadow', 'Dollar_Volume', 'Intraday_Return', 'Dist_HMA_12', 
            'Hour', 'TRIX_15', 'Dist_Donchian_Upper', 'Rolling_Skew', 'IBS', 
            'Donchian_Width', 'VWAP_Dist'
        ]
    ),
}

ap = argparse.ArgumentParser()
ap.add_argument('--tf', required=True, choices=['15min', '1h', '1h_v3', '1h_v3_d4', '1h_roll', '1h_roll_v21', '1h_roll_v23'])
args = ap.parse_args()
c = CFG[args.tf]
DATA_FILE, RET_COL, MODEL_DIR = c['data'], c['ret_col'], c['model_dir']
os.makedirs(MODEL_DIR, exist_ok=True)
LONG_MODEL_PATH  = f'{MODEL_DIR}/xgb_long_model.json'
SHORT_MODEL_PATH = f'{MODEL_DIR}/xgb_short_model.json'
META_PATH        = f'{MODEL_DIR}/metadata.json'
SCALER_PATH      = f'{MODEL_DIR}/scaler.pkl'

print("=" * 64)
print(f"CLEAN {args.tf.upper()} RANKING TRAINING — Walk-Forward + Early Stopping")
print("=" * 64)
print(f"Loading {DATA_FILE} ...")
df = pd.read_parquet(DATA_FILE) if DATA_FILE.endswith('.parquet') else pd.read_csv(DATA_FILE)
print(f"Loaded {df.shape[0]:,} rows")
df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
unique_months = sorted(df['YearMonth'].unique())
print(f"Spans {len(unique_months)} months: {unique_months[0]} -> {unique_months[-1]}")

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']

if 'selected_features_long' in c and 'selected_features_short' in c:
    feature_cols = list(dict.fromkeys(c['selected_features_long'] + c['selected_features_short']))
    feature_cols = [col for col in feature_cols if col in df.columns]
elif 'selected_features' in c:
    feature_cols = [col for col in c['selected_features'] if col in df.columns]
else:
    feature_cols = [col for col in df.columns if col not in exclude_cols]

has_split_feats = 'selected_features_long' in c and 'selected_features_short' in c
if has_split_feats:
    idx_long = [feature_cols.index(f) for f in c['selected_features_long'] if f in feature_cols]
    idx_short = [feature_cols.index(f) for f in c['selected_features_short'] if f in feature_cols]
else:
    idx_long = list(range(len(feature_cols)))
    idx_short = list(range(len(feature_cols)))

print(f"Features: {len(feature_cols)} | Samples: {df.shape[0]:,} | Queries: {df['Query_ID'].nunique():,}")

X = df[feature_cols].values.astype(np.float64)
y_returns = df[RET_COL].values
query_ids = df['Query_ID'].values

nan_mask = np.isnan(X) | np.isinf(X)
if nan_mask.any():
    print(f"Replacing {int(nan_mask.sum())} NaN/Inf values...")
    for ci in range(X.shape[1]):
        col = X[:, ci]; bad = np.isnan(col) | np.isinf(col)
        if bad.any():
            good = col[~bad]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0

def get_integer_ranks(y_vals, qids, invert=False):
    y_int = np.zeros_like(y_vals, dtype=int)
    for qid in np.unique(qids):
        m = qids == qid
        if m.sum() == 0: continue
        vals = -y_vals[m] if invert else y_vals[m]
        y_int[m] = rankdata(vals, method='ordinal') - 1
    return y_int

# GPU detection
device = 'cpu'
try:
    d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
    xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, d, num_boost_round=1)
    device = 'cuda'; print("  CUDA GPU detected.")
except Exception:
    print("  CPU training.")

params = {
    'objective': 'rank:pairwise', 'eta': 0.03,
    'max_depth': c['params']['max_depth'], 'subsample': 0.8, 'colsample_bytree': 0.8,
    'alpha': 1.0, 'lambda': 2.0, 'min_child_weight': c['params']['min_child_weight'],
    'random_state': 42, 'verbosity': 0, 'eval_metric': 'ndcg@3', 'ndcg_exp_gain': False,
    'tree_method': 'hist', 'device': device,
}

def compute_spearman_rho(df_eval, score_col, invert=False):
    rhos = []
    for qid in df_eval['Query_ID'].unique():
        q = df_eval[df_eval['Query_ID'] == qid]
        if len(q) > 1:
            y = -q[RET_COL].values if invert else q[RET_COL].values
            rho, _ = spearmanr(q[score_col].values, y)
            if not np.isnan(rho): rhos.append(rho)
    return np.mean(rhos) if rhos else 0.0

def evaluate(df_eval, long_scores, short_scores):
    d = df_eval.copy()
    d['long_score'] = long_scores; d['short_score'] = short_scores
    qids = d['Query_ID'].unique()
    long_rho = compute_spearman_rho(d, 'long_score', False)
    short_rho = compute_spearman_rho(d, 'short_score', True)
    lwr, swr = {}, {}
    for k in [1, 3, 5]:
        lh = lt = sh = st = 0
        for qid in qids:
            q = d[d['Query_ID'] == qid]
            if len(q) < k + 1: continue
            ar = q[RET_COL].values; med = np.median(ar)
            li = np.argsort(q['long_score'].values)[::-1][:k]
            lh += (ar[li] > med).sum(); lt += k
            si = np.argsort(q['short_score'].values)[::-1][:k]
            sh += (ar[si] < med).sum(); st += k
        lwr[k] = lh / lt if lt else 0.0
        swr[k] = sh / st if st else 0.0
    tl, ts, tr = [], [], []
    for qid in qids:
        q = d[d['Query_ID'] == qid]
        if len(q) < 4: continue
        ar = q[RET_COL].values
        tl.append(ar[np.argsort(q['long_score'].values)[::-1][:3]].mean())
        ts.append(-ar[np.argsort(q['short_score'].values)[::-1][:3]].mean())
        tr.append(ar.mean())
    if tl:
        al, ash, arnd = np.mean(tl), np.mean(ts), np.mean(tr)
        le, se, ce = al - arnd, ash + arnd, al + ash
    else:
        al = ash = arnd = le = se = ce = 0.0
    return dict(long_rho=long_rho, short_rho=short_rho, long_win_rates=lwr, short_win_rates=swr,
                long_edge=le, short_edge=se, combined_edge=ce)

# walk-forward folds
min_train_months, horizon = 18, 2
folds = []
for i in range(min_train_months, len(unique_months) - horizon, 4):
    folds.append(dict(fold=len(folds)+1, train=unique_months[:i],
                      val=[unique_months[i]], test=unique_months[i+1:i+horizon+1]))
print(f"\nWalk-forward folds: {len(folds)}")

wf = []
for cfg in folds:
    tr_m, val_m, te_m = cfg['train'], cfg['val'], cfg['test']
    print(f"\n--- FOLD {cfg['fold']} --- train {tr_m[0]}->{tr_m[-1]} ({len(tr_m)}m) | val {val_m[0]} | test {te_m[0]}->{te_m[-1]}")
    trm = df['YearMonth'].isin(tr_m).values
    vam = df['YearMonth'].isin(val_m).values
    tem = df['YearMonth'].isin(te_m).values
    Xtr, ytr, qtr = X[trm], y_returns[trm], query_ids[trm]
    Xva, yva, qva = X[vam], y_returns[vam], query_ids[vam]
    Xte = X[tem]; dfte = df[tem]
    gtr = pd.Series(qtr).groupby(qtr).size().values
    gva = pd.Series(qva).groupby(qva).size().values
    # long
    dtl = xgb.DMatrix(Xtr[:, idx_long], label=get_integer_ranks(ytr, qtr, False)); dtl.set_group(gtr)
    dvl = xgb.DMatrix(Xva[:, idx_long], label=get_integer_ranks(yva, qva, False)); dvl.set_group(gva)
    bl = xgb.train(params, dtl, num_boost_round=500, evals=[(dvl, 'val')], early_stopping_rounds=50, verbose_eval=False)
    # short
    dts = xgb.DMatrix(Xtr[:, idx_short], label=get_integer_ranks(ytr, qtr, True)); dts.set_group(gtr)
    dvs = xgb.DMatrix(Xva[:, idx_short], label=get_integer_ranks(yva, qva, True)); dvs.set_group(gva)
    bs = xgb.train(params, dts, num_boost_round=500, evals=[(dvs, 'val')], early_stopping_rounds=50, verbose_eval=False)
    
    dte_long = xgb.DMatrix(Xte[:, idx_long])
    dte_short = xgb.DMatrix(Xte[:, idx_short])
    m = evaluate(dfte, bl.predict(dte_long), bs.predict(dte_short)); m['fold'] = cfg['fold']
    wf.append(m)
    print(f"    Long Rho {m['long_rho']:.4f} | Short Rho {m['short_rho']:.4f} | L-WR@3 {m['long_win_rates'][3]:.1%} | S-WR@3 {m['short_win_rates'][3]:.1%}")

avg = lambda key: float(np.mean([r[key] for r in wf]))
avg_l_rho = avg('long_rho'); avg_s_rho = avg('short_rho')
avg_l_wr3 = float(np.mean([r['long_win_rates'][3] for r in wf]))
avg_s_wr3 = float(np.mean([r['short_win_rates'][3] for r in wf]))
avg_l_edge = avg('long_edge'); avg_s_edge = avg('short_edge'); avg_c_edge = avg('combined_edge')

print("\n" + "=" * 64)
print("AGGREGATE WALK-FORWARD")
print(f"  Long Rho {avg_l_rho:.4f} | Short Rho {avg_s_rho:.4f}")
print(f"  L-WR@3 {avg_l_wr3:.1%} | S-WR@3 {avg_s_wr3:.1%}")
print(f"  Long edge {avg_l_edge*100:+.4f}% | Short edge {avg_s_edge*100:+.4f}% | Combined {avg_c_edge*100:+.4f}% per bar")

# production models
print("\nTraining production models (Strict 80% Train/Val, 20% untouched Test)...")
split_idx = int(len(unique_months) * 0.8)
ptr = df['YearMonth'].isin(unique_months[:split_idx-1]).values
pva = df['YearMonth'].isin([unique_months[split_idx-1]]).values
Xptr, yptr, qptr = X[ptr], y_returns[ptr], query_ids[ptr]
Xpva, ypva, qpva = X[pva], y_returns[pva], query_ids[pva]
gptr = pd.Series(qptr).groupby(qptr).size().values
gpva = pd.Series(qpva).groupby(qpva).size().values

dptl = xgb.DMatrix(Xptr[:, idx_long], label=get_integer_ranks(yptr, qptr, False)); dptl.set_group(gptr)
dpvl = xgb.DMatrix(Xpva[:, idx_long], label=get_integer_ranks(ypva, qpva, False)); dpvl.set_group(gpva)
prod_long = xgb.train(params, dptl, num_boost_round=500, evals=[(dpvl, 'val')], early_stopping_rounds=50, verbose_eval=50)
prod_long.save_model(LONG_MODEL_PATH)

dpts = xgb.DMatrix(Xptr[:, idx_short], label=get_integer_ranks(yptr, qptr, True)); dpts.set_group(gptr)
dpvs = xgb.DMatrix(Xpva[:, idx_short], label=get_integer_ranks(ypva, qpva, True)); dpvs.set_group(gpva)
prod_short = xgb.train(params, dpts, num_boost_round=500, evals=[(dpvs, 'val')], early_stopping_rounds=50, verbose_eval=50)
prod_short.save_model(SHORT_MODEL_PATH)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(StandardScaler(with_mean=False, with_std=False), f)

def imp(bst):
    try:
        s = bst.get_score(importance_type='gain'); out = {}
        for k, v in s.items():
            i = int(k.replace('f', ''))
            if i < len(feature_cols): out[feature_cols[i]] = float(v)
        return dict(sorted(out.items(), key=lambda x: -x[1])[:20])
    except Exception:
        return {}

metadata = {
    'description': c['desc'], 'type': 'ranking', 
    'features': feature_cols,
    'features_long': c.get('selected_features_long', feature_cols),
    'features_short': c.get('selected_features_short', feature_cols),
    'num_features': len(feature_cols), 'data_source': f'upstox_{args.tf}_clean',
    'data_file': DATA_FILE, 'total_rows': int(df.shape[0]),
    'walk_forward_summary': {'avg_long_spearman': avg_l_rho, 'avg_short_spearman': avg_s_rho,
                             'avg_long_win_rate_k3': avg_l_wr3, 'avg_short_win_rate_k3': avg_s_wr3},
    'long_test_spearman': avg_l_rho, 'short_test_spearman': avg_s_rho,
    'long_model': LONG_MODEL_PATH, 'short_model': SHORT_MODEL_PATH, 'meta': META_PATH, 'scaler': SCALER_PATH,
    'walk_forward_folds': [{'fold': int(r['fold']), 'long_rho': float(r['long_rho']), 'short_rho': float(r['short_rho'])} for r in wf],
    'top_features_long': imp(prod_long), 'top_features_short': imp(prod_short),
    'params': params, 'trained_at': datetime.now().isoformat(),
}
with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 64)
print(f"DONE — {args.tf} clean models saved to {MODEL_DIR}")
print(f"  WF Long Rho {avg_l_rho:.4f} | Short Rho {avg_s_rho:.4f} | L-WR@3 {avg_l_wr3:.1%} | S-WR@3 {avg_s_wr3:.1%}")
print("=" * 64)
