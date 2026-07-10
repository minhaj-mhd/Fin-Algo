"""
Optuna hyperparameter search for the v20 rolling-1h XGBoost ranker (long + short).

OBJECTIVE (maximize): walk-forward mean rank-IC (Spearman), FLOORED by the worst fold —
  obj = 0.5*mean(per_fold_combined_rho) + 0.5*min(per_fold_combined_rho)
where per_fold_combined_rho = (long_rho + short_rho)/2 over the SAME 8 month-based folds the
production v20 recipe uses (scripts/training/train_ranking_clean.py --tf 1h_roll).

DISCIPLINE (repo rules — read AGENTS.md "Model Metric Discipline"):
  * Optuna has NO verdict authority. Output is EXPLORATORY / ⚠️ UNVERIFIED. Only the Validation
    Gauntlet can certify "better than v20".
  * Gauntlet runs are not free (they deflate the dataset family's t-thresholds). So: tune freely
    HERE (off-Gauntlet), pick ONE winner, and only then — with explicit user approval + a
    pre-registered hypothesis — spend ONE Gauntlet run.
  * The certified models/research/v20_rolling_1h/ and its stamp are NOT touched. This script only
    writes a study db + best-params json.
  * Overlapping panel inflates significance (effective N ~1/4) — these rhos RANK configs only;
    they are not trustworthy point estimates. The winner is re-checked on the :15 non-overlap
    subset (and ultimately the Gauntlet) before any claim.

Run:  python scripts/training/tune_v20_xgb_optuna.py --n_trials 120 --timeout 7200
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

DATA = 'data/research/v20_rolling_1h/panel.parquet'
RET_COL = 'Next_Hour_Return'
SEED = 42
NUM_ROUNDS = 500
ES_ROUNDS = 50

# current production v20 params (the v10 hand-me-down recipe) — enqueued as the baseline trial
BASELINE = dict(objective='rank:pairwise', eta=0.03, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, alpha=1.0, reg_lambda=2.0, min_child_weight=10, gamma=0.0)


def device_ok():
    try:
        d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'},
                  d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


def group_blocks(qids):
    """Contiguous-run group sizes + per-query row-index lists (panel is sorted by query)."""
    _, first = np.unique(qids, return_index=True)
    starts = np.sort(first)
    ends = np.append(starts[1:], len(qids))
    sizes = (ends - starts).astype(int)
    groups = [np.arange(s, e) for s, e in zip(starts, ends)]
    return sizes, groups


def integer_ranks(y, qids, invert):
    """Per-query ordinal ranks (XGBRanker relevance labels). Query-local → fold-slice safe."""
    out = np.zeros(len(y), dtype=int)
    _, first = np.unique(qids, return_index=True)
    starts = np.sort(first)
    ends = np.append(starts[1:], len(qids))
    for s, e in zip(starts, ends):
        vals = -y[s:e] if invert else y[s:e]
        out[s:e] = rankdata(vals, method='ordinal') - 1
    return out


def mean_spearman(scores, ret, qgroups):
    """Mean per-query Spearman ρ(scores, ret) via rank-Pearson (no scipy per call)."""
    rhos = []
    for idx in qgroups:
        if len(idx) < 2:
            continue
        a = rankdata(scores[idx]); b = rankdata(ret[idx])
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            rhos.append(float((a * b).sum() / den))
    return float(np.mean(rhos)) if rhos else 0.0


# ── one-time setup: load panel, build folds, cache param-independent slices ───
print("Loading v20 panel …")
df = pd.read_parquet(DATA)
df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
months = sorted(df['YearMonth'].unique())
exclude = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
           'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']
feat_cols = [c for c in df.columns if c not in exclude]
X = df[feat_cols].to_numpy(dtype=np.float32)
np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
y = df[RET_COL].to_numpy(dtype=np.float64)
qids = df['Query_ID'].to_numpy()
ym = df['YearMonth'].to_numpy()
print(f"  {len(df):,} rows, {len(feat_cols)} features, {len(months)} months "
      f"({months[0]}..{months[-1]})")

# global per-query ranks (query is fully inside one fold-slice → safe to slice afterwards)
y_long = integer_ranks(y, qids, invert=False)
y_short = integer_ranks(y, qids, invert=True)

# same fold scheme as train_ranking_clean.py: 18m min train, horizon 2, step 4
MIN_TRAIN, HORIZON = 18, 2
fold_defs = []
for i in range(MIN_TRAIN, len(months) - HORIZON, 4):
    fold_defs.append((months[:i], [months[i]], months[i + 1:i + HORIZON + 1]))
print(f"  walk-forward folds: {len(fold_defs)}")

FOLDS = []
for fi, (tr_m, va_m, te_m) in enumerate(fold_defs):
    trm = np.isin(ym, tr_m); vam = np.isin(ym, va_m); tem = np.isin(ym, te_m)
    gtr, _ = group_blocks(qids[trm])
    gva, _ = group_blocks(qids[vam])
    _, te_groups = group_blocks(qids[tem])
    FOLDS.append(dict(
        Xtr=np.ascontiguousarray(X[trm]), ytr_l=y_long[trm], ytr_s=y_short[trm], gtr=gtr,
        Xva=np.ascontiguousarray(X[vam]), yva_l=y_long[vam], yva_s=y_short[vam], gva=gva,
        Xte=np.ascontiguousarray(X[tem]), te_ret=y[tem], te_groups=te_groups))
    print(f"    fold {fi+1}: train {tr_m[0]}..{tr_m[-1]} val {va_m[0]} test {te_m[0]}..{te_m[-1]}"
          f"  (tr={trm.sum():,} te={tem.sum():,})")

DEVICE = device_ok()
print(f"  device={DEVICE}")


def fixed(params):
    p = dict(params)
    p.update(eval_metric='ndcg@3', ndcg_exp_gain=False, tree_method='hist',
             device=DEVICE, verbosity=0, random_state=SEED)
    return p


def run_folds(params, trial=None):
    long_rhos, short_rhos = [], []
    for fi, F in enumerate(FOLDS):
        dtl = xgb.DMatrix(F['Xtr'], label=F['ytr_l']); dtl.set_group(F['gtr'])
        dvl = xgb.DMatrix(F['Xva'], label=F['yva_l']); dvl.set_group(F['gva'])
        bl = xgb.train(params, dtl, NUM_ROUNDS, evals=[(dvl, 'v')],
                       early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
        dts = xgb.DMatrix(F['Xtr'], label=F['ytr_s']); dts.set_group(F['gtr'])
        dvs = xgb.DMatrix(F['Xva'], label=F['yva_s']); dvs.set_group(F['gva'])
        bs = xgb.train(params, dts, NUM_ROUNDS, evals=[(dvs, 'v')],
                       early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
        dte = xgb.DMatrix(F['Xte'])
        ls = bl.predict(dte, iteration_range=(0, bl.best_iteration + 1))
        ss = bs.predict(dte, iteration_range=(0, bs.best_iteration + 1))
        long_rhos.append(mean_spearman(ls, F['te_ret'], F['te_groups']))
        short_rhos.append(mean_spearman(ss, -F['te_ret'], F['te_groups']))
        del dtl, dvl, dts, dvs, dte
        if trial is not None:
            comb = [(l + s) / 2 for l, s in zip(long_rhos, short_rhos)]
            trial.report(float(np.mean(comb)), fi)        # running mean for pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
    return long_rhos, short_rhos


def score(long_rhos, short_rhos):
    comb = np.array([(l + s) / 2 for l, s in zip(long_rhos, short_rhos)])
    return 0.5 * float(comb.mean()) + 0.5 * float(comb.min())   # mean + worst-fold floor


def objective(trial):
    t0 = time.time()
    params = fixed(dict(
        objective=trial.suggest_categorical('objective', ['rank:pairwise', 'rank:ndcg']),
        eta=trial.suggest_float('eta', 0.01, 0.3, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 8),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        alpha=trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 30),
        gamma=trial.suggest_float('gamma', 0.0, 5.0),
    ))
    lr, sr = run_folds(params, trial)
    obj = score(lr, sr)
    trial.set_user_attr('mean_long_rho', float(np.mean(lr)))
    trial.set_user_attr('mean_short_rho', float(np.mean(sr)))
    trial.set_user_attr('worst_fold', float(np.min([(l + s) / 2 for l, s in zip(lr, sr)])))
    trial.set_user_attr('secs', round(time.time() - t0, 1))
    print(f"  trial {trial.number}: obj={obj:+.4f} Lρ={np.mean(lr):+.4f} Sρ={np.mean(sr):+.4f} "
          f"worst={trial.user_attrs['worst_fold']:+.4f} ({trial.user_attrs['secs']}s)")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_trials', type=int, default=120)
    ap.add_argument('--timeout', type=int, default=7200, help='seconds (default 2h)')
    ap.add_argument('--study', default='v20_xgb')
    args = ap.parse_args()

    os.makedirs('artifacts', exist_ok=True)
    storage = 'sqlite:///artifacts/optuna_v20_study.db'
    study = optuna.create_study(
        study_name=args.study, storage=storage, load_if_exists=True, direction='maximize',
        sampler=TPESampler(seed=SEED, n_startup_trials=12),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=3))
    if not study.trials:
        study.enqueue_trial(BASELINE)        # reference: current production v20 recipe
        print("enqueued BASELINE (current v20 params) as trial 0")
    print(f"study '{args.study}' @ {storage}  n_trials={args.n_trials} timeout={args.timeout}s")

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    done = [t for t in study.trials if t.value is not None]
    bt = study.best_trial
    print(f"\n{'='*70}\nfinished: {len(study.trials)} trials ({len(done)} scored)")
    print(f"BEST obj={bt.value:+.4f}  params={bt.params}")
    print(f"  attrs: {bt.user_attrs}")
    # baseline reference (trial 0 if it ran)
    base = next((t for t in study.trials if t.number == 0 and t.value is not None), None)
    if base:
        print(f"BASELINE obj={base.value:+.4f}  Lρ={base.user_attrs.get('mean_long_rho'):+.4f} "
              f"Sρ={base.user_attrs.get('mean_short_rho'):+.4f}")
    out = {'best_value': bt.value, 'best_params': bt.params, 'best_user_attrs': bt.user_attrs,
           'baseline_value': base.value if base else None,
           'baseline_attrs': base.user_attrs if base else None,
           'n_trials': len(study.trials), 'n_scored': len(done)}
    json.dump(out, open('artifacts/optuna_v20_best.json', 'w'), indent=2, default=float)
    print("saved -> artifacts/optuna_v20_best.json")


if __name__ == '__main__':
    main()
