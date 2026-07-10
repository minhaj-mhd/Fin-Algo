"""
Optuna hyperparameter search for the daily_macro_v3 XGBoost ranker — LONG and SHORT
tuned as TWO INDEPENDENT studies (separate best params per side).

OBJECTIVE (maximize, per side): walk-forward mean rank-IC (Spearman) FLOORED by the worst fold
  obj_side = 0.5*mean(per_fold_rho) + 0.5*min(per_fold_rho)
over the SAME 4 month-based walk-forward folds the production v3 recipe uses
(scripts/training/train_daily_xgboost_v3.py: val_size=6, test_size=6, expanding train).
  • LONG  side: rho = per-query Spearman( long_score ,  Label_1D )
  • SHORT side: rho = per-query Spearman( short_score, -Label_1D )   (short model trained on inverted ranks)

DISCIPLINE (repo rules — AGENTS.md "Model Metric Discipline"):
  * Optuna has NO verdict authority. Output is EXPLORATORY / ⚠️ UNVERIFIED. Only the Validation
    Gauntlet can certify "better than v3". Tune freely HERE (off-Gauntlet); pick ONE winner per
    side; only then — with explicit user approval + a pre-registered hypothesis — spend ONE
    Gauntlet run.
  * This script is READ-ONLY w.r.t. production: it NEVER writes to models/daily_macro_v3/. It only
    writes an Optuna study db + a best-params json under artifacts/.
  * 4 folds is a small-N walk-forward — these rhos RANK configs only; they are not trustworthy
    point estimates. The winner is re-trained by the real pipeline and Gauntleted before any claim.

Run:  python scripts/training/tune_daily_v3_optuna.py --n_trials 80 --timeout 14400
Smoke: python scripts/training/tune_daily_v3_optuna.py --smoke
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

DATA = 'data/ranking_data_daily_macro_v3.csv'
RET_COL = 'Label_1D'
SEED = 42

# current production v3 params (shared hand-tuned recipe) — enqueued as each side's baseline trial
BASELINE = dict(objective='rank:pairwise', eta=0.01, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, alpha=2.0, reg_lambda=4.0, min_child_weight=40, gamma=0.0)

# columns the v3 trainer excludes from the feature matrix (must match exactly)
EXCLUDE = ['DateTime', 'Query_ID', 'Ticker', 'Open', 'High', 'Low', 'Close',
           'Volume', 'Label_1D', 'Sector', 'YearMonth']


def device_ok():
    try:
        d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'},
                  d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


def group_blocks(qids):
    """Per-query group sizes + per-query row-index lists (contiguous runs; panel sorted by query)."""
    _, first = np.unique(qids, return_index=True)
    starts = np.sort(first)
    ends = np.append(starts[1:], len(qids))
    sizes = (ends - starts).astype(int)
    groups = [np.arange(s, e) for s, e in zip(starts, ends)]
    return sizes, groups


def integer_ranks(y, qids, invert):
    """Per-query ordinal ranks (XGBRanker relevance labels)."""
    out = np.zeros(len(y), dtype=int)
    _, first = np.unique(qids, return_index=True)
    starts = np.sort(first)
    ends = np.append(starts[1:], len(qids))
    for s, e in zip(starts, ends):
        vals = -y[s:e] if invert else y[s:e]
        out[s:e] = rankdata(vals, method='ordinal') - 1
    return out


def mean_spearman(scores, ret, qgroups):
    """Mean per-query Spearman ρ(scores, ret) via rank-Pearson."""
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


def build():
    """Load v3 csv, mean-impute (matches production), build the 4 walk-forward folds, cache slices."""
    print(f"Loading {DATA} …")
    df = pd.read_csv(DATA)
    df['YearMonth'] = df['DateTime'].str[:7]
    df = df.sort_values(['Query_ID']).reset_index(drop=True)   # ensure contiguous queries
    months = sorted(df['YearMonth'].unique())
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feat_cols].to_numpy(dtype=np.float32)

    # mean-impute NaN/Inf per column (faithful to train_daily_xgboost_v3.py)
    bad = ~np.isfinite(X)
    if bad.any():
        col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        X[bad] = np.take(col_mean, np.where(bad)[1])

    y = df[RET_COL].to_numpy(dtype=np.float64)
    qids = df['Query_ID'].to_numpy()
    ym = df['YearMonth'].to_numpy()
    y_long = integer_ranks(y, qids, invert=False)
    y_short = integer_ranks(y, qids, invert=True)
    print(f"  {len(df):,} rows, {len(feat_cols)} features, {len(months)} months "
          f"({months[0]}..{months[-1]})")

    # SAME fold scheme as the v3 trainer: val_size=6, test_size=6, 4 expanding folds
    VAL, TEST = 6, 6
    fold_defs = []
    for fi in range(1, 5):
        te_end = len(months) - (4 - fi) * TEST
        te_start = te_end - TEST
        va_end = te_start
        va_start = va_end - VAL
        tr_end = va_start
        fold_defs.append((months[:tr_end], months[va_start:va_end], months[te_start:te_end]))

    folds = []
    for fi, (tr_m, va_m, te_m) in enumerate(fold_defs):
        trm = np.isin(ym, tr_m); vam = np.isin(ym, va_m); tem = np.isin(ym, te_m)
        gtr, _ = group_blocks(qids[trm])
        gva, _ = group_blocks(qids[vam])
        _, te_groups = group_blocks(qids[tem])
        folds.append(dict(
            Xtr=np.ascontiguousarray(X[trm]), ytr_l=y_long[trm], ytr_s=y_short[trm], gtr=gtr,
            Xva=np.ascontiguousarray(X[vam]), yva_l=y_long[vam], yva_s=y_short[vam], gva=gva,
            Xte=np.ascontiguousarray(X[tem]), te_ret=y[tem], te_groups=te_groups))
        print(f"    fold {fi+1}: train {tr_m[0]}..{tr_m[-1]} | val {va_m[0]}..{va_m[-1]} | "
              f"test {te_m[0]}..{te_m[-1]}  (tr={trm.sum():,} te={tem.sum():,})")
    return folds, feat_cols


def fixed(params, device):
    p = dict(params)
    p.update(eval_metric='ndcg@5', ndcg_exp_gain=False, tree_method='hist',
             device=device, verbosity=0, random_state=SEED)
    return p


def run_side_folds(params, side, folds, n_rounds, es_rounds, device, trial=None):
    """Train ONE side per fold, return list of per-fold test rho for that side."""
    rhos = []
    inv = (side == 'short')
    ylab = 'ytr_s' if inv else 'ytr_l'
    yval = 'yva_s' if inv else 'yva_l'
    p = fixed(params, device)
    for fi, F in enumerate(folds):
        dtr = xgb.DMatrix(F['Xtr'], label=F[ylab]); dtr.set_group(F['gtr'])
        dva = xgb.DMatrix(F['Xva'], label=F[yval]); dva.set_group(F['gva'])
        bst = xgb.train(p, dtr, n_rounds, evals=[(dva, 'v')],
                        early_stopping_rounds=es_rounds, verbose_eval=False)
        dte = xgb.DMatrix(F['Xte'])
        sc = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
        ref = -F['te_ret'] if inv else F['te_ret']
        rhos.append(mean_spearman(sc, ref, F['te_groups']))
        del dtr, dva, dte
        if trial is not None:
            trial.report(float(np.mean(rhos)), fi)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return rhos


def score(rhos):
    a = np.array(rhos)
    return 0.5 * float(a.mean()) + 0.5 * float(a.min())   # mean + worst-fold floor


def make_objective(side, folds, n_rounds, es_rounds, device):
    def objective(trial):
        t0 = time.time()
        params = dict(
            objective=trial.suggest_categorical('objective', ['rank:pairwise', 'rank:ndcg']),
            eta=trial.suggest_float('eta', 0.005, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 9),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            alpha=trial.suggest_float('alpha', 1e-3, 20.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 20.0, log=True),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 100),
            gamma=trial.suggest_float('gamma', 0.0, 5.0),
        )
        try:
            rhos = run_side_folds(params, side, folds, n_rounds, es_rounds, device, trial)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  trial {trial.number} [{side}] FAILED: {e}")
            return -1.0
        obj = score(rhos)
        trial.set_user_attr('fold_rhos', [round(r, 5) for r in rhos])
        trial.set_user_attr('mean_rho', float(np.mean(rhos)))
        trial.set_user_attr('worst_fold', float(np.min(rhos)))
        trial.set_user_attr('secs', round(time.time() - t0, 1))
        print(f"  [{side}] trial {trial.number}: obj={obj:+.4f} mean_rho={np.mean(rhos):+.4f} "
              f"worst={np.min(rhos):+.4f} ({trial.user_attrs['secs']}s)")
        return obj
    return objective


def tune_side(side, folds, args, device, storage):
    study = optuna.create_study(
        study_name=f'daily_v3_{side}', storage=storage, load_if_exists=True, direction='maximize',
        sampler=TPESampler(seed=SEED, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=6, n_warmup_steps=2))
    if not study.trials:
        study.enqueue_trial(BASELINE)
        print(f"[{side}] enqueued BASELINE (current v3 params) as trial 0")
    print(f"\n=== tuning {side.upper()} | n_trials={args.n_trials} timeout={args.timeout_side}s ===")
    study.optimize(make_objective(side, folds, args.max_rounds, args.es, device),
                   n_trials=args.n_trials, timeout=args.timeout_side, gc_after_trial=True)

    done = [t for t in study.trials if t.value is not None]
    bt = study.best_trial
    base = next((t for t in study.trials if t.number == 0 and t.value is not None), None)
    print(f"\n[{side}] finished: {len(study.trials)} trials ({len(done)} scored)")
    print(f"[{side}] BEST obj={bt.value:+.4f} mean_rho={bt.user_attrs.get('mean_rho'):+.4f} "
          f"worst={bt.user_attrs.get('worst_fold'):+.4f}")
    print(f"[{side}] BEST params={bt.params}")
    if base:
        print(f"[{side}] BASELINE obj={base.value:+.4f} mean_rho={base.user_attrs.get('mean_rho'):+.4f}")
    return {
        'best_value': bt.value, 'best_params': bt.params, 'best_user_attrs': bt.user_attrs,
        'baseline_value': base.value if base else None,
        'baseline_attrs': base.user_attrs if base else None,
        'n_trials': len(study.trials), 'n_scored': len(done),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_trials', type=int, default=80, help='trials PER SIDE')
    ap.add_argument('--timeout', type=int, default=14400, help='TOTAL seconds (split across both sides)')
    ap.add_argument('--max_rounds', type=int, default=1000)
    ap.add_argument('--es', type=int, default=100, help='early-stopping rounds')
    ap.add_argument('--sides', default='long,short')
    ap.add_argument('--smoke', action='store_true', help='2 trials, 2 folds, 60 rounds — wiring check only')
    args = ap.parse_args()

    sides = [s.strip() for s in args.sides.split(',') if s.strip()]
    if args.smoke:
        args.n_trials, args.max_rounds, args.es = 2, 60, 20
        print("[SMOKE] 2 trials/side, 60 rounds, 2 folds — wiring check (results meaningless)")

    device = device_ok()
    print(f"device={device}")
    folds, feat_cols = build()
    if args.smoke:
        folds = folds[:2]
    args.timeout_side = max(60, args.timeout // max(1, len(sides)))

    os.makedirs('artifacts', exist_ok=True)
    storage = 'sqlite:///artifacts/optuna_daily_v3_study.db'
    out = {'data': DATA, 'num_features': len(feat_cols), 'device': device,
           'baseline_params': BASELINE, 'smoke': args.smoke, 'sides': {}}
    for side in sides:
        out['sides'][side] = tune_side(side, folds, args, device, storage)

    out_path = 'artifacts/optuna_daily_v3_best.json'
    json.dump(out, open(out_path, 'w'), indent=2, default=float)
    print(f"\n{'='*70}\nsaved -> {out_path}")
    for side, r in out['sides'].items():
        b = r.get('baseline_value'); v = r.get('best_value')
        delta = (v - b) if (b is not None and v is not None) else None
        print(f"  {side.upper():5s}: best obj={v:+.4f}  baseline={b if b is None else f'{b:+.4f}'}"
              + (f"  Δ={delta:+.4f}" if delta is not None else ""))
    print("⚠️ EXPLORATORY / UNVERIFIED — ranks configs only; no Gauntlet authority. models/ untouched.")


if __name__ == '__main__':
    main()
