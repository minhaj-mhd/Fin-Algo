"""
Optuna tuning of the v20 ranker for the TRADEABLE metric — Top-K return — not rank-IC.

Why: the rank-IC tuner (tune_v20_xgb_optuna.py) found a config with better average ordering but
WORSE top-of-book return (Top-1 gross 5.12→3.56) — rank-IC and Top-K diverge. Since net = gross −
constant cost, maximising net Top-K ≡ maximising gross Top-K; the real fix is optimising sharpness
at the few names we trade, not whole-cross-section ordering. See [[project_v20_xgb_tuning_deadend]].

OBJECTIVE (maximize): WF mean combined Top-3 GROSS (½·long + ½·short), floored by the worst fold:
  obj_bps = 0.5*mean(fold_top3_combined) + 0.5*min(fold_top3_combined)   (in bps)
evaluated on the :15 NON-overlapping grid (honest, no overlap inflation), over the FIRST 6 folds.
The LAST 2 folds (2025-10..11, 2026-02..03) are HELD OUT and only scored once, for the best config
vs baseline — the analog of a frozen test, because a noisy Top-K objective over many trials will
otherwise find a lucky winner.

DISCIPLINE: exploratory, no Gauntlet verdict, certified v20 untouched. Writes only study db + json.
Run: python scripts/training/tune_v20_topk_optuna.py --n_trials 100 --timeout 5400
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
ANCHORS = {'10:15', '11:15', '12:15', '13:15', '14:15'}
SEED, NUM_ROUNDS, ES_ROUNDS = 42, 500, 50
COSTS, KS = [6.0, 10.0], [1, 3, 5]
HOLDOUT = 2                     # last 2 folds reserved for confirmation
BASELINE = dict(objective='rank:pairwise', eta=0.03, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, alpha=1.0, reg_lambda=2.0, min_child_weight=10, gamma=0.0)


def device_ok():
    try:
        d = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); d.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


def group_blocks(qids):
    _, first = np.unique(qids, return_index=True)
    starts = np.sort(first); ends = np.append(starts[1:], len(qids))
    return (ends - starts).astype(int), [np.arange(s, e) for s, e in zip(starts, ends)]


def integer_ranks(y, qids, invert):
    out = np.zeros(len(y), dtype=int)
    _, first = np.unique(qids, return_index=True)
    starts = np.sort(first); ends = np.append(starts[1:], len(qids))
    for s, e in zip(starts, ends):
        out[s:e] = rankdata(-y[s:e] if invert else y[s:e], method='ordinal') - 1
    return out


def mean_spearman(scores, ret, qgroups):
    rhos = []
    for idx in qgroups:
        if len(idx) < 2:
            continue
        a = rankdata(scores[idx]); b = rankdata(ret[idx]); a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            rhos.append(float((a * b).sum() / den))
    return float(np.mean(rhos)) if rhos else 0.0


def fixed(params, device):
    p = dict(params)
    p.update(eval_metric='ndcg@3', ndcg_exp_gain=False, tree_method='hist',
             device=device, verbosity=0, random_state=SEED)
    return p


# ── load :15 subset + build folds ────────────────────────────────────────────
print("Loading panel, filtering to :15 anchors …")
df = pd.read_parquet(DATA)
df = df[pd.to_datetime(df['DateTime']).dt.strftime('%H:%M').isin(ANCHORS)].copy()
df = df.sort_values('DateTime').reset_index(drop=True)
df['Query_ID'] = df.groupby('DateTime').ngroup()
df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
months = sorted(df['YearMonth'].unique())
exclude = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
           'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']
feat_cols = [c for c in df.columns if c not in exclude]
X = df[feat_cols].to_numpy(np.float32); np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
y = df[RET_COL].to_numpy(np.float64); qids = df['Query_ID'].to_numpy(); ym = df['YearMonth'].to_numpy()
y_long = integer_ranks(y, qids, False); y_short = integer_ranks(y, qids, True)
print(f"  :15 subset: {len(df):,} rows, {df['Query_ID'].nunique():,} queries, {len(months)} months")

MIN_TRAIN, HORIZON = 18, 2
FOLDS = []
for i in range(MIN_TRAIN, len(months) - HORIZON, 4):
    tr_m, va_m, te_m = months[:i], [months[i]], months[i + 1:i + HORIZON + 1]
    trm = np.isin(ym, tr_m); vam = np.isin(ym, va_m); tem = np.isin(ym, te_m)
    gtr, _ = group_blocks(qids[trm]); gva, _ = group_blocks(qids[vam]); _, teg = group_blocks(qids[tem])
    FOLDS.append(dict(Xtr=np.ascontiguousarray(X[trm]), ytr_l=y_long[trm], ytr_s=y_short[trm], gtr=gtr,
                      Xva=np.ascontiguousarray(X[vam]), yva_l=y_long[vam], yva_s=y_short[vam], gva=gva,
                      Xte=np.ascontiguousarray(X[tem]), te_ret=y[tem], teg=teg,
                      label=f"{te_m[0]}..{te_m[-1]}"))
OPT_FOLDS, HOLD_FOLDS = FOLDS[:-HOLDOUT], FOLDS[-HOLDOUT:]
DEVICE = device_ok()
print(f"  total folds={len(FOLDS)}  optimize on {len(OPT_FOLDS)}  HOLD OUT {len(HOLD_FOLDS)} "
      f"({', '.join(f['label'] for f in HOLD_FOLDS)})  device={DEVICE}")


def eval_fold(ls, ss, ret, qgroups):
    out = {'long_rho': mean_spearman(ls, ret, qgroups), 'short_rho': mean_spearman(ss, -ret, qgroups)}
    g = {k: {'l': [], 's': []} for k in KS}
    for idx in qgroups:
        if len(idx) < max(KS) + 1:
            continue
        r = ret[idx]; lo = np.argsort(-ls[idx]); so = np.argsort(-ss[idx])
        for k in KS:
            g[k]['l'].append(r[lo[:k]].mean()); g[k]['s'].append((-r[so[:k]]).mean())
    for k in KS:
        out[f'l_g{k}'] = float(np.mean(g[k]['l'])); out[f's_g{k}'] = float(np.mean(g[k]['s']))
    out['comb3_bps'] = (out['l_g3'] + out['s_g3']) / 2 * 1e4
    return out


def train_fold(p, F):
    dtl = xgb.DMatrix(F['Xtr'], label=F['ytr_l']); dtl.set_group(F['gtr'])
    dvl = xgb.DMatrix(F['Xva'], label=F['yva_l']); dvl.set_group(F['gva'])
    bl = xgb.train(p, dtl, NUM_ROUNDS, evals=[(dvl, 'v')], early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    dts = xgb.DMatrix(F['Xtr'], label=F['ytr_s']); dts.set_group(F['gtr'])
    dvs = xgb.DMatrix(F['Xva'], label=F['yva_s']); dvs.set_group(F['gva'])
    bs = xgb.train(p, dts, NUM_ROUNDS, evals=[(dvs, 'v')], early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    dte = xgb.DMatrix(F['Xte'])
    ls = bl.predict(dte, iteration_range=(0, bl.best_iteration + 1))
    ss = bs.predict(dte, iteration_range=(0, bs.best_iteration + 1))
    return eval_fold(ls, ss, F['te_ret'], F['teg'])


def objective(trial):
    t0 = time.time()
    p = fixed(dict(
        objective=trial.suggest_categorical('objective', ['rank:pairwise', 'rank:ndcg']),
        eta=trial.suggest_float('eta', 0.01, 0.3, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 8),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        alpha=trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 30),
        gamma=trial.suggest_float('gamma', 0.0, 5.0),
    ), DEVICE)
    comb = []
    for fi, F in enumerate(OPT_FOLDS):
        comb.append(train_fold(p, F)['comb3_bps'])
        trial.report(float(np.mean(comb)), fi)
        if trial.should_prune():
            raise optuna.TrialPruned()
    obj = 0.5 * float(np.mean(comb)) + 0.5 * float(np.min(comb))
    trial.set_user_attr('mean_comb3_bps', float(np.mean(comb)))
    trial.set_user_attr('worst_comb3_bps', float(np.min(comb)))
    trial.set_user_attr('secs', round(time.time() - t0, 1))
    print(f"  trial {trial.number}: obj={obj:+.2f}bps mean={np.mean(comb):+.2f} worst={np.min(comb):+.2f} "
          f"({p['objective']} d{p['max_depth']}) ({trial.user_attrs['secs']}s)")
    return obj


def confirm(params, tag):
    """Full breakdown on the HELD-OUT folds (rho + Top-K gross/net)."""
    p = fixed(params, DEVICE)
    rows = [train_fold(p, F) for F in HOLD_FOLDS]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"\n[{tag} | HELD-OUT {', '.join(f['label'] for f in HOLD_FOLDS)}] "
          f"rank-IC L={agg['long_rho']:+.4f} S={agg['short_rho']:+.4f}")
    for k in KS:
        lg, sg = agg[f'l_g{k}'] * 1e4, agg[f's_g{k}'] * 1e4
        for c in COSTS:
            print(f"   K={k} @{int(c)}bps: LONG net={lg-c:+.2f} (gross {lg:+.2f})  "
                  f"SHORT net={sg-c:+.2f} (gross {sg:+.2f})")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_trials', type=int, default=100)
    ap.add_argument('--timeout', type=int, default=5400)
    ap.add_argument('--study', default='v20_topk')
    args = ap.parse_args()
    os.makedirs('artifacts', exist_ok=True)
    storage = 'sqlite:///artifacts/optuna_v20_topk_study.db'
    study = optuna.create_study(study_name=args.study, storage=storage, load_if_exists=True,
                                direction='maximize', sampler=TPESampler(seed=SEED, n_startup_trials=12),
                                pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=2))
    if not study.trials:
        study.enqueue_trial(BASELINE)
        print("enqueued BASELINE (current v20 params) as trial 0")
    print(f"study '{args.study}'  n_trials={args.n_trials} timeout={args.timeout}s")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    bt = study.best_trial
    print(f"\n{'='*72}\nfinished: {len(study.trials)} trials, best opt-obj={bt.value:+.2f}bps")
    print(f"BEST params: {bt.params}\n  attrs: {bt.user_attrs}")
    print("=" * 72 + "\nHELD-OUT CONFIRMATION (the honest test)\n" + "=" * 72)
    base_h = confirm(BASELINE, 'BASELINE')
    best_h = confirm(bt.params, 'TUNED-BEST')
    print("\n" + "=" * 72 + "\nLIFT on held-out (tuned − baseline)\n" + "=" * 72)
    for k in KS:
        dl = (best_h[f'l_g{k}'] - base_h[f'l_g{k}']) * 1e4
        ds = (best_h[f's_g{k}'] - base_h[f's_g{k}']) * 1e4
        print(f"  K={k} gross Δ: LONG {dl:+.2f}bps  SHORT {ds:+.2f}bps  "
              f"(tuned LONG net@6 {best_h[f'l_g{k}']*1e4-6:+.2f}, @10 {best_h[f'l_g{k}']*1e4-10:+.2f})")
    json.dump({'best_params': bt.params, 'best_opt_obj_bps': bt.value, 'best_attrs': bt.user_attrs,
               'heldout_baseline': base_h, 'heldout_tuned': best_h,
               'heldout_folds': [f['label'] for f in HOLD_FOLDS], 'n_trials': len(study.trials)},
              open('artifacts/optuna_v20_topk_best.json', 'w'), indent=2, default=float)
    print("\nsaved -> artifacts/optuna_v20_topk_best.json")


if __name__ == '__main__':
    main()
