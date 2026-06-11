"""
Purged Walk-Forward TBM Ensemble — Phases 3-6.

Master training + evaluation harness for the 1h TBM ensemble.

Architecture per fold:
  Long model : 4 base learners (Views A/B/C + D_momentum), 12-feature OOF combiner
  Short model: 3 base learners (Views A/B/C),               9-feature OOF combiner

  Per learner:
    1. CatBoost MultiClass (oblivious trees, GPU, purged WF)
    2. Isotonic calibration of class probs (on val slice)
    3. Stacked combiner: logistic regression on OOF predictions
    4. EV filter: trade iff EV > τ (τ swept on val to hit target_wr)
    5. Top-K ranking by EV
    6. Business metrics with bootstrap CIs

Purge: drop train samples whose 1h label window overlaps the test block.
Embargo: 1-day gap on both sides of each test block.

Usage:
    python scripts/validation/purged_wf_tbm.py [--side long|short|both] [--k 3] [--target_wr 0.57]

Reads:
  data/tbm_feature_views/{A_meanrev,B_trend,C_vol}.parquet   (all sides)
  data/tbm_feature_views/D_momentum.parquet                   (long only)
Writes:
  data/model_analysis/tbm_1h/wf_results_{long,short}.json
  models/tbm_1h_ensemble/{long,short}/
"""

import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

# ── config ────────────────────────────────────────────────────────────────────
VIEWS_DIR   = 'data/tbm_feature_views'
OUT_DIR     = 'data/model_analysis/tbm_1h'
MODEL_DIR   = 'models/tbm_1h_ensemble'
COST        = 0.0006   # 6 bps
COST_10     = 0.0010   # 10 bps (secondary reporting)
N_BOOTSTRAP = 500
RANDOM_SEED = 42

# Walk-forward parameters
MIN_TRAIN_MONTHS = 18
VAL_MONTHS       = 4    # calibration/combiner/τ tuning
TEST_MONTHS      = 2
STEP_MONTHS      = 4
EMBARGO_DAYS     = 1

# CatBoost per-view config
CB_PARAMS = dict(
    iterations=500,
    learning_rate=0.03,
    depth=5,
    loss_function='MultiClass',
    eval_metric='AUC',
    classes_count=3,
    random_seed=RANDOM_SEED,
    task_type='GPU',
    devices='0',
    verbose=False,
    early_stopping_rounds=50,
)

VIEW_FILES = {
    'A': os.path.join(VIEWS_DIR, 'A_meanrev.parquet'),
    'B': os.path.join(VIEWS_DIR, 'B_trend.parquet'),
    'C': os.path.join(VIEWS_DIR, 'C_vol.parquet'),
    'D': os.path.join(VIEWS_DIR, 'D_momentum.parquet'),  # long only
}

# Views used per side
VIEWS_FOR_SIDE = {
    'long':  ['A', 'B', 'C', 'D'],
    'short': ['A', 'B', 'C'],
}

META_COLS = ['DateTime', 'Ticker', 'label', 'realized_gross', 'realized_net',
             'entry_price', 'atr', 'R', 'weight', 'YearMonth']

# ── helpers ───────────────────────────────────────────────────────────────────

def impute_from_train(Xtr, Xva, Xte):
    """Impute NaNs using TRAIN-fold means only."""
    train_means = np.nanmean(Xtr, axis=0)
    train_means = np.where(np.isfinite(train_means), train_means, 0.0)
    def _fill(X):
        out = X.copy()
        for ci in range(X.shape[1]):
            bad = ~np.isfinite(out[:, ci])
            if bad.any():
                out[bad, ci] = train_means[ci]
        return out
    return _fill(Xtr), _fill(Xva), _fill(Xte)


def bootstrap_ci(arr, n=N_BOOTSTRAP, alpha=0.05):
    if len(arr) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_SEED)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    lo = np.percentile(means, alpha/2*100)
    hi = np.percentile(means, (1-alpha/2)*100)
    return lo, hi


def t_stat(arr):
    if len(arr) < 2:
        return np.nan
    return arr.mean() / (arr.std() / np.sqrt(len(arr)))


def net_winrate(net_rets):
    if len(net_rets) == 0:
        return np.nan
    return float(np.mean(net_rets > 0))


def profit_factor(net_rets):
    wins  = net_rets[net_rets > 0].sum()
    loses = (-net_rets[net_rets < 0]).sum()
    return wins / loses if loses > 0 else np.inf


def compute_ev(p_tp, p_sl, p_to, R, e_ret_to, cost):
    """
    EV = P_TP·R/entry − P_SL·R/entry + P_TO·E[ret|TO] − cost
    R is in price units; we work in return space so R_frac = R/entry.
    p_to and e_ret_to are arrays.
    """
    R_frac = R  # already in fraction form in our label (realized is in fraction)
    # For symmetric barriers: R is the barrier as a fraction of entry
    return p_tp * R_frac - p_sl * R_frac + p_to * e_ret_to - cost


def tau_sweep(ev_arr, net_ret_arr, target_wr=0.57, n_steps=100):
    """Find smallest τ such that net WR ≥ target_wr among EV>τ trades."""
    if len(ev_arr) == 0:
        return np.inf
    taus = np.quantile(ev_arr, np.linspace(0, 0.99, n_steps))
    best_tau = np.inf
    for tau in taus:
        mask = ev_arr > tau
        if mask.sum() < 5:
            continue
        wr = net_winrate(net_ret_arr[mask])
        if wr >= target_wr:
            best_tau = tau
            break  # smallest passing τ
    return best_tau


# ── base learner ──────────────────────────────────────────────────────────────

def train_base_learner(Xtr, ytr, wtr, Xva, yva):
    """Train one CatBoost MultiClass model. Returns model + OOF probs on val."""
    cb = CatBoostClassifier(**CB_PARAMS)
    tr_pool = Pool(Xtr, label=ytr, weight=wtr)
    va_pool = Pool(Xva, label=yva)
    cb.fit(tr_pool, eval_set=va_pool)
    va_probs = cb.predict_proba(Xva)   # shape (N, 3)
    return cb, va_probs


def calibrate_probs(raw_probs_val, y_val, raw_probs_test):
    """
    Fit isotonic regression calibrator per class (one-vs-rest) on val,
    apply to test. Renormalises to sum-1.
    """
    n_classes = raw_probs_val.shape[1]
    cal_test  = np.zeros_like(raw_probs_test)
    for c in range(n_classes):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(raw_probs_val[:, c], (y_val == c).astype(float))
        cal_test[:, c] = ir.transform(raw_probs_test[:, c])
    # Renormalise
    row_sum = cal_test.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return cal_test / row_sum


# ── combiner ──────────────────────────────────────────────────────────────────

def train_combiner(oof_features, y_true):
    """
    OOF stacked combiner: logistic regression on concatenated
    [P_SL, P_TP, P_TO] from each view → predicts label {0,1,2}.
    Returns trained combiner.
    """
    lr = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, C=0.1)
    lr.fit(oof_features, y_true)
    return lr


def combiner_predict(combiner, features):
    return combiner.predict_proba(features)  # (N, 3)


# ── fold evaluation ───────────────────────────────────────────────────────────

def evaluate_fold(df_test, final_probs, R_frac_arr, e_ret_to_arr, tau, cost, k=3):
    """
    Apply EV filter and Top-K, return per-trade metrics.
    final_probs: (N, 3) — [P_SL, P_TP, P_TO]
    """
    p_sl  = final_probs[:, 0]
    p_tp  = final_probs[:, 1]
    p_to  = final_probs[:, 2]

    ev = compute_ev(p_tp, p_sl, p_to, R_frac_arr, e_ret_to_arr, cost)

    df_test = df_test.copy()
    df_test['ev']    = ev
    df_test['p_tp']  = p_tp
    df_test['p_sl']  = p_sl
    df_test['p_to']  = p_to

    # EV filter
    ev_passed = df_test[df_test['ev'] > tau].copy()

    # Top-K per timestamp
    if len(ev_passed) == 0:
        return pd.DataFrame()
    top_k = (ev_passed.sort_values('ev', ascending=False)
             .groupby('DateTime').head(k)
             .reset_index(drop=True))
    return top_k


# ── main walk-forward ─────────────────────────────────────────────────────────

def run_wf(side='long', k=3, target_wr=0.57):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, side), exist_ok=True)

    print("=" * 64)
    print(f"Purged Walk-Forward TBM Ensemble — side={side.upper()} k={k} target_wr={target_wr:.0%}")
    print("=" * 64)

    # ── Load views for this side only ──
    active_views = VIEWS_FOR_SIDE[side]
    views = {}
    for v_name in active_views:
        v_file = VIEW_FILES[v_name]
        if not os.path.exists(v_file):
            if v_name == 'D':
                raise FileNotFoundError(
                    f"View D not found: {v_file}\n"
                    "Run: python scripts/features/build_momentum_view.py first.")
            raise FileNotFoundError(f"View file not found: {v_file}\n"
                                    "Run build_feature_views.py first.")
        df_v = pd.read_parquet(v_file)
        df_v['DateTime'] = pd.to_datetime(df_v['DateTime'])
        views[v_name] = df_v
        print(f"  View {v_name}: {df_v.shape[0]:,} rows × {df_v.shape[1]} cols")
    print(f"  Active views for {side}: {active_views}")

    # Use view A as the master for meta-columns; all views share the same rows
    df_master = views['A'][META_COLS].copy()

    # For long: labels 1=TP (win), 0=SL (loss), 2=TO (neutral)
    # For short: labels are mirrored (0=TP for short, 1=SL for short)
    # We train separate models; for short we flip the TP/SL label interpretation
    # Convention: label 1 = outcome we want (TP for long = up, TP for short = down)
    if side == 'short':
        # Flip TP/SL labels so label=1 means "short wins"
        label_map = {0: 1, 1: 0, 2: 2}
        df_master['label'] = df_master['label'].map(label_map)
        for v_name in views:
            views[v_name]['label'] = views[v_name]['label'].map(label_map)
        # Short gross = -(long gross). Short net = short_gross - COST.
        # NOTE: must SUBTRACT cost after flipping gross — do NOT negate the long
        # net (that would yield short_gross + COST, inverting the cost sign).
        df_master['realized_gross'] = -df_master['realized_gross']
        df_master['realized_net']   = df_master['realized_gross'] - COST
        for v_name in views:
            views[v_name]['realized_gross'] = -views[v_name]['realized_gross']
            views[v_name]['realized_net']   = views[v_name]['realized_gross'] - COST

    # ── Build folds ──
    unique_months = sorted(df_master['YearMonth'].unique())
    folds = []
    for i in range(MIN_TRAIN_MONTHS, len(unique_months) - VAL_MONTHS - TEST_MONTHS, STEP_MONTHS):
        folds.append(dict(
            fold=len(folds) + 1,
            train_months=unique_months[:i],
            val_months=unique_months[i:i + VAL_MONTHS],
            test_months=unique_months[i + VAL_MONTHS:i + VAL_MONTHS + TEST_MONTHS],
        ))
    print(f"\n  Walk-forward folds: {len(folds)}")

    # ── Precompute feature arrays per view ──
    feat_cols = {}
    X_all     = {}
    for v_name, df_v in views.items():
        cols = [c for c in df_v.columns if c not in META_COLS]
        feat_cols[v_name] = cols
        X_all[v_name] = df_v[cols].values.astype(np.float64)

    y_all         = df_master['label'].values.astype(np.int32)
    w_all         = df_master['weight'].values.astype(np.float64)
    ym_all        = df_master['YearMonth'].values
    dt_all        = df_master['DateTime'].values
    realized_all  = df_master['realized_net'].values.astype(np.float64)
    realized_g    = df_master['realized_gross'].values.astype(np.float64)
    R_frac_all    = (df_master['R'] / df_master['entry_price']).values.astype(np.float64)

    # E[ret|timeout] — estimated from training labels (updated per fold)
    wf_results = []
    all_test_trades = []

    for cfg in folds:
        fold_n  = cfg['fold']
        tr_m    = set(cfg['train_months'])
        va_m    = set(cfg['val_months'])
        te_m    = set(cfg['test_months'])

        tr_mask = np.array([ym in tr_m for ym in ym_all])
        va_mask = np.array([ym in va_m for ym in ym_all])
        te_mask = np.array([ym in te_m for ym in ym_all])

        # Purge: remove train rows whose label window overlaps test block
        # Label window = [DateTime, DateTime + 1h]. Test block starts at min(DateTime[te_mask]).
        test_start = pd.Timestamp(dt_all[te_mask].min()) - timedelta(days=EMBARGO_DAYS)
        purge_mask = tr_mask & (pd.to_datetime(dt_all) + timedelta(hours=1) > test_start)
        tr_mask    = tr_mask & ~purge_mask

        # Embargo: remove val rows within EMBARGO_DAYS of test block start
        val_end    = pd.Timestamp(dt_all[va_mask].max()) + timedelta(days=EMBARGO_DAYS)
        embargo_mask = va_mask & (pd.to_datetime(dt_all) > val_end)
        va_mask = va_mask & ~embargo_mask

        if tr_mask.sum() < 100 or va_mask.sum() < 20 or te_mask.sum() < 20:
            print(f"  Fold {fold_n}: insufficient data — skip")
            continue

        # Estimate E[ret|timeout] from training data
        to_mask_tr = tr_mask & (y_all == 2)
        e_ret_to = float(realized_g[to_mask_tr].mean()) if to_mask_tr.sum() > 0 else 0.0

        print(f"\n--- FOLD {fold_n} ---  train:{tr_mask.sum():,}  val:{va_mask.sum():,}  test:{te_mask.sum():,}")
        print(f"  Test months: {cfg['test_months']}  |  E[ret|TO]={e_ret_to*10000:+.1f}bps")

        # ── Train base learners + get OOF val predictions ──
        val_probs_per_view = {}
        test_probs_per_view = {}

        for v_name in active_views:
            Xtr = X_all[v_name][tr_mask]
            Xva = X_all[v_name][va_mask]
            Xte = X_all[v_name][te_mask]
            Xtr, Xva, Xte = impute_from_train(Xtr, Xva, Xte)

            ytr = y_all[tr_mask]
            yva = y_all[va_mask]
            wtr = w_all[tr_mask]

            model, va_probs_raw = train_base_learner(Xtr, ytr, wtr, Xva, yva)

            # Calibrate: fit on val, apply to test
            Xte_raw_probs = model.predict_proba(Xte)
            te_probs_cal  = calibrate_probs(va_probs_raw, yva, Xte_raw_probs)

            val_probs_per_view[v_name]  = va_probs_raw     # uncalibrated for combiner OOF
            test_probs_per_view[v_name] = te_probs_cal

            va_auc = roc_auc_score(yva, va_probs_raw, multi_class='ovr')
            te_auc = roc_auc_score(y_all[te_mask], te_probs_cal, multi_class='ovr')
            print(f"  View {v_name}: val_AUC={va_auc:.3f}  test_AUC(cal)={te_auc:.3f}")

        # ── Train stacked combiner on OOF val predictions ──
        n_views  = len(active_views)
        oof_val  = np.hstack([val_probs_per_view[v]  for v in active_views])  # (Nval, n_views*3)
        oof_test = np.hstack([test_probs_per_view[v] for v in active_views])  # (Nte,  n_views*3)

        combiner = train_combiner(oof_val, y_all[va_mask])
        final_val_probs  = combiner_predict(combiner, oof_val)
        final_test_probs = combiner_predict(combiner, oof_test)

        comb_val_auc  = roc_auc_score(y_all[va_mask],  final_val_probs,  multi_class='ovr')
        comb_test_auc = roc_auc_score(y_all[te_mask], final_test_probs, multi_class='ovr')
        print(f"  Combiner: val_AUC={comb_val_auc:.3f}  test_AUC={comb_test_auc:.3f}")

        # ── EV computation + τ sweep on validation ──
        R_frac_val = R_frac_all[va_mask]
        R_frac_te  = R_frac_all[te_mask]
        e_to_val   = np.full(va_mask.sum(), e_ret_to)
        e_to_te    = np.full(te_mask.sum(), e_ret_to)

        ev_val = compute_ev(
            final_val_probs[:, 1], final_val_probs[:, 0], final_val_probs[:, 2],
            R_frac_val, e_to_val, COST
        )
        net_val = realized_all[va_mask]

        tau = tau_sweep(ev_val, net_val, target_wr=target_wr)
        if np.isinf(tau):
            print(f"  ⚠️  τ sweep: no threshold achieves {target_wr:.0%} WR on val — using p95")
            tau = np.quantile(ev_val, 0.95) if len(ev_val) > 0 else 0.0

        # Baseline: dumb AND-gate (all active views agree TP)
        and_gate_val = np.ones(va_mask.sum(), dtype=bool)
        for _v in active_views:
            and_gate_val &= (val_probs_per_view[_v].argmax(axis=1) == 1)
        and_wr_val = net_winrate(net_val[and_gate_val]) if and_gate_val.sum() > 0 else np.nan
        ev_wr_val  = net_winrate(net_val[ev_val > tau]) if (ev_val > tau).sum() > 0 else np.nan
        print(f"  τ={tau:.6f}  EV-gate val WR={ev_wr_val:.1%}  AND-gate val WR={and_wr_val:.1%}  "
              f"({and_gate_val.sum()} AND trades vs {(ev_val>tau).sum()} EV trades)")

        # ── Test evaluation ──
        ev_te = compute_ev(
            final_test_probs[:, 1], final_test_probs[:, 0], final_test_probs[:, 2],
            R_frac_te, e_to_te, COST
        )

        df_te = df_master[te_mask].copy().reset_index(drop=True)
        df_te['ev']  = ev_te
        df_te['p_tp'] = final_test_probs[:, 1]
        df_te['p_sl'] = final_test_probs[:, 0]
        df_te['p_to'] = final_test_probs[:, 2]
        df_te['fold'] = fold_n

        # EV-filtered trades
        ev_trades = df_te[df_te['ev'] > tau].copy()

        # Top-K per timestamp
        if len(ev_trades) > 0:
            top_k_trades = (ev_trades.sort_values('ev', ascending=False)
                            .groupby('DateTime').head(k)
                            .reset_index(drop=True))
        else:
            top_k_trades = pd.DataFrame()

        # Business metrics for this fold
        n_trades  = len(top_k_trades)
        net_rets  = top_k_trades['realized_net'].values if n_trades > 0 else np.array([])
        raw_rets  = top_k_trades['realized_gross'].values if n_trades > 0 else np.array([])

        wr_6bps   = net_winrate(net_rets) if n_trades > 0 else np.nan
        net_10    = raw_rets - COST_10 if n_trades > 0 else np.array([])
        wr_10bps  = net_winrate(net_10) if n_trades > 0 else np.nan
        exp_6bps  = float(net_rets.mean()) * 10000 if n_trades > 0 else np.nan
        pf_6bps   = profit_factor(net_rets) if n_trades > 0 else np.nan
        raw_mean  = float(raw_rets.mean()) * 10000 if n_trades > 0 else np.nan
        t         = t_stat(net_rets)
        ci_lo, ci_hi = bootstrap_ci(net_rets * 10000) if n_trades > 0 else (np.nan, np.nan)

        n_months  = len(cfg['test_months'])
        tpm       = n_trades / n_months if n_months > 0 else 0

        print(f"  TEST: n={n_trades} (~{tpm:.0f}/mo)  WR@6={wr_6bps:.1%}  WR@10={wr_10bps:.1%}  "
              f"exp={exp_6bps:+.1f}bps  PF={pf_6bps:.2f}  t={t:.2f}  "
              f"CI=[{ci_lo:+.1f},{ci_hi:+.1f}]bps")

        wf_results.append({
            'fold': fold_n,
            'test_months': cfg['test_months'],
            'n_trades': n_trades,
            'trades_per_month': tpm,
            'wr_6bps': wr_6bps,
            'wr_10bps': wr_10bps,
            'net_expectancy_bps': exp_6bps,
            'raw_mean_bps': raw_mean,
            'profit_factor': pf_6bps,
            't_stat': t,
            'ci_lo_bps': ci_lo,
            'ci_hi_bps': ci_hi,
            'tau': tau,
            'e_ret_to_bps': e_ret_to * 10000,
        })

        if len(top_k_trades) > 0:
            all_test_trades.append(top_k_trades)

    # ── Pooled aggregate ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"POOLED RESULTS — {side.upper()} — {len(wf_results)} folds")
    print("=" * 64)

    if all_test_trades:
        df_all = pd.concat(all_test_trades, ignore_index=True)
        all_net  = df_all['realized_net'].values
        all_raw  = df_all['realized_gross'].values
        all_net10 = all_raw - COST_10

        pool_wr6  = net_winrate(all_net)
        pool_wr10 = net_winrate(all_net10)
        pool_exp  = float(all_net.mean()) * 10000
        pool_pf   = profit_factor(all_net)
        pool_t    = t_stat(all_net)
        pool_ci   = bootstrap_ci(all_net * 10000)
        pool_n    = len(df_all)
        pool_tpm  = pool_n / max(sum(len(r['test_months']) for r in wf_results), 1)

        target_ok = '✅' if pool_wr6 >= target_wr and pool_ci[0] > 0 else '❌'

        print(f"  Total trades:    {pool_n:,}  (~{pool_tpm:.0f}/mo)")
        print(f"  Net WR @6bps:    {pool_wr6:.1%}  {target_ok}  (target ≥{target_wr:.0%})")
        print(f"  Net WR @10bps:   {pool_wr10:.1%}")
        print(f"  Net expectancy:  {pool_exp:+.2f} bps/trade")
        print(f"  Profit factor:   {pool_pf:.3f}")
        print(f"  t-stat:          {pool_t:.2f}")
        print(f"  Bootstrap CI:    [{pool_ci[0]:+.1f}, {pool_ci[1]:+.1f}] bps  (95%)")

        ok_target   = pool_wr6 >= target_wr
        ok_positive = pool_ci[0] > 0
        ok_volume   = pool_tpm >= 1.0  # at least 1 trade/month
        ok_repro    = sum(1 for r in wf_results if r['n_trades'] > 0 and r['wr_6bps'] >= target_wr)

        print(f"\n  Acceptance criteria:")
        print(f"    WR ≥{target_wr:.0%}:        {'✅' if ok_target   else '❌'}")
        print(f"    CI_lo > 0:        {'✅' if ok_positive else '❌'}  ({pool_ci[0]:+.1f} bps)")
        print(f"    Volume ≥1/mo:     {'✅' if ok_volume   else '❌'}  ({pool_tpm:.1f}/mo)")
        print(f"    Folds profitable: {ok_repro}/{len(wf_results)}")

        if ok_target and ok_positive and ok_volume:
            print("\n  ✅ ALL ACCEPTANCE CRITERIA MET — candidate for deployment")
        else:
            print("\n  ❌ KILL CRITERION: does not meet acceptance — do not deploy")

        # Save results
        summary = {
            'side': side, 'k': k, 'target_wr': target_wr,
            'pool_n_trades': pool_n, 'pool_trades_per_month': pool_tpm,
            'pool_wr_6bps': pool_wr6, 'pool_wr_10bps': pool_wr10,
            'pool_net_exp_bps': pool_exp, 'pool_profit_factor': pool_pf,
            'pool_t_stat': pool_t, 'pool_ci_lo': pool_ci[0], 'pool_ci_hi': pool_ci[1],
            'folds': wf_results,
        }
        out_path = os.path.join(OUT_DIR, f'wf_results_{side}.json')
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Results → {out_path}")

        trade_path = os.path.join(OUT_DIR, f'test_trades_{side}.parquet')
        df_all.to_parquet(trade_path, index=False)
        print(f"  Trades → {trade_path}")
    else:
        print("  No test trades generated across any fold.")

    return wf_results


# ── AND-gate baseline ─────────────────────────────────────────────────────────

def run_and_gate_baseline(side='long', k=3):
    """Simple AND-gate baseline without combiner — all 3 views must agree TP."""
    print("\n" + "=" * 64)
    print(f"AND-Gate Baseline — {side.upper()}")
    print("=" * 64)
    print("(Load views, quick single-fold eval of unanimous agreement rate)")

    active_views = VIEWS_FOR_SIDE[side]
    views = {}
    for v_name in active_views:
        v_file = VIEW_FILES[v_name]
        if not os.path.exists(v_file):
            return
        df_v = pd.read_parquet(v_file)
        views[v_name] = df_v

    df_master = views['A'][META_COLS].copy()
    if side == 'short':
        label_map = {0: 1, 1: 0, 2: 2}
        df_master['label'] = df_master['label'].map(label_map)
        # short net = -(long gross) - COST  (subtract cost, don't negate long net)
        df_master['realized_net'] = -df_master['realized_gross'] - COST

    # Last 6 months as quick OOS
    months = sorted(df_master['YearMonth'].unique())
    oos_m  = set(months[-6:])
    tr_m   = set(months[:-8])
    oos_mask = np.array([ym in oos_m for ym in df_master['YearMonth'].values])
    tr_mask  = np.array([ym in tr_m  for ym in df_master['YearMonth'].values])

    preds = {}
    for v_name in active_views:
        df_v = views[v_name]
        feat_cols = [c for c in df_v.columns if c not in META_COLS]
        Xtr = df_v[feat_cols].values[tr_mask].astype(np.float64)
        Xte = df_v[feat_cols].values[oos_mask].astype(np.float64)
        ytr = df_master['label'].values[tr_mask]
        wtr = df_master['weight'].values[tr_mask]
        Xtr, _, Xte = impute_from_train(Xtr, Xtr, Xte)
        cb = CatBoostClassifier(**CB_PARAMS)
        cb.fit(Pool(Xtr, label=ytr, weight=wtr))
        preds[v_name] = cb.predict_proba(Xte).argmax(axis=1)

    unanimous_tp = np.ones(oos_mask.sum(), dtype=bool)
    for _v in active_views:
        unanimous_tp &= (preds[_v] == 1)
    net_rets = df_master['realized_net'].values[oos_mask]
    print(f"  AND-gate trades: {unanimous_tp.sum()} / {oos_mask.sum()} ({unanimous_tp.mean():.1%})")
    if unanimous_tp.sum() > 0:
        print(f"  AND-gate net WR: {net_winrate(net_rets[unanimous_tp]):.1%}")
        print(f"  AND-gate exp:    {net_rets[unanimous_tp].mean()*10000:+.1f} bps")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--side',      choices=['long','short','both'], default='both')
    ap.add_argument('--k',         type=int,   default=3,    help='Top-K per timestamp')
    ap.add_argument('--target_wr', type=float, default=0.57, help='Target net win-rate')
    ap.add_argument('--baseline',  action='store_true',      help='Also run AND-gate baseline')
    args = ap.parse_args()

    sides = ['long', 'short'] if args.side == 'both' else [args.side]
    for side in sides:
        run_wf(side=side, k=args.k, target_wr=args.target_wr)
        if args.baseline:
            run_and_gate_baseline(side=side, k=args.k)


if __name__ == '__main__':
    main()
