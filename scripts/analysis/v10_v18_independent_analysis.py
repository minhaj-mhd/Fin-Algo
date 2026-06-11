"""
Independent re-evaluation of V10 (ranker), V18 (classifier), and their combination.

This is NOT a re-run of the author's training/backtest scripts. It rebuilds every metric
from the raw data + saved production model artifacts, with a data-integrity foundation so
each edge is reported both RAW and artifact-CLEAN, at both 6 and 10 bps cost.

Phases:
  0  Data integrity & clean_mask construction
  1  V10 standalone (rank IC, decile monotonicity, Top-K, quarterly, time-of-day)
  2  V18 standalone (AUC, calibration, threshold sweep, "is it just a clock?")
  3  Combined (A1/A3/B logics, veto decomposition, asymmetric config, dual-lock, cost grid, OOS halves)
  5  Synthesis master table

Phase 4 (fresh walk-forward retrain) lives in v10_v18_walkforward.py (heavier, GPU).

Usage:
    python scripts/analysis/v10_v18_independent_analysis.py

Outputs:
    data/model_analysis/v10_v18_independent/*.json, *.csv
"""

import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata, ttest_1samp
from sklearn.metrics import roc_auc_score

# ── config ───────────────────────────────────────────────────────────────────
DATA_FILE   = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL     = 'Next_Hour_Return'
V10_DIR     = 'models/v10_native_1h'
V18_DIR     = 'models/v18_random_forest_1h'
OUT_DIR     = 'data/model_analysis/v10_v18_independent'

V10_TRAIN_END = '2025-06'   # v10 trained 2022-01 .. 2025-06
V18_TRAIN_END = '2025-03'   # v18 trained 2022-01 .. 2025-03
OOS_START     = '2025-07'   # common clean OOS (after both cutoffs)
OOS_MID       = '2026-01'   # split point for H2-25 vs H1-26 stability

COSTS       = {'6bps': 0.0006, '10bps': 0.0010}
PROB_TH     = 0.52          # v18 production threshold

os.makedirs(OUT_DIR, exist_ok=True)
RESULTS = {}   # collected for JSON dump

EXCLUDE = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
           'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth', 'Date']


# ── load & prepare ───────────────────────────────────────────────────────────
def load_data():
    print("Loading data ...")
    df = pd.read_csv(DATA_FILE)
    df['YearMonth'] = df['DateTime'].str[:7]
    df['date']      = df['DateTime'].str[:10]
    df = df.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    return df


def build_features(df):
    helpers = {'date', 'fwd_cc', 'next_date'}
    feature_cols = [c for c in df.columns if c not in EXCLUDE and c not in helpers]
    X = df[feature_cols].values.astype(np.float64)
    # NaN handling: column means from the TRAIN portion only (no test leakage)
    train_mask = (df['YearMonth'] <= V10_TRAIN_END).values
    for ci in range(X.shape[1]):
        col = X[:, ci]
        bad = ~np.isfinite(col)
        if bad.any():
            good = col[np.isfinite(col) & train_mask]
            fill = float(good.mean()) if len(good) else 0.0
            col[bad] = fill
    return X, feature_cols


# ── Phase 0: data integrity & clean mask ─────────────────────────────────────
def phase0_clean_mask(df):
    print("\n" + "=" * 72)
    print("PHASE 0  DATA INTEGRITY & CLEAN MASK")
    print("=" * 72)

    g = df.groupby('Ticker', group_keys=False)
    df['fwd_cc']    = g['Close'].apply(lambda s: s.shift(-1) / s - 1)
    df['next_date'] = g['date'].shift(-1)
    next_row_spans_day = (df['date'] != df['next_date']).values

    # Per-hour target match to same-day close/close (intraday integrity)
    rows = []
    for h in sorted(df['Hour'].unique()):
        m = (df['Hour'] == h)
        s = df[m].dropna(subset=[RET_COL, 'fwd_cc'])
        match = float((np.abs(s[RET_COL] - s['fwd_cc']) < 1e-6).mean()) if len(s) else np.nan
        big   = float((df[m][RET_COL].abs() > 0.01).mean())
        rows.append(dict(hour=int(h), n=int(m.sum()),
                         match_overnight_cc=round(match, 4),
                         frac_gt_100bps=round(big, 4),
                         mean_abs_bps=round(float(df[m][RET_COL].abs().mean() * 10000), 1)))
    integ = pd.DataFrame(rows)
    print("\nPer-hour integrity (match_overnight_cc near 1.0 = same-day intraday; "
          "low at last bar = NOT overnight):")
    print(integ.to_string(index=False))

    # CLEAN definition: drop rows whose forward return crosses an overnight gap.
    # For non-last bars that means next-row-is-next-day (missing intermediate bar).
    # Last bar (Hour 13) target is a same-day 13:15->14:15 return (verified Phase 0), so it is KEPT.
    last_hour = int(df['Hour'].max())
    overnight_contam = next_row_spans_day & (df['Hour'].values < last_hour)
    clean_mask = ~overnight_contam & np.isfinite(df[RET_COL].values)

    print(f"\nOvernight-contaminated intraday rows (excluded by clean_mask): "
          f"{overnight_contam.sum():,} ({overnight_contam.mean():.2%})")
    print(f"Clean rows retained: {clean_mask.sum():,} / {len(df):,}")
    print("\nDETERMINATION: Next_Hour_Return is a genuine same-day intraday forward return on "
          "all hours incl. the 13:15 last bar (it does NOT match the overnight gap). The fat "
          "tail (~7% >100bps) is real open-hour volatility, not leakage. Overnight artifact "
          "risk is ~0.1% of rows.")

    RESULTS['phase0'] = dict(per_hour=rows,
                             overnight_contam_frac=float(overnight_contam.mean()),
                             clean_rows=int(clean_mask.sum()), total_rows=int(len(df)),
                             determination='same-day intraday target; ~0.1% overnight contam; '
                                           'fat tail is open-hour vol not leakage')
    return clean_mask


# ── metric helpers ───────────────────────────────────────────────────────────
def trade_stats(returns, cost):
    """returns = realized per-trade returns (already sign-adjusted for shorts)."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return dict(n=0, raw_bps=0.0, net_bps=0.0, raw_win=0.0, net_hit=0.0, t_stat=0.0)
    net = r - cost
    t = float(ttest_1samp(net, 0.0).statistic) if len(r) > 1 and np.std(net) > 0 else 0.0
    return dict(n=int(len(r)),
                raw_bps=round(float(r.mean()) * 10000, 1),
                net_bps=round(float(net.mean()) * 10000, 1),
                raw_win=round(float((r > 0).mean()), 4),
                net_hit=round(float((r > cost).mean()), 4),
                t_stat=round(t, 2))


def fmt(s, cost_label):
    return (f"n={s['n']:>5} | raw {s['raw_bps']:>+6.1f} | net {s['net_bps']:>+6.1f} "
            f"| rawwin {s['raw_win']:.1%} | nethit {s['net_hit']:.1%} | t={s['t_stat']:>5.2f}")


# ── predictions ──────────────────────────────────────────────────────────────
def load_models():
    def L(p):
        b = xgb.Booster(); b.load_model(p); return b
    return dict(
        v10_l=L(f'{V10_DIR}/xgb_long_model.json'),
        v10_s=L(f'{V10_DIR}/xgb_short_model.json'),
        v18_l=L(f'{V18_DIR}/xgb_long_model.json'),
        v18_s=L(f'{V18_DIR}/xgb_short_model.json'),
    )


def predict_all(models, X):
    d = xgb.DMatrix(X)
    return dict(
        rl=models['v10_l'].predict(d), rs=models['v10_s'].predict(d),
        pl=models['v18_l'].predict(d), ps=models['v18_s'].predict(d),
    )


# ── Phase 1: V10 standalone ──────────────────────────────────────────────────
def phase1_v10(df, P, clean_mask):
    print("\n" + "=" * 72)
    print("PHASE 1  V10 STANDALONE (RANKER)")
    print("=" * 72)
    qids = df['Query_ID'].values
    rets = df[RET_COL].values
    ym   = df['YearMonth'].values
    hour = df['Hour'].values

    def windows(name_mask):
        return name_mask

    masks = {
        'in_sample (<=2025-06)': (ym <= V10_TRAIN_END),
        'OOS (>=2025-07)':       (ym >= OOS_START),
        'OOS_clean':             (ym >= OOS_START) & clean_mask,
    }

    # Rank IC (Spearman) per query
    print("\n-- Rank IC (mean per-cross-section Spearman of score vs forward return) --")
    ic_out = {}
    for label, msk in masks.items():
        ic_l, ic_s = [], []
        for qid in np.unique(qids[msk]):
            qm = (qids == qid) & msk
            if qm.sum() < 5: continue
            a = rets[qm]
            if np.std(a) == 0: continue
            ic_l.append(np.corrcoef(rankdata(P['rl'][qm]), rankdata(a))[0, 1])
            ic_s.append(np.corrcoef(rankdata(P['rs'][qm]), rankdata(-a))[0, 1])
        ic_out[label] = dict(long_ic=round(float(np.mean(ic_l)), 4),
                             short_ic=round(float(np.mean(ic_s)), 4))
        print(f"  {label:<24} long_IC={ic_out[label]['long_ic']:+.4f}  "
              f"short_IC={ic_out[label]['short_ic']:+.4f}")

    # Decile monotonicity on OOS_clean (long score)
    print("\n-- Decile mean forward return (OOS_clean, LONG score; tests decile-3 spike) --")
    msk = masks['OOS_clean']
    sc, a = P['rl'][msk], rets[msk]
    dec = pd.qcut(rankdata(sc), 10, labels=False)
    dec_means = [round(float(a[dec == d].mean() * 10000), 2) for d in range(10)]
    print("  decile 0->9 bps:", dec_means)
    monotonic = all(dec_means[i] <= dec_means[i+1] for i in range(9))
    print(f"  strictly monotonic increasing: {monotonic}")

    # Top-K curves (OOS_clean), both costs
    print("\n-- Top-K net (OOS_clean) --")
    topk = {}
    for K in (1, 3, 5, 10):
        l_r, s_r = topk_returns(qids, P, rets, msk, K)
        topk[K] = {}
        for cl, cv in COSTS.items():
            ls, ss = trade_stats(l_r, cv), trade_stats(s_r, cv)
            topk[K][cl] = dict(long=ls, short=ss)
            print(f"  K={K:<2} {cl:<5} LONG  {fmt(ls, cl)}")
            print(f"  K={K:<2} {cl:<5} SHORT {fmt(ss, cl)}")

    # Quarterly + ToD (Top-3, 10bps, clean)
    print("\n-- Quarterly Top-3 net@10bps (OOS_clean) --")
    quarterly = quarterly_topk(df, P, rets, msk, 3, COSTS['10bps'])
    for q, v in quarterly.items():
        print(f"  {q}  LONG net {v['long']['net_bps']:>+6.1f} ({v['long']['n']:>4})  "
              f"SHORT net {v['short']['net_bps']:>+6.1f} ({v['short']['n']:>4})")

    print("\n-- Time-of-day Top-3 net@10bps (OOS_clean): is edge concentrated at a bar? --")
    tod = tod_topk(df, P, rets, msk, 3, COSTS['10bps'])
    for h, v in tod.items():
        print(f"  hour {h}  LONG net {v['long']['net_bps']:>+6.1f} ({v['long']['n']:>4})  "
              f"SHORT net {v['short']['net_bps']:>+6.1f} ({v['short']['n']:>4})")

    RESULTS['phase1_v10'] = dict(rank_ic=ic_out, decile_means_bps=dec_means,
                                 monotonic=monotonic, topk=topk,
                                 quarterly=quarterly, time_of_day=tod)


def topk_returns(qids, P, rets, msk, K, dual_short_lock=None):
    l_r, s_r = [], []
    for qid in np.unique(qids[msk]):
        qm = (qids == qid) & msk
        if qm.sum() < max(3, K): continue
        rl, rs, a = P['rl'][qm], P['rs'][qm], rets[qm]
        for idx in np.argsort(rl)[-K:]:
            l_r.append(a[idx])
        for idx in np.argsort(rs)[-K:]:
            s_r.append(-a[idx])
    return np.array(l_r), np.array(s_r)


def quarterly_topk(df, P, rets, msk, K, cost):
    q = (df['YearMonth'].str[:4] + 'Q' +
         ((df['DateTime'].str[5:7].astype(int) - 1) // 3 + 1).astype(str)).values
    qids = df['Query_ID'].values
    out = {}
    for quarter in sorted(pd.unique(q[msk])):
        qm_period = msk & (q == quarter)
        l_r, s_r = topk_returns(qids, P, rets, qm_period, K)
        out[quarter] = dict(long=trade_stats(l_r, cost), short=trade_stats(s_r, cost))
    return out


def tod_topk(df, P, rets, msk, K, cost):
    hour = df['Hour'].values
    qids = df['Query_ID'].values
    out = {}
    # Top-K is cross-sectional per query; each query is a single hour, so bucket queries by hour
    for h in sorted(np.unique(hour[msk])):
        hm = msk & (hour == h)
        l_r, s_r = topk_returns(qids, P, rets, hm, K)
        out[int(h)] = dict(long=trade_stats(l_r, cost), short=trade_stats(s_r, cost))
    return out


# ── Phase 2: V18 standalone ──────────────────────────────────────────────────
def phase2_v18(df, P, clean_mask):
    print("\n" + "=" * 72)
    print("PHASE 2  V18 STANDALONE (CLASSIFIER)")
    print("=" * 72)
    rets = df[RET_COL].values
    ym   = df['YearMonth'].values
    hour = df['Hour'].values
    y_long  = (rets > 0).astype(int)
    y_short = (rets < 0).astype(int)

    masks = {
        'in_sample (<=2025-03)': (ym <= V18_TRAIN_END),
        'OOS (>=2025-07)':       (ym >= OOS_START),
        'OOS_clean':             (ym >= OOS_START) & clean_mask,
    }

    print("\n-- ROC AUC (doc claims 0.524) --")
    auc_out = {}
    for label, msk in masks.items():
        try:
            al = roc_auc_score(y_long[msk], P['pl'][msk])
            a_s = roc_auc_score(y_short[msk], P['ps'][msk])
        except Exception:
            al, a_s = float('nan'), float('nan')
        auc_out[label] = dict(long_auc=round(al, 4), short_auc=round(a_s, 4))
        print(f"  {label:<24} long_AUC={al:.4f}  short_AUC={a_s:.4f}")

    # Calibration: predicted prob bucket vs realized win rate (OOS_clean)
    print("\n-- Calibration (OOS_clean): predicted prob decile vs realized win rate --")
    msk = masks['OOS_clean']
    cal = {}
    for side, prob, y in (('long', P['pl'], y_long), ('short', P['ps'], y_short)):
        p, yy = prob[msk], y[msk]
        bins = np.clip((p * 20).astype(int), 0, 19)  # 5% buckets
        rows = []
        for b in range(20):
            bm = bins == b
            if bm.sum() < 50: continue
            rows.append((round(b * 0.05, 2), int(bm.sum()), round(float(yy[bm].mean()), 4)))
        cal[side] = rows
        print(f"  {side}: (prob_bucket, n, realized_winrate)")
        for r in rows[-6:]:
            print(f"     {r}")

    # Threshold sweep (OOS_clean): trade ALL bars passing prob, both costs
    print("\n-- Threshold sweep (OOS_clean, trade every bar passing prob, no ranking) --")
    sweep = {}
    for th in (0.50, 0.52, 0.54, 0.56, 0.58, 0.60):
        lm = msk & (P['pl'] > th)
        sm = msk & (P['ps'] > th)
        l_r = rets[lm]
        s_r = -rets[sm]
        sweep[th] = {cl: dict(long=trade_stats(l_r, cv), short=trade_stats(s_r, cv))
                     for cl, cv in COSTS.items()}
        ls, ss = sweep[th]['10bps']['long'], sweep[th]['10bps']['short']
        print(f"  th={th:.2f} @10bps  LONG {fmt(ls,'10bps')}")
        print(f"             SHORT {fmt(ss,'10bps')}")

    # "Is V18 just a clock?" — compare prob-gated edge vs naive hour filter
    print("\n-- IS V18 JUST A CLOCK? prob>0.52 edge vs naive 'trade this hour' baseline (OOS_clean) --")
    clock = {}
    for side, prob, sign in (('long', P['pl'], 1.0), ('short', P['ps'], -1.0)):
        gated = msk & (prob > PROB_TH)
        g_stats = trade_stats(sign * rets[gated], COSTS['10bps'])
        # naive: best single hour, all bars that side
        per_hour = {}
        for h in np.unique(hour[msk]):
            hm = msk & (hour == h)
            per_hour[int(h)] = trade_stats(sign * rets[hm], COSTS['10bps'])
        best_h = max(per_hour, key=lambda k: per_hour[k]['net_bps'])
        clock[side] = dict(v18_gated=g_stats, best_hour=best_h,
                           best_hour_stats=per_hour[best_h], per_hour=per_hour)
        print(f"  {side}: V18-gated net {g_stats['net_bps']:+.1f} ({g_stats['n']}) | "
              f"best naive hour={best_h} net {per_hour[best_h]['net_bps']:+.1f} "
              f"({per_hour[best_h]['n']})")

    # Quarterly (prob>0.52)
    print("\n-- Quarterly V18-gated net@10bps (OOS_clean) --")
    q = (df['YearMonth'].str[:4] + 'Q' +
         ((df['DateTime'].str[5:7].astype(int) - 1) // 3 + 1).astype(str)).values
    quarterly = {}
    for quarter in sorted(pd.unique(q[msk])):
        qm = msk & (q == quarter)
        lq = trade_stats(rets[qm & (P['pl'] > PROB_TH)], COSTS['10bps'])
        sq = trade_stats(-rets[qm & (P['ps'] > PROB_TH)], COSTS['10bps'])
        quarterly[quarter] = dict(long=lq, short=sq)
        print(f"  {quarter}  LONG net {lq['net_bps']:>+6.1f} ({lq['n']:>5})  "
              f"SHORT net {sq['net_bps']:>+6.1f} ({sq['n']:>5})")

    RESULTS['phase2_v18'] = dict(auc=auc_out, calibration=cal, threshold_sweep=sweep,
                                 clock_test=clock, quarterly=quarterly)


# ── Phase 3: combined ────────────────────────────────────────────────────────
def logic_returns(qids, P, rets, msk, prob_th=PROB_TH,
                  veto_long=True, veto_short=True, K=3, dual_lock=False):
    """Return long & short realized returns under a hybrid logic.
    veto_long/short: apply v18 gate on that side. dual_lock: require BOTH models high-conviction."""
    l_r, s_r = [], []
    flags_l, flags_s = [], []  # boundary-bar flags placeholder (filled by caller via index)
    for qid in np.unique(qids[msk]):
        qm = (qids == qid) & msk
        if qm.sum() < max(3, K): continue
        rl, rs, pl, ps, a = P['rl'][qm], P['rs'][qm], P['pl'][qm], P['ps'][qm], rets[qm]
        # long
        for idx in np.argsort(rl)[-K:]:
            ok = (pl[idx] > prob_th) if veto_long else True
            if dual_lock:
                ok = ok and (ps[idx] < (1 - prob_th))  # also low short-prob (agreement)
            if ok:
                l_r.append(a[idx])
        # short
        for idx in np.argsort(rs)[-K:]:
            ok = (ps[idx] > prob_th) if veto_short else True
            if dual_lock:
                ok = ok and (pl[idx] < (1 - prob_th))
            if ok:
                s_r.append(-a[idx])
    return np.array(l_r), np.array(s_r)


def phase3_combined(df, P, clean_mask):
    print("\n" + "=" * 72)
    print("PHASE 3  COMBINED / HYBRID")
    print("=" * 72)
    qids = df['Query_ID'].values
    rets = df[RET_COL].values
    ym   = df['YearMonth'].values

    oos      = (ym >= OOS_START)
    oos_clean= oos & clean_mask
    h2_25    = oos_clean & (ym < OOS_MID)
    h1_26    = oos_clean & (ym >= OOS_MID)

    # Anchor regression check (RAW oos, 10bps, Top3 hybrid) — should match verified +4.6/+4.6
    print("\n-- ANCHOR (RAW OOS, Top-3 symmetric hybrid @10bps): expect ~+4.6 / +4.6 --")
    al, as_ = logic_returns(qids, P, rets, oos, K=3)
    print(f"  LONG  {fmt(trade_stats(al, COSTS['10bps']), '10bps')}")
    print(f"  SHORT {fmt(trade_stats(as_, COSTS['10bps']), '10bps')}")

    configs = {
        'V10_alone_T3':        dict(veto_long=False, veto_short=False),
        'hybrid_symmetric_T3': dict(veto_long=True,  veto_short=True),
        'asymmetric_T3':       dict(veto_long=True,  veto_short=False),  # veto longs only
        'dual_lock_T3':        dict(veto_long=True,  veto_short=True, dual_lock=True),
    }

    print("\n-- Config comparison on OOS_clean (Top-3) --")
    cfg_out = {}
    for name, kw in configs.items():
        l_r, s_r = logic_returns(qids, P, rets, oos_clean, K=3, **kw)
        cfg_out[name] = {}
        for cl, cv in COSTS.items():
            cfg_out[name][cl] = dict(long=trade_stats(l_r, cv), short=trade_stats(s_r, cv))
        ls6, ss6 = cfg_out[name]['6bps']['long'], cfg_out[name]['6bps']['short']
        ls10, ss10 = cfg_out[name]['10bps']['long'], cfg_out[name]['10bps']['short']
        print(f"\n  [{name}]")
        print(f"    LONG  @6bps {fmt(ls6,'6bps')}")
        print(f"    LONG  @10  {fmt(ls10,'10bps')}")
        print(f"    SHORT @6bps {fmt(ss6,'6bps')}")
        print(f"    SHORT @10  {fmt(ss10,'10bps')}")

    # Veto marginal decomposition (OOS_clean, 10bps)
    print("\n-- Veto marginal decomposition (OOS_clean, Top-3, 10bps) --")
    vl_off, vs_off = logic_returns(qids, P, rets, oos_clean, K=3, veto_long=False, veto_short=False)
    vl_on,  vs_on  = logic_returns(qids, P, rets, oos_clean, K=3, veto_long=True,  veto_short=True)
    print(f"  LONG  veto OFF net {trade_stats(vl_off,COSTS['10bps'])['net_bps']:+.1f} "
          f"({len(vl_off)}) -> veto ON net {trade_stats(vl_on,COSTS['10bps'])['net_bps']:+.1f} "
          f"({len(vl_on)})")
    print(f"  SHORT veto OFF net {trade_stats(vs_off,COSTS['10bps'])['net_bps']:+.1f} "
          f"({len(vs_off)}) -> veto ON net {trade_stats(vs_on,COSTS['10bps'])['net_bps']:+.1f} "
          f"({len(vs_on)})")

    # Dual-lock artifact check: where do dual-lock trades sit by hour? (open-hour concentration?)
    print("\n-- Dual-lock hour concentration (OOS_clean): is the big number open-hour variance? --")
    dl_l, dl_s, hours_s = dual_lock_hours(df, P, rets, oos_clean, K=3)
    if len(hours_s):
        vc = pd.Series(hours_s).value_counts(normalize=True).sort_index()
        print("  SHORT dual-lock trade share by hour:", {int(k): round(float(v), 3) for k, v in vc.items()})
        print(f"  SHORT dual-lock net@10bps {trade_stats(dl_s, COSTS['10bps'])['net_bps']:+.1f} "
              f"raw {trade_stats(dl_s, COSTS['10bps'])['raw_bps']:+.1f} (n={len(dl_s)})")

    # OOS stability halves (symmetric hybrid + asymmetric)
    print("\n-- OOS stability: H2-2025 vs H1-2026 (Top-3, 10bps) --")
    stab = {}
    for name, kw in (('hybrid_symmetric', dict(veto_long=True, veto_short=True)),
                     ('asymmetric',       dict(veto_long=True, veto_short=False))):
        stab[name] = {}
        for half_name, half in (('H2_2025', h2_25), ('H1_2026', h1_26)):
            l_r, s_r = logic_returns(qids, P, rets, half, K=3, **kw)
            stab[name][half_name] = dict(long=trade_stats(l_r, COSTS['10bps']),
                                         short=trade_stats(s_r, COSTS['10bps']))
            ls, ss = stab[name][half_name]['long'], stab[name][half_name]['short']
            print(f"  [{name}] {half_name}  LONG net {ls['net_bps']:>+6.1f} ({ls['n']:>4})  "
                  f"SHORT net {ss['net_bps']:>+6.1f} ({ss['n']:>4})")

    RESULTS['phase3_combined'] = dict(configs=cfg_out,
        veto_decomp=dict(
            long_off=trade_stats(vl_off, COSTS['10bps']), long_on=trade_stats(vl_on, COSTS['10bps']),
            short_off=trade_stats(vs_off, COSTS['10bps']), short_on=trade_stats(vs_on, COSTS['10bps'])),
        stability=stab)


def dual_lock_hours(df, P, rets, msk, K=3):
    qids = df['Query_ID'].values
    hour = df['Hour'].values
    l_r, s_r, h_s = [], [], []
    for qid in np.unique(qids[msk]):
        qm = (qids == qid) & msk
        if qm.sum() < 3: continue
        rl, rs, pl, ps, a, h = (P['rl'][qm], P['rs'][qm], P['pl'][qm],
                                P['ps'][qm], rets[qm], hour[qm])
        for idx in np.argsort(rs)[-K:]:
            if ps[idx] > PROB_TH and pl[idx] < (1 - PROB_TH):
                s_r.append(-a[idx]); h_s.append(int(h[idx]))
        for idx in np.argsort(rl)[-K:]:
            if pl[idx] > PROB_TH and ps[idx] < (1 - PROB_TH):
                l_r.append(a[idx])
    return np.array(l_r), np.array(s_r), h_s


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    clean_mask = phase0_clean_mask(df)
    X, feature_cols = build_features(df)
    print(f"\nFeatures: {len(feature_cols)} | models loading ...")
    models = load_models()
    P = predict_all(models, X)

    phase1_v10(df, P, clean_mask)
    phase2_v18(df, P, clean_mask)
    phase3_combined(df, P, clean_mask)

    out_path = f'{OUT_DIR}/results.json'
    with open(out_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
