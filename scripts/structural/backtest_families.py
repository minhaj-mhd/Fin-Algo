"""
Structural family backtest orchestrator.

Pipeline per candidate:
  family trigger -> composite gate (v10 prior + 15m confirm) -> exit sim -> aggregate

Run family-by-family (longs first) then merged.

Usage:
    python scripts/structural/backtest_families.py
    python scripts/structural/backtest_families.py --sides long        # longs only
    python scripts/structural/backtest_families.py --family 103        # single family
    python scripts/structural/backtest_families.py --no-gate           # skip ML gate (raw triggers)
"""
import os, sys, json, warnings, gc, argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

from scripts.structural.detectors import run_all_detectors, DETECTORS, FAMILY_NAMES
from scripts.structural.exit_sim import run_simulation

PANEL_FILE      = 'data/structural_panel_15m.parquet'
V10_CACHE       = 'data/v10_macro_regime_oos_cache.parquet'
V15_SCORES_FILE = 'data/v3_15min_oos_scores.parquet'
RAW_CACHE_DIR   = 'data/raw_upstox_cache_15min_3y'
DATA_15M        = 'data/ranking_data_upstox_15min_3y_clean.csv'
META_15M        = 'models/v3_15min_clean/metadata.json'
RESULTS_OUT     = 'data/structural_family_results.json'

H_TEST    = 4
MIN_TRAIN = 18
RNG       = np.random.default_rng(0)


# ── stat reporter (same as wf_1h_base) ────────────────────────────────────────
def stat(net, fold):
    n = len(net)
    if n < 20:
        return None
    t = net.mean() / (net.std() / np.sqrt(n) + 1e-12)
    bs = [net[RNG.integers(0, n, n)].mean() * 1e4 for _ in range(1500)]
    fm = [net[fold == fi].mean() for fi in np.unique(fold) if (fold == fi).sum() > 0]
    sig = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.64 else ''))
    return dict(n=n, wr=(net > 0).mean() * 100, bps=net.mean() * 1e4,
                ci=(np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
                t=t, sig=sig, pos=sum(1 for x in fm if x > 0), nf=len(fm))


def print_stat(label, st, exit_reasons=None):
    if st is None:
        print(f"  {label:<45}  [< 20 trades]")
        return
    reason_str = ''
    if exit_reasons is not None:
        rc = exit_reasons.value_counts(normalize=True)
        parts = [f"{k}:{v*100:.0f}%" for k, v in rc.items()]
        reason_str = '  exits=[' + ', '.join(parts) + ']'
    print(f"  {label:<45}  N={st['n']:>5}  WR={st['wr']:>5.1f}%  "
          f"NetBps={st['bps']:>+7.2f}  [{st['ci'][0]:>+6.1f},{st['ci'][1]:>+5.1f}]  "
          f"t={st['t']:>5.2f}{st['sig']:<3}  +folds={st['pos']}/{st['nf']}"
          f"{reason_str}")


# ── 15m walk-forward OOS scores ───────────────────────────────────────────────
def build_15m_oos_scores():
    """Walk-forward v3_15min_clean scores, cached to parquet."""
    if os.path.exists(V15_SCORES_FILE):
        print(f"Loading cached 15m OOS scores from {V15_SCORES_FILE} ...")
        return pd.read_parquet(V15_SCORES_FILE)

    print("Building 15m walk-forward OOS scores (v3_15min_clean) ...")
    with open(META_15M) as f:
        meta = json.load(f)
    fe, params = meta['features'], meta['params']

    df = pd.concat([c for c in pd.read_csv(DATA_15M, chunksize=200_000)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime'])
    df['ym'] = df['DateTime'].str[:7]
    df = df.dropna(subset=['Next_15Min_Return']).reset_index(drop=True)

    def Xmat(d):
        X = d[fe].values.astype(float)
        for ci in range(X.shape[1]):
            c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
            if b.any():
                X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
        return X

    def iranks(y, q, inv=False):
        out = np.zeros_like(y, dtype=int)
        for qid in np.unique(q):
            m = q == qid; v = -y[m] if inv else y[m]
            out[m] = rankdata(v, method='ordinal') - 1
        return out

    def fitdm(X, y, q, inv):
        d = xgb.DMatrix(X, label=iranks(y, q, inv))
        d.set_group(pd.Series(q).groupby(q).size().values)
        return d

    X = Xmat(df)
    months = sorted(df['ym'].unique())
    folds = []
    i = MIN_TRAIN + 1
    while i + 1 <= len(months):
        folds.append((months[:i-1], months[i-1], months[i:i+H_TEST]))
        i += H_TEST
    print(f"  {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

    rows = []
    for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
        tr = df['ym'].isin(tr_m).values
        va = df['ym'].isin([val_m]).values
        te = df['ym'].isin(te_m).values
        bl = xgb.train(params,
                       fitdm(X[tr], df['Next_15Min_Return'].values[tr], df['Query_ID'].values[tr], False), 500,
                       evals=[(fitdm(X[va], df['Next_15Min_Return'].values[va], df['Query_ID'].values[va], False), 'v')],
                       early_stopping_rounds=50, verbose_eval=False)
        bs = xgb.train(params,
                       fitdm(X[tr], df['Next_15Min_Return'].values[tr], df['Query_ID'].values[tr], True), 500,
                       evals=[(fitdm(X[va], df['Next_15Min_Return'].values[va], df['Query_ID'].values[va], True), 'v')],
                       early_stopping_rounds=50, verbose_eval=False)
        sub = df[te].copy()
        sub['sL'] = bl.predict(xgb.DMatrix(X[te]))
        sub['sS'] = bs.predict(xgb.DMatrix(X[te]))
        sub['long_pct']   = sub.groupby('dt')['sL'].rank(pct=True)
        sub['short_rank'] = sub.groupby('dt')['sS'].rank(ascending=False, method='first')
        sub['fold'] = fi
        rows.append(sub[['fold','DateTime','Ticker','long_pct','short_rank']].rename(columns={'DateTime':'ts'}))
        print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]} done")
        del sub; gc.collect()

    scores = pd.concat(rows, ignore_index=True)
    scores['ts'] = pd.to_datetime(scores['ts'])
    scores.to_parquet(V15_SCORES_FILE, index=False)
    print(f"  Saved {V15_SCORES_FILE}  rows={len(scores):,}")
    return scores


# ── v10 prior join ────────────────────────────────────────────────────────────
def load_v10_prior():
    """Load v10 1h OOS cache and return posL/posS/sL/sS per (DateTime, Ticker)."""
    if not os.path.exists(V10_CACHE):
        print(f"[WARN] v10 cache not found at {V10_CACHE}. v10 gate will be skipped.")
        return None
    v10 = pd.read_parquet(V10_CACHE)
    v10['DateTime'] = pd.to_datetime(v10['DateTime'])
    # Ensure raw scores are present (added by wf_1h_macro_regime.py)
    has_raw = 'sL' in v10.columns and 'sS' in v10.columns
    if not has_raw:
        print("[WARN] v10 cache missing raw scores sL/sS. Using posL/posS only.")
    return v10


def gate_v10(triggers, v10, side='long'):
    """Add v10 prior check. Returns filtered triggers."""
    if v10 is None:
        return triggers

    # Map trigger ts -> 1h bar timestamp (floor to :15)
    def to_1h_ts(ts):
        h = ts.replace(minute=15, second=0, microsecond=0)
        if ts.minute < 15:
            h = h - pd.Timedelta(hours=1)
        return h

    triggers = triggers.copy()
    triggers['v10_ts'] = triggers['ts'].apply(to_1h_ts)

    # Build lookup from v10: key=(DateTime_1h, Ticker)
    v10_key = v10.copy()
    v10_key['v10_ts'] = pd.to_datetime(v10_key['DateTime'])
    v10_key = v10_key.set_index(['v10_ts','Ticker'])

    pos_col  = 'posL' if side == 'long' else 'posS'
    raw_col  = 'sL'   if side == 'long' else 'sS'
    has_raw  = raw_col in v10_key.columns

    def check(row):
        key = (row['v10_ts'], row['Ticker'])
        if key not in v10_key.index:
            return True  # no v10 data -> permissive (let through)
        rec = v10_key.loc[key]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        pos  = rec.get(pos_col, 999)
        raw  = rec.get(raw_col, 0) if has_raw else 0
        return (pos <= 15) or (raw > 0)

    mask = triggers.apply(check, axis=1)
    return triggers[mask].reset_index(drop=True)


def gate_15m(triggers, ml_scores, side='long'):
    """15m confirmation: at trigger bar, long_pct strong or short weak."""
    if ml_scores is None or len(ml_scores) == 0:
        return triggers

    triggers = triggers.copy()
    ml_scores = ml_scores.copy()
    ml_scores['ts'] = pd.to_datetime(ml_scores['ts'])

    key_map = ml_scores.set_index(['ts','Ticker'])

    def check(row):
        key = (row['ts'], row['Ticker'])
        if key not in key_map.index:
            return True  # permissive if no score available
        rec = key_map.loc[key]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        lp = rec.get('long_pct', 0.5)
        sr = rec.get('short_rank', 999)
        if side == 'long':
            return (lp >= 0.5) and (sr > 30)   # 15m at least neutral or long
        else:
            return (lp <= 0.5) and (sr <= 30)

    mask = triggers.apply(check, axis=1)
    return triggers[mask].reset_index(drop=True)


def gate_opposite_family(triggers):
    """Remove candidates where the opposite-direction family also fires on the same bar."""
    opp = triggers.copy()
    long_ts  = set(zip(opp[opp['side']=='long']['ts'],  opp[opp['side']=='long']['Ticker'],  opp[opp['side']=='long']['strategy_id']))
    short_ts = set(zip(opp[opp['side']=='short']['ts'], opp[opp['side']=='short']['Ticker'], opp[opp['side']=='short']['strategy_id']))
    conflict = long_ts & short_ts
    if not conflict:
        return triggers
    mask = ~triggers.apply(lambda r: (r['ts'], r['Ticker'], r['strategy_id']) in conflict, axis=1)
    return triggers[mask].reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sides', default='long', choices=['long','short','both'])
    ap.add_argument('--family', type=int, default=None, help='Run a single family ID (101-105)')
    ap.add_argument('--no-gate', action='store_true', help='Skip ML gates (raw triggers only)')
    ap.add_argument('--no-flip', action='store_true', help='Disable conviction-flip exit (ATR TP/SL + 1h close only)')
    args = ap.parse_args()
    sides = ['long','short'] if args.sides == 'both' else [args.sides]

    # ── load panel ────────────────────────────────────────────────────────────
    if not os.path.exists(PANEL_FILE):
        print(f"Panel not found. Building from scratch ...")
        from scripts.structural.build_panel import build_panel
        build_panel()
    print(f"Loading panel from {PANEL_FILE} ...")
    panel = pd.read_parquet(PANEL_FILE)
    panel['ts'] = pd.to_datetime(panel['ts'])
    print(f"  Panel: {len(panel):,} rows  {panel['ts'].min()} -> {panel['ts'].max()}")

    # ── ML gates ─────────────────────────────────────────────────────────────
    v10 = None if args.no_gate else load_v10_prior()
    ml_scores = None if args.no_gate else build_15m_oos_scores()

    # ── run detectors ─────────────────────────────────────────────────────────
    families_to_run = [args.family] if args.family else list(DETECTORS.keys())
    print(f"\nRunning detectors: families={families_to_run}  sides={sides}")
    all_triggers = run_all_detectors(panel, sides=sides)
    if args.family:
        all_triggers = all_triggers[all_triggers['strategy_id'] == args.family]
    print(f"  Raw triggers: {len(all_triggers):,}")

    if not args.no_gate:
        # Gates
        all_triggers = gate_opposite_family(all_triggers)
        all_triggers_long  = all_triggers[all_triggers['side']=='long']
        all_triggers_short = all_triggers[all_triggers['side']=='short']
        all_triggers_long  = gate_v10(all_triggers_long,  v10, 'long')
        all_triggers_short = gate_v10(all_triggers_short, v10, 'short')
        all_triggers_long  = gate_15m(all_triggers_long,  ml_scores, 'long')
        all_triggers_short = gate_15m(all_triggers_short, ml_scores, 'short')
        all_triggers = pd.concat([all_triggers_long, all_triggers_short], ignore_index=True)
        print(f"  After gates: {len(all_triggers):,}")

    # ── simulate exits ────────────────────────────────────────────────────────
    flip_scores = None if args.no_flip else ml_scores
    trades = run_simulation(all_triggers, RAW_CACHE_DIR, ml_scores_df=flip_scores)
    if len(trades) == 0:
        print("\nNo trades simulated. Check panel/trigger/gate settings.")
        return

    # Assign fold from 15m OOS scores using ym-based mapping (avoids timestamp precision issues)
    trades['trigger_ts'] = pd.to_datetime(trades['trigger_ts'])
    trades['ym'] = trades['trigger_ts'].dt.strftime('%Y-%m')
    if ml_scores is not None:
        ym_fold_map = (ml_scores.assign(ym=ml_scores['ts'].dt.strftime('%Y-%m'))
                       .drop_duplicates('ym').set_index('ym')['fold'].to_dict())
        trades['fold'] = trades['ym'].map(ym_fold_map).fillna(-1).astype(int)
    else:
        trades['fold'] = 1

    # -- Overlap guard ---------------------------------------------------------
    # Only evaluate trades where BOTH ML gates had genuine OOS scores available.
    # Trades before the 15m OOS window (2024-09) had no v3 gate active and may
    # have had no v10 gate either -- those trades are silently unfiltered and
    # would pollute the aggregate. Restrict to fold > 0 (trades inside the 15m
    # OOS window) to ensure every evaluated trade passed through real ML gates.
    n_total = len(trades)
    if not args.no_gate and ml_scores is not None:
        trades = trades[trades['fold'] > 0].reset_index(drop=True)
        n_dropped = n_total - len(trades)
        if n_dropped > 0:
            print(f"  [overlap guard] dropped {n_dropped:,} trades outside 15m OOS window "
                  f"(kept {len(trades):,} trades in genuinely-gated period)")
    else:
        trades['fold'] = 1

    # ── report ────────────────────────────────────────────────────────────────
    sep = '=' * 110
    print(f"\n{sep}")
    print(f"  STRUCTURAL FAMILY BACKTEST  (10 bps cost, path-based ATR TP/SL + conviction flip + 1h close)")
    print(f"{sep}")

    all_results = {}

    for sid in families_to_run:
        fname = FAMILY_NAMES.get(sid, f'Family {sid}')
        fam_trades = trades[trades['strategy_id'] == sid]
        if len(fam_trades) == 0:
            print(f"\n  [{sid}] {fname}: no trades")
            continue

        print(f"\n  [{sid}] {fname}")
        print(f"  {'-'*100}")
        fam_res = {}
        for side in sides:
            sub = fam_trades[fam_trades['side'] == side]
            if len(sub) == 0:
                continue
            net = sub['net_return'].values
            fld = sub['fold'].values
            st  = stat(net, fld)
            print_stat(f"  {side.upper():<6} {fname}", st, sub['exit_reason'])
            fam_res[side] = {
                'stat': st,
                'exit_reasons': sub['exit_reason'].value_counts().to_dict(),
                'n_trades': len(sub),
            }
        all_results[str(sid)] = fam_res

    # Merged
    if len(families_to_run) > 1:
        print(f"\n  MERGED (all families)")
        print(f"  {'-'*100}")
        for side in sides:
            sub = trades[trades['side'] == side]
            if len(sub) == 0:
                continue
            net = sub['net_return'].values
            fld = sub['fold'].values
            st  = stat(net, fld)
            print_stat(f"  {side.upper():<6} Merged", st, sub['exit_reason'])

    print(f"\n{sep}")
    print(f"  INTERPRETATION: CI spanning 0 => not significant.  * p<.10  ** p<.05  *** p<.01")
    print(f"  Trust only: significantly positive AND majority of folds AND >50 trades AND sane exit mix.")
    print(f"{sep}\n")

    # ── save results ──────────────────────────────────────────────────────────
    def _ser(o):
        if o is None: return None
        out = {}
        for k, v in o.items():
            if isinstance(v, (np.floating, float)): out[k] = float(v)
            elif isinstance(v, (tuple, list)):       out[k] = [float(x) for x in v]
            else:                                    out[k] = v
        return out

    serializable = {}
    for sid_str, sides_res in all_results.items():
        serializable[sid_str] = {}
        for side, data in sides_res.items():
            serializable[sid_str][side] = {
                'stat': _ser(data['stat']),
                'exit_reasons': data['exit_reasons'],
                'n_trades': data['n_trades'],
            }

    with open(RESULTS_OUT, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved -> {RESULTS_OUT}")

    # Also save full trades table
    trades_out = RESULTS_OUT.replace('.json', '_trades.parquet')
    trades.to_parquet(trades_out, index=False)
    print(f"Trades saved  -> {trades_out}")


if __name__ == '__main__':
    main()
