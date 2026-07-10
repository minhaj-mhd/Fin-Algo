"""
N-HOUR HORIZON evaluation of the v20/v21 rolling-1h ranker — does a longer hold cross the cost line?
("2h" in the filename is the original default; `--hours` runs any horizon, e.g. --hours 3.)

Same purged monthly walk-forward + same XGBoost rank:pairwise recipe as eval_v21_vs_v20.py, on the
v21 panel augmented with `Next_{H}Hour_Return` (scripts/research/build_2h_labels.py --hours H).
Features are the proven v20/v21 rolling-1h set UNCHANGED — only the prediction HORIZON varies, so any
difference is the horizon, not the feature window. Reported per side (long / short), all on the
IDENTICAL test entries (rows carrying BOTH a 1h and an Nh label):

  A) 1h model | held 1h   — the existing v20/v21 ranker, its native horizon (baseline).
  B) 1h model | held Nh   — SAME picks, just held N times as long (the holding-period lever).
  C) Nh model | held Nh   — a ranker TRAINED on the Nh label (the "Nh model").
  D) Nh model SHUFFLED    — neg-control; rho must collapse to ~0 or the Nh label leaks.

Cost is 10bps round-trip for EVERY row regardless of hold length (cost is per-trade, not per-hour) —
exactly why a longer hold *could* help: it scales the gross while cost stays fixed.

RESEARCH ONLY (AGENTS.md): overlapping windows => effective N ~1/4 rows; point estimates, NO t-tests,
no Gauntlet, no verdict authority.

Run: python scripts/research/eval_2h_v20.py --hours 3
"""
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

RET1 = 'Next_Hour_Return'
COST = 0.001          # 10 bps round-trip, hold-length independent
KS = [1, 3, 5]
SEED = 42
BASE_EXCLUDE = {'DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET1, 'YearMonth'}
PARAMS = {'objective': 'rank:pairwise', 'eta': 0.03, 'max_depth': 5, 'subsample': 0.8,
          'colsample_bytree': 0.8, 'alpha': 1.0, 'lambda': 2.0, 'min_child_weight': 10,
          'random_state': SEED, 'verbosity': 0, 'eval_metric': 'ndcg@3', 'ndcg_exp_gain': False,
          'tree_method': 'hist', 'device': 'cpu'}


def _gpu():
    try:
        m = xgb.DMatrix(np.random.randn(10, 2), label=np.arange(10)); m.set_group([10])
        xgb.train({'objective': 'rank:pairwise', 'device': 'cuda', 'tree_method': 'hist'}, m, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'


def int_ranks(y, q, invert=False):
    out = np.zeros_like(y, dtype=int)
    for qi in np.unique(q):
        m = q == qi
        out[m] = rankdata(-y[m] if invert else y[m], method='ordinal') - 1
    return out


def folds_of(df):
    months = sorted(pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m').unique())
    ym = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m').values
    F = []
    for i in range(18, len(months) - 2, 4):
        F.append((np.isin(ym, months[:i]), np.isin(ym, [months[i]]),
                  np.isin(ym, months[i + 1:i + 3])))
    return F


def train_side(X, y, q, tr_mask, va_mask, invert):
    Xtr, ytr, qtr = X[tr_mask], y[tr_mask], q[tr_mask]
    Xva, yva, qva = X[va_mask], y[va_mask], q[va_mask]
    gtr = pd.Series(qtr).groupby(qtr).size().values
    gva = pd.Series(qva).groupby(qva).size().values
    dtr = xgb.DMatrix(Xtr, label=int_ranks(ytr, qtr, invert)); dtr.set_group(gtr)
    dva = xgb.DMatrix(Xva, label=int_ranks(yva, qva, invert)); dva.set_group(gva)
    return xgb.train(PARAMS, dtr, 500, evals=[(dva, 'v')], early_stopping_rounds=50, verbose_eval=False)


def topk_net(score, ret, q, short=False):
    sgn = -1.0 if short else 1.0
    acc = {k: [] for k in KS}
    for qi in np.unique(q):
        m = q == qi
        if m.sum() < max(KS) + 1:
            continue
        sc = score[m]; r = sgn * ret[m]
        order = np.argsort(-sc)
        for k in KS:
            acc[k].append(r[order[:k]].mean())
    out = {}
    for k in KS:
        g = float(np.mean(acc[k])) if acc[k] else 0.0
        out[k] = (g * 1e4, (g - COST) * 1e4)
    return out


def rho_of(score, ret, q, short=False):
    sgn = -1.0 if short else 1.0
    rs = []
    for qi in np.unique(q):
        m = q == qi
        if m.sum() < 2 or np.std(score[m]) == 0:
            continue
        r = spearmanr(score[m], sgn * ret[m]).correlation
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else 0.0


def shuffle_within_q(y, q, seed):
    y = y.copy(); rng = np.random.default_rng(seed)
    for qi in np.unique(q):
        idx = np.where(q == qi)[0]
        y[idx] = y[idx][rng.permutation(len(idx))]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=int, default=2)
    args = ap.parse_args()
    H = args.hours
    RET2 = f'Next_{H}Hour_Return'
    PANEL = f'data/research/v21_rolling_1h/panel_{H}h.parquet'
    meta_exclude = BASE_EXCLUDE | {RET2}

    PARAMS['device'] = _gpu()
    print(f"device={PARAMS['device']}  horizon={H}h  label={RET2}  panel={PANEL}")
    df = pd.read_parquet(PANEL)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    feats = [c for c in df.columns if c not in meta_exclude]
    print(f"panel rows={len(df):,}  feats={len(feats)}  {H}h-labeled={df[RET2].notna().mean()*100:.1f}%")

    both = df[df[RET1].notna() & df[RET2].notna()].copy().reset_index(drop=True)
    both['Query_ID'] = both.groupby('DateTime').ngroup()
    X = both[feats].values.astype(np.float64)
    if not np.isfinite(X).all():
        cm = np.nan_to_num(np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0))
        bad = np.where(~np.isfinite(X)); X[bad] = np.take(cm, bad[1])
    y1 = both[RET1].values.astype(np.float64)
    yN = both[RET2].values.astype(np.float64)
    q = both['Query_ID'].values
    tod = sorted(pd.to_datetime(both['DateTime']).dt.strftime('%H:%M').unique())
    print(f"both-labeled rows={len(both):,}  queries={both['Query_ID'].nunique():,} "
          f"(window-closes {tod[0]}..{tod[-1]})\n")

    keys = ['A_1h_hold1h', 'B_1h_holdNh', 'C_Nh_holdNh', 'D_Nh_shuffle']
    rows = {k: {} for k in keys}
    for r in rows.values():
        for side in ('long', 'short'):
            r[side] = {'rho': [], **{k: {'g': [], 'n': []} for k in KS}}

    folds = folds_of(both)
    print(f"purged monthly WF: {len(folds)} folds\n")
    for fi, (tr, va, te) in enumerate(folds):
        qte = pd.Series(q[te]).groupby(q[te]).ngroup().values if te.sum() else np.array([])
        for inv, side in ((False, 'long'), (True, 'short')):
            b1 = train_side(X, y1, q, tr, va, inv)
            b2 = train_side(X, yN, q, tr, va, inv)
            yNs = shuffle_within_q(yN, q, SEED + fi)
            b2sh = train_side(X, yNs, q, tr, va, inv)
            dte = xgb.DMatrix(X[te])
            s1 = b1.predict(dte); s2 = b2.predict(dte); s2sh = b2sh.predict(dte)
            r1, rN = y1[te], yN[te]
            short = (side == 'short')
            spec = [('A_1h_hold1h', s1, r1), ('B_1h_holdNh', s1, rN),
                    ('C_Nh_holdNh', s2, rN), ('D_Nh_shuffle', s2sh, rN)]
            for name, sc, rr in spec:
                rows[name][side]['rho'].append(rho_of(sc, rr, qte, short))
                tk = topk_net(sc, rr, qte, short)
                for k in KS:
                    rows[name][side][k]['g'].append(tk[k][0])
                    rows[name][side][k]['n'].append(tk[k][1])
        print(f"  fold {fi+1}/{len(folds)} done (train {tr.sum():,} / test {te.sum():,})")

    def agg(cell, key):
        return float(np.mean(cell[key])) if cell[key] else 0.0

    print("\n" + "=" * 96)
    print(f"{H}-HOUR HORIZON RESULTS  (cost=10bps round-trip, identical test entries, net bps)")
    print("=" * 96)
    label_names = {'A_1h_hold1h': 'A) 1h model | held 1h  (baseline)',
                   'B_1h_holdNh': f'B) 1h model | held {H}h  (hold lever)',
                   'C_Nh_holdNh': f'C) {H}h model | held {H}h  ({H}h model)',
                   'D_Nh_shuffle': f'D) {H}h model SHUFFLED   (neg-control)'}
    for side in ('long', 'short'):
        print(f"\n--- {side.upper()} ---")
        hdr = f"  {'variant':36s} {'rho':>8s}"
        for k in KS:
            hdr += f"  {'K'+str(k)+'_net':>9s}{'(gross)':>9s}"
        print(hdr)
        for name in keys:
            c = rows[name][side]
            line = f"  {label_names[name]:36s} {agg(c,'rho'):+8.4f}"
            for k in KS:
                line += f"  {agg(c[k],'n'):+9.2f}{agg(c[k],'g'):+9.2f}"
            print(line)

    print("\n=== READ ===")
    for side in ('long', 'short'):
        a = agg(rows['A_1h_hold1h'][side][3], 'n')
        b = agg(rows['B_1h_holdNh'][side][3], 'n')
        c = agg(rows['C_Nh_holdNh'][side][3], 'n')
        nc = agg(rows['D_Nh_shuffle'][side], 'rho')
        print(f"  {side:5s} K3 net@10bps:  hold1h {a:+.2f}  ->  hold{H}h {b:+.2f}  ({b-a:+.2f})  |  "
              f"{H}h-model {c:+.2f}  |  neg-ctrl rho {nc:+.4f}")

    out = {name: {side: {'rho': agg(rows[name][side], 'rho'),
                         **{f'K{k}_net': agg(rows[name][side][k], 'n') for k in KS},
                         **{f'K{k}_gross': agg(rows[name][side][k], 'g') for k in KS}}
                  for side in ('long', 'short')} for name in rows}
    path = f'data/research/v21_rolling_1h/eval_{H}h_summary.json'
    json.dump(out, open(path, 'w'), indent=2, default=float)
    print(f"\nsaved {path}")


if __name__ == '__main__':
    main()
