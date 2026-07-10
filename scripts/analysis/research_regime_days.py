"""
POSITIVE vs NEGATIVE DAY research for the v21 1h ranker book.

Reads data/research/regime_days/daily_pnl.csv (contiguous OOS daily book P&L from build_daily_pnl).
Question: can we DETECT good vs bad days EX-ANTE (and thus skip the bad ones / size up the good)?

Discipline: predictors must be causally available BEFORE the day's trades. Macro features
(ranking_data_daily_macro_v3.csv) are merged with merge_asof STRICTLY BACKWARD (prior trading day's
row only) -> no look-ahead. Plus the strategy's OWN lagged daily P&L and day-of-week.

Targets:
  net  = combined long+short daily net bps (the deployable book; what we'd gate on)
  ls   = long_bps - short_bps (the market-DIRECTION axis; long & short days are -0.75 correlated)

Outputs: (1) Spearman correlation scan of each ex-ante feature vs net & vs ls;
         (2) OOS day-gate: walk-forward predict net, trade only predicted-good days, compare realized
             net of gated book vs trade-all vs a label-shuffle control.

RESEARCH ONLY (AGENTS.md). Run: python scripts/analysis/research_regime_days.py
"""
import os, sys, warnings
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

DAILY = 'data/research/regime_days/daily_pnl.csv'
MACRO = 'data/ranking_data_daily_macro_v3.csv'
COST = 10.0
SEED = 42
rng = np.random.default_rng(SEED)
# market-level (one value per date) macro features — the same 28 the transformer uses
MACRO_COLS = ['Breadth_AD_Ratio', 'Breadth_Pct_Above_SMA_50', 'Breadth_Pct_Above_SMA_200',
    'Breadth_Pct_Near_52W_High', 'Breadth_Return_Dispersion', 'Nifty50_Dist_SMA_20',
    'Nifty50_Dist_SMA_50', 'Nifty50_Dist_SMA_200', 'Nifty50_Return_5D', 'Nifty50_Return_20D',
    'Nifty500_Return_5D', 'Nifty500_Return_20D', 'VIX_Level', 'VIX_Change_5D', 'VIX_Percentile_1Y',
    'SP500_Return_1D', 'SP500_Change_5D', 'NASDAQ_Return_1D', 'NASDAQ_Change_5D', 'NIKKEI_Return_1D',
    'NIKKEI_Change_5D', 'HSI_Return_1D', 'HSI_Change_5D', 'USDINR_Change_5D', 'BRENT_Change_5D',
    'GOLD_Change_5D', 'DXY_Change_5D', 'US10Y_Change_5D']


def load():
    d = pd.read_csv(DAILY, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    d['ls'] = d['long_bps'] - d['short_bps']
    m = pd.read_csv(MACRO, usecols=['DateTime'] + MACRO_COLS)
    m['date'] = pd.to_datetime(m['DateTime']).dt.normalize()
    for c in MACRO_COLS:
        m[c] = pd.to_numeric(m[c], errors='coerce')
    m = m.dropna(subset=MACRO_COLS, how='all').drop_duplicates('date').sort_values('date').reset_index(drop=True)
    macro_cols = MACRO_COLS
    # STRICTLY BACKWARD asof: each strategy day d gets the prior trading day's macro row (ex-ante)
    merged = pd.merge_asof(d, m[['date'] + macro_cols], on='date',
                           direction='backward', allow_exact_matches=False)
    # strategy's own lagged P&L (shifted -> ex-ante) + calendar
    merged['pnl_lag1'] = merged['net_bps'].shift(1)
    merged['pnl_lag2'] = merged['net_bps'].shift(2)
    merged['pnl_roll5'] = merged['net_bps'].shift(1).rolling(5).mean()
    merged['dow'] = merged['date'].dt.dayofweek.astype(float)
    feat_cols = macro_cols + ['pnl_lag1', 'pnl_lag2', 'pnl_roll5', 'dow']
    return merged, feat_cols, macro_cols


def corr_scan(df, feat_cols):
    print("\n" + "=" * 92)
    print("EX-ANTE CORRELATION SCAN (Spearman rho; |rho| sorted). Targets: net (book), ls (long-short dir)")
    print("=" * 92)
    rows = []
    for f in feat_cols:
        sub = df.dropna(subset=[f, 'net_bps'])
        if len(sub) < 50 or sub[f].std() == 0:
            continue
        rn = spearmanr(sub[f], sub['net_bps']).correlation
        rl = spearmanr(sub[f], sub['ls']).correlation
        rows.append((f, rn, rl, len(sub)))
    rows.sort(key=lambda x: -abs(x[1]))
    print(f"  {'feature':28s} {'rho(net)':>9s} {'rho(ls)':>9s}  N")
    for f, rn, rl, n in rows[:15]:
        print(f"  {f:28s} {rn:>+9.3f} {rl:>+9.3f}  {n}")
    print("\n  (top |rho(ls)| — the market-direction axis)")
    for f, rn, rl, n in sorted(rows, key=lambda x: -abs(x[2]))[:8]:
        print(f"  {f:28s} {rn:>+9.3f} {rl:>+9.3f}  {n}")
    return rows


def oos_gate(df, feat_cols):
    """Walk-forward: expanding train, predict next block's daily net, trade only predicted-good days."""
    print("\n" + "=" * 92)
    print("OOS DAY-GATE: predict daily net (xgb reg), trade only predicted-good days vs trade-all")
    print("=" * 92)
    d = df.dropna(subset=['pnl_roll5']).reset_index(drop=True)   # drop warmup
    X = d[feat_cols].values.astype(np.float64); yv = d['net_bps'].values
    n = len(d); init = int(n * 0.5); step = 21
    params = {'objective': 'reg:squarederror', 'eta': 0.03, 'max_depth': 3, 'subsample': 0.8,
              'colsample_bytree': 0.7, 'min_child_weight': 20, 'lambda': 2.0, 'verbosity': 0,
              'tree_method': 'hist'}
    pred = np.full(n, np.nan)
    i = init
    while i < n:
        j = min(i + step, n)
        dtr = xgb.DMatrix(X[:i], label=yv[:i])
        bst = xgb.train(params, dtr, 200)
        pred[i:j] = bst.predict(xgb.DMatrix(X[i:j]))
        i = j
    m = ~np.isnan(pred)
    yt = yv[m]; pt = pred[m]
    ic = spearmanr(pt, yt).correlation
    print(f"  OOS days scored={m.sum()}  rank-IC(pred, realized net)={ic:+.3f}")
    allmean = yt.mean()
    for q in (0.5, 0.7, 0.8):                                    # trade only top-(1-q) predicted days
        thr = np.quantile(pt, q)
        sel = pt >= thr
        gated = yt[sel].mean() if sel.any() else float('nan')
        # shuffle control: random same-size subset
        ctrl = np.mean([yt[rng.choice(len(yt), sel.sum(), replace=False)].mean() for _ in range(500)])
        print(f"  trade top {int((1-q)*100)}% predicted days (n={sel.sum():>3}): realized net={gated:>+6.2f}bps "
              f"| trade-all={allmean:+.2f} | random-subset={ctrl:+.2f}")
    # also: does predicting the >0 net day work at all? (only ~5% positive -> hard)
    pos_rate = (yt > 0).mean()
    print(f"  base rate net>0 days = {pos_rate*100:.1f}%  (trading any subset still pays full cost)")


def side_timing_oos(df, feat_cols):
    """The one weak signal was market-trend -> long-vs-short axis. Test if timing the favored SIDE
    (long-only on predicted long-favored days, short-only otherwise) is monetizable OOS."""
    print("\n" + "=" * 92)
    print("OOS SIDE-TIMING: predict ls=long-short (xgb reg), trade the predicted-favored side only")
    print("=" * 92)
    d = df.dropna(subset=['pnl_roll5']).reset_index(drop=True)
    X = d[feat_cols].values.astype(np.float64); yls = d['ls'].values
    L = d['long_bps'].values; S = d['short_bps'].values
    n = len(d); init = int(n * 0.5); step = 21
    params = {'objective': 'reg:squarederror', 'eta': 0.03, 'max_depth': 3, 'subsample': 0.8,
              'colsample_bytree': 0.7, 'min_child_weight': 20, 'lambda': 2.0, 'verbosity': 0, 'tree_method': 'hist'}
    pred = np.full(n, np.nan); i = init
    while i < n:
        j = min(i + step, n)
        bst = xgb.train(params, xgb.DMatrix(X[:i], label=yls[:i]), 200)
        pred[i:j] = bst.predict(xgb.DMatrix(X[i:j])); i = j
    m = ~np.isnan(pred)
    ic = spearmanr(pred[m], yls[m]).correlation
    timed = np.where(pred[m] > np.median(pred[m]), L[m], S[m])     # long on predicted-long-favored else short
    print(f"  OOS days={m.sum()}  rank-IC(pred ls, realized ls)={ic:+.3f}")
    print(f"  timed-side net={timed.mean():+.2f}bps | long-only={L[m].mean():+.2f} | short-only={S[m].mean():+.2f} "
          f"| hedged(book)={df['net_bps'][df.index.isin(d.index[m])].mean():+.2f}")
    print("  (timing the side helps only if timed-side > both long-only and short-only, and clears 0)")


def main():
    df, feat_cols, macro_cols = load()
    print(f"days={len(df)}  ex-ante features={len(feat_cols)} ({len(macro_cols)} macro + own-lag + dow)")
    print(f"daily net: mean {df['net_bps'].mean():+.2f}bps  std {df['net_bps'].std():.2f}  "
          f"pos-day% {(df['net_bps']>0).mean()*100:.1f}")
    corr_scan(df, feat_cols)
    oos_gate(df, feat_cols)
    side_timing_oos(df, feat_cols)
    print("\nREAD: detectable iff (a) some feature has stable |rho| AND (b) the gated book's realized net "
          "beats trade-all AND a random subset, ideally clearing 0. ls-axis = market direction (timing bet).")


if __name__ == '__main__':
    main()
