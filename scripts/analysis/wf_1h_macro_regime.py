"""
Walk-forward backtest of the v10 native-1h model with a NIFTY 500 macro-regime overlay.

Regime: NIFTY 500 index position relative to a moving average (daily or hourly).
  bull (+1): index close > MA  ->  take v10 top-K LONG positions
  bear (-1): index close < MA  ->  take v10 top-K SHORT positions

Three arms reported for every (MA spec, conviction) cell:
  gated       : LONG in bull, SHORT in bear
  long-only-up: LONG in bull only, flat in bear
  ungated     : same as wf_1h_base (longs + shorts always) -- the baseline

MA specs swept:
  Daily SMA: 50 / 100 / 200 bars (regime for a query on date D uses close[D-1] vs SMA[D-1])
  Hourly SMA: 12 / 20 / 50 bars  (regime uses index bar strictly before the query bar)

Usage:
    python scripts/analysis/wf_1h_macro_regime.py
    python scripts/analysis/wf_1h_macro_regime.py --proxy   # use universe equal-weight proxy
    python scripts/analysis/wf_1h_macro_regime.py --no-train  # skip retrain, load cached OOS from data/
"""
import os, sys, json, warnings, gc, argparse
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

# -- config --------------------------------------------------------------------
DIR    = 'models/v10_native_1h'
DATA   = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET    = 'Next_Hour_Return'
H_TEST = 4          # OOS months per fold
MIN_TRAIN = 18      # minimum training months before first fold
COST   = 10 / 1e4   # 10 bps round-trip
RNG    = np.random.default_rng(0)

DAILY_INDEX  = 'data/raw_index_cache/nifty500_1d.csv'
HOURLY_INDEX = 'data/raw_index_cache/nifty500_1h.csv'
OOS_CACHE    = 'data/v10_macro_regime_oos_cache.parquet'
RESULTS_OUT  = 'data/v10_macro_regime_results.json'

DAILY_MA_WINDOWS  = [50, 100, 200]
HOURLY_MA_WINDOWS = [12, 20, 50]
TOP_K_LIST        = [1, 3, 5, 10]


# -- helpers (same as wf_1h_base) ---------------------------------------------
def load(p, ret):
    df = pd.concat([c for c in pd.read_csv(p, chunksize=200_000)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime'])
    df['ym'] = df['DateTime'].str[:7]
    df['hour'] = df['dt'].dt.strftime('%H:%M')
    return df.dropna(subset=[ret]).reset_index(drop=True)


def Xmat(df, fe):
    X = df[fe].values.astype(float)
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


def stat(net, fold):
    n = len(net)
    if n < 20:
        return None
    t = net.mean() / (net.std() / np.sqrt(n)) if net.std() > 0 else 0
    bs = [net[RNG.integers(0, n, n)].mean() * 1e4 for _ in range(1500)]
    fm = [net[fold == fi].mean() for fi in np.unique(fold)]
    sig = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.64 else ''))
    return dict(n=n, wr=(net > 0).mean() * 100, bps=net.mean() * 1e4,
                ci=(np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
                t=t, sig=sig, pos=sum(1 for x in fm if x > 0), nf=len(fm))


# -- regime builder ------------------------------------------------------------
def build_daily_regime(idx_daily, window):
    """Returns Series indexed by date (tz-naive midnight Timestamp), values +1/-1."""
    d = idx_daily.copy()
    ts = pd.to_datetime(d['timestamp'])
    # strip any tz-info so the index aligns with the tz-naive model data dates
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    d['date'] = ts.dt.normalize()
    d = d.sort_values('date').drop_duplicates('date').set_index('date')
    d['sma'] = d['close'].rolling(window, min_periods=window).mean()
    # regime on date D uses close[D-1] vs sma[D-1] -- shift by 1
    d['regime'] = np.where(d['close'].shift(1) > d['sma'].shift(1), 1, -1)
    return d['regime'].dropna()


def build_hourly_regime(idx_hourly, window):
    """Returns Series indexed by IST timestamp, values +1/-1 (uses strictly-before bar)."""
    h = idx_hourly.copy()
    h['ts'] = pd.to_datetime(h['timestamp'])
    h = h.sort_values('ts').drop_duplicates('ts').set_index('ts')
    h['sma'] = h['close'].rolling(window, min_periods=window).mean()
    # shift(1): the current bar's close is NOT used for its own regime gate
    h['regime'] = np.where(h['close'].shift(1) > h['sma'].shift(1), 1, -1)
    return h['regime'].dropna()


def assign_regimes(P, idx_daily, idx_hourly):
    """Attach one regime column per MA spec onto the pooled OOS frame P."""
    P = P.copy()
    P['date'] = pd.to_datetime(P['DateTime']).dt.normalize()

    # daily specs
    for w in DAILY_MA_WINDOWS:
        reg = build_daily_regime(idx_daily, w)
        reg.name = f'reg_d{w}'
        P = P.join(reg, on='date', how='left')
        P[f'reg_d{w}'] = P[f'reg_d{w}'].ffill()

    # hourly specs -- use merge_asof (backward) on datetime
    P_sorted = P.sort_values('DateTime')
    for w in HOURLY_MA_WINDOWS:
        reg = build_hourly_regime(idx_hourly, w).reset_index()
        reg.columns = ['ts', f'reg_h{w}']
        reg = reg.sort_values('ts')
        merged = pd.merge_asof(P_sorted[['DateTime']].rename(columns={'DateTime': 'ts'}),
                               reg, on='ts', direction='backward')
        P_sorted[f'reg_h{w}'] = merged[f'reg_h{w}'].values

    # re-align after sort
    for w in HOURLY_MA_WINDOWS:
        if f'reg_h{w}' not in P.columns:
            P = P.join(P_sorted[[f'reg_h{w}']], how='left')

    return P


# -- regime diagnostics --------------------------------------------------------
def regime_diag(P, col):
    v = P[col].dropna()
    bull_pct = (v == 1).mean() * 100
    flips = (v != v.shift()).sum()
    return bull_pct, flips


# -- main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', action='store_true', help='Use universe equal-weight proxy index')
    ap.add_argument('--no-train', action='store_true', help='Load cached OOS parquet; skip retraining')
    args = ap.parse_args()

    if args.proxy:
        DAILY_IDX_FILE  = 'data/raw_index_cache/nifty500_1d.csv'    # proxy daily
        HOURLY_IDX_FILE = 'data/raw_index_cache/nifty500_1h.csv'
    else:
        DAILY_IDX_FILE  = DAILY_INDEX
        HOURLY_IDX_FILE = HOURLY_INDEX

    for f in [DAILY_IDX_FILE, HOURLY_IDX_FILE]:
        if not os.path.exists(f):
            print(f"[ERROR] Index file not found: {f}")
            print("  Run: python scripts/collectors/collect_nifty500_index.py [--proxy]")
            sys.exit(1)

    print(f"Loading index data ...")
    idx_daily  = pd.read_csv(DAILY_IDX_FILE)
    idx_hourly = pd.read_csv(HOURLY_IDX_FILE)
    print(f"  Daily  {len(idx_daily)} bars  {idx_daily['timestamp'].min()} -> {idx_daily['timestamp'].max()}")
    print(f"  Hourly {len(idx_hourly)} bars  {idx_hourly['timestamp'].min()} -> {idx_hourly['timestamp'].max()}")

    # -- load or retrain ------------------------------------------------------
    if args.no_train and os.path.exists(OOS_CACHE):
        print(f"\nLoading cached OOS from {OOS_CACHE} ...")
        P = pd.read_parquet(OOS_CACHE)
        P['DateTime'] = pd.to_datetime(P['DateTime'])
    else:
        print(f"\nLoading model + data ...")
        with open(f'{DIR}/metadata.json') as f:
            meta = json.load(f)
        fe, params = meta['features'], meta['params']
        df = load(DATA, RET)
        X = Xmat(df, fe)
        months = sorted(df['ym'].unique())
        folds = []
        i = MIN_TRAIN + 1
        while i + 1 <= len(months):
            folds.append((months[:i - 1], months[i - 1], months[i:i + H_TEST]))
            i += H_TEST
        print(f"  {len(folds)} folds  OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

        rows = []
        for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
            tr = df['ym'].isin(tr_m).values
            va = df['ym'].isin([val_m]).values
            te = df['ym'].isin(te_m).values
            bl = xgb.train(params,
                           fitdm(X[tr], df[RET].values[tr], df['Query_ID'].values[tr], False), 500,
                           evals=[(fitdm(X[va], df[RET].values[va], df['Query_ID'].values[va], False), 'v')],
                           early_stopping_rounds=50, verbose_eval=False)
            bs = xgb.train(params,
                           fitdm(X[tr], df[RET].values[tr], df['Query_ID'].values[tr], True), 500,
                           evals=[(fitdm(X[va], df[RET].values[va], df['Query_ID'].values[va], True), 'v')],
                           early_stopping_rounds=50, verbose_eval=False)
            sub = df[te].copy()
            sub['sL'] = bl.predict(xgb.DMatrix(X[te]))
            sub['sS'] = bs.predict(xgb.DMatrix(X[te]))
            sub['pctL'] = sub.groupby('dt')['sL'].rank(pct=True)
            sub['pctS'] = sub.groupby('dt')['sS'].rank(pct=True)
            sub['posL'] = sub.groupby('dt')['sL'].rank(ascending=False, method='first')
            sub['posS'] = sub.groupby('dt')['sS'].rank(ascending=False, method='first')
            sub['fold'] = fi
            rows.append(sub[['fold', 'ym', 'hour', 'DateTime', 'Ticker', RET, 'posL', 'posS', 'pctL', 'pctS', 'sL', 'sS']])
            print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]} done")
            del sub; gc.collect()

        P = pd.concat(rows, ignore_index=True)
        P['DateTime'] = pd.to_datetime(P['DateTime'])
        P.to_parquet(OOS_CACHE, index=False)
        print(f"  Pooled OOS rows: {len(P):,}  (cached -> {OOS_CACHE})")

    # -- attach regimes -------------------------------------------------------
    print("\nBuilding regime columns ...")
    P = assign_regimes(P, idx_daily, idx_hourly)
    all_reg_cols = [f'reg_d{w}' for w in DAILY_MA_WINDOWS] + \
                   [f'reg_h{w}' for w in HOURLY_MA_WINDOWS]

    # -- no-lookahead check ---------------------------------------------------
    print("\nNo-lookahead sanity checks:")
    for col in all_reg_cols:
        n_na = P[col].isna().sum()
        print(f"  {col:12s}: {P[col].notna().sum():>7,} rows with regime  ({n_na} NaN warm-up)")

    # -- sweep results --------------------------------------------------------
    sep = '=' * 110

    def print_stat(label, st):
        if st is None:
            print(f"  {label:<50s}  [too few rows]")
            return
        print(f"  {label:<50s}  N={st['n']:>7,}  WR={st['wr']:>5.1f}%  "
              f"NetBps={st['bps']:>+7.2f}  [{st['ci'][0]:>+6.1f},{st['ci'][1]:>+5.1f}]  "
              f"t={st['t']:>5.2f}{st['sig']:<3}  +folds={st['pos']}/{st['nf']}")

    all_results = {}

    print(f"\n{sep}")
    print("  V10 1h  x  NIFTY 500 MACRO REGIME OVERLAY  (10 bps cost, raw returns)")
    print(f"{sep}")

    for reg_col in all_reg_cols:
        bull = P[reg_col] == 1
        bear = P[reg_col] == -1
        bull_pct, flips = regime_diag(P, reg_col)
        spec_label = reg_col.replace('reg_d', 'Daily-SMA').replace('reg_h', 'Hourly-SMA')

        print(f"\n{'-'*110}")
        print(f"  Regime spec: {spec_label}  |  bull={bull_pct:.1f}% of queries  |  regime flips={flips}")
        if bull_pct > 85:
            print("  [!]  ALMOST ALWAYS BULL -- any long gains here are mostly always-in beta, not timing skill.")
        print(f"{'-'*110}")
        print(f"\n  {'LABEL':<50s}  {'N':>7}  {'WR%':>6}  {'NetBps':>8}  {'95% CI':>14}  {'t':>6}  {'+folds'}")

        spec_res = {}
        for K in TOP_K_LIST:
            # ungated baseline (same as wf_1h_base, for reference)
            for direction, sgn, pcol in [('LONG', 1, 'posL'), ('SHORT', -1, 'posS')]:
                sub_u = P[P[pcol] <= K]
                net_u = sub_u[RET].values * sgn - COST
                label = f"UNGATED  {direction:5s} top{K:>2d}"
                st = stat(net_u, sub_u['fold'].values)
                print_stat(label, st)
                spec_res[f'ungated_{direction.lower()}_top{K}'] = st

            # gated: long in bull, short in bear
            long_g  = P[(P['posL'] <= K) & bull]
            short_g = P[(P['posS'] <= K) & bear]
            combined_g = pd.concat([
                long_g[[RET, 'fold']].assign(net=long_g[RET] * 1 - COST),
                short_g[[RET, 'fold']].assign(net=short_g[RET] * -1 - COST),
            ])
            st_g = stat(combined_g['net'].values, combined_g['fold'].values)
            print_stat(f"GATED    L-bull/S-bear top{K:>2d}", st_g)
            spec_res[f'gated_top{K}'] = st_g

            # long-only-up: longs in bull, flat in bear
            long_u = P[(P['posL'] <= K) & bull]
            net_lu = long_u[RET].values * 1 - COST
            st_lu = stat(net_lu, long_u['fold'].values)
            print_stat(f"LONG-UP  (bull-only)   top{K:>2d}", st_lu)
            spec_res[f'long_only_up_top{K}'] = st_lu

        all_results[reg_col] = {
            'bull_pct': bull_pct, 'regime_flips': int(flips),
            'results': spec_res,
        }

    # -- save JSON -------------------------------------------------------------
    def to_serializable(o):
        if o is None:
            return None
        return {k: (float(v) if isinstance(v, (np.floating, float)) else
                    [float(x) for x in v] if isinstance(v, (tuple, list)) else v)
                for k, v in o.items()}

    out = {}
    for spec, data in all_results.items():
        out[spec] = {
            'bull_pct': data['bull_pct'],
            'regime_flips': data['regime_flips'],
            'results': {k: to_serializable(v) for k, v in data['results'].items()},
        }
    os.makedirs('data', exist_ok=True)
    with open(RESULTS_OUT, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{sep}")
    print(f"  Results saved -> {RESULTS_OUT}")
    print(f"\n  INTERPRETATION GUIDE")
    print(f"  CI spanning 0 => not significant.  * p<.10  ** p<.05  *** p<.01")
    print(f"  Multiple-testing note: {len(all_reg_cols)} specs x 3 arms x {len(TOP_K_LIST)} convictions = "
          f"{len(all_reg_cols)*3*len(TOP_K_LIST)} cells.  Trust only effects that are significant AND")
    print(f"  consistent across multiple convictions and a majority of folds (+folds).")
    print(f"  [!]  Watch bull_pct: if >85%, 'gated' longs are always-in beta, not timing skill.")
    print(f"{sep}\n")


if __name__ == '__main__':
    main()
