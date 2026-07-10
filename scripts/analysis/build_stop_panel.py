"""
Build a TRADE PANEL WITH THE INTRABAR (15-min low/high) PATH for honest stop-loss research.

The dualtf panel only has 15-min CLOSE checkpoints, which makes tight stops look like they rarely
trigger (a resolution artifact). Here we attach, for each of the ranker's actual Top-K trades, the
real 15-min OHLC path from the raw cache out to 3h: per 15-min step we record the bar's CLOSE
(for horizon exit / checkpoints) AND its LOW & HIGH (for the worst intrabar excursion a stop would
actually hit). Fill assumption for a stop is then "-level" only if the bar's low/high pierced it.

Trades = v21 1h ranker (rank:pairwise, same recipe as eval_2h_v20) Top-K LONG and Top-K SHORT per
decision time, purged-monthly walk-forward OOS. We ALSO record K RANDOM names per decision time as a
control (does a stop "help" the random book as much as the model book? -> stop is just tail-truncation
on a negative-drift distribution, not alpha).

Output: data/research/stop_research/trade_path_panel.parquet  (one row per trade)
  cols: fold, T, ticker, side('long'|'short'|'rand_long'|'rand_short'), entry,
        rc_1..rc_12 (close ret vs entry), rl_1..rl_12 (low ret), rh_1..rh_12 (high ret)   [12 x 15min = 3h]
RESEARCH ONLY (AGENTS.md). Run: python scripts/analysis/build_stop_panel.py [--k 10]
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
from scripts.research.eval_2h_v20 import folds_of, int_ranks, _gpu, PARAMS, BASE_EXCLUDE, RET1
from scripts.research.build_v21_rolling_1h_panel import _load_raw

PANEL = 'data/research/v21_rolling_1h/panel.parquet'
RAW   = 'data/raw_upstox_cache_15min_3y'
UNIV  = 'data/research/v21_rolling_1h/universe.json'
OUTDIR = 'data/research/stop_research'
OUT = os.path.join(OUTDIR, 'trade_path_panel.parquet')
NSTEP = 12                     # 12 x 15min = 180min = 3h path
STEP = pd.Timedelta('15min')
SEED = 42
os.makedirs(OUTDIR, exist_ok=True)


def train_side(X, y, q, tr, va, invert):
    Xtr, ytr, qtr = X[tr], y[tr], q[tr]; Xva, yva, qva = X[va], y[va], q[va]
    gtr = pd.Series(qtr).groupby(qtr).size().values; gva = pd.Series(qva).groupby(qva).size().values
    dtr = xgb.DMatrix(Xtr, label=int_ranks(ytr, qtr, invert)); dtr.set_group(gtr)
    dva = xgb.DMatrix(Xva, label=int_ranks(yva, qva, invert)); dva.set_group(gva)
    return xgb.train(PARAMS, dtr, 500, evals=[(dva, 'v')], early_stopping_rounds=50, verbose_eval=False)


def pick_trades(df_te, sL, sS, K, rng):
    """Per decision time (Query_ID): top-K long, top-K short, K random. Returns list of (T,ticker,side)."""
    out = []
    df_te = df_te.assign(_sL=sL, _sS=sS)
    for _, g in df_te.groupby('Query_ID'):
        if len(g) < K + 1:
            continue
        gi = g.reset_index(drop=True)
        T = gi['DateTime'].iloc[0]
        long_idx = np.argsort(-gi['_sL'].values)[:K]
        short_idx = np.argsort(-gi['_sS'].values)[:K]
        rand_idx = rng.choice(len(gi), size=min(K, len(gi)), replace=False)
        for i in long_idx:
            out.append((T, gi['Ticker'].iloc[i], 'long'))
        for i in short_idx:
            out.append((T, gi['Ticker'].iloc[i], 'short'))
        for i in rand_idx:
            out.append((T, gi['Ticker'].iloc[i], 'rand'))
    return out


def attach_paths(trades, entry_lookup):
    """trades: DataFrame[T,ticker,side,fold]. Attach entry + 12-step close/low/high returns per ticker
    from the raw 15-min OHLC (bar start == step time). Vectorized per ticker."""
    cols_rc = [f'rc_{k}' for k in range(1, NSTEP + 1)]
    cols_rl = [f'rl_{k}' for k in range(1, NSTEP + 1)]
    cols_rh = [f'rh_{k}' for k in range(1, NSTEP + 1)]
    frames = []
    for tk, g in trades.groupby('ticker'):
        fp = os.path.join(RAW, tk + '.csv')
        if not os.path.exists(fp):
            continue
        raw = _load_raw(fp, hygiene=False)                    # all bars (path must be continuous)
        raw = raw.drop_duplicates('DateTime').set_index('DateTime').sort_index()
        C = raw['Close']; L = raw['Low']; H = raw['High']
        g = g.copy()
        T = pd.to_datetime(g['T'].values)
        entry = g['T'].map(lambda t: entry_lookup.get((tk, pd.Timestamp(t)), np.nan)).values.astype(float)
        # forward bar-start times: T + (k-1)*15min closes at T + k*15min
        rc = np.full((len(g), NSTEP), np.nan); rl = np.full((len(g), NSTEP), np.nan); rh = np.full((len(g), NSTEP), np.nan)
        for k in range(NSTEP):
            starts = pd.DatetimeIndex(T) + k * STEP           # bar covering [T+k*15, T+(k+1)*15]
            cc = C.reindex(starts).values; ll = L.reindex(starts).values; hh = H.reindex(starts).values
            rc[:, k] = cc / entry - 1.0; rl[:, k] = ll / entry - 1.0; rh[:, k] = hh / entry - 1.0
        gg = g[['fold', 'T', 'ticker', 'side']].copy()
        gg['entry'] = entry
        gg[cols_rc] = rc; gg[cols_rl] = rl; gg[cols_rh] = rh
        frames.append(gg)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int, default=10); args = ap.parse_args()
    K = args.k
    PARAMS['device'] = _gpu(); rng = np.random.default_rng(SEED)
    print(f"device={PARAMS['device']}  K={K}")
    df = pd.read_parquet(PANEL); df['DateTime'] = pd.to_datetime(df['DateTime'])
    feats = [c for c in df.columns if c not in BASE_EXCLUDE]
    print(f"panel rows={len(df):,} feats={len(feats)}")
    X = df[feats].values.astype(np.float64)
    if not np.isfinite(X).all():
        cm = np.nan_to_num(np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0))
        bad = np.where(~np.isfinite(X)); X[bad] = np.take(cm, bad[1])
    y = df[RET1].values.astype(np.float64); q = df['Query_ID'].values
    entry_lookup = {(t, dt): c for t, dt, c in zip(df['Ticker'].values,
                    pd.to_datetime(df['DateTime']).values, df['Close'].values)}
    entry_lookup = {(t, pd.Timestamp(dt)): c for (t, dt), c in entry_lookup.items()}

    all_trades = []
    folds = folds_of(df)
    print(f"WF folds={len(folds)}")
    for fi, (tr, va, te) in enumerate(folds):
        bL = train_side(X, y, q, tr, va, invert=False)
        bS = train_side(X, y, q, tr, va, invert=True)
        dte = xgb.DMatrix(X[te])
        sL = bL.predict(dte); sS = bS.predict(dte)
        df_te = df.iloc[np.where(te)[0]][['DateTime', 'Ticker', 'Query_ID']].reset_index(drop=True)
        trd = pick_trades(df_te, sL, sS, K, rng)
        t = pd.DataFrame(trd, columns=['T', 'ticker', 'side']); t['fold'] = fi + 1
        all_trades.append(t)
        print(f"  fold {fi+1}/{len(folds)}: {len(t):,} trade-legs (test queries)")
    trades = pd.concat(all_trades, ignore_index=True)
    # split 'rand' into rand_long/rand_short later in the sweep (sign-agnostic here; keep as 'rand')
    print(f"total trade-legs={len(trades):,}  attaching intrabar paths from raw cache ...")
    out = attach_paths(trades, entry_lookup)
    for c in out.columns:
        if c.startswith(('rc_', 'rl_', 'rh_', 'entry')):
            out[c] = out[c].astype('float32')
    out.to_parquet(OUT, index=False)
    cov = out[[f'rc_{NSTEP}']].notna().mean().iloc[0]
    print(f"\nsaved {OUT}  rows={len(out):,}  side counts={out['side'].value_counts().to_dict()}")
    print(f"  3h-close path present on {cov*100:.1f}% of legs (rest hit session end before 3h)")


if __name__ == '__main__':
    main()
