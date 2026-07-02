"""
Build the v21 ROLLING-1h RESEARCH panel — the "cleanest" v20.

v21 = v20's rolling-1h recipe + data-cleaning + tweaks. Source is the SAME Upstox 15-min
cache (`data/raw_upstox_cache_15min_3y/`), which is ALREADY split/bonus-adjusted (verified:
no discontinuity at known ex-dates), so NO corporate-action adjustment is applied — doing so
would double-adjust. v21 differs from v20 (`build_rolling_1h_panel.py`) by:

  Phase 1   Liquidity universe  — keep the top-N names by median daily dollar-volume; drop the
            illiquid tail that both creates degenerate-bar fills and breaks the 10bps cost floor.
            Bar hygiene — drop frozen (High==Low) and zero-volume bars before windowing; the
            contiguity guard then naturally excludes any window that would have contained one.
  Phase 1.5 Session-boundary candles — the overnight transition is no longer silently dropped.
            Emit `close_stub` (15:15→15:30 last bar) and `overnight` (prior close→next open,
            volume 0) as flagged NON-TRADABLE candles to a sidecar (`boundary_candles.parquet`,
            "masked-token ready" for a future sequence model). The tradable `panel.parquet`
            stays intraday-only but gains CAUSAL features: `Overnight_Gap_Prior` (cross-sectional,
            z-scored — the overnight-reversal signal), `Time_To_Close`, `Is_Last_Tradable_Hr` (raw).
  Phase 3   mask-not-fill — via compute_features(clean_v21=True): warmup/degenerate values are
            left NaN (dropped here) instead of filled with 0.0/0.5 that z-score into fake signal.
  Phase 4   T1 robust scoring — cross-sectional (x-median)/(1.4826*MAD) clipped to ±5 (outlier
            resistant) instead of (x-mean)/std.  T2 wall-clock lookback — in compute_features.
            T3 sector-graph neighbor feature — merged by a separate step (build_v21_graph_feature).

RESEARCH ONLY (AGENTS.md): no Gauntlet, no registry stamp. Overlapping windows → effective N ~1/4
of rows; point estimates only, no t-tests.

Output: data/research/v21_rolling_1h/{panel.parquet, boundary_candles.parquet, universe.json}
Run:    python scripts/research/build_v21_rolling_1h_panel.py [--top_n 110]
"""
import os, sys, glob, json, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

from scripts.feature_utils import compute_features

SRC_DIR     = 'data/raw_upstox_cache_15min_3y'        # already split/bonus-adjusted
OUT_DIR     = 'data/research/v21_rolling_1h'
OUT_PARQUET = os.path.join(OUT_DIR, 'panel.parquet')
OUT_BOUND   = os.path.join(OUT_DIR, 'boundary_candles.parquet')
OUT_UNIV    = os.path.join(OUT_DIR, 'universe.json')
STEP        = pd.Timedelta('15min')
HOUR        = pd.Timedelta('60min')
MIN_BARS    = 300
MIN_PER_Q   = 5
TOP_N       = 110            # liquidity universe size (of 172)
MAX_GAP_DAYS = 4            # a true "overnight" gap spans <= this many calendar days
os.makedirs(OUT_DIR, exist_ok=True)


def _load_raw(fp, hygiene=True):
    """Raw 15-min cache CSV -> clean naive-IST OHLCV frame; bar hygiene optional (ablation)."""
    raw = pd.read_csv(fp)
    dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = pd.DataFrame({
        'DateTime': dt,
        'Open': raw['open'].astype(float), 'High': raw['high'].astype(float),
        'Low': raw['low'].astype(float), 'Close': raw['close'].astype(float),
        'Volume': raw['volume'].astype(float),
    }).dropna(subset=['DateTime', 'Open', 'High', 'Low', 'Close'])
    df = df.drop_duplicates('DateTime').sort_values('DateTime').reset_index(drop=True)
    if hygiene:
        # ── BAR HYGIENE: drop frozen (High==Low) and non-positive-volume bars ──
        bad = (df['High'] <= df['Low']) | (df['Volume'] <= 0) | (~np.isfinite(df['Volume']))
        df = df[~bad].reset_index(drop=True)
    return df


GRAPH_EDGES   = 'data/research/graph/edges.csv'
AGG_FEATS_V21 = ['Return', 'MOM_12_pct', 'RSI_14', 'Volume_Zscore', 'Dist_SMA_50']


def _build_adjacency(universe):
    """Binary group & sector adjacency over the ordered `universe`, reusing the exogenous
    relation graph (build_relation_graph.py: group w=1.0, sector w=0.3 — used here binary,
    aggregated separately, matching the gate1 recipe)."""
    idx = {t: i for i, t in enumerate(universe)}
    N = len(universe)
    A_grp = np.zeros((N, N)); A_sec = np.zeros((N, N))
    e = pd.read_csv(GRAPH_EDGES)
    for _, r in e.iterrows():
        if r['src'] in idx and r['dst'] in idx:
            i, j = idx[r['src']], idx[r['dst']]
            A = A_grp if r['type'] == 'group' else A_sec
            A[i, j] = A[j, i] = 1.0
    return A_grp, A_sec, idx


def add_neighbor_features(df_all, universe, permute_seed=None):
    """Phase 4-T3: dynamic 1-layer message-pass. Per timestamp, append the group- and
    sector-neighbor MEAN (nan-aware) of a few momentum features, computed on RAW features
    BEFORE cross-sectional z-scoring (build_ranking then z-scores them like any feature).
    permute_seed!=None shuffles ticker<->node identity = negative control. Static topology
    is intentionally excluded (it memorizes in-sample per the Gate-1 finding)."""
    A_grp, A_sec, idx = _build_adjacency(universe)
    if permute_seed is not None:
        perm = np.random.default_rng(permute_seed).permutation(len(universe))
        A_grp = A_grp[np.ix_(perm, perm)]; A_sec = A_sec[np.ix_(perm, perm)]
    feats = [f for f in AGG_FEATS_V21 if f in df_all.columns]
    node = df_all['Ticker'].map(idx).values
    out = {f'nb_{tag}_{f}': np.zeros(len(df_all)) for f in feats for tag in ('grp', 'sec')}
    fvals = {f: df_all[f].values for f in feats}
    for _, g in df_all.groupby('DateTime'):
        rows = g.index.values
        present = node[rows]
        keep = ~pd.isna(present)
        if keep.sum() < 2:
            continue
        rows = rows[keep]; p = present[keep].astype(int)
        V = np.column_stack([fvals[f][rows] for f in feats])
        valid = (~np.isnan(V)).astype(float); Vf = np.nan_to_num(V)
        for tag, A in (('grp', A_grp), ('sec', A_sec)):
            sub = A[np.ix_(p, p)]
            num = sub @ Vf; den = sub @ valid
            nb = np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)
            for fi, f in enumerate(feats):
                out[f'nb_{tag}_{f}'][rows] = nb[:, fi]
    for c, arr in out.items():
        df_all[c] = arr
    return df_all, list(out.keys())


def build_universe(files):
    """Top-N base tickers by MEDIAN daily dollar-volume. Writes universe.json."""
    adv = {}
    for fp in tqdm(files, desc='ADV scan'):
        tk = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = _load_raw(fp)
            if len(df) < MIN_BARS:
                continue
            dv = (df['Close'] * df['Volume'])
            day = df['DateTime'].dt.date
            adv[tk] = float(dv.groupby(day).sum().median())
        except Exception:
            continue
    ranked = sorted(adv.items(), key=lambda x: -x[1])
    keep = [t for t, _ in ranked[:TOP_N]]
    with open(OUT_UNIV, 'w') as f:
        json.dump({'top_n': TOP_N, 'metric': 'median_daily_dollar_volume_inr',
                   'tickers': keep, 'adv_inr': {t: adv[t] for t in keep}}, f, indent=2)
    print(f"Universe: kept {len(keep)}/{len(adv)} tickers by ADV -> {OUT_UNIV}")
    print(f"  most liquid: {ranked[0][0]} ({ranked[0][1]/1e7:.1f} cr)  | "
          f"cutoff #{TOP_N}: {ranked[min(TOP_N, len(ranked))-1][0]} ({ranked[min(TOP_N, len(ranked))-1][1]/1e7:.1f} cr)")
    return keep


def build_ticker(ticker, df, clean_feats=True, gap_feats=True):
    """Hygiene'd 15-min OHLCV -> (intraday rolling-1h feature rows, boundary candle rows).
    clean_feats -> compute_features(clean_v21=...); gap_feats -> add causal session/gap features
    (both toggleable for the leave-one-out ablation; default True = full v21)."""
    if len(df) < MIN_BARS:
        return None, None
    t = df['DateTime']

    # ── INTRADAY rolling 1h windows (= 4 consecutive 15-min bars), v20-faithful ──
    win = pd.DataFrame({
        'DateTime': t + STEP,                       # entry/close time T
        'Open':   df['Open'].shift(3),
        'High':   df['High'].rolling(4).max(),
        'Low':    df['Low'].rolling(4).min(),
        'Close':  df['Close'],
        'Volume': df['Volume'].rolling(4).sum(),
    })
    contiguous = (t - t.shift(3)) == (3 * STEP)
    win = win[contiguous.values].dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    win = win.drop_duplicates('DateTime').sort_values('DateTime').set_index('DateTime')
    if len(win) < MIN_BARS:
        return None, None

    feat = compute_features(win[['Open', 'High', 'Low', 'Close', 'Volume']].copy(),
                            legacy=False, clean_v21=clean_feats)
    close_at = feat['Close']
    fwd = close_at.reindex(close_at.index + HOUR)
    feat['Next_Hour_Return'] = fwd.values / close_at.values - 1.0   # session-masked (no overnight leak)

    # ── per-day open/close for causal gap features + boundary candles ──
    day = df['DateTime'].dt.date
    day_first = df.groupby(day).first()          # first bar of each session
    day_last  = df.groupby(day).last()           # last bar (the 15:15->15:30 close stub)
    days = list(day_first.index)
    first_open  = pd.Series(day_first['Open'].values, index=pd.Index(days))
    first_dt    = pd.Series(day_first['DateTime'].values, index=pd.Index(days))
    last_close  = pd.Series(day_last['Close'].values, index=pd.Index(days))
    prev_close  = last_close.shift(1)
    prev_day    = pd.Series(days, index=pd.Index(days)).shift(1)
    gap_days    = pd.to_datetime(pd.Series(days)).values - pd.to_datetime(prev_day.values)
    gap_ok      = pd.Series([(pd.notna(p) and (g / np.timedelta64(1, 'D')) <= MAX_GAP_DAYS)
                             for p, g in zip(prev_day.values, gap_days)], index=pd.Index(days))
    overnight_gap = (first_open / prev_close - 1.0).where(gap_ok)    # causal, realized at the open

    # CAUSAL session/gap features onto each intraday window by its session date (Phase 1.5).
    # Toggleable for the ablation; the per-day block above still feeds the boundary candles.
    if gap_feats:
        wdate = pd.Series(feat.index.date, index=feat.index)
        feat['Overnight_Gap_Prior'] = wdate.map(overnight_gap).values
        mins_to_close = (15 * 60 + 30) - (feat.index.hour * 60 + feat.index.minute)
        feat['Time_To_Close'] = np.clip(mins_to_close / 60.0, 0, None)
        # last tradable window of each session = latest window that still has a forward label
        # (late-day windows with NaN Next_Hour_Return are dropped in build_ranking, so flag the
        # last LABELED one — else this flag would always land on a row that gets dropped).
        ts = pd.Series(feat.index, index=feat.index).where(feat['Next_Hour_Return'].notna())
        day_max_T = ts.groupby(feat.index.date).transform('max')
        feat['Is_Last_Tradable_Hr'] = (feat.index == day_max_T.values).astype(float)

    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    feat['candle_type'] = 'intraday'
    feat['tradable'] = 1

    # ── BOUNDARY candles (non-tradable; sidecar for sequence models) ──
    bdfs = []
    # close_stub: the last 15-min bar of each session
    cs = day_last.copy()
    cs = pd.DataFrame({'DateTime': cs['DateTime'].values,
                       'Open': cs['Open'].values, 'High': cs['High'].values,
                       'Low': cs['Low'].values, 'Close': cs['Close'].values,
                       'Volume': cs['Volume'].values})
    cs['candle_type'] = 'close_stub'
    bdfs.append(cs)
    # overnight: synthetic gap candle prior-close -> next-open
    on = pd.DataFrame({
        'DateTime': pd.to_datetime(first_dt.values),     # keyed at the next session open
        'Open': prev_close.values, 'Close': first_open.values,
        'High': np.nanmax([prev_close.values, first_open.values], axis=0),
        'Low':  np.nanmin([prev_close.values, first_open.values], axis=0),
        'Volume': 0.0,
    }).iloc[1:]   # first session has no prior
    on = on[gap_ok.values[1:]]                           # only true overnight gaps
    on['candle_type'] = 'overnight'
    bdfs.append(on)
    bound = pd.concat(bdfs, ignore_index=True)
    bound['Ticker'] = ticker
    bound['tradable'] = 0
    return feat, bound


def build_ranking(df_all, robust=True):
    """Per-query cross-sectional scoring. robust=True -> median/MAD winsorized ±5 (Phase 4 T1);
    robust=False -> v20-style mean/std (for the ablation). Overnight_Gap_Prior is z-scored
    (cross-sectional); session-position flags stay raw."""
    df_all = df_all.copy()
    df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    df_all = df_all.dropna(subset=['Next_Hour_Return']).sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    sizes = df_all.groupby('Query_ID').size()
    df_all = df_all[df_all['Query_ID'].isin(sizes[sizes >= MIN_PER_Q].index)].copy()
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
    df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
    df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
    df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)
    # columns that are metadata or kept-raw (NOT z-scored). Session-position flags are constant
    # across a query, so z-scoring them would zero them out — keep raw for absolute context.
    exclude = {'DateTime', 'Query_ID', 'Ticker', 'Next_Hour_Return', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Market_Mean_Return', 'Relative_Return', 'Market_Mean_Volatility', 'Relative_Volatility',
               'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close',
               'Is_Last_Tradable_Hr', 'candle_type', 'tradable'}
    feat_cols = [c for c in df_all.columns if c not in exclude]
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    g = df_all.groupby('Query_ID')
    for col in tqdm(feat_cols, desc='  z-score', leave=False):
        if robust:
            med = g[col].transform('median')
            mad = (df_all[col] - med).abs().groupby(df_all['Query_ID']).transform('median')
            df_all[col] = ((df_all[col] - med) / (1.4826 * mad + 1e-8)).clip(-5.0, 5.0)
        else:
            mean = g[col].transform('mean'); std = g[col].transform('std')
            df_all[col] = (df_all[col] - mean) / (std + 1e-8)
    return df_all.dropna(subset=feat_cols), feat_cols


def main(top_n, graph_mode='on', robust=True):
    global TOP_N
    TOP_N = top_n
    files = sorted(glob.glob(os.path.join(SRC_DIR, '*.csv')))
    print(f"Source: {SRC_DIR} ({len(files)} tickers)  |  top_n={TOP_N}  |  graph={graph_mode}  |  robust={robust}")
    universe = build_universe(files)
    keep = set(universe)
    files = [fp for fp in files if os.path.splitext(os.path.basename(fp))[0] in keep]

    frames, bounds, ok, skip = [], [], 0, 0
    for fp in tqdm(files, desc='Tickers'):
        ticker = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = _load_raw(fp)
            f, b = build_ticker(ticker, df)
            if f is not None and len(f):
                frames.append(f); bounds.append(b); ok += 1
            else:
                skip += 1
        except Exception as e:
            skip += 1; tqdm.write(f"  [skip] {ticker}: {str(e)[:80]}")
    print(f"  tickers OK={ok} skip={skip}")

    df_all = pd.concat(frames, ignore_index=True)
    if graph_mode != 'off':
        ps = 42 if graph_mode == 'permute' else None
        df_all, nb_cols = add_neighbor_features(df_all, universe, permute_seed=ps)
        print(f"  graph: +{len(nb_cols)} neighbor features (mode={graph_mode})")
    final, fc = build_ranking(df_all, robust=robust)

    fcast = fc + ['Next_Hour_Return', 'Market_Mean_Return', 'Relative_Return',
                  'Market_Mean_Volatility', 'Relative_Volatility', 'Time_To_Close',
                  'Is_Last_Tradable_Hr', 'Open', 'High', 'Low', 'Close', 'Volume']
    for c in fcast:
        if c in final.columns:
            final[c] = final[c].astype('float32')
    # tradable panel is all intraday/tradable by construction; drop the constant tag columns
    # (candle_type is non-numeric and would break the trainer). Sidecar retains them.
    final = final.drop(columns=['candle_type', 'tradable'], errors='ignore')
    final.to_parquet(OUT_PARQUET, index=False)

    boundary = pd.concat(bounds, ignore_index=True)
    boundary.to_parquet(OUT_BOUND, index=False)

    months = sorted(pd.to_datetime(final['DateTime']).dt.strftime('%Y-%m').unique())
    print(f"\nSaved {OUT_PARQUET}")
    print(f"  rows={len(final):,}  queries={final['Query_ID'].nunique():,}  feats={len(fc)}")
    print(f"  span: {months[0]} -> {months[-1]} ({len(months)} months)")
    print(f"  avg tickers/query: {final.groupby('Query_ID').size().mean():.1f}")
    print(f"  Overnight_Gap_Prior present on {final['Overnight_Gap_Prior'].notna().mean()*100:.1f}% of rows")
    print(f"Saved {OUT_BOUND}  rows={len(boundary):,}  types={boundary['candle_type'].value_counts().to_dict()}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--top_n', type=int, default=TOP_N)
    ap.add_argument('--graph', choices=['on', 'off', 'permute'], default='on')
    ap.add_argument('--robust', choices=['on', 'off'], default='on',
                    help="cross-sectional scoring: on=median/MAD winsorized, off=v20 mean/std")
    args = ap.parse_args()
    main(args.top_n, args.graph, robust=(args.robust == 'on'))
