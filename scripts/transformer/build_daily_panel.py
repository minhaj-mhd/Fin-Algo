"""
Build the daily cross-sectional tensor panel for the DAILY veto-overlay transformer
(Conv-2026-06-12-Daily-Transformer-Veto).

Single resolution (DAILY only -- no 1h/15m branch). Consumes the SAME features that the
Gauntlet-certified `daily_macro_v2` XGBoost ranker trains on, so the overlay is apples-to-apples:
  data/ranking_data_daily_macro_v2.csv  (83 features, already per-day cross-sectionally z-scored)

Two corrections vs naively reusing the CSV (verified 2026-06-12):
  * 33 market-level columns (VIX/Breadth/index/global) are ZEROED by the per-day cross-sectional
    z-scoring (constant within a day -> z=0). They carry no cross-sectional signal, so we DROP them
    from the per-ticker sequence and instead source a real macro FiLM vector from the raw daily macro
    file (data/ranking_data_daily_macro_v3.csv), exactly as the intraday panel does.
  * Label_3D is the RAW 3-day forward close-to-close return (usable for bps net-edge accounting).

Leak discipline: stock features are already per-day z-scored (each day independent -> no temporal
lookahead). Macro is stored RAW here; train-only normalization happens per-fold in train_daily.py.
Production CSVs / models are NOT touched.

Outputs (data/daily_transformer_panel/):
  X_daily.npy (T, N, Fs)   Y_3d.npy (T, N)   macro_raw.npy (T, Mc)
  dow.npy (T,)   ts_days.npy (T,)   sector_ids.npy (N,)   meta.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DATA_FILE  = 'data/ranking_data_daily_macro_v2.csv'
MACRO_FILE = 'data/ranking_data_daily_macro_v3.csv'
OUT_DIR    = 'data/daily_transformer_panel'

EXCLUDE = ['DateTime', 'Query_ID', 'Ticker', 'Open', 'High', 'Low', 'Close',
           'Volume', 'Label_3D', 'Sector', 'YearMonth']

# Raw daily macro columns for the FiLM context (same set the intraday panel uses).
MACRO_COLS = [
    'Breadth_AD_Ratio', 'Breadth_Pct_Above_SMA_50', 'Breadth_Pct_Above_SMA_200',
    'Breadth_Pct_Near_52W_High', 'Breadth_Return_Dispersion',
    'Nifty50_Dist_SMA_20', 'Nifty50_Dist_SMA_50', 'Nifty50_Dist_SMA_200',
    'Nifty50_Return_5D', 'Nifty50_Return_20D', 'Nifty500_Return_5D', 'Nifty500_Return_20D',
    'VIX_Level', 'VIX_Change_5D', 'VIX_Percentile_1Y',
    'SP500_Return_1D', 'SP500_Change_5D', 'NASDAQ_Return_1D', 'NASDAQ_Change_5D',
    'NIKKEI_Return_1D', 'NIKKEI_Change_5D', 'HSI_Return_1D', 'HSI_Change_5D',
    'USDINR_Change_5D', 'BRENT_Change_5D', 'GOLD_Change_5D', 'DXY_Change_5D', 'US10Y_Change_5D',
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("DAILY CROSS-SECTIONAL PANEL BUILDER (veto overlay on daily_macro_v2)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    print(f"loaded {df.shape[0]:,} rows  cols={df.shape[1]}")
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.normalize()

    all_feats = [c for c in df.columns if c not in EXCLUDE]
    # Drop the cross-sectionally-zeroed market columns (constant within each day -> z=0).
    stds = df[all_feats].std(axis=0)
    dead = [c for c in all_feats if stds[c] < 1e-6]
    stock_feats = [c for c in all_feats if c not in dead]
    print(f"features: {len(all_feats)} total  ->  {len(stock_feats)} stock-level kept, "
          f"{len(dead)} dead/zeroed dropped")
    assert len(stock_feats) >= 40, "unexpected stock-feature count"

    tickers = sorted(df['Ticker'].unique())
    tmap = {t: i for i, t in enumerate(tickers)}
    days = np.sort(df['DateTime'].unique())
    dmap = {pd.Timestamp(d): i for i, d in enumerate(days)}
    T, N, Fs = len(days), len(tickers), len(stock_feats)
    print(f"grid: T={T} days  N={N} tickers  Fs={Fs} features  "
          f"({pd.Timestamp(days[0]).date()} .. {pd.Timestamp(days[-1]).date()})")

    ti = df['DateTime'].map(dmap).to_numpy()
    ni = df['Ticker'].map(tmap).to_numpy()

    X = np.full((T, N, Fs), np.nan, dtype=np.float32)
    X[ti, ni, :] = df[stock_feats].to_numpy(dtype=np.float32)
    Y = np.full((T, N), np.nan, dtype=np.float32)
    Y[ti, ni] = df['Label_3D'].to_numpy(dtype=np.float32)

    # sector id per ticker (mode of its Sector rows; 'MISC' fallback)
    sec_by_tk = df.groupby('Ticker')['Sector'].agg(lambda s: s.dropna().mode().iloc[0]
                                                   if len(s.dropna()) else 'MISC')
    sectors = sorted(sec_by_tk.unique().tolist())
    if 'MISC' not in sectors:
        sectors.append('MISC')
    secmap = {s: i for i, s in enumerate(sectors)}
    sector_ids = np.array([secmap.get(sec_by_tk.get(t, 'MISC'), secmap['MISC']) for t in tickers],
                          dtype=np.int32)

    ts_days = days.astype('datetime64[ns]').astype('int64')
    dow = pd.DatetimeIndex(days).dayofweek.to_numpy().astype(np.int32)   # 0=Mon..6=Sun (NSE has rare Sat sessions)

    # ── raw macro (FiLM), aligned by date; normalized train-only later ──────────────
    mdf = pd.read_csv(MACRO_FILE, usecols=['DateTime'] + MACRO_COLS)
    mdf['DateTime'] = pd.to_datetime(mdf['DateTime']).dt.normalize()
    mdf = mdf.drop_duplicates('DateTime').set_index('DateTime')
    macro = np.full((T, len(MACRO_COLS)), np.nan, dtype=np.float32)
    mre = mdf.reindex(pd.DatetimeIndex(days))
    macro[:] = mre[MACRO_COLS].to_numpy(dtype=np.float32)
    cov = np.isfinite(macro).all(axis=1).mean()
    print(f"macro: {macro.shape}  days with full macro: {cov*100:.1f}%")

    for name, arr in [('X_daily', X), ('Y_3d', Y), ('macro_raw', macro), ('dow', dow),
                      ('ts_days', ts_days), ('sector_ids', sector_ids)]:
        np.save(f'{OUT_DIR}/{name}.npy', arr)
    meta = {
        'stock_feats': stock_feats, 'n_stock_feats': Fs, 'dead_feats': dead,
        'macro_cols': MACRO_COLS, 'n_macro': len(MACRO_COLS),
        'tickers': tickers, 'n_tickers': N, 'sectors': sectors, 'n_sectors': len(sectors),
        'n_days': T, 'n_dow': 7,
        'label': 'Label_3D = raw 3-day fwd close-to-close return (Close.shift(-3)/Close - 1)',
        'shapes': {'X_daily': [T, N, Fs], 'Y_3d': [T, N], 'macro_raw': [T, len(MACRO_COLS)]},
        'note': 'daily-only; stock feats already per-day z-scored; 33 zeroed macro cols dropped; '
                'macro stored RAW (train-only normalization in train_daily.py)',
    }
    with open(f'{OUT_DIR}/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("=" * 70)
    print(f"SAVED -> {OUT_DIR}/   X_daily {X.shape}  Y_3d {Y.shape}")
    print(f"  finite labels: {np.isfinite(Y).sum():,}  up-rate {np.nanmean(Y > 0)*100:.2f}%")
    print(f"  Label_3D: mean {np.nanmean(Y):+.5f} std {np.nanstd(Y):.5f} "
          f"min {np.nanmin(Y):+.3f} max {np.nanmax(Y):+.3f}  (note extreme +tail outliers)")
    print(f"  sectors ({len(sectors)}): {sectors}")
    print("=" * 70)


if __name__ == '__main__':
    main()
