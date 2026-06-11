"""
Live 2-day hybrid trade generator: what trades the SAVED v10 + v18 models produce
for the most recent sessions (Mon 2026-06-08 + today 2026-06-09 if available).

The training dataset (data/ranking_data_upstox_1h_v3_3y.csv) ends 2026-06-05, so this
fetches fresh native-1h candles from Upstox V3, merges them with the cached history (needed
for rolling features like SMA_50 / 52W high-low), then rebuilds the feature panel using the
EXACT collector pipeline (compute_features + session-masked fwd + cross-sectional z-score),
and runs the saved models with the production hybrid logic.

Feature construction is copied verbatim from scripts/collectors/collect_upstox_1h_v3.py to
guarantee train/serve consistency.

Usage:
    python scripts/analysis/live_2day_hybrid_trades.py

Outputs:
    data/model_analysis/v10_v18_independent/live_2day_trades.csv
"""
import os, sys, time, json, glob, warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

import upstox_client
from scripts.upstox_broker import UpstoxSandboxBroker
from scripts.feature_utils import compute_features
from scripts.tickers import TICKERS
import xgboost as xgb

RAW_CACHE_DIR = 'data/raw_upstox_cache_1h_v3'
OUT_DIR       = 'data/model_analysis/v10_v18_independent'
V10_DIR, V18_DIR = 'models/v10_native_1h', 'models/v18_random_forest_1h'
VALID_TODS    = {'09:15', '10:15', '11:15', '12:15', '13:15', '14:15'}
MIN_BARS      = 60
RATE_PAUSE    = 0.20
FETCH_FROM    = '2026-04-01'          # recent window to refresh (cache covers deep history)
FETCH_TO      = date.today().strftime('%Y-%m-%d')
TARGET_DAYS   = ['2026-06-08', '2026-06-09']
PROB_TH       = 0.52
COSTS         = {'6bps': 0.0006, '10bps': 0.0010}
os.makedirs(OUT_DIR, exist_ok=True)

broker = UpstoxSandboxBroker()
v3 = upstox_client.HistoryV3Api(broker.data_api_client)


# ── feature construction (verbatim from collect_upstox_1h_v3.py) ──────────────
def session_masked_fwd(close):
    by_day = close.groupby(close.index.normalize())
    return by_day.shift(-1) / close - 1.0


def build_ticker(ticker, raw):
    dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df = pd.DataFrame({
        'DateTime': dt,
        'Open': raw['open'].astype(float), 'High': raw['high'].astype(float),
        'Low': raw['low'].astype(float), 'Close': raw['close'].astype(float),
        'Volume': raw['volume'].astype(float),
    }).dropna(subset=['DateTime', 'Open', 'Close'])
    df = df.drop_duplicates('DateTime').sort_values('DateTime').set_index('DateTime')
    df = df[pd.Index(df.index).strftime('%H:%M').isin(VALID_TODS)]
    if len(df) < MIN_BARS:
        return None
    feat = compute_features(df[['Open', 'High', 'Low', 'Close', 'Volume']].copy(), legacy=False)
    feat['Next_Hour_Return'] = session_masked_fwd(feat['Close'])
    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    return feat


def build_ranking(df_all):
    df_all = df_all.copy()
    df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])
    # NOTE: keep rows even if Next_Hour_Return is NaN (today's last bars) — we still want trades.
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    sizes = df_all.groupby('Query_ID').size()
    df_all = df_all[df_all['Query_ID'].isin(sizes[sizes >= 5].index)].copy()
    df_all = df_all.sort_values('DateTime')
    df_all['Query_ID'] = df_all.groupby('DateTime').ngroup()
    df_all['Market_Mean_Return']     = df_all.groupby('Query_ID')['Return'].transform('mean')
    df_all['Relative_Return']        = df_all['Return'] - df_all['Market_Mean_Return']
    df_all['Market_Mean_Volatility'] = df_all.groupby('Query_ID')['HL_Range'].transform('mean')
    df_all['Relative_Volatility']    = df_all['HL_Range'] / (df_all['Market_Mean_Volatility'] + 1e-8)
    exclude = {'DateTime', 'Query_ID', 'Ticker', 'Next_Hour_Return', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Market_Mean_Return', 'Relative_Return', 'Market_Mean_Volatility', 'Relative_Volatility',
               'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close'}
    feat_cols = [c for c in df_all.columns if c not in exclude]
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    for col in feat_cols:
        g = df_all.groupby('Query_ID')[col]
        df_all[col] = (df_all[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
    return df_all


# ── fetch recent candles, merge with cache ───────────────────────────────────
def fetch_recent_merge(ticker):
    cache_path = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    cached = pd.read_csv(cache_path) if os.path.exists(cache_path) else None
    ik = broker.get_instrument_key(ticker)
    rows = []
    for attempt in range(3):
        try:
            resp = v3.get_historical_candle_data1(ik, 'hours', '1', FETCH_TO, FETCH_FROM)
            if resp.status == 'success' and resp.data and resp.data.candles:
                rows = resp.data.candles
            break
        except Exception as e:
            if '429' in str(e) or 'Too Many' in str(e):
                time.sleep(3); continue
            break
    time.sleep(RATE_PAUSE)
    fresh = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']) if rows else None
    parts = [p for p in (cached, fresh) if p is not None and len(p)]
    if not parts:
        return None
    merged = pd.concat(parts, ignore_index=True).drop_duplicates(subset='timestamp')
    return merged


def main():
    print(f"Fetching recent 1h candles ({FETCH_FROM} -> {FETCH_TO}) + merging cache for {len(TICKERS)} tickers...")
    frames, ok, skip = [], 0, 0
    for i, ticker in enumerate(TICKERS, 1):
        try:
            raw = fetch_recent_merge(ticker)
            if raw is None: skip += 1; continue
            f = build_ticker(ticker, raw)
            if f is not None: frames.append(f); ok += 1
            else: skip += 1
        except Exception as e:
            skip += 1
        if i % 40 == 0:
            print(f"  {i}/{len(TICKERS)} (ok={ok} skip={skip})")
    print(f"  done. ok={ok} skip={skip}")

    df_all = pd.concat(frames, ignore_index=True)
    panel = build_ranking(df_all)
    panel['date'] = panel['DateTime'].astype(str).str[:10]
    avail = sorted(panel['date'].unique())
    print(f"\nPanel built. Latest dates available: {avail[-6:]}")

    target = panel[panel['date'].isin(TARGET_DAYS)].copy()
    if len(target) == 0:
        print(f"\nNo rows for {TARGET_DAYS}. Latest available is {avail[-1]}. Nothing to trade.")
        return
    print(f"Target rows: {len(target)} across days {sorted(target['date'].unique())}")

    # feature matrix in model's training order
    meta = json.load(open(f'{V10_DIR}/metadata.json'))
    feats = meta['features']
    X = target[feats].values.astype(np.float64)
    # impute residual NaNs per-column (cross-section already z-scored)
    for ci in range(X.shape[1]):
        bad = ~np.isfinite(X[:, ci])
        if bad.any():
            good = X[np.isfinite(X[:, ci]), ci]
            X[bad, ci] = float(good.mean()) if len(good) else 0.0
    d = xgb.DMatrix(X)

    def L(p): b = xgb.Booster(); b.load_model(p); return b
    rl = L(f'{V10_DIR}/xgb_long_model.json').predict(d)
    rs = L(f'{V10_DIR}/xgb_short_model.json').predict(d)
    pl = L(f'{V18_DIR}/xgb_long_model.json').predict(d)
    ps = L(f'{V18_DIR}/xgb_short_model.json').predict(d)

    target = target.reset_index(drop=True)
    target['v10_long'], target['v10_short'] = rl, rs
    target['v18_long'], target['v18_short'] = pl, ps
    ret = target['Next_Hour_Return'].values  # may be NaN for the very last bar of an unfinished day

    # ── produce trades: Top-3 ranked per (day,hour) cross-section + v18 veto ──
    trades = []
    for qid, g in target.groupby('Query_ID'):
        g = g.reset_index(drop=True)
        if len(g) < 5:
            continue
        dt = g['DateTime'].iloc[0]
        # longs: top-3 by v10_long, veto v18_long>0.52
        for idx in np.argsort(g['v10_long'].values)[-3:][::-1]:
            r = g.iloc[idx]
            trades.append(dict(DateTime=str(dt), side='LONG', rank_in_top3=True,
                               Ticker=r['Ticker'], v10_score=round(float(r['v10_long']), 4),
                               v18_prob=round(float(r['v18_long']), 4),
                               veto_pass=bool(r['v18_long'] > PROB_TH),
                               realized_bps=(None if not np.isfinite(r['Next_Hour_Return'])
                                             else round(float(r['Next_Hour_Return']) * 10000, 1))))
        # shorts: top-3 by v10_short (raw V10 = the asymmetric recommendation), veto shown but not applied
        for idx in np.argsort(g['v10_short'].values)[-3:][::-1]:
            r = g.iloc[idx]
            trades.append(dict(DateTime=str(dt), side='SHORT', rank_in_top3=True,
                               Ticker=r['Ticker'], v10_score=round(float(r['v10_short']), 4),
                               v18_prob=round(float(r['v18_short']), 4),
                               veto_pass=bool(r['v18_short'] > PROB_TH),
                               realized_bps=(None if not np.isfinite(r['Next_Hour_Return'])
                                             else round(-float(r['Next_Hour_Return']) * 10000, 1))))
    td = pd.DataFrame(trades)
    td.to_csv(f'{OUT_DIR}/live_2day_trades.csv', index=False)

    # ── report ──
    print("\n" + "=" * 78)
    print("HYBRID TRADES — SAVED v10 (rank) + v18 (veto) — Top-3 per hourly cross-section")
    print("=" * 78)
    for day in sorted(td['DateTime'].str[:10].unique()):
        dd = td[td['DateTime'].str[:10] == day]
        print(f"\n#### {day} ####")
        for side in ('LONG', 'SHORT'):
            sd = dd[dd['side'] == side]
            print(f"\n  --- {side} (Top-3/hour by v10; veto = v18>{PROB_TH}) ---")
            for _, r in sd.iterrows():
                rb = 'n/a' if r['realized_bps'] is None else f"{r['realized_bps']:+.1f}bps"
                vp = 'PASS' if r['veto_pass'] else 'veto '
                print(f"    {r['DateTime'][11:16]} {r['Ticker']:<14} v10={r['v10_score']:+.3f} "
                      f"v18={r['v18_prob']:.1%} [{vp}] realized {rb}")

    # summary: hybrid-long (veto applied) vs raw-short, where realized known
    print("\n" + "=" * 78)
    print("REALIZED SUMMARY (only bars with known next-hour return)")
    print("=" * 78)
    def summ(label, mask_df, use_veto):
        s = mask_df.dropna(subset=['realized_bps'])
        if use_veto:
            s = s[s['veto_pass']]
        if len(s) == 0:
            print(f"  {label}: no settled trades"); return
        r = s['realized_bps'].values / 10000.0
        for cl, cv in COSTS.items():
            net = r.mean() - cv
            print(f"  {label} @{cl}: n={len(s)} raw {r.mean()*10000:+.1f} net {net*10000:+.1f} "
                  f"win {(r>0).mean():.0%}")
    summ('LONG hybrid (veto on) ', td[td['side'] == 'LONG'], True)
    summ('SHORT raw V10 (no veto)', td[td['side'] == 'SHORT'], False)
    print(f"\nSaved -> {OUT_DIR}/live_2day_trades.csv")


if __name__ == '__main__':
    main()
