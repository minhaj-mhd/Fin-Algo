"""
Augmented panel builder = the vetted dual-res panel + explicit price-action / SMC
features (Conv-2026-06-15-Price-Action-SMC-Transformer).

Reuses every leaf helper from build_tensor_panel.py (load, cross-sectional z-score,
alignment, macro, pivot) UNCHANGED — the only differences are:
  * per-ticker frames get scripts/features/price_action.add_price_action_features
    appended (81 TA -> 108 = 81 TA + 27 PA), and
  * output goes to data/transformer_panel_smc/  (production panel is NOT touched).

The shared helpers read build_tensor_panel.FEATURES as a module global, so we extend
that list ONCE here; cross_sectional/pivot_panel then z-score + pivot the PA columns
exactly like the TA ones.  Use --limit N for a fast integration smoke run.

    python scripts/transformer/build_tensor_panel_smc.py --limit 6     # smoke
    python scripts/transformer/build_tensor_panel_smc.py               # full
"""
import os, sys, glob, json, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import scripts.transformer.build_tensor_panel as B
from scripts.feature_utils import compute_features
from scripts.features.price_action import add_price_action_features, PA_FEATURES
from scripts.sector_map import SECTOR_MAP

OUT_DIR = 'data/transformer_panel_smc'

# Extend the feature list the SHARED helpers (B.cross_sectional, B.pivot_panel) read.
B.FEATURES = list(dict.fromkeys(B.FEATURES + PA_FEATURES))


# ── PA-augmented per-ticker builders (mirror B.build_15m / B.build_1h) ───────────
def build_15m(ohlc, ticker):
    feat = compute_features(ohlc[['Open', 'High', 'Low', 'Close', 'Volume']].copy(), legacy=False)
    feat = add_price_action_features(feat)                    # OHLC kept by compute_features
    feat['Next_Ret'] = B.session_masked_fwd(feat['Close'])
    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    return feat


def build_1h(ohlc, ticker):
    h1 = ohlc[['Open', 'High', 'Low', 'Close', 'Volume']].resample(
        '1h', origin='start_day', offset='15min'
    ).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    h1 = h1.dropna(subset=['Open', 'Close'])
    h1 = h1[pd.Index(h1.index).strftime('%H:%M').isin(B.VALID_1H_TODS)]
    if len(h1) < B.MIN_BARS:
        return None
    feat = compute_features(h1, legacy=False)
    feat = add_price_action_features(feat)
    feat['Next_Ret'] = B.session_masked_fwd(feat['Close'])
    feat['DateTime'] = feat.index
    feat['Ticker'] = ticker
    return feat


def main(limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("SMC PANEL BUILDER (81 TA + 27 price-action features)")
    print("=" * 70)
    print(f"Features: {len(B.FEATURES)}  (PA: {len(PA_FEATURES)})")

    tickers = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(f'{B.CACHE_DIR}/*.csv'))
    if limit:
        tickers = tickers[:limit]
    print(f"Tickers: {len(tickers)}{'  [SMOKE]' if limit else ''}")

    frames_15m, frames_1h = [], []
    for tk in tqdm(tickers, desc='features'):
        ohlc = B.load_ticker_ohlcv(f'{B.CACHE_DIR}/{tk}.csv')
        if ohlc is None:
            continue
        try:
            frames_15m.append(build_15m(ohlc, tk))
            f1 = build_1h(ohlc, tk)
            if f1 is not None:
                frames_1h.append(f1)
        except Exception as e:
            tqdm.write(f"  [skip] {tk}: {str(e)[:80]}")

    print("\n[15m] cross-sectional z-score + pivot")
    d15 = B.cross_sectional(pd.concat(frames_15m, ignore_index=True))
    del frames_15m
    X15, _, ts15 = B.pivot_panel(d15, tickers, with_label=False)
    del d15
    slot15 = B.tod_slot(ts15, 15)
    print(f"   X_15m {X15.shape}  slots {slot15.min()}..{slot15.max()}")

    print("\n[1h] cross-sectional z-score + pivot")
    d1 = B.cross_sectional(pd.concat(frames_1h, ignore_index=True))
    del frames_1h
    X1, Y, ts1 = B.pivot_panel(d1, tickers, with_label=True)
    del d1
    slot1 = B.tod_slot(ts1, 60)
    print(f"   X_1h {X1.shape}  finite labels {np.isfinite(Y).sum():,}")

    print("\n[align] 1h@T <-> 15m@T+45m")
    ts15_map = {int(t): i for i, t in enumerate(ts15)}
    off45 = np.int64(45 * 60 * 1_000_000_000)
    end15 = np.array([ts15_map.get(int(t + off45), -1) for t in ts1], dtype=np.int32)
    aligned = end15 >= 0
    chk = aligned.nonzero()[0]
    lhs = ts15[end15[chk]] + np.int64(15 * 60 * 1_000_000_000)
    rhs = ts1[chk] + np.int64(60 * 60 * 1_000_000_000)
    assert np.all(lhs == rhs), "ALIGNMENT VIOLATION"
    print(f"   aligned {aligned.mean()*100:.1f}%  [OK] close-time assertion passed")

    print("\n[macro] daily VIX/breadth/global")
    mdf = pd.read_csv(B.MACRO_FILE, usecols=['DateTime'] + B.MACRO_COLS)
    mdf['DateTime'] = pd.to_datetime(mdf['DateTime']).dt.normalize()
    mdf = mdf.drop_duplicates('DateTime').sort_values('DateTime').reset_index(drop=True)
    macro = mdf[B.MACRO_COLS].to_numpy(dtype=np.float32)
    macro_dates = mdf['DateTime'].to_numpy().astype('datetime64[ns]').astype('int64')
    mdate_map = {int(d): i for i, d in enumerate(macro_dates)}
    date_idx = np.array([mdate_map.get(int(pd.Timestamp(t).normalize().value), -1) for t in ts1],
                        dtype=np.int32)
    print(f"   macro {macro.shape}  1h bars with macro: {(date_idx>=0).mean()*100:.1f}%")

    sec_norm = {k.replace('.NS', ''): v for k, v in SECTOR_MAP.items()}
    sectors = sorted(set(sec_norm.values()))
    secmap = {s: i for i, s in enumerate(sectors)}
    sector_ids = np.array([secmap.get(sec_norm.get(t, 'MISC'), secmap['MISC']) for t in tickers],
                          dtype=np.int32)

    for name, arr in [('X_1h', X1), ('X_15m', X15), ('Y_ret', Y), ('slot_1h', slot1),
                      ('slot_15m', slot15), ('end15', end15), ('ts_1h', ts1), ('ts_15m', ts15),
                      ('date_idx', date_idx), ('macro', macro), ('macro_dates', macro_dates),
                      ('sector_ids', sector_ids)]:
        np.save(f'{OUT_DIR}/{name}.npy', arr)
    meta = {
        'features': B.FEATURES, 'n_features': len(B.FEATURES),
        'pa_features': PA_FEATURES, 'n_pa': len(PA_FEATURES),
        'tickers': tickers, 'n_tickers': len(tickers),
        'sectors': sectors, 'macro_cols': B.MACRO_COLS,
        'n_slots_1h': 6, 'n_slots_15m': 25,
        'shapes': {'X_1h': list(X1.shape), 'X_15m': list(X15.shape), 'macro': list(macro.shape)},
        'note': 'SMC-augmented: 81 TA + 27 price-action; per-query z-scored; 14:15 context kept',
        'smoke_limit': limit,
    }
    with open(f'{OUT_DIR}/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SAVED -> {OUT_DIR}/   X_1h {X1.shape}  X_15m {X15.shape}")
    print(f"  up-rate {np.nanmean(Y > 0)*100:.2f}%   finite labels {np.isfinite(Y).sum():,}")
    # quick PA sanity: cross-sectional std of the PA feature block after z-scoring
    pa_idx = [B.FEATURES.index(c) for c in PA_FEATURES]
    pa_block = X1[:, :, pa_idx]
    print(f"  PA block finite frac {np.isfinite(pa_block).mean()*100:.1f}%  "
          f"mean|z| {np.nanmean(np.abs(pa_block)):.3f}")
    print("=" * 70)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='only first N tickers (smoke)')
    args = ap.parse_args()
    main(limit=args.limit)
