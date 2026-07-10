"""
Build the dual-resolution tensor panel on the **v21 CLEAN rolling-1h decision grid**.

This is build_tensor_panel_v20.py with exactly TWO changes, so the only difference vs the v20
DualRes panel is the v21 data-cleaning on the 1h decision side:
  1. Tickers restricted to the v21 LIQUIDITY UNIVERSE (data/research/v21_rolling_1h/universe.json).
  2. 1h decision features built via build_v21_rolling_1h_panel (clean_feats=True): mask-not-fill +
     the wall-clock lookback fix (the only ablation-positive lever). Source bars get v21 hygiene
     (frozen/zero-vol dropped) via the v21 loader.

The 15m CONTEXT side is intentionally LEFT as the proven native-15m build (compute_features
legacy=False): clean_v21's x4 lookback scale is correct only for the 15-min-stepped rolling-1h grid,
NOT for native 15m bars — applying it there would mis-scale every 15m lookback. Macro/sector/align
logic is reused verbatim. Explicit overnight-gap awareness is Phase B (a sequence token), not a
feature here (a single gap feature would force a 1h/15m feature-dim mismatch).

============================  RESEARCH ONLY  ============================
No Gauntlet, no registry stamp, no verdict authority (AGENTS.md Model Metric Discipline).
Overlapping windows -> consecutive rows ~75% identical: do NOT t-test panel point estimates.

Output (data/transformer_panel_v21/): identical file layout to data/transformer_panel_v20/ so
scripts/transformer/train.py runs unchanged via TRANSFORMER_PANEL=data/transformer_panel_v21.
"""
import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# v21 CLEAN rolling-1h candle/feature construction (hygiene + clean_v21 features)
from scripts.research.build_v21_rolling_1h_panel import _load_raw as v21_load, build_ticker as build_roll_1h_v21
# reuse the DualRes panel's 15m builder, cross-sectional z-score, pivot, macro/sector logic verbatim
from scripts.transformer.build_tensor_panel import (
    load_ticker_ohlcv, build_15m, cross_sectional, pivot_panel, tod_slot,
    FEATURES, MACRO_COLS, CACHE_DIR, MACRO_FILE,
)
from scripts.sector_map import SECTOR_MAP

OUT_DIR  = 'data/transformer_panel_v21'
UNIV_JSON = 'data/research/v21_rolling_1h/universe.json'


def build_1h_rolling_v21(path, ticker):
    """v21 clean rolling-1h features keyed by window-close T, with the DualRes label column name."""
    df = v21_load(path, hygiene=True)                       # v21 bar hygiene
    f, _ = build_roll_1h_v21(ticker, df, clean_feats=True, gap_feats=True)
    if f is None or not len(f):
        return None
    return f.rename(columns={'Next_Hour_Return': 'Next_Ret'})  # pivot/cross_sectional expect 'Next_Ret'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("DUAL-RES PANEL BUILDER -- v21 CLEAN ROLLING-1h DECISION GRID")
    print("=" * 70)
    print(f"Features: {len(FEATURES)}")

    universe = set(json.load(open(UNIV_JSON))['tickers'])
    all_tk = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f'{CACHE_DIR}/*.csv'))
    tickers = [t for t in all_tk if t in universe]                 # CHANGE 1: liquidity universe
    print(f"Tickers: {len(tickers)} (v21 liquidity universe of {len(all_tk)} cached)")

    frames_15m, frames_1h = [], []
    for tk in tqdm(tickers, desc='features'):
        path = f'{CACHE_DIR}/{tk}.csv'
        ohlc = load_ticker_ohlcv(path)                             # 15m context: native build (unchanged)
        if ohlc is None:
            continue
        try:
            frames_15m.append(build_15m(ohlc, tk))
            f1 = build_1h_rolling_v21(path, tk)                    # CHANGE 2: clean v21 1h features
            if f1 is not None:
                frames_1h.append(f1)
        except Exception as e:
            tqdm.write(f"  [skip] {tk}: {str(e)[:60]}")

    # ── 15m context panel (verbatim DualRes path) ──────────────────────────────
    print("\n[15m] cross-sectional z-score + pivot")
    d15 = cross_sectional(pd.concat(frames_15m, ignore_index=True))
    del frames_15m
    X15, _, ts15 = pivot_panel(d15, tickers, with_label=False)
    del d15
    slot15 = tod_slot(ts15, 15)
    print(f"   X_15m {X15.shape}  slots {slot15.min()}..{slot15.max()}")

    # ── 1h clean-rolling panel (keep NaN-label session-end context rows) ───────
    print("\n[1h-roll v21] cross-sectional z-score + pivot")
    d1 = cross_sectional(pd.concat(frames_1h, ignore_index=True))
    del frames_1h
    X1, Y, ts1 = pivot_panel(d1, tickers, with_label=True)
    del d1
    slot1 = tod_slot(ts1, 15)                                      # window-close clock slot (15-min grid)
    n_slots_1h = int(slot1.max()) + 1
    print(f"   X_1h {X1.shape}  slots {slot1.min()}..{slot1.max()} (n_slots_1h={n_slots_1h})  "
          f"decision rows (finite label): {np.isfinite(Y).sum():,}")

    # ── align 15m to 1h: the 15m bar that CLOSES at T has start ts15 == T - 15m ─
    print("\n[align] 1h-roll close@T <-> 15m close@T (15m start == T-15m)")
    ts15_map = {int(t): i for i, t in enumerate(ts15)}
    off15 = np.int64(15 * 60 * 1_000_000_000)
    end15 = np.array([ts15_map.get(int(t - off15), -1) for t in ts1], dtype=np.int32)
    aligned = end15 >= 0
    chk = aligned.nonzero()[0]
    lhs = ts15[end15[chk]] + off15
    rhs = ts1[chk]
    assert np.all(lhs == rhs), "ALIGNMENT VIOLATION"
    print(f"   aligned {aligned.mean()*100:.1f}%  [OK] close-time assertion passed")

    # ── macro (daily, market-level, PIT) -- verbatim DualRes path ──────────────
    print("\n[macro] daily VIX/breadth/global")
    mdf = pd.read_csv(MACRO_FILE, usecols=['DateTime'] + MACRO_COLS)
    mdf['DateTime'] = pd.to_datetime(mdf['DateTime']).dt.normalize()
    mdf = mdf.drop_duplicates('DateTime').sort_values('DateTime').reset_index(drop=True)
    macro = mdf[MACRO_COLS].to_numpy(dtype=np.float32)
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

    # ── save (same layout as data/transformer_panel_v20/) ──────────────────────
    for name, arr in [('X_1h', X1), ('X_15m', X15), ('Y_ret', Y), ('slot_1h', slot1),
                      ('slot_15m', slot15), ('end15', end15), ('ts_1h', ts1), ('ts_15m', ts15),
                      ('date_idx', date_idx), ('macro', macro), ('macro_dates', macro_dates),
                      ('sector_ids', sector_ids)]:
        np.save(f'{OUT_DIR}/{name}.npy', arr)
    meta = {
        'features': FEATURES, 'n_features': len(FEATURES), 'tickers': tickers, 'n_tickers': len(tickers),
        'sectors': sectors, 'macro_cols': MACRO_COLS,
        'n_slots_1h': n_slots_1h, 'n_slots_15m': 25,
        'shapes': {'X_1h': list(X1.shape), 'X_15m': list(X15.shape), 'macro': list(macro.shape)},
        'note': 'v21 CLEAN rolling-1h decision grid (110-liquid universe, clean_v21 1h features); '
                '15m context = native v20 build; ts_1h = window CLOSE; 15-min 1h slots',
    }
    with open(f'{OUT_DIR}/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SAVED -> {OUT_DIR}/   X_1h {X1.shape}  X_15m {X15.shape}")
    print(f"  up-rate {np.nanmean(Y > 0)*100:.2f}%   finite labels {np.isfinite(Y).sum():,}")
    qpd = pd.Series(pd.to_datetime(ts1[np.isfinite(Y).any(1)])).dt.date.value_counts()
    print(f"  avg decision timestamps/day: {qpd.mean():.1f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
