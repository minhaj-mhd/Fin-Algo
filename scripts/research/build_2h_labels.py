"""
Add an N-HOUR forward-return label to the v21 rolling-1h research panel — WITHOUT touching
the features (the "2h" in the filename is just the original default; `--hours` does any horizon).
Hypothesis: the binding cost is 10bps round-trip *regardless of hold length*, so holding a 1h-ranked
pick for N hours could ~Nx the gross edge (if the signal persists) while the cost stays fixed ->
cost-per-unit-time falls, the one clean lever that could cross the cost line without new data. To
isolate the HORIZON effect we keep the proven v20/v21 rolling-1h FEATURES unchanged and add only a
longer-horizon LABEL.

The label is reconstructed on the EXACT same window-close grid the panel/transformer use
(`_load_raw(hygiene=True)` + the v21 rolling-1h window = 4 consecutive 15-min bars), so it merges
1:1 onto every (Ticker, DateTime) row. Forward return uses the same exact-timestamp reindex as the
1h label (`close(T+Nh)/close(T) - 1`), NaN whenever T+Nh is not a same-session window close ==
session mask, no overnight leak. So labels exist only up to (15:30 - Nh): 2h -> 13:30, 3h -> 12:30.

Outputs (H = hours):
  data/research/v21_rolling_1h/panel_{H}h.parquet  = v21 panel + Next_{H}Hour_Return column
  data/transformer_panel_v21/Y_ret_{H}h.npy        = [T, N] N-h-return tensor on the ts_1h x ticker grid

RESEARCH ONLY (AGENTS.md): no Gauntlet, no registry stamp.
Run: python scripts/research/build_2h_labels.py --hours 3
"""
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

from scripts.research.build_v21_rolling_1h_panel import _load_raw, STEP, HOUR

SRC_DIR    = 'data/raw_upstox_cache_15min_3y'
V21_DIR    = 'data/research/v21_rolling_1h'
PANEL_IN   = os.path.join(V21_DIR, 'panel.parquet')
UNIV_JSON  = os.path.join(V21_DIR, 'universe.json')
TPANEL_DIR = 'data/transformer_panel_v21'


def window_close_series(df):
    """Hygiene'd 15-min OHLCV -> the rolling-1h window CLOSE price series keyed by window-close T.
    Byte-identical T grid + Close values to build_v21_rolling_1h_panel.build_ticker (the window is
    the 4 consecutive 15-min bars [k-3..k], keyed at t[k]+15m; valid only when contiguous)."""
    t = df['DateTime']
    win = pd.DataFrame({'DateTime': t + STEP, 'Close': df['Close']})
    contiguous = (t - t.shift(3)) == (3 * STEP)
    win = (win[contiguous.values].dropna(subset=['Close'])
              .drop_duplicates('DateTime').sort_values('DateTime').set_index('DateTime')['Close'])
    return win


def forward_return(close_at, horizon):
    """close(T+horizon)/close(T)-1 via exact-match reindex (session mask; NaN across the gap)."""
    fwd = close_at.reindex(close_at.index + horizon)
    return fwd.values / close_at.values - 1.0


def build_label_frame(universe, hours, ret_col):
    """Per-ticker N-h (and 1h, as a self-check) forward returns on the window-close grid."""
    horizon = hours * HOUR
    rows, skip = [], 0
    for tk in tqdm(universe, desc=f'{hours}h labels'):
        fp = os.path.join(SRC_DIR, tk + '.csv')
        if not os.path.exists(fp):
            skip += 1; continue
        try:
            df = _load_raw(fp, hygiene=True)
            close_at = window_close_series(df)
            if len(close_at) == 0:
                skip += 1; continue
            rN = forward_return(close_at, horizon)
            r1 = forward_return(close_at, HOUR)
            rows.append(pd.DataFrame({'Ticker': tk, 'DateTime': close_at.index,
                                      ret_col: rN, 'Next_1Hour_Return_chk': r1}))
        except Exception as e:
            skip += 1; tqdm.write(f"  [skip] {tk}: {str(e)[:70]}")
    lab = pd.concat(rows, ignore_index=True)
    lab['DateTime'] = pd.to_datetime(lab['DateTime'])
    print(f"  label rows={len(lab):,}  tickers_skipped={skip}")
    return lab


def merge_xgb_panel(lab, ret_col, panel_out):
    print(f"\n[XGB] merging {ret_col} onto {PANEL_IN}")
    panel = pd.read_parquet(PANEL_IN)
    panel['DateTime'] = pd.to_datetime(panel['DateTime'])
    n0 = len(panel)
    merged = panel.merge(lab[['Ticker', 'DateTime', ret_col, 'Next_1Hour_Return_chk']],
                         on=['Ticker', 'DateTime'], how='left')
    assert len(merged) == n0, "merge changed row count"
    m = merged['Next_1Hour_Return_chk'].notna() & merged['Next_Hour_Return'].notna()
    maxdiff = float((merged.loc[m, 'Next_1Hour_Return_chk'].astype('float64')
                     - merged.loc[m, 'Next_Hour_Return'].astype('float64')).abs().max())
    print(f"  1h-return reconstruction self-check: max|diff| vs panel = {maxdiff:.2e} (float32 noise ~1e-6)")
    assert maxdiff < 1e-4, f"1h reconstruction mismatch {maxdiff:.2e} -> grid/Close drift, ABORT"
    covN = merged[ret_col].notna().mean()
    merged = merged.drop(columns=['Next_1Hour_Return_chk'])
    merged[ret_col] = merged[ret_col].astype('float32')
    merged.to_parquet(panel_out, index=False)
    print(f"  saved {panel_out}  rows={len(merged):,}  {ret_col} present on {covN*100:.1f}% of rows")
    tod = pd.to_datetime(merged.loc[merged[ret_col].notna(), 'DateTime']).dt.strftime('%H:%M')
    print(f"  labeled window-close times: {sorted(tod.unique())}")


def build_transformer_tensor(lab, ret_col, hours):
    if not os.path.isdir(TPANEL_DIR):
        print(f"\n[TF] {TPANEL_DIR} not found -> skipping transformer tensor")
        return
    out = f'{TPANEL_DIR}/Y_ret_{hours}h.npy'
    print(f"\n[TF] building Y_ret_{hours}h.npy aligned to {TPANEL_DIR} grid")
    ts1 = np.load(f'{TPANEL_DIR}/ts_1h.npy')
    meta = json.load(open(f'{TPANEL_DIR}/meta.json'))
    tickers = meta['tickers']
    Y1 = np.load(f'{TPANEL_DIR}/Y_ret.npy')
    T, N = Y1.shape
    assert N == len(tickers) and len(ts1) == T
    ts_pos = {int(v): i for i, v in enumerate(ts1)}
    col = {tk: j for j, tk in enumerate(tickers)}
    YN = np.full((T, N), np.nan, dtype=np.float32)
    labN = lab[lab[ret_col].notna()]
    lab_ns = labN['DateTime'].values.astype('datetime64[ns]').astype(np.int64)
    miss = 0
    for tk, t_ns, rN in zip(labN['Ticker'].values, lab_ns, labN[ret_col].values):
        i = ts_pos.get(int(t_ns)); j = col.get(tk)
        if i is None or j is None:
            miss += 1; continue
        YN[i, j] = rN
    np.save(out, YN)
    print(f"  saved {out}  shape={YN.shape}  finite 1h={int(np.isfinite(Y1).sum()):,}  "
          f"finite {hours}h={int(np.isfinite(YN).sum()):,}  label rows not on grid={miss:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=int, default=2)
    args = ap.parse_args()
    H = args.hours
    ret_col = f'Next_{H}Hour_Return'
    panel_out = os.path.join(V21_DIR, f'panel_{H}h.parquet')
    universe = json.load(open(UNIV_JSON))['tickers']
    print(f"universe: {len(universe)} tickers  |  horizon={H}h  |  label={ret_col}")
    lab = build_label_frame(universe, H, ret_col)
    merge_xgb_panel(lab, ret_col, panel_out)
    build_transformer_tensor(lab, ret_col, H)


if __name__ == '__main__':
    main()
