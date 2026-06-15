"""
Export v20 rolling-1h data as FLOAT64 CSVs for the Validation Gauntlet.

Two variants:
  --which anchor : v10's 5 NON-overlapping :15 decision moments {10:15,11:15,12:15,13:15,14:15}.
                   Non-overlapping 1h labels -> independent queries -> HONEST Gauntlet significance.
                   This is the AUTHORITATIVE certification dataset (v20 at v10's cadence).
  --which full   : all 18 overlapping entry-times/day. DIAGNOSTIC ONLY — overlapping windows make
                   the Gauntlet's bootstrap CI / t-stats over-optimistic (no verdict authority).

Float64 (no downcast) so the Stage-0 label-integrity check (recompute target/close-1 vs stored,
atol=1e-9) passes exactly. Reuses the exact build_ticker/build_ranking pipeline (identical features).
"""
import os, sys, glob, argparse
import pandas as pd
from tqdm import tqdm
sys.path.append(os.getcwd())
from scripts.research.build_rolling_1h_panel import build_ticker, build_ranking, SRC_DIR

OUT_DIR = 'data/research/v20_rolling_1h'
SHARED_15 = {'10:15', '11:15', '12:15', '13:15', '14:15'}

ap = argparse.ArgumentParser()
ap.add_argument('--which', required=True, choices=['anchor', 'full'])
args = ap.parse_args()
out_csv = os.path.join(OUT_DIR, 'panel_15anchor.csv' if args.which == 'anchor' else 'panel_full.csv')

files = sorted(glob.glob(os.path.join(SRC_DIR, '*.csv')))
print(f"Building '{args.which}' from {len(files)} tickers (float64)...")
frames = []
for fp in tqdm(files, desc='Tickers'):
    t = os.path.splitext(os.path.basename(fp))[0]
    try:
        raw = pd.read_csv(fp)
        f = build_ticker(t, raw)
        if f is not None and len(f):
            frames.append(f)
    except Exception as e:
        tqdm.write(f"  [skip] {t}: {str(e)[:60]}")

final, fc = build_ranking(pd.concat(frames, ignore_index=True))   # float64, z-scored

if args.which == 'anchor':
    hhmm = pd.to_datetime(final['DateTime']).dt.strftime('%H:%M')
    final = final[hhmm.isin(SHARED_15)].copy()
    # re-number Query_ID contiguously after subsetting (bijection qid<->timestamp preserved)
    final = final.sort_values('DateTime')
    final['Query_ID'] = final.groupby('DateTime').ngroup()

final.to_csv(out_csv, index=False)
qpd = final.groupby(pd.to_datetime(final['DateTime']).dt.date)['Query_ID'].nunique().mean()
print(f"\nSaved {out_csv}")
print(f"  rows={len(final):,}  queries={final['Query_ID'].nunique():,}  avg entries/day={qpd:.1f}")
print(f"  tickers/query={final.groupby('Query_ID').size().mean():.1f}")
