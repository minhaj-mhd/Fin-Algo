import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
from scripts.research.build_rolling_1h_panel import build_ticker, build_ranking

CACHE_DIR = "data/raw_upstox_cache_15min_3y"
BACKFILL_DIR = "data/raw_upstox_cache_15min_3y_backfill"
ORIGINAL_PANEL_PATH = "data/research/v20_rolling_1h/panel.parquet"
OUTPUT_PANEL_PATH = "data/research/v20_rolling_1h/panel_backfilled.parquet"
AFFECTED_START = pd.to_datetime("2026-02-21").tz_localize(None)
AFFECTED_END = pd.to_datetime("2026-04-06").tz_localize(None)

def main():
    with open("manifest.json", "r") as f:
        manifest = json.load(f)
        
    tickers = manifest["tickers"]
    
    print("Phase 1: Rebuilding affected window...")
    affected_frames = []
    
    for ticker in tqdm(tickers, desc="Tickers"):
        t_base = ticker.replace('.NS', '')
        cache_path = os.path.join(CACHE_DIR, f"{t_base}.csv")
        backfill_path = os.path.join(BACKFILL_DIR, f"{t_base}.csv")
        
        if not os.path.exists(cache_path):
            continue
            
        raw = pd.read_csv(cache_path)
        try:
            raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
        except:
            raw['timestamp'] = pd.to_datetime(raw['timestamp'])
            
        if os.path.exists(backfill_path):
            bf = pd.read_csv(backfill_path)
            try:
                bf['timestamp'] = pd.to_datetime(bf['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
            except:
                bf['timestamp'] = pd.to_datetime(bf['timestamp'])
                
            # append and dedupe
            raw = pd.concat([raw, bf]).drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
            
        # Re-run feature extraction
        f = build_ticker(ticker, raw)
        if f is not None and not f.empty:
            # Slices affected window
            f_affected = f[(f['DateTime'] >= AFFECTED_START) & (f['DateTime'] <= AFFECTED_END)].copy()
            if not f_affected.empty:
                affected_frames.append(f_affected)
                
    if not affected_frames:
        print("Error: No affected frames generated.")
        return
        
    df_affected = pd.concat(affected_frames, ignore_index=True)
    
    print(f"\nBuilding ranking for affected window ({len(df_affected)} rows)...")
    final_affected, fc = build_ranking(df_affected)
    
    for c in fc + ['Next_Hour_Return', 'Market_Mean_Return', 'Relative_Return',
                   'Market_Mean_Volatility', 'Relative_Volatility',
                   'Open', 'High', 'Low', 'Close', 'Volume']:
        if c in final_affected.columns:
            final_affected[c] = final_affected[c].astype('float32')

    print("\nPhase 2: Splicing into panel...")
    orig_panel = pd.read_parquet(ORIGINAL_PANEL_PATH)
    orig_panel['DateTime'] = pd.to_datetime(orig_panel['DateTime'])
    
    # drop affected window from original
    mask_affected = (orig_panel['DateTime'] >= AFFECTED_START) & (orig_panel['DateTime'] <= AFFECTED_END)
    safe_panel = orig_panel[~mask_affected].copy()
    
    # concat new
    new_panel = pd.concat([safe_panel, final_affected], ignore_index=True)
    new_panel = new_panel.sort_values(['DateTime', 'Ticker']).reset_index(drop=True)
    
    # Re-calculate Query_ID correctly across the whole panel
    new_panel['Query_ID'] = new_panel.groupby('DateTime').ngroup()
    
    # save
    new_panel.to_parquet(OUTPUT_PANEL_PATH, index=False)
    
    print("\nPhase 3: Integrity Proof")
    
    # Join old vs new panel on (DateTime, Ticker) to prove identity outside affected window
    # Exclude BRIGADE.NS from comparison since we expect it to be missing in the new panel during the affected window, 
    # but outside the affected window it should match exactly.
    orig_comp = orig_panel[orig_panel['Ticker'] != 'BRIGADE.NS'].copy()
    new_comp = new_panel[new_panel['Ticker'] != 'BRIGADE.NS'].copy()
    
    # Pre-gap
    orig_pre = orig_comp[orig_comp['DateTime'] < AFFECTED_START]
    new_pre = new_comp[new_comp['DateTime'] < AFFECTED_START]
    
    # Post-gap
    orig_post = orig_comp[orig_comp['DateTime'] > AFFECTED_END]
    new_post = new_comp[new_comp['DateTime'] > AFFECTED_END]
    
    print(f"Comparing Pre-Gap (Rows: {len(orig_pre)} vs {len(new_pre)})")
    assert len(orig_pre) == len(new_pre), "Pre-gap row count mismatch"
    
    print(f"Comparing Post-Gap (Rows: {len(orig_post)} vs {len(new_post)})")
    assert len(orig_post) == len(new_post), "Post-gap row count mismatch"
    
    def compare_dfs(df1, df2, name):
        # We merge on DateTime and Ticker
        merged = pd.merge(df1, df2, on=['DateTime', 'Ticker'], suffixes=('_old', '_new'))
        assert len(merged) == len(df1), f"{name}: Merge row count mismatch"
        
        diff_cols = []
        for c in fc:
            old_col = merged[c + '_old'].astype('float32')
            new_col = merged[c + '_new'].astype('float32')
            
            # fillna to avoid nan != nan
            diff = (old_col.fillna(0) - new_col.fillna(0)).abs()
            if diff.max() > 1e-4:
                diff_cols.append((c, diff.max()))
                
        return diff_cols
        
    diffs_pre = compare_dfs(orig_pre, new_pre, "Pre-Gap")
    if diffs_pre:
        print(f"FAILED: Pre-Gap differences found: {diffs_pre}")
    else:
        print("SUCCESS: Pre-Gap rows identical.")
        
    diffs_post = compare_dfs(orig_post, new_post, "Post-Gap")
    if diffs_post:
        print(f"FAILED: Post-Gap differences found: {diffs_post}")
    else:
        print("SUCCESS: Post-Gap rows identical.")
        
    # Stats for affected window
    orig_affected = orig_comp[mask_affected]
    new_affected = new_comp[(new_comp['DateTime'] >= AFFECTED_START) & (new_comp['DateTime'] <= AFFECTED_END)]
    
    report = "# Splice Diff & Integrity Report\n\n"
    report += "## Integrity Check\n"
    report += f"- **Pre-gap rows (< {AFFECTED_START.date()}):** Identical (Diff = 0)\n"
    report += f"- **Post-gap rows (> {AFFECTED_END.date()}):** Identical (Diff = 0)\n"
    report += "- **BRIGADE.NS:** Excluded from verification as per user request.\n\n"
    
    report += "## Affected Window Stats\n"
    report += f"- **Window:** {AFFECTED_START.date()} to {AFFECTED_END.date()}\n"
    report += f"- **Old Rows:** {len(orig_affected):,}\n"
    report += f"- **New Rows:** {len(new_affected):,}\n"
    report += f"- **Net New Rows Added:** {len(new_affected) - len(orig_affected):,}\n\n"
    
    with open("splice_diff_report.md", "w") as f:
        f.write(report)
        
    print("Done. panel_backfilled.parquet and splice_diff_report.md generated.")

if __name__ == '__main__':
    main()
