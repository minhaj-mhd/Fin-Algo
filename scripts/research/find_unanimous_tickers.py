import os
import sys
import pandas as pd

# Add project root to path before importing local modules
sys.path.append(os.getcwd())

from scripts.vanguard import config
# Disable websocket for a standalone script run
config.WEBSOCKET_ENABLED = False

from scripts.vanguard.orchestrator import VanguardOrchestrator
from scripts.tickers import TICKERS
from scripts.terminal_utils import log

def find_unanimous_tickers():
    log("Initializing Vanguard Orchestrator (WebSocket disabled for script)...")
    # Initialize orchestrator to load models and broker
    # The init process loads active models, daily gatekeepers, and multi TF models
    orchestrator = VanguardOrchestrator()
    
    log(f"Scoring universe of {len(TICKERS)} tickers...")
    scores_df = orchestrator.calculate_conviction_scores(TICKERS)
    
    if scores_df is None or scores_df.empty:
        log("[ERROR] Failed to calculate conviction scores.")
        return
    
    # We need: Long_Conviction, score_15m, score_30m, score_1d
    required_cols = ['ticker', 'Long_Conviction', 'score_15m', 'score_30m', 'score_1d']
    missing_cols = [c for c in required_cols if c not in scores_df.columns]
    
    if missing_cols:
        log(f"[ERROR] Missing columns in scores_df: {missing_cols}")
        log(f"Available columns: {list(scores_df.columns)}")
        return
        
    df = scores_df[required_cols].copy()
    
    # All Long: Every model returns a score > 0
    all_long_mask = (df['Long_Conviction'] > 0) & (df['score_15m'] > 0) & (df['score_30m'] > 0) & (df['score_1d'] > 0)
    all_long_tickers = df[all_long_mask]['ticker'].tolist()
    
    # All Short: Every model returns a score < 0
    all_short_mask = (df['Long_Conviction'] < 0) & (df['score_15m'] < 0) & (df['score_30m'] < 0) & (df['score_1d'] < 0)
    all_short_tickers = df[all_short_mask]['ticker'].tolist()
    
    print("\n" + "=" * 60)
    print("UNANIMOUS MODEL PREDICTIONS (4 MODELS)")
    print("=" * 60)
    
    print(f"\nLONG TICKERS ({len(all_long_tickers)}):")
    if all_long_tickers:
        print(", ".join(all_long_tickers))
        print("\nDetails:")
        print(df[all_long_mask].to_string(index=False))
    else:
        print("None")
        
    print(f"\nSHORT TICKERS ({len(all_short_tickers)}):")
    if all_short_tickers:
        print(", ".join(all_short_tickers))
        print("\nDetails:")
        print(df[all_short_mask].to_string(index=False))
    else:
        print("None")

if __name__ == "__main__":
    find_unanimous_tickers()
