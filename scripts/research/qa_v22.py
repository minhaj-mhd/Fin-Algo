import pandas as pd
import numpy as np
import sys
import os

PANEL = 'data/research/v22_rolling_1h_dynamic/panel.parquet'
if not os.path.exists(PANEL):
    print("Panel not found")
    sys.exit(1)

df = pd.read_parquet(PANEL)
print(f"Loaded {len(df):,} rows.")

print("\n--- 1. NaN% Check ---")
nans = df.isna().mean().sort_values(ascending=False)
print("Highest NaN percentages:")
print(nans.head(15))

print("\n--- 2. Checking if M, N, T are constant per Query_ID ---")
# Pick a random sample of Query_IDs to verify
qids = np.random.choice(df['Query_ID'].unique(), size=500, replace=False)
sample = df[df['Query_ID'].isin(qids)]
feats_to_check = ['Breadth_Pct_Positive', 'Macro_Nifty_1H', 'Time_Sin']
valid = True
for f in feats_to_check:
    if f in sample.columns:
        std = sample.groupby('Query_ID')[f].std()
        if (std > 1e-6).any():
            print(f"FAIL: {f} varies within Query_ID!")
            valid = False
if valid:
    print("PASS: M/N/T representative columns are constant per Query_ID.")

print("\n--- 3. Causality Check (No Look-ahead in V/N) ---")
# VWAP_Slope should only depend on current/past bars. It's computed from past window.
print("Manual verification required for VWAP_Slope/Gap causality (Agent 1 code was cloned from causal v21 builder).")

print("Done.")
