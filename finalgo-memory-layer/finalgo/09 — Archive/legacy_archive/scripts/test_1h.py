import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

TEST_MONTH = "2026-05"
TRANSACTION_COST_PCT = 0.06

def load_and_filter_csv(path, month_prefixes):
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000):
        mask = chunk['DateTime'].str.startswith(tuple(month_prefixes))
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

print("Loading Data...")
df_daily = load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", ["2026-04", "2026-05"])
df_1h = load_and_filter_csv("data/ranking_data_upstox_3y.csv", [TEST_MONTH])

# Simplistic scoring using pre-existing predictions (we assume we have them, wait no, we need to run xgb or use pre-scored. Let's just use the loaded models like in the main script).
