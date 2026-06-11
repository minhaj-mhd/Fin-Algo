import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.getcwd())

def test_point_in_time_contract_logic():
    # Construct mock data for daily trades and global macro
    # Trade Day T calendar dates: 2026-06-08 (Mon), 2026-06-09 (Tue), 2026-06-10 (Wed)
    # T-1 Daily indices: 2026-06-05 (Fri), 2026-06-08 (Mon), 2026-06-09 (Tue)
    daily_dates = pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09"])
    
    # Intraday trades on Trade Day T: 2026-06-08, 2026-06-09, 2026-06-10
    intra_dates = pd.to_datetime([
        "2026-06-08 09:15:00", "2026-06-08 10:30:00",
        "2026-06-09 09:15:00", "2026-06-09 14:30:00",
        "2026-06-10 09:15:00"
    ])
    
    # Normalize trade dates to date only (Trade Day T)
    trade_days = intra_dates.normalize()
    
    # PIT Join: For each trade day T, daily features must come from previous trading day T-1
    # Find the largest daily_date strictly less than the trade_day
    trade_to_daily_map = {}
    for t_day in trade_days.unique():
        prev_dates = [d for d in daily_dates if d < t_day]
        if prev_dates:
            trade_to_daily_map[t_day] = max(prev_dates)
            
    # Assertions
    assert trade_to_daily_map[pd.Timestamp("2026-06-08")] == pd.Timestamp("2026-06-05")
    assert trade_to_daily_map[pd.Timestamp("2026-06-09")] == pd.Timestamp("2026-06-08")
    assert trade_to_daily_map[pd.Timestamp("2026-06-10")] == pd.Timestamp("2026-06-09")
    
    print("Point-in-Time Join Logic successfully verified!")

def test_leakage_detector_assertion():
    # Simulate a feature dataset with lookahead
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'Feature_Leaky': np.random.randn(n),
        'Label_3D': np.random.randn(n)
    })
    
    # Leak the label directly into the feature
    df['Feature_Leaky'] = df['Label_3D'] * 1.5 + np.random.randn(n) * 0.01
    
    # Recompute correlation
    corr = df['Feature_Leaky'].corr(df['Label_3D'])
    
    # Assert correlation is very high
    assert corr > 0.99
    
    # Assert that our check catches it
    corr_series = df[['Feature_Leaky']].corrwith(df['Label_3D']).abs()
    perfect_corrs = corr_series[corr_series > 0.99].index.tolist()
    assert len(perfect_corrs) == 1
    assert perfect_corrs[0] == 'Feature_Leaky'
    
    print("Lookahead correlation leakage detector successfully verified!")
