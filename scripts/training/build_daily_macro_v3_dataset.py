"""
Build the Daily Macro Gatekeeper V3 dataset (10-year lookback).
Combines:
1. Lean single-stock features (~53 features).
2. Market breadth features (advance/decline, % above SMA, dispersion).
3. Market index context (Nifty 50, Nifty 500, India VIX).
4. Sector relative strength features.
5. Global macro features (US SP500, NASDAQ, Nikkei, HSI, USDINR, Brent, Gold, DXY, US10Y).

Applies:
- Strict Point-in-Time joins (all features in row T-1 are known before 09:00 IST T).
- 1-bar forward close-to-close returns label: Label_1D = Close(T)/Close(T-1) - 1.0 (shift(-1)).
- Cross-sectional Z-scoring per trading day.
- Programmatic lookahead assertions.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.sector_map import SECTOR_MAP
from scripts.feature_utils import compute_features_daily_xgb

# Config
DATA_DIR = "data/raw_daily_10y"
GLOBAL_DIR = "data/raw_global_daily"
OUTPUT_CSV = "data/ranking_data_daily_macro_v3.csv"
MIN_BARS = 100

# Sector to Index mapping
SECTOR_TO_INDEX = {
    "BANK": "NIFTY_BANK",
    "IT": "NIFTY_IT",
    "PHARMA": "NIFTY_PHARMA",
    "AUTO": "NIFTY_AUTO",
    "METAL": "NIFTY_METAL",
    "ENERGY": "NIFTY_ENERGY",
    "FMCG": "NIFTY_FMCG",
    "REALTY": "NIFTY_REALTY",
    "FINANCE": "NIFTY_BANK",
    "CEMENT": "NIFTY_500",
    "INFRA": "NIFTY_500",
    "CHEMICAL": "NIFTY_500",
    "CONSUMER": "NIFTY_500",
    "TELECOM": "NIFTY_500",
    "DEFENCE": "NIFTY_500",
    "MISC": "NIFTY_500"
}

LEAN_FEATURES = [
    'Return', 'Log_Return', 'HL_Range', 'OC_Range',
    'Return_5D', 'Return_10D', 'Return_20D',
    'ROC_5', 'ROC_10', 'ROC_20',
    'Dist_SMA_5', 'Dist_SMA_10', 'Dist_SMA_20', 'Dist_SMA_50', 'Dist_SMA_100', 'Dist_SMA_200',
    'Dist_EMA_12', 'Dist_EMA_26', 'Dist_EMA_50', 'Dist_HMA_20',
    'RSI_14', 'RSI_7', 'CCI_20', 'WPR_14', 'Ultimate_Osc',
    'Stoch_K', 'Stoch_D', 'IBS', 'IBS_3', 'IBS_5',
    'ATR_14_Pct', 'Volatility_20D', 'PercentB', 'Dist_BB_Upper', 'Dist_BB_Lower', 'BB_Width',
    'Dist_Donchian_Upper', 'Dist_Donchian_Lower', 'Donchian_Width',
    'Volume_Zscore', 'RVOL', 'Buy_Pressure',
    'Up_Streak', 'Down_Streak',
    'Dist_52W_High', 'Dist_52W_Low', 'Position_In_52W',
    'Direction_Consistency_5', 'Direction_Consistency_10',
    'Return_lag1', 'Return_lag2', 'RSI_lag1', 'Volume_Zscore_lag1'
]

def load_clean_parquet(path):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(pd.to_datetime(df['timestamp']).dt.date)
    df = df.rename(columns={
        'timestamp': 'DateTime', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    df = df.drop_duplicates(subset=['DateTime']).sort_values('DateTime')
    return df

def main():
    print("=" * 70)
    print("DAILY MACRO DATASET BUILDER (V3 - 1-DAY LABEL)")
    print("=" * 70)

    # 1. Load and process Indian Indices & VIX
    print("Loading Indian indices & VIX...")
    indices_dfs = {}
    for idx_name in SECTOR_TO_INDEX.values():
        if idx_name not in indices_dfs:
            path = os.path.join(DATA_DIR, f"{idx_name}.parquet")
            df = load_clean_parquet(path)
            if df is not None:
                indices_dfs[idx_name] = df

    nifty50 = load_clean_parquet(os.path.join(DATA_DIR, "NIFTY_50.parquet"))
    nifty500 = load_clean_parquet(os.path.join(DATA_DIR, "NIFTY_500.parquet"))
    vix = load_clean_parquet(os.path.join(DATA_DIR, "INDIA_VIX.parquet"))

    assert nifty50 is not None, "Nifty 50 is required!"
    assert nifty500 is not None, "Nifty 500 is required!"
    assert vix is not None, "India VIX is required!"

    # Compute Index features
    nifty50['Nifty50_SMA_20'] = nifty50['Close'].rolling(20).mean()
    nifty50['Nifty50_SMA_50'] = nifty50['Close'].rolling(50).mean()
    nifty50['Nifty50_SMA_200'] = nifty50['Close'].rolling(200).mean()
    
    nifty50['Nifty50_Dist_SMA_20'] = (nifty50['Close'] - nifty50['Nifty50_SMA_20']) / nifty50['Close']
    nifty50['Nifty50_Dist_SMA_50'] = (nifty50['Close'] - nifty50['Nifty50_SMA_50']) / nifty50['Close']
    nifty50['Nifty50_Dist_SMA_200'] = (nifty50['Close'] - nifty50['Nifty50_SMA_200']) / nifty50['Close']
    nifty50['Nifty50_Return_5D'] = nifty50['Close'].pct_change(5)
    nifty50['Nifty50_Return_20D'] = nifty50['Close'].pct_change(20)
    
    nifty_context = nifty50[['DateTime', 'Nifty50_Dist_SMA_20', 'Nifty50_Dist_SMA_50', 
                             'Nifty50_Dist_SMA_200', 'Nifty50_Return_5D', 'Nifty50_Return_20D']].copy()

    nifty500['Nifty500_Return_5D'] = nifty500['Close'].pct_change(5)
    nifty500['Nifty500_Return_20D'] = nifty500['Close'].pct_change(20)
    nifty500_context = nifty500[['DateTime', 'Nifty500_Return_5D', 'Nifty500_Return_20D']].copy()

    vix['VIX_Level'] = vix['Close']
    vix['VIX_Change_5D'] = vix['Close'] - vix['Close'].shift(5)
    vix['VIX_Percentile_1Y'] = vix['Close'].rolling(250).rank(pct=True)
    vix_context = vix[['DateTime', 'VIX_Level', 'VIX_Change_5D', 'VIX_Percentile_1Y']].copy()

    sector_index_returns = {}
    for idx_name, df in indices_dfs.items():
        df_copy = df.copy()
        df_copy[f'{idx_name}_Return_5D'] = df_copy['Close'].pct_change(5)
        df_copy[f'{idx_name}_Return_20D'] = df_copy['Close'].pct_change(20)
        sector_index_returns[idx_name] = df_copy[['DateTime', f'{idx_name}_Return_5D', f'{idx_name}_Return_20D']].copy()

    # 2. Load and process Global Macro Assets
    print("Loading global macro assets...")
    global_assets = ["SP500", "NASDAQ", "NIKKEI", "HSI", "USDINR", "BRENT", "GOLD", "DXY", "US10Y"]
    global_dfs = {}
    for name in global_assets:
        path = os.path.join(GLOBAL_DIR, f"{name}.parquet")
        df = load_clean_parquet(path)
        if df is not None:
            df_copy = df.copy()
            if name == "US10Y":
                df_copy[f'{name}_Change_5D'] = df_copy['Close'] - df_copy['Close'].shift(5)
            else:
                df_copy[f'{name}_Change_5D'] = df_copy['Close'].pct_change(5)
                
            if name in ["SP500", "NASDAQ", "NIKKEI", "HSI"]:
                df_copy[f'{name}_Return_1D'] = df_copy['Close'].pct_change(1)
                global_dfs[name] = df_copy[['DateTime', f'{name}_Return_1D', f'{name}_Change_5D']].copy()
            else:
                global_dfs[name] = df_copy[['DateTime', f'{name}_Change_5D']].copy()

    # 3. Load tickers and compute features
    print("Processing individual tickers...")
    all_ticker_dfs = []
    
    for ticker in tqdm(TICKERS, desc="Single Ticker Features"):
        path = os.path.join(DATA_DIR, f"{ticker.replace('.NS', '')}.parquet")
        df = load_clean_parquet(path)
        if df is None or len(df) < MIN_BARS:
            continue

        df.set_index('DateTime', inplace=True)
        df_feat = compute_features_daily_xgb(df)
        
        # 1-bar forward close-to-close return label (Close(T)/Close(T-1) - 1.0)
        df_feat['Label_1D'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1.0
        
        df_feat['Ticker'] = ticker
        df_feat['DateTime'] = df_feat.index
        
        keep_cols = ['DateTime', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Label_1D'] + LEAN_FEATURES
        existing_cols = [c for c in keep_cols if c in df_feat.columns]
        df_feat = df_feat[existing_cols].copy()
        
        all_ticker_dfs.append(df_feat)

    if not all_ticker_dfs:
        print("[FATAL] No ticker data successfully processed.")
        sys.exit(1)

    df_master = pd.concat(all_ticker_dfs, ignore_index=True)
    df_master = df_master.sort_values(['DateTime', 'Ticker']).reset_index(drop=True)

    # 4. Market Breadth Features (at DateTime = T-1)
    print("Calculating market breadth features...")
    breadth_records = []
    for dt, group in tqdm(df_master.groupby('DateTime'), desc="Daily Breadth"):
        advances = (group['Return'] > 0).sum()
        declines = (group['Return'] < 0).sum()
        ad_ratio = advances / (declines + 1e-8)
        
        pct_above_50 = (group['Dist_SMA_50'] > 0).mean()
        pct_above_200 = (group['Dist_SMA_200'] > 0).mean()
        pct_near_52w = (group['Dist_52W_High'] > -0.05).mean()
        dispersion = group['Return'].std()
        
        breadth_records.append({
            'DateTime': dt,
            'Breadth_AD_Ratio': ad_ratio,
            'Breadth_Pct_Above_SMA_50': pct_above_50,
            'Breadth_Pct_Above_SMA_200': pct_above_200,
            'Breadth_Pct_Near_52W_High': pct_near_52w,
            'Breadth_Return_Dispersion': dispersion
        })
    df_breadth = pd.DataFrame(breadth_records)
    
    df_master = pd.merge(df_master, df_breadth, on='DateTime', how='left')

    # 5. Join Indices (Nifty 50, Nifty 500, India VIX)
    df_master = pd.merge(df_master, nifty_context, on='DateTime', how='left')
    df_master = pd.merge(df_master, nifty500_context, on='DateTime', how='left')
    df_master = pd.merge(df_master, vix_context, on='DateTime', how='left')

    # 6. Sector relative strength features
    df_master['Sector'] = df_master['Ticker'].map(lambda t: SECTOR_MAP.get(t, "MISC"))
    df_master['Sector_Index'] = df_master['Sector'].map(lambda s: SECTOR_TO_INDEX.get(s, "NIFTY_500"))
    
    for idx_name, df_ret in sector_index_returns.items():
        df_master = pd.merge(
            df_master,
            df_ret.rename(columns={
                f'{idx_name}_Return_5D': 'temp_sec_5d',
                f'{idx_name}_Return_20D': 'temp_sec_20d'
            }),
            on='DateTime',
            how='left'
        )
        
        mask = df_master['Sector_Index'] == idx_name
        if 'Sector_Return_5D' not in df_master.columns:
            df_master['Sector_Return_5D'] = np.nan
            df_master['Sector_Return_20D'] = np.nan
            
        df_master.loc[mask, 'Sector_Return_5D'] = df_master.loc[mask, 'temp_sec_5d']
        df_master.loc[mask, 'Sector_Return_20D'] = df_master.loc[mask, 'temp_sec_20d']
        df_master.drop(columns=['temp_sec_5d', 'temp_sec_20d'], inplace=True)

    df_master['Sector_Relative_Strength_5D'] = df_master['Return_5D'] - df_master['Sector_Return_5D']
    df_master['Sector_Relative_Strength_20D'] = df_master['Return_20D'] - df_master['Sector_Return_20D']
    df_master.drop(columns=['Sector_Return_5D', 'Sector_Return_20D', 'Sector_Index'], inplace=True)

    # 7. Merge Global Macro Features (Forward filled to handle holidays)
    unique_dates = pd.DataFrame({'DateTime': sorted(df_master['DateTime'].unique())})
    for name, df_glob in global_dfs.items():
        unique_dates = pd.merge(unique_dates, df_glob, on='DateTime', how='left')
    unique_dates = unique_dates.sort_values('DateTime').ffill().fillna(0)
    df_master = pd.merge(df_master, unique_dates, on='DateTime', how='left')

    # 8. Clean up and Z-score
    print("Filtering and scoring...")
    df_master = df_master.dropna(subset=['Label_1D'])
    
    df_master = df_master.sort_values('DateTime')
    df_master['Query_ID'] = df_master.groupby(df_master['DateTime'].dt.date).ngroup()
    
    query_sizes = df_master.groupby('Query_ID').size()
    valid_queries = query_sizes[query_sizes >= 5].index
    df_master = df_master[df_master['Query_ID'].isin(valid_queries)].copy()
    
    df_master = df_master.sort_values('DateTime')
    df_master['Query_ID'] = df_master.groupby(df_master['DateTime'].dt.date).ngroup()

    exclude_cols = {
        'DateTime', 'Query_ID', 'Ticker', 'Label_1D', 'Sector',
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Hour', 'DayOfWeek', 'DayOfMonth', 'MonthOfYear',
        'Is_Month_Start', 'Is_Month_End', 'WeekOfMonth'
    }
    feature_cols = [c for c in df_master.columns if c not in exclude_cols]

    for col in tqdm(feature_cols, desc="Cross-sectional Z-scoring"):
        grp_mean = df_master.groupby('Query_ID')[col].transform('mean')
        grp_std = df_master.groupby('Query_ID')[col].transform('std')
        df_master[col] = (df_master[col] - grp_mean) / (grp_std + 1e-8)

    df_master[feature_cols] = df_master[feature_cols].fillna(0)

    # 9. Programmatic point-in-time assertion
    corr_to_label = df_master[feature_cols].corrwith(df_master['Label_1D']).abs()
    perfect_corrs = corr_to_label[corr_to_label > 0.99].index.tolist()
    assert len(perfect_corrs) == 0, f"Possible lookahead leakage in features: {perfect_corrs}"

    # Save
    print(f"Saving dataset to {OUTPUT_CSV}...")
    df_master.to_csv(OUTPUT_CSV, index=False)
    
    print("\nDataset stats:")
    print(f"  Total rows      : {len(df_master):,}")
    print(f"  Total features  : {len(feature_cols)}")
    print(f"  Total queries   : {df_master['Query_ID'].nunique()}")
    print(f"  Date span       : {df_master['DateTime'].min().date()} -> {df_master['DateTime'].max().date()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
