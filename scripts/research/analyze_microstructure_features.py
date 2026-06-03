"""
Analyze microstructure features and their predictive power for Next_Hour_Return.

Tests:
1. IBS (Internal Bar Strength) = (Close - Low) / (High - Low) 
2. Body-to-Range Ratio = |Close - Open| / (High - Low)
3. Upper/Lower Shadow Ratios
4. Buy Pressure accumulation
5. Squeeze detection (BB inside Keltner)
6. ATR compression ratio
7. Momentum change features (RSI acceleration, Return acceleration)
8. Alpha persistence (rolling relative return)

Compares IC of these NEW features vs existing features.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.tickers import TICKERS
from scripts.feature_utils import compute_features, ATR, Bollinger_Bands, Keltner_Channel

RAW_CACHE_DIR = "data/raw_upstox_cache"
SAMPLE_TICKERS = TICKERS[:30]

print("=" * 70)
print("MICROSTRUCTURE FEATURE ANALYSIS")
print("=" * 70)

# Load and prepare data
all_raw = []
for ticker in tqdm(SAMPLE_TICKERS, desc="Loading data"):
    cache_file = os.path.join(RAW_CACHE_DIR, f"{ticker.replace('.NS','')}.csv")
    if not os.path.exists(cache_file):
        continue
    try:
        df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        if len(df) < 100:
            continue
        df = df.set_index('timestamp')
        df_1h = df.resample('1h', origin='start_day').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna(subset=['open', 'close'])
        
        if df_1h.index.tz is not None:
            df_1h.index = df_1h.index.tz_convert('Asia/Kolkata')
        df_1h = df_1h[(df_1h.index.hour >= 9) & (df_1h.index.hour < 16)]
        
        if len(df_1h) < 50:
            continue
        
        df_1h = df_1h.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        
        # Compute standard features
        df_feat = compute_features(df_1h, legacy=False)
        df_feat['Next_Hour_Return'] = df_feat['Close'].shift(-1) / df_feat['Close'] - 1
        df_feat['Ticker'] = ticker
        df_feat['DateTime_Hour'] = df_feat.index.floor('h')
        
        # ============================================================
        # NEW MICROSTRUCTURE FEATURES
        # ============================================================
        
        # 1. IBS (Internal Bar Strength) — THE most predictive OHLCV feature
        df_feat['IBS'] = (df_feat['Close'] - df_feat['Low']) / (df_feat['High'] - df_feat['Low'] + 1e-8)
        
        # 2. Body-to-Range Ratio (directional conviction)
        df_feat['Body_Ratio'] = abs(df_feat['Close'] - df_feat['Open']) / (df_feat['High'] - df_feat['Low'] + 1e-8)
        
        # 3. Upper Shadow Ratio (selling pressure at highs)
        df_feat['Upper_Shadow'] = (df_feat['High'] - np.maximum(df_feat['Close'], df_feat['Open'])) / (df_feat['High'] - df_feat['Low'] + 1e-8)
        
        # 4. Lower Shadow Ratio (buying at lows)
        df_feat['Lower_Shadow'] = (np.minimum(df_feat['Close'], df_feat['Open']) - df_feat['Low']) / (df_feat['High'] - df_feat['Low'] + 1e-8)
        
        # 5. Rolling IBS (smoothed over 3 bars)
        df_feat['IBS_3'] = df_feat['IBS'].rolling(3).mean()
        
        # 6. Buy Pressure (IBS * RVOL — volume-weighted close position)
        rvol = df_feat['Volume'] / (df_feat['Volume'].rolling(20).mean() + 1e-8)
        df_feat['Buy_Pressure'] = df_feat['IBS'] * rvol
        
        # 7. Squeeze Detection (BB inside Keltner)
        bb_w = df_feat['BB_Width']
        kw = df_feat['Keltner_Width']
        df_feat['Squeeze'] = (bb_w < kw).astype(int)
        # Duration of squeeze
        df_feat['Squeeze_Duration'] = df_feat['Squeeze'].groupby(
            (df_feat['Squeeze'] != df_feat['Squeeze'].shift()).cumsum()
        ).cumcount() * df_feat['Squeeze']
        
        # 8. ATR Compression Ratio
        atr_14 = ATR(df_feat, 14)
        df_feat['ATR_Ratio'] = atr_14 / (atr_14.rolling(20).mean() + 1e-8)
        
        # 9. RSI Momentum (3-bar RSI change — is momentum accelerating?)
        df_feat['RSI_Momentum'] = df_feat['RSI_14'] - df_feat['RSI_14'].shift(3)
        
        # 10. Return Acceleration
        df_feat['Return_Accel'] = df_feat['Return'] - df_feat['Return'].shift(1)
        
        # 11. Volume Surge (volume spike detection)
        vol_z = df_feat['Volume_Zscore']
        df_feat['Vol_Surge'] = vol_z - vol_z.shift(1)
        
        # 12. Alpha Persistence (rolling 3H and 6H relative return)
        market_ret = df_feat['Return'].rolling(20).mean()  # Approximate
        alpha = df_feat['Return'] - market_ret
        df_feat['Alpha_3H'] = alpha.rolling(3).sum()
        df_feat['Alpha_6H'] = alpha.rolling(6).sum()
        
        # 13. Gap from previous session close
        date_int = df_feat.index.year * 10000 + df_feat.index.month * 100 + df_feat.index.day
        day_open = df_feat.groupby(date_int)['Open'].transform('first')
        prev_close = df_feat.groupby(date_int)['Close'].transform('last').shift(1)
        df_feat['Gap_Pct'] = (day_open / (prev_close + 1e-8)) - 1
        
        # 14. Consecutive bar direction consistency
        direction = np.sign(df_feat['Return'])
        df_feat['Direction_Consistency_3'] = direction.rolling(3).sum() / 3  # -1 to +1
        df_feat['Direction_Consistency_5'] = direction.rolling(5).sum() / 5
        
        # 15. Price range expansion (current range vs avg range)
        df_feat['Range_Expansion'] = df_feat['HL_Range'] / (df_feat['HL_Range'].rolling(20).mean() + 1e-8)
        
        all_raw.append(df_feat)
    except Exception as e:
        continue

df_all = pd.concat(all_raw, ignore_index=False)
df_all = df_all.dropna(subset=['Next_Hour_Return'])
df_all['Query_ID'] = df_all.groupby('DateTime_Hour').ngroup()

# Filter queries with >= 5 tickers
query_sizes = df_all.groupby('Query_ID').size()
valid_queries = query_sizes[query_sizes >= 5].index
df_all = df_all[df_all['Query_ID'].isin(valid_queries)].copy()

print(f"\nDataset: {len(df_all):,} rows, {df_all['Query_ID'].nunique():,} queries")

# ============================================================
# COMPUTE IC FOR ALL FEATURES (raw, not z-scored)
# ============================================================

new_features = [
    'IBS', 'Body_Ratio', 'Upper_Shadow', 'Lower_Shadow', 'IBS_3',
    'Buy_Pressure', 'Squeeze', 'Squeeze_Duration', 'ATR_Ratio',
    'RSI_Momentum', 'Return_Accel', 'Vol_Surge',
    'Alpha_3H', 'Alpha_6H', 'Gap_Pct',
    'Direction_Consistency_3', 'Direction_Consistency_5', 'Range_Expansion'
]

# Also include key existing features for comparison
existing_features = [
    'RSI_14', 'PercentB', 'CCI_20', 'CMF_20', 'Volume_Zscore', 
    'VWAP_Dist', 'BB_Width', 'ROC_12', 'Stoch_K', 'WPR_14',
    'Return', 'HL_Range', 'Price_Zscore', 'Dist_52W_High',
    'PPO', 'Elder_Bull', 'Elder_Bear'
]

all_features = new_features + existing_features

def compute_ic(df, feature_col):
    """Compute average Spearman IC across queries."""
    ics = []
    for qid, group in df.groupby('Query_ID'):
        if len(group) < 5:
            continue
        feat = group[feature_col].values
        target = group['Next_Hour_Return'].values
        valid = np.isfinite(feat) & np.isfinite(target)
        if valid.sum() < 5 or np.std(feat[valid]) < 1e-10:
            continue
        corr, _ = spearmanr(feat[valid], target[valid])
        if not np.isnan(corr):
            ics.append(corr)
    return np.mean(ics) if ics else 0.0, np.std(ics) if ics else 0.0, len(ics)

print(f"\n{'=' * 70}")
print("INFORMATION COEFFICIENT COMPARISON: NEW vs EXISTING FEATURES")
print(f"{'=' * 70}")

results = []
for feat in tqdm(all_features, desc="Computing IC"):
    if feat not in df_all.columns:
        continue
    mean_ic, std_ic, count = compute_ic(df_all, feat)
    is_new = feat in new_features
    results.append({
        'Feature': feat, 
        'Mean_IC': mean_ic, 
        'Abs_IC': abs(mean_ic),
        'Std_IC': std_ic, 
        'ICIR': abs(mean_ic) / (std_ic + 1e-8),
        'Count': count,
        'Type': 'NEW' if is_new else 'EXISTING',
        't_stat': abs(mean_ic) / (std_ic / np.sqrt(count) + 1e-8) if count > 0 else 0
    })

results_df = pd.DataFrame(results).sort_values('Abs_IC', ascending=False)

print(f"\n{'Feature':<30} {'Type':>8} {'Mean IC':>10} {'|IC|':>8} {'ICIR':>8} {'t-stat':>8} {'Sig?':>6}")
print("-" * 85)
for _, row in results_df.iterrows():
    sig = "***" if row['t_stat'] > 3.0 else ("**" if row['t_stat'] > 2.0 else ("*" if row['t_stat'] > 1.5 else ""))
    marker = ">>>" if row['Type'] == 'NEW' else "   "
    print(f"{marker}{row['Feature']:<27} {row['Type']:>8} {row['Mean_IC']:>10.5f} {row['Abs_IC']:>8.5f} {row['ICIR']:>8.4f} {row['t_stat']:>8.2f} {sig:>6}")

# Summary
new_results = results_df[results_df['Type'] == 'NEW']
existing_results = results_df[results_df['Type'] == 'EXISTING']

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  NEW features avg |IC|:      {new_results['Abs_IC'].mean():.5f}")
print(f"  EXISTING features avg |IC|: {existing_results['Abs_IC'].mean():.5f}")
print(f"  NEW features avg ICIR:      {new_results['ICIR'].mean():.4f}")
print(f"  EXISTING features avg ICIR: {existing_results['ICIR'].mean():.4f}")

sig_new = len(new_results[new_results['t_stat'] > 2.0])
sig_existing = len(existing_results[existing_results['t_stat'] > 2.0])
print(f"  NEW features statistically significant (t>2):      {sig_new}/{len(new_results)}")
print(f"  EXISTING features statistically significant (t>2): {sig_existing}/{len(existing_results)}")

# Top 10 NEW features
print(f"\n{'=' * 70}")
print("TOP 10 NEW MICROSTRUCTURE FEATURES")
print(f"{'=' * 70}")
for i, (_, row) in enumerate(new_results.head(10).iterrows()):
    print(f"  {i+1}. {row['Feature']:<30} IC={row['Mean_IC']:>+.5f}  ICIR={row['ICIR']:.4f}  t={row['t_stat']:.2f}")

# Save
results_df.to_csv('data/microstructure_feature_analysis.csv', index=False)
print(f"\nSaved to data/microstructure_feature_analysis.csv")
