"""
strategy_25x_backtest.py — Comprehensive Intraday Backtester for 25 Strategies

Uses 4 XGBoost models:
1. Daily: models/daily_xgb/
2. 1-Hour: models/v8_upstox_3y/
3. 30-Min: models/v1_30min/
4. 15-Min: models/v1_15min/

Holdout Period: 2026-05 (for intraday). Daily loading includes 2026-04 and 2026-05.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)

sys.path.append(os.getcwd())

# ========================================
# CONFIG
# ========================================
TEST_MONTH = "2026-05"
TRANSACTION_COST_PCT = 0.06  # 0.06% round-trip (including slippage)

# ========================================
# Helper: Memory-Optimized Loading
# ========================================
def load_and_filter_csv(path, month_prefixes):
    print(f"Loading and filtering {path} for months {month_prefixes}...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100000):
        mask = chunk['DateTime'].str.startswith(tuple(month_prefixes))
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

# ========================================
# Helper: Predict Scores
# ========================================
def predict_timeframe_scores(df_tf, model_key, meta_path, long_model_path, short_model_path, scaler_path=None):
    print(f"Scoring {model_key} dataset...")
    df_tf = df_tf.copy()
    
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    
    # Check column existence
    missing_cols = [c for c in feature_cols if c not in df_tf.columns]
    if missing_cols:
        print(f"  [WARN] Missing features in dataset: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            df_tf[col] = 0.0
            
    # Load models
    bst_long = xgb.Booster()
    bst_long.load_model(long_model_path)
    bst_long.set_param({'device': 'cpu'})
    
    bst_short = xgb.Booster()
    bst_short.load_model(short_model_path)
    bst_short.set_param({'device': 'cpu'})
    
    # Load scaler if it exists and apply
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as sf:
            scaler = pickle.load(sf)
            
    X = df_tf[feature_cols].values
    X_clean = np.nan_to_num(X)
    
    scaler_is_fitted = (
        scaler is not None
        and hasattr(scaler, 'scale_')
        and scaler.scale_ is not None
    )
    if scaler_is_fitted:
        X_final = scaler.transform(X_clean)
        print("  Scaler applied")
    else:
        X_final = X_clean
        
    dmat = xgb.DMatrix(X_final, feature_names=feature_cols)
    df_tf['long_score'] = bst_long.predict(dmat)
    df_tf['short_score'] = bst_short.predict(dmat)
    
    # Compute Convictions and Ranks
    df_tf['long_conv'] = df_tf['long_score'] - df_tf['short_score']
    df_tf['short_conv'] = df_tf['short_score'] - df_tf['long_score']
    df_tf['long_rank'] = df_tf.groupby('DateTime')['long_conv'].rank(ascending=False)
    df_tf['short_rank'] = df_tf.groupby('DateTime')['short_conv'].rank(ascending=False)
    
    # Sort and add lag 1
    df_tf = df_tf.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    df_tf['long_conv_lag1'] = df_tf.groupby('Ticker')['long_conv'].shift(1)
    df_tf['short_conv_lag1'] = df_tf.groupby('Ticker')['short_conv'].shift(1)
    
    print(f"  Scoring complete: {df_tf.shape[0]:,} rows.")
    return df_tf

# ========================================
# Helper: Find Yesterday's Daily Date
# ========================================
def find_yesterday_daily_date(trading_date, sorted_daily_dates):
    for d in reversed(sorted_daily_dates):
        if d < trading_date:
            return d
    return None

# ========================================
# Helper: Align Timeframes (15M -> 1H, 30M)
# ========================================
def align_timeframes(t_str):
    date_part = t_str[:10]
    time_part = t_str[11:16]
    h, m = map(int, time_part.split(':'))
    minutes = h * 60 + m
    
    # 1H bars start at 09:30, 10:30, 11:30, 12:30, 13:30, 14:30
    # They close at 10:30, 11:30, 12:30, 13:30, 14:30, 15:30
    hourly_bars = [
        (9, 30),
        (10, 30),
        (11, 30),
        (12, 30),
        (13, 30),
        (14, 30),
    ]
    t_1h = None
    for bar_h, bar_m in hourly_bars:
        bar_start_min = bar_h * 60 + bar_m
        bar_close_min = bar_start_min + 60
        if minutes >= bar_close_min:
            t_1h = f"{date_part} {bar_h:02d}:{bar_m:02d}:00+05:30"
            
    # 30M bars start at 09:15, 09:45, 10:15, 10:45, ..., 15:15
    t_30m = None
    bar_start = 9 * 60 + 15
    while bar_start < 15 * 60 + 30:
        bar_close = bar_start + 30
        if minutes >= bar_close:
            bh, bm = divmod(bar_start, 60)
            t_30m = f"{date_part} {bh:02d}:{bm:02d}:00+05:30"
        bar_start += 30
        
    return t_1h, t_30m

# ========================================
# Helper: Evaluate Strategy Trades
# ========================================
def evaluate_strategy_trades(trades, name, slots=5):
    if not trades:
        return {
            'strategy': name,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'long_win_rate': 0.0,
            'short_win_rate': 0.0,
            'long_trades': 0,
            'short_trades': 0,
            'total_return': 0.0,
            'avg_return': 0.0,
            'avg_trades_per_day': 0.0,
            'green_day_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'avg_bars_held': 0.0
        }
        
    df_t = pd.DataFrame(trades)
    
    # Calculate daily statistics
    daily_stats = []
    for date, group in df_t.groupby('date'):
        wins = group['is_win'].sum()
        total = len(group)
        pnl = group['net_return'].sum()
        daily_stats.append({
            'date': date, 'trades': total, 'wins': wins, 'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0, 'pnl': pnl, 'is_green': pnl > 0
        })
    df_d = pd.DataFrame(daily_stats)
    
    total_trades = len(df_t)
    total_wins = int(df_t['is_win'].sum())
    total_losses = total_trades - total_wins
    overall_wr = total_wins / total_trades
    
    avg_trades_day = df_d['trades'].mean() if len(df_d) > 0 else 0
    
    # Capital-Adjusted Returns
    df_t['net_return'] = df_t['net_return'] / slots
    total_net_pnl = df_t['net_return'].sum()
    avg_net_trade = df_t['net_return'].mean()
    
    green_days = int(df_d['is_green'].sum()) if len(df_d) > 0 else 0
    total_days = len(df_d) if len(df_d) > 0 else 1
    green_day_rate = green_days / total_days
    
    winners = df_t[df_t['is_win']]
    losers = df_t[~df_t['is_win']]
    avg_win_size = winners['net_return'].mean() if len(winners) > 0 else 0
    avg_loss_size = losers['net_return'].mean() if len(losers) > 0 else 0
    
    profit_factor = abs(winners['net_return'].sum() / losers['net_return'].sum()) if len(losers) > 0 and losers['net_return'].sum() != 0 else float('inf')
    
    # Maximum Drawdown (cumulative sum of trades)
    cum_returns = df_t['net_return'].cumsum()
    max_dd = (cum_returns - cum_returns.cummax()).min() if len(cum_returns) > 0 else 0.0
    
    # Avg bars held
    avg_bars = df_t['bars_held'].mean() if 'bars_held' in df_t.columns else 1.0
    
    # Long vs Short breakdown
    longs = df_t[df_t['side'] == 'LONG']
    shorts = df_t[df_t['side'] == 'SHORT']
    long_wr = longs['is_win'].mean() if len(longs) > 0 else 0.0
    short_wr = shorts['is_win'].mean() if len(shorts) > 0 else 0.0
    
    return {
        'strategy': name,
        'total_trades': int(total_trades),
        'wins': int(total_wins),
        'losses': int(total_losses),
        'win_rate': float(overall_wr),
        'long_win_rate': float(long_wr),
        'short_win_rate': float(short_wr),
        'long_trades': int(len(longs)),
        'short_trades': int(len(shorts)),
        'total_return': float(total_net_pnl),
        'avg_return': float(avg_net_trade),
        'avg_trades_per_day': float(avg_trades_day),
        'green_day_rate': float(green_day_rate),
        'profit_factor': float(profit_factor),
        'max_drawdown': float(max_dd),
        'avg_bars_held': float(avg_bars)
    }

# ========================================
# MAIN SIMULATION SCRIPT
# ========================================
def main():
    print("=" * 70)
    print("25X INTRADAY TRADING STRATEGIES BACKTEST SIMULATION")
    print("=" * 70)

    from datetime import timedelta
    print("\nLoading 5-minute candles cache for intrabar stops...")
    cache_dir = "data/raw_upstox_cache_5min"
    dict_5m_candles = {}
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            if filename.endswith(".csv"):
                ticker = filename.replace(".csv", "") + ".NS"
                path = os.path.join(cache_dir, filename)
                df_5m = pd.read_csv(path)
                df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
                ticker_map = {}
                for _, row in df_5m.iterrows():
                    ticker_map[row['timestamp']] = {
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close'])
                    }
                dict_5m_candles[ticker] = ticker_map

    # 1. LOAD DATASETS
    # Memory-optimized loading using chunks
    df_daily_raw = load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", ["2026-04", "2026-05"])
    df_1h_raw = load_and_filter_csv("data/ranking_data_upstox_3y.csv", [TEST_MONTH])
    df_30m_raw = load_and_filter_csv("data/ranking_data_upstox_30min_1y.csv", [TEST_MONTH])
    df_15m_raw = load_and_filter_csv("data/ranking_data_upstox_15min_1y.csv", [TEST_MONTH])
    
    print("\nDatasets loaded:")
    print(f"  Daily: {df_daily_raw.shape[0]:,} rows")
    print(f"  1H:    {df_1h_raw.shape[0]:,} rows")
    print(f"  30M:   {df_30m_raw.shape[0]:,} rows")
    print(f"  15M:   {df_15m_raw.shape[0]:,} rows")
    
    # 2. RUN ML PREDICTIONS
    df_daily = predict_timeframe_scores(
        df_daily_raw, "Daily",
        "models/daily_xgb/metadata.json",
        "models/daily_xgb/xgb_long_model.json",
        "models/daily_xgb/xgb_short_model.json",
        "models/daily_xgb/scaler.pkl"
    )
    
    df_1h = predict_timeframe_scores(
        df_1h_raw, "1H",
        "models/v8_upstox_3y/metadata.json",
        "models/v8_upstox_3y/xgb_long_model.json",
        "models/v8_upstox_3y/xgb_short_model.json",
        "models/scaler.pkl"
    )
    
    df_30m = predict_timeframe_scores(
        df_30m_raw, "30M",
        "models/v1_30min/metadata.json",
        "models/v1_30min/xgb_long_model.json",
        "models/v1_30min/xgb_short_model.json",
        "models/v1_30min/scaler.pkl"
    )
    
    df_15m = predict_timeframe_scores(
        df_15m_raw, "15M",
        "models/v1_15min/metadata.json",
        "models/v1_15min/xgb_long_model.json",
        "models/v1_15min/xgb_short_model.json",
        "models/v1_15min/scaler.pkl"
    )
    
    # Find ticker intersection
    t_daily = set(df_daily['Ticker'].unique())
    t_1h = set(df_1h['Ticker'].unique())
    t_30m = set(df_30m['Ticker'].unique())
    t_15m = set(df_15m['Ticker'].unique())
    common_tickers = sorted(list(t_daily.intersection(t_1h).intersection(t_30m).intersection(t_15m)))
    print(f"\nTickers universe intersection: {len(common_tickers)} symbols")
    
    df_daily = df_daily[df_daily['Ticker'].isin(common_tickers)].copy()
    df_1h = df_1h[df_1h['Ticker'].isin(common_tickers)].copy()
    df_30m = df_30m[df_30m['Ticker'].isin(common_tickers)].copy()
    df_15m = df_15m[df_15m['Ticker'].isin(common_tickers)].copy()
    
    # Sort and rebuild daily dates list
    sorted_daily_dates = sorted(df_daily['DateTime'].str[:10].unique())
    
    # 3. PRE-COMPUTE TECHNICAL INDICATORS ON 15M DATASET
    print("\nComputing indicators and lags on 15M dataset...")
    df_15m = df_15m.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    
    # Bollinger Bands (20 periods, 2 std dev) on Close. Include 'bb_upper' and 'bb_lower'.
    df_15m['ma20'] = df_15m.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df_15m['std20'] = df_15m.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20, min_periods=1).std())
    df_15m['bb_upper'] = df_15m['ma20'] + 2 * df_15m['std20']
    df_15m['bb_lower'] = df_15m['ma20'] - 2 * df_15m['std20']
    
    # Persistence of 15M ranks
    df_15m['long_rank_le_5'] = (df_15m['long_rank'] <= 5).astype(int)
    df_15m['short_rank_le_5'] = (df_15m['short_rank'] <= 5).astype(int)
    df_15m['persist_long_3bar_5'] = df_15m.groupby('Ticker')['long_rank_le_5'].transform(lambda x: x.rolling(3).sum() == 3)
    df_15m['persist_short_3bar_5'] = df_15m.groupby('Ticker')['short_rank_le_5'].transform(lambda x: x.rolling(3).sum() == 3)
    
    df_15m['long_rank_le_3'] = (df_15m['long_rank'] <= 3).astype(int)
    df_15m['short_rank_le_3'] = (df_15m['short_rank'] <= 3).astype(int)
    df_15m['persist_long_4bar_3'] = df_15m.groupby('Ticker')['long_rank_le_3'].transform(lambda x: x.rolling(4).sum() == 4)
    df_15m['persist_short_4bar_3'] = df_15m.groupby('Ticker')['short_rank_le_3'].transform(lambda x: x.rolling(4).sum() == 4)
    
    # Daily high conviction for S22
    df_15m['Date'] = df_15m['DateTime'].str[:10]
    df_15m['long_conv_daily_max'] = df_15m.groupby(['Date', 'Ticker'])['long_conv'].cummax()
    df_15m['short_conv_daily_max'] = df_15m.groupby(['Date', 'Ticker'])['short_conv'].cummax()
    
    # Day Open for S23
    df_15m['day_open'] = df_15m.groupby(['Date', 'Ticker'])['Open'].transform('first')
    
    # Conviction MA3 for S15
    df_15m['long_conv_ma3'] = df_15m.groupby('Ticker')['long_conv'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df_15m['short_conv_ma3'] = df_15m.groupby('Ticker')['short_conv'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    # A. Score momentum lags
    df_15m['long_conv_lag1'] = df_15m.groupby('Ticker')['long_conv'].shift(1)
    df_15m['long_conv_lag2'] = df_15m.groupby('Ticker')['long_conv'].shift(2)
    df_15m['short_conv_lag1'] = df_15m.groupby('Ticker')['short_conv'].shift(1)
    df_15m['short_conv_lag2'] = df_15m.groupby('Ticker')['short_conv'].shift(2)
    
    # B. Conviction Spread & Z-score
    df_15m['spread'] = df_15m['long_conv'] - df_15m['short_conv']
    df_15m['spread_mean_20'] = df_15m.groupby('Ticker')['spread'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df_15m['spread_std_20'] = df_15m.groupby('Ticker')['spread'].transform(lambda x: x.rolling(20, min_periods=5).std())
    df_15m['spread_zscore'] = (df_15m['spread'] - df_15m['spread_mean_20']) / (df_15m['spread_std_20'] + 1e-8)
    
    # C. ATR 14%
    df_15m['prev_close'] = df_15m.groupby('Ticker')['Close'].shift(1)
    df_15m['tr'] = np.maximum(
        df_15m['High'] - df_15m['Low'],
        np.maximum(
            (df_15m['High'] - df_15m['prev_close']).abs(),
            (df_15m['Low'] - df_15m['prev_close']).abs()
        )
    )
    df_15m['atr_14'] = df_15m.groupby('Ticker')['tr'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    df_15m['atr_14_pct'] = (df_15m['atr_14'] / df_15m['Close']) * 100.0
    
    # D. Volatility Regime
    print("Calculating rolling market volatility for S7/18/19/20...")
    bar_volatility = {}
    for dt, g in df_15m.groupby('DateTime'):
        top_10 = g.nlargest(10, 'Dollar_Volume')
        mean_atr_pct = top_10['atr_14_pct'].mean()
        bar_volatility[dt] = mean_atr_pct
        
    vol_df = pd.DataFrame(list(bar_volatility.items()), columns=['DateTime', 'avg_atr'])
    vol_df = vol_df.sort_values('DateTime').reset_index(drop=True)
    vol_df['rolling_vol'] = vol_df['avg_atr'].rolling(20, min_periods=1).mean()
    
    vol_df['p30_vol'] = vol_df['rolling_vol'].expanding(min_periods=1).quantile(0.30)
    vol_df['p50_vol'] = vol_df['rolling_vol'].expanding(min_periods=1).quantile(0.50)
    vol_df['p70_vol'] = vol_df['rolling_vol'].expanding(min_periods=1).quantile(0.70)
    
    dict_rolling_vol = dict(zip(vol_df['DateTime'], vol_df['rolling_vol']))
    dict_p30_vol = dict(zip(vol_df['DateTime'], vol_df['p30_vol'].fillna(0.0)))
    dict_p50_vol = dict(zip(vol_df['DateTime'], vol_df['p50_vol'].fillna(0.0)))
    dict_p70_vol = dict(zip(vol_df['DateTime'], vol_df['p70_vol'].fillna(0.0)))
    
    final_p30 = vol_df['p30_vol'].iloc[-1] if len(vol_df) > 0 else 0.0
    final_p50 = vol_df['p50_vol'].iloc[-1] if len(vol_df) > 0 else 0.0
    final_p70 = vol_df['p70_vol'].iloc[-1] if len(vol_df) > 0 else 0.0
    print(f"  Final Regime Thresholds: P30={final_p30:.4f}%, P50={final_p50:.4f}%, P70={final_p70:.4f}%")
    
    # E. Opening Range (9:15 - 9:45)
    print("Computing opening ranges for S8...")
    df_op = df_15m[df_15m['DateTime'].str[11:16].isin(['09:15', '09:30'])]
    op_range = df_op.groupby(['Date', 'Ticker']).agg(
        or_high=('High', 'max'),
        or_low=('Low', 'min')
    ).reset_index()
    
    dict_or = {}
    for _, row in op_range.iterrows():
        d = row['Date']
        t = row['Ticker']
        if d not in dict_or:
            dict_or[d] = {}
        dict_or[d][t] = (row['or_high'], row['or_low'])
        
    # F. Conviction percentiles per 15M bar (for S1, S3)
    print("Pre-computing conviction percentiles...")
    p60_long_series = df_15m.groupby('DateTime')['long_conv'].transform(lambda x: x.quantile(0.60))
    p60_short_series = df_15m.groupby('DateTime')['short_conv'].transform(lambda x: x.quantile(0.60))
    p70_long_series = df_15m.groupby('DateTime')['long_conv'].transform(lambda x: x.quantile(0.70))
    p70_short_series = df_15m.groupby('DateTime')['short_conv'].transform(lambda x: x.quantile(0.70))
    
    df_15m['p60_long'] = p60_long_series
    df_15m['p60_short'] = p60_short_series
    df_15m['p70_long'] = p70_long_series
    df_15m['p70_short'] = p70_short_series

    # Dollar volume percentile rank per bar for S14
    df_15m['vol_rank_pct'] = df_15m.groupby('DateTime')['Dollar_Volume'].rank(pct=True)

    # Gap percentage for S16
    df_15m['gap_pct'] = (df_15m['Open'] - df_15m['prev_close']) / (df_15m['prev_close'] + 1e-10)

    # 4. INDEX DICTIONARIES FOR FASTER O(1) LOOKUPS
    print("\nIndexing predictions for O(1) simulation lookups...")
    
    dict_daily = {}
    for date_str, group in df_daily.groupby(df_daily['DateTime'].str[:10]):
        dict_daily[date_str] = {}
        for _, row in group.iterrows():
            dict_daily[date_str][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'long_conv': float(row['long_conv']),
                'short_conv': float(row['short_conv']),
                'count': len(group)
            }
            
    dict_1h = {}
    for dt, group in df_1h.groupby('DateTime'):
        dict_1h[dt] = {}
        for _, row in group.iterrows():
            dict_1h[dt][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'long_conv': float(row['long_conv']),
                'short_conv': float(row['short_conv']),
                'long_conv_lag1': float(row['long_conv_lag1']) if pd.notna(row['long_conv_lag1']) else 0.0,
                'short_conv_lag1': float(row['short_conv_lag1']) if pd.notna(row['short_conv_lag1']) else 0.0,
                'count': len(group)
            }
            
    dict_30m = {}
    for dt, group in df_30m.groupby('DateTime'):
        dict_30m[dt] = {}
        for _, row in group.iterrows():
            dict_30m[dt][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'long_conv': float(row['long_conv']),
                'short_conv': float(row['short_conv']),
                'long_conv_lag1': float(row['long_conv_lag1']) if pd.notna(row['long_conv_lag1']) else 0.0,
                'short_conv_lag1': float(row['short_conv_lag1']) if pd.notna(row['short_conv_lag1']) else 0.0,
                'count': len(group)
            }

    # Align daily, 1H, and 30M convictions & ranks to df_15m for blended scoring
    print("Aligning multi-timeframe convictions and ranks to 15M...")
    unique_dates = df_15m['Date'].unique()
    date_to_yesterday = {}
    for d in unique_dates:
        date_to_yesterday[d] = find_yesterday_daily_date(d, sorted_daily_dates)
        
    aligned_daily_long_conv = []
    aligned_daily_short_conv = []
    aligned_daily_long_rank = []
    aligned_daily_short_rank = []
    
    aligned_1h_long_conv = []
    aligned_1h_short_conv = []
    aligned_1h_long_rank = []
    aligned_1h_short_rank = []
    aligned_1h_long_conv_lag1 = []
    aligned_1h_short_conv_lag1 = []
    
    aligned_30m_long_conv = []
    aligned_30m_short_conv = []
    aligned_30m_long_rank = []
    aligned_30m_short_rank = []
    aligned_30m_long_conv_lag1 = []
    aligned_30m_short_conv_lag1 = []
    
    for t, dt, d in zip(df_15m['Ticker'], df_15m['DateTime'], df_15m['Date']):
        # Daily
        y_date = date_to_yesterday.get(d)
        d_val = dict_daily.get(y_date, {}).get(t) if y_date else None
        if d_val:
            aligned_daily_long_conv.append(d_val['long_conv'])
            aligned_daily_short_conv.append(d_val['short_conv'])
            aligned_daily_long_rank.append(d_val['long_rank'])
            aligned_daily_short_rank.append(d_val['short_rank'])
        else:
            aligned_daily_long_conv.append(0.0)
            aligned_daily_short_conv.append(0.0)
            aligned_daily_long_rank.append(9999)
            aligned_daily_short_rank.append(9999)
            
        # Align timeframes
        t_1h, t_30m = align_timeframes(dt)
        
        # 1H
        h_val = dict_1h.get(t_1h, {}).get(t) if t_1h else None
        if h_val:
            aligned_1h_long_conv.append(h_val['long_conv'])
            aligned_1h_short_conv.append(h_val['short_conv'])
            aligned_1h_long_rank.append(h_val['long_rank'])
            aligned_1h_short_rank.append(h_val['short_rank'])
            aligned_1h_long_conv_lag1.append(h_val['long_conv_lag1'])
            aligned_1h_short_conv_lag1.append(h_val['short_conv_lag1'])
        else:
            aligned_1h_long_conv.append(0.0)
            aligned_1h_short_conv.append(0.0)
            aligned_1h_long_rank.append(9999)
            aligned_1h_short_rank.append(9999)
            aligned_1h_long_conv_lag1.append(0.0)
            aligned_1h_short_conv_lag1.append(0.0)
            
        # 30M
        m_val = dict_30m.get(t_30m, {}).get(t) if t_30m else None
        if m_val:
            aligned_30m_long_conv.append(m_val['long_conv'])
            aligned_30m_short_conv.append(m_val['short_conv'])
            aligned_30m_long_rank.append(m_val['long_rank'])
            aligned_30m_short_rank.append(m_val['short_rank'])
            aligned_30m_long_conv_lag1.append(m_val['long_conv_lag1'])
            aligned_30m_short_conv_lag1.append(m_val['short_conv_lag1'])
        else:
            aligned_30m_long_conv.append(0.0)
            aligned_30m_short_conv.append(0.0)
            aligned_30m_long_rank.append(9999)
            aligned_30m_short_rank.append(9999)
            aligned_30m_long_conv_lag1.append(0.0)
            aligned_30m_short_conv_lag1.append(0.0)
            
    df_15m['daily_long_conv'] = aligned_daily_long_conv
    df_15m['daily_short_conv'] = aligned_daily_short_conv
    df_15m['daily_long_rank'] = aligned_daily_long_rank
    df_15m['daily_short_rank'] = aligned_daily_short_rank
    
    df_15m['h1_long_conv'] = aligned_1h_long_conv
    df_15m['h1_short_conv'] = aligned_1h_short_conv
    df_15m['h1_long_rank'] = aligned_1h_long_rank
    df_15m['h1_short_rank'] = aligned_1h_short_rank
    df_15m['h1_long_conv_lag1'] = aligned_1h_long_conv_lag1
    df_15m['h1_short_conv_lag1'] = aligned_1h_short_conv_lag1
    
    df_15m['m30_long_conv'] = aligned_30m_long_conv
    df_15m['m30_short_conv'] = aligned_30m_short_conv
    df_15m['m30_long_rank'] = aligned_30m_long_rank
    df_15m['m30_short_rank'] = aligned_30m_short_rank
    df_15m['m30_long_conv_lag1'] = aligned_30m_long_conv_lag1
    df_15m['m30_short_conv_lag1'] = aligned_30m_short_conv_lag1
    
    # Compute blended scores: 0.1 * daily_conv + 0.2 * 1h_conv + 0.3 * 30m_conv + 0.5 * 15m_conv
    df_15m['blend_long_score'] = (
        0.1 * df_15m['daily_long_conv'] +
        0.2 * df_15m['h1_long_conv'] +
        0.3 * df_15m['m30_long_conv'] +
        0.5 * df_15m['long_conv']
    )
    df_15m['blend_short_score'] = (
        0.1 * df_15m['daily_short_conv'] +
        0.2 * df_15m['h1_short_conv'] +
        0.3 * df_15m['m30_short_conv'] +
        0.5 * df_15m['short_conv']
    )
    
    # Rank stocks by blended scores per 15M bar
    df_15m['blend_long_rank'] = df_15m.groupby('DateTime')['blend_long_score'].rank(ascending=False)
    df_15m['blend_short_rank'] = df_15m.groupby('DateTime')['blend_short_score'].rank(ascending=False)

    # Pre-build list of dictionaries for each ticker's 15M series
    ticker_bars = {t: [] for t in common_tickers}
    ticker_bar_map = {t: {} for t in common_tickers}
    
    print("Building structured price lists...")
    for idx, row in df_15m.iterrows():
        t = row['Ticker']
        if t not in ticker_bars:
            continue
        bar_dict = {
            'ticker': t,
            'datetime': row['DateTime'],
            'date': row['Date'],
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'ibs': float((row['Close'] - row['Low']) / (row['High'] - row['Low'] + 1e-10)),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv_lag1': float(row['long_conv_lag1']) if pd.notna(row['long_conv_lag1']) else 0.0,
            'long_conv_lag2': float(row['long_conv_lag2']) if pd.notna(row['long_conv_lag2']) else 0.0,
            'short_conv_lag1': float(row['short_conv_lag1']) if pd.notna(row['short_conv_lag1']) else 0.0,
            'short_conv_lag2': float(row['short_conv_lag2']) if pd.notna(row['short_conv_lag2']) else 0.0,
            'spread_zscore': float(row['spread_zscore']) if pd.notna(row['spread_zscore']) else 0.0,
            'atr_14_pct': float(row['atr_14_pct']) if pd.notna(row['atr_14_pct']) else 0.0,
            'p60_long': float(row['p60_long']),
            'p60_short': float(row['p60_short']),
            'p70_long': float(row['p70_long']),
            'p70_short': float(row['p70_short']),
            
            # Bollinger Bands
            'bb_upper': float(row['bb_upper']),
            'bb_lower': float(row['bb_lower']),
            
            # Multi-TF ranks
            'daily_long_rank': int(row['daily_long_rank']),
            'daily_short_rank': int(row['daily_short_rank']),
            'h1_long_rank': int(row['h1_long_rank']),
            'h1_short_rank': int(row['h1_short_rank']),
            'm30_long_rank': int(row['m30_long_rank']),
            'm30_short_rank': int(row['m30_short_rank']),
            'm15_long_rank': int(row['long_rank']),
            'm15_short_rank': int(row['short_rank']),
            
            # Blends & Persistence
            'blend_long_score': float(row['blend_long_score']),
            'blend_short_score': float(row['blend_short_score']),
            'blend_long_rank': int(row['blend_long_rank']),
            'blend_short_rank': int(row['blend_short_rank']),
            'persist_long_3bar_5': bool(row['persist_long_3bar_5']),
            'persist_short_3bar_5': bool(row['persist_short_3bar_5']),
            'persist_long_4bar_3': bool(row['persist_long_4bar_3']),
            'persist_short_4bar_3': bool(row['persist_short_4bar_3']),
            
            # S12, S15, S16, S22, S23 auxiliary indicators
            'vol_rank_pct': float(row['vol_rank_pct']),
            'gap_pct': float(row['gap_pct']),
            'long_conv_daily_max': float(row['long_conv_daily_max']),
            'short_conv_daily_max': float(row['short_conv_daily_max']),
            'day_open': float(row['day_open']),
            'long_conv_ma3': float(row['long_conv_ma3']),
            'short_conv_ma3': float(row['short_conv_ma3']),
            
            # Triple TF lag 1 convictions
            'h1_long_conv': float(row['h1_long_conv']),
            'h1_short_conv': float(row['h1_short_conv']),
            'h1_long_conv_lag1': float(row['h1_long_conv_lag1']),
            'h1_short_conv_lag1': float(row['h1_short_conv_lag1']),
            'm30_long_conv': float(row['m30_long_conv']),
            'm30_short_conv': float(row['m30_short_conv']),
            'm30_long_conv_lag1': float(row['m30_long_conv_lag1']),
            'm30_short_conv_lag1': float(row['m30_short_conv_lag1']),
            
            'idx': len(ticker_bars[t])
        }
        ticker_bars[t].append(bar_dict)
        ticker_bar_map[t][row['DateTime']] = bar_dict
        
    unique_15m_times = sorted(df_15m['DateTime'].unique())
    print(f"Total active 15M bar timestamps: {len(unique_15m_times)}")

    # Pre-build a dict of 15M bar lists for each timestamp to make loops faster
    dict_15m = {}
    for dt, group in df_15m.groupby('DateTime'):
        dict_15m[dt] = []
        for _, row in group.iterrows():
            t = row['Ticker']
            if t in ticker_bar_map and dt in ticker_bar_map[t]:
                dict_15m[dt].append(ticker_bar_map[t][dt])

    results = {}

    # =========================================================================
    # SIMULATION FUNCTION
    # =========================================================================
    def simulate_strategy(strategy_id, name):
        print(f"\nSimulating Strategy {strategy_id}: {name}...")
        
        # Max daily trade limits dictionary
        max_daily = {
            1: 6, 2: 4, 3: 6, 4: 8, 5: 4, 6: 6, 8: 6, 9: 6, 10: 4,
            11: 6, 12: 6, 13: 6, 14: 6, 15: 4, 16: 4, 17: 6, 18: 4,
            19: 6, 20: 6, 21: 6, 22: 4, 23: 4, 24: 6, 25: 6
        }
        limit = max_daily.get(strategy_id, 6)
        
        trades = []
        active_trades = [] # list of active trade dicts
        
        # Track active pairs for S6
        active_pairs = []
        
        # Trade counter per day
        daily_trade_count = {}
        # Strategy 8: Tickers traded per day
        daily_tickers_traded = {}
        
        for bar_idx, T in enumerate(unique_15m_times):
            date_str = T[:10]
            time_str = T[11:16]
            h, m = map(int, time_str.split(':'))
            t_min = h * 60 + m
            
            # Reset daily trade limits if it's a new day
            if date_str not in daily_trade_count:
                daily_trade_count[date_str] = 0
                daily_tickers_traded[date_str] = set()
                
            # Align timeframes
            t_1h, t_30m = align_timeframes(T)
            # Find yesterday's daily predictions date
            d_daily = find_yesterday_daily_date(date_str, sorted_daily_dates)
            
            # Identify if it is the last bar of the day
            is_last_bar = (bar_idx == len(unique_15m_times) - 1) or (unique_15m_times[bar_idx + 1][:10] != date_str) or (time_str == "15:15")
            
            # ─────────────────────────────────────────────────────────────
            # A. MANAGE ACTIVE TRADE EXITS
            # ─────────────────────────────────────────────────────────────
            remaining_trades = []
            
            if strategy_id == 6:
                # Handle exits for pairs strategy (S6)
                remaining_pairs = []
                for p in active_pairs:
                    p['bars_held'] += 1
                    
                    # Look up current bars
                    bar_long = ticker_bar_map[p['long_ticker']].get(T)
                    bar_short = ticker_bar_map[p['short_ticker']].get(T)
                    
                    # If data gap, force exit
                    if not bar_long or not bar_short:
                        for leg_ticker, leg_side, leg_entry_price, last_close in [
                            (p['long_ticker'], 'LONG', p['entry_price_long'], p['entry_price_long']),
                            (p['short_ticker'], 'SHORT', p['entry_price_short'], p['entry_price_short'])
                        ]:
                            g_ret = 0.0
                            n_ret = g_ret - TRANSACTION_COST_PCT / 100
                            trades.append({
                                'date': date_str, 'entry_time': p['entry_time'], 'exit_time': T,
                                'ticker': leg_ticker, 'side': leg_side, 'entry_price': leg_entry_price,
                                'exit_price': last_close, 'bars_held': p['bars_held'], 'exit_reason': 'DATA_GAP',
                                'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
                            })
                        continue
                        
                    # Calculate returns
                    p_ret_long = (bar_long['close'] / p['entry_price_long']) - 1.0
                    p_ret_short = 1.0 - (bar_short['close'] / p['entry_price_short'])
                    
                    t_dt = pd.to_datetime(T)
                    sub_5m = [t_dt, t_dt + timedelta(minutes=5), t_dt + timedelta(minutes=10)]
                    long_stop_hit = False
                    short_stop_hit = False
                    
                    for t5 in sub_5m:
                        b5_long = dict_5m_candles.get(p['long_ticker'], {}).get(t5)
                        b5_short = dict_5m_candles.get(p['short_ticker'], {}).get(t5)
                        
                        if b5_long and b5_long['low'] <= p['entry_price_long'] * 0.995:
                            long_stop_hit = True
                        if b5_short and b5_short['high'] >= p['entry_price_short'] * 1.005:
                            short_stop_hit = True
                            
                        if long_stop_hit or short_stop_hit:
                            break
                    
                    if not (long_stop_hit or short_stop_hit):
                        # Backup with 15m just in case 5m data was missing
                        long_stop_hit = bar_long['low'] <= p['entry_price_long'] * 0.995
                        short_stop_hit = bar_short['high'] >= p['entry_price_short'] * 1.005

                    exit_reason = None
                    if long_stop_hit or short_stop_hit:
                        exit_reason = 'STOP_LOSS'
                    elif p['bars_held'] >= 4:
                        exit_reason = 'TIME_EXPIRY'
                    elif is_last_bar:
                        exit_reason = 'FORCE_CLOSE_EOD'
                        
                    if exit_reason:
                        # Close both legs of the pair
                        leg_exits = []
                        if exit_reason == 'STOP_LOSS':
                            # Determine stopped vs market exits
                            if long_stop_hit:
                                exit_p_long = p['entry_price_long'] * 0.995
                                ret_long = -0.005
                            else:
                                exit_p_long = bar_long['close']
                                ret_long = p_ret_long
                                
                            if short_stop_hit:
                                exit_p_short = p['entry_price_short'] * 1.005
                                ret_short = -0.005
                            else:
                                exit_p_short = bar_short['close']
                                ret_short = p_ret_short
                            
                            leg_exits = [
                                (p['long_ticker'], 'LONG', p['entry_price_long'], exit_p_long, ret_long),
                                (p['short_ticker'], 'SHORT', p['entry_price_short'], exit_p_short, ret_short)
                            ]
                        else:
                            leg_exits = [
                                (p['long_ticker'], 'LONG', p['entry_price_long'], bar_long['close'], p_ret_long),
                                (p['short_ticker'], 'SHORT', p['entry_price_short'], bar_short['close'], p_ret_short)
                            ]
                            
                        for leg_ticker, leg_side, leg_entry_price, exit_price, g_ret in leg_exits:
                            n_ret = g_ret - TRANSACTION_COST_PCT / 100
                            trades.append({
                                'date': date_str, 'entry_time': p['entry_time'], 'exit_time': T,
                                'ticker': leg_ticker, 'side': leg_side, 'entry_price': leg_entry_price,
                                'exit_price': exit_price, 'bars_held': p['bars_held'], 'exit_reason': exit_reason,
                                'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
                            })
                    else:
                        remaining_pairs.append(p)
                active_pairs = remaining_pairs
                
            else:
                # Handle exits for standard single-trade strategies
                for t in active_trades:
                    t['bars_held'] += 1
                    
                    # Look up ticker's current bar at T
                    bar = ticker_bar_map[t['ticker']].get(T)
                    if not bar:
                        # Force exit due to data gap
                        g_ret = 0.0
                        n_ret = g_ret - TRANSACTION_COST_PCT / 100
                        trades.append({
                            'date': date_str, 'entry_time': t['entry_time'], 'exit_time': T,
                            'ticker': t['ticker'], 'side': t['side'], 'entry_price': t['entry_price'],
                            'exit_price': t['entry_price'], 'bars_held': t['bars_held'], 'exit_reason': 'DATA_GAP',
                            'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
                        })
                        continue
                        
                    exit_reason = None
                    exit_price = bar['close']
                    
                    # UNIFORM 4-BAR HOLD TEST FOR S26-S35
                    target_hold = 4
                    
                    trailing_stop = None
                    tp_pct = None
                    sl_pct = 0.010 # 1.0% stop loss
                    
                    t_dt = pd.to_datetime(T)
                    sub_5m = [t_dt, t_dt + timedelta(minutes=5), t_dt + timedelta(minutes=10)]
                    
                    for t5 in sub_5m:
                        b5 = dict_5m_candles.get(t['ticker'], {}).get(t5)
                        if not b5: continue
                        
                        if t['side'] == 'LONG':
                            l_pnl = (b5['low'] / t['entry_price']) - 1.0
                            h_pnl = (b5['high'] / t['entry_price']) - 1.0
                            t['peak_pnl'] = max(t['peak_pnl'], h_pnl)
                            
                            if sl_pct and l_pnl <= -sl_pct:
                                exit_reason = 'STOP_LOSS'
                                exit_price = t['entry_price'] * (1.0 - sl_pct)
                                break
                        else:
                            # Short side
                            low_price_pnl = 1.0 - (b5['low'] / t['entry_price'])
                            t['peak_pnl'] = max(t['peak_pnl'], low_price_pnl)
                            high_price_pnl = 1.0 - (b5['high'] / t['entry_price'])
                            
                            if sl_pct and high_price_pnl <= -sl_pct:
                                exit_reason = 'STOP_LOSS'
                                exit_price = t['entry_price'] * (1.0 + sl_pct)
                                break
                                
                    # If stopped out intrabar, exit logic ends here for this trade.
                    if not exit_reason:
                        # Update peak PnL with 15m bar extremes just in case 5m data was missing
                        if t['side'] == 'LONG':
                            t['peak_pnl'] = max(t['peak_pnl'], (bar['high'] / t['entry_price']) - 1.0)
                        else:
                            t['peak_pnl'] = max(t['peak_pnl'], 1.0 - (bar['low'] / t['entry_price']))
                            
                        # Uniform time expiry based on avg hold
                        if t['bars_held'] >= target_hold:
                            exit_reason = 'TIME_EXPIRY'
                    
                    if not exit_reason and is_last_bar:
                        exit_reason = 'FORCE_CLOSE_EOD'
                        
                    if exit_reason:
                        # Exit trade
                        # Only set exit_price to bar['close'] if it wasn't already set to the exact exit level by the intrabar check loop
                        if exit_reason not in ['STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP']:
                            exit_price = bar['close']
                                
                        if t['side'] == 'LONG':
                            g_ret = (exit_price / t['entry_price']) - 1.0
                        else:
                            g_ret = 1.0 - (exit_price / t['entry_price'])
                            
                        n_ret = g_ret - TRANSACTION_COST_PCT / 100
                        trades.append({
                            'date': date_str, 'entry_time': t['entry_time'], 'exit_time': T,
                            'ticker': t['ticker'], 'side': t['side'], 'entry_price': t['entry_price'],
                            'exit_price': exit_price, 'bars_held': t['bars_held'], 'exit_reason': exit_reason,
                            'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
                        })
                    else:
                        remaining_trades.append(t)
                active_trades = remaining_trades

            # ─────────────────────────────────────────────────────────────
            # B. EVALUATE NEW ENTRIES
            # ─────────────────────────────────────────────────────────────
            if time_str > "14:45":
                continue
                
            # Skip if data for this 15M bar is missing
            if T not in dict_15m:
                continue
                
            active_15m = dict_15m[T]
            if len(active_15m) == 0:
                continue
                
            active_tickers = {at['ticker'] for at in active_trades}
                
            # Filter tickers that have daily predictions if required
            has_daily_req = strategy_id in [1, 2, 5, 10, 11, 15, 16, 23]
            if has_daily_req and (not d_daily or d_daily not in dict_daily):
                continue
                
            # Hourly and 30M required signals check
            has_1h_req = strategy_id in [3, 6, 8, 10, 11, 13, 25]
            if has_1h_req and (not t_1h or t_1h not in dict_1h):
                continue
            has_30m_req = strategy_id in [2, 6, 7, 10, 13, 25]
            if has_30m_req and (not t_30m or t_30m not in dict_30m):
                continue
                
            # Sort 15M tickers by ranks
            top_15m_longs = sorted(active_15m, key=lambda x: x['long_rank'])
            top_15m_shorts = sorted(active_15m, key=lambda x: x['short_rank'])
            

            # =================================================================
            # STRATEGY 26: The Morning Gap Reversal
            # =================================================================
            if strategy_id == 26:
                if time_str == "09:15":
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_longs:
                            if x['gap_pct'] < -0.01 and x['h1_long_rank'] <= 3 and x['daily_long_rank'] <= 3:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 27: Consecutive Conviction Acceleration
            # =================================================================
            elif strategy_id == 27:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs:
                        if x['long_rank'] <= 10 and x['long_conv'] > x['long_conv_lag1'] and x['long_conv_lag1'] > x['long_conv_lag2'] and x['long_conv_lag2'] > 0:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break
                    for x in top_15m_shorts:
                        if daily_trade_count[date_str] >= limit: break
                        if x['short_rank'] <= 10 and x['short_conv'] > x['short_conv_lag1'] and x['short_conv_lag1'] > x['short_conv_lag2'] and x['short_conv_lag2'] > 0:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1

            # =================================================================
            # STRATEGY 28: Midday Volatility Squeeze
            # =================================================================
            elif strategy_id == 28:
                curr_vol = dict_rolling_vol.get(T, 0.0)
                if curr_vol < dict_p30_vol.get(T, 0.0): # LOW VOL
                    if time_str in ["11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00"]:
                        if daily_trade_count[date_str] < limit:
                            for x in top_15m_longs:
                                if x['long_rank'] <= 3 and 0.4 <= x['ibs'] <= 0.6:
                                    ticker = x['ticker']
                                    if ticker not in active_tickers:
                                        active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                        active_tickers.add(ticker)
                                        daily_trade_count[date_str] += 1
                                        if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 29: Contrarian High-Vol Fade
            # =================================================================
            elif strategy_id == 29:
                curr_vol = dict_rolling_vol.get(T, 0.0)
                if curr_vol > dict_p70_vol.get(T, 0.0): # HIGH VOL
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 5 and x['close'] >= x['bb_upper']:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 30: Macro Alignment Scalper
            # =================================================================
            elif strategy_id == 30:
                if daily_trade_count[date_str] < limit:
                    for x in top_15m_longs:
                        if x['long_rank'] <= 5 and x['h1_short_rank'] <= 5 and x['daily_short_rank'] <= 5:
                            # 15M says Buy, but macro says massive Short. We short it.
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 31: Extreme Z-Score Reversion
            # =================================================================
            elif strategy_id == 31:
                if daily_trade_count[date_str] < limit:
                    for x in active_15m:
                        if x['spread_zscore'] > 2.5: # Extremely overvalued conviction
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break
                        elif x['spread_zscore'] < -2.5: # Extremely undervalued conviction
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 32: The Persistent Anchor
            # =================================================================
            elif strategy_id == 32:
                if time_str == "13:00":
                    if daily_trade_count[date_str] < limit:
                        for x in active_15m:
                            if x['persist_long_4bar_3']:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 33: EOD Retail Liquidity Trap
            # =================================================================
            elif strategy_id == 33:
                if time_str >= "14:15":
                    if daily_trade_count[date_str] < limit:
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 3 and x['ibs'] > 0.85 and x['short_conv'] > x['long_conv']:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 34: Triple-Timeframe Laggard
            # =================================================================
            elif strategy_id == 34:
                if daily_trade_count[date_str] < limit:
                    for x in active_15m:
                        if x['h1_long_rank'] <= 3 and x['m30_long_rank'] <= 3 and x['long_rank'] > 20:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 35: Volatility Contraction Breakout
            # =================================================================
            elif strategy_id == 35:
                if daily_trade_count[date_str] < limit:
                    for x in active_15m:
                        if x['atr_14_pct'] < 0.002 and x['blend_long_rank'] <= 5:
                            ticker = x['ticker']
                            if ticker not in active_tickers:
                                active_trades.append({'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'], 'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0})
                                active_tickers.add(ticker)
                                daily_trade_count[date_str] += 1
                                if daily_trade_count[date_str] >= limit: break


        # End of simulation for this strategy, evaluate trades
        eval_res = evaluate_strategy_trades(trades, name, slots=limit)
        results[f"strategy_{strategy_id}"] = eval_res
        results[f"trades_{strategy_id}"] = trades
        print(f"  Trades generated: {len(trades)} | Net P&L: {eval_res['total_return']*100:+.2f}%")
        return eval_res

    # Run simulations sequentially
    strategies_info = [
        (26, "The Morning Gap Reversal"),
        (27, "Consecutive Conviction Acceleration"),
        (28, "Midday Volatility Squeeze"),
        (29, "Contrarian High-Vol Fade"),
        (30, "Macro Alignment Scalper"),
        (31, "Extreme Z-Score Reversion"),
        (32, "The Persistent Anchor"),
        (33, "EOD Retail Liquidity Trap"),
        (34, "Triple-Timeframe Laggard"),
        (35, "Volatility Contraction Breakout")
    ]
    
    for s_id, s_name in strategies_info:
        simulate_strategy(s_id, s_name)
        
    # Save the output results json
    output_path = "data/strategy_10_new_results.json"
    with open(output_path, "w") as f:
        json.dump({
            'holdout_month': TEST_MONTH,
            'tickers_universe_size': len(common_tickers),
            'strategies': results,
            'backtested_at': datetime.now().isoformat()
        }, f, indent=2)
        
    print(f"\n[SUCCESS] Saved backtest outcomes to {output_path}")
    
    # 5. PRINT SUMMARY ASCII MARKDOWN TABLE
    print("\n" + "=" * 145)
    print("BACKTEST SIMULATION SUMMARY TABLE (NEW S26-S35)")
    print("=" * 145)
    headers = ["ID", "Strategy Name", "Trades", "L-Trds", "S-Trds", "WR %", "L-WR %", "S-WR %", "Return %", "Profit Factor", "Max DD %", "Avg Hold"]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-|-'.join(['-' * len(h) for h in headers])}|")
    for s_id, s_name in strategies_info:
        res = results[f"strategy_{s_id}"]
        s_name_trunc = s_name[:35]
        pnl_str = f"{res['total_return']*100:+.2f}%"
        wr_str = f"{res['win_rate']*100:.1f}%"
        l_wr_str = f"{res['long_win_rate']*100:.1f}%" if res['long_trades'] > 0 else "N/A"
        s_wr_str = f"{res['short_win_rate']*100:.1f}%" if res['short_trades'] > 0 else "N/A"
        pf_str = f"{res['profit_factor']:.2f}" if res['profit_factor'] != float('inf') else "inf"
        dd_str = f"{res['max_drawdown']*100:.2f}%"
        print(f"| {s_id:<2} | {s_name_trunc:<35} | {res['total_trades']:<6} | {res['long_trades']:<6} | {res['short_trades']:<6} | {wr_str:<6} | {l_wr_str:<6} | {s_wr_str:<6} | {pnl_str:<8} | {pf_str:<13} | {dd_str:<8} | {res['avg_bars_held']:<8.1f} |")
    print("=" * 145)

if __name__ == "__main__":
    main()
