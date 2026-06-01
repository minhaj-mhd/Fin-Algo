"""
strategy_10x_backtest.py — Comprehensive Intraday Backtester for 10 Strategies

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
def evaluate_strategy_trades(trades, name):
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
    print("10X INTRADAY TRADING STRATEGIES BACKTEST SIMULATION")
    print("=" * 70)

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
    
    # 3. PRE-COMPUTE TECHNICAL INDICATORS ON 15M DATASET
    print("\nComputing indicators and lags on 15M dataset...")
    df_15m = df_15m.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    
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
    print("Calculating rolling market volatility for S7...")
    bar_volatility = {}
    for dt, g in df_15m.groupby('DateTime'):
        top_10 = g.nlargest(10, 'Dollar_Volume')
        mean_atr_pct = top_10['atr_14_pct'].mean()
        bar_volatility[dt] = mean_atr_pct
        
    vol_df = pd.DataFrame(list(bar_volatility.items()), columns=['DateTime', 'avg_atr'])
    vol_df = vol_df.sort_values('DateTime').reset_index(drop=True)
    vol_df['rolling_vol'] = vol_df['avg_atr'].rolling(20, min_periods=1).mean()
    
    dict_rolling_vol = dict(zip(vol_df['DateTime'], vol_df['rolling_vol']))
    all_rolling_vols = vol_df['rolling_vol'].dropna().values
    p30_vol = np.percentile(all_rolling_vols, 30) if len(all_rolling_vols) > 0 else 0.0
    p70_vol = np.percentile(all_rolling_vols, 70) if len(all_rolling_vols) > 0 else 0.0
    print(f"  Regime Thresholds: P30={p30_vol:.4f}%, P70={p70_vol:.4f}%")
    
    # E. Opening Range (9:15 - 9:45)
    print("Computing opening ranges for S8...")
    df_15m['Date'] = df_15m['DateTime'].str[:10]
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

    # 4. INDEX DICTIONARIES FOR FASTER O(1) LOOKUPS
    print("\nIndexing predictions for O(1) simulation lookups...")
    
    sorted_daily_dates = sorted(df_daily['DateTime'].str[:10].unique())
    
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
                'count': len(group)
            }

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
    def simulate_strategy(strategy_id, name, apply_gatekeeper=False):
        pass_name = "GATED" if apply_gatekeeper else "UNGATED"
        print(f"\nSimulating Strategy {strategy_id}: {name} [{pass_name}]...")
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
                    
                    # Track high/low check for stop loss
                    long_stop_hit = bar_long['low'] <= p['entry_price_long'] * 0.995 # -0.5%
                    short_stop_hit = bar_short['high'] >= p['entry_price_short'] * 1.005 # +0.5% (loss for short)
                    
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
                        
                    # Calculate current returns
                    if t['side'] == 'LONG':
                        current_pnl = (bar['close'] / t['entry_price']) - 1.0
                        high_pnl = (bar['high'] / t['entry_price']) - 1.0
                        t['peak_pnl'] = max(t['peak_pnl'], high_pnl)
                    else:
                        current_pnl = 1.0 - (bar['close'] / t['entry_price'])
                        low_pnl = 1.0 - (bar['low'] / t['entry_price'])
                        t['peak_pnl'] = max(t['peak_pnl'], low_pnl)
                        
                    exit_reason = None
                    
                    # Strategy-specific exits
                    if strategy_id == 1:
                        if t['bars_held'] >= 1:
                            exit_reason = 'TIME_EXPIRY'
                            
                    elif strategy_id == 2:
                        stop_hit = bar['high'] >= t['entry_price'] * 1.006 # Short stop loss (+0.6%)
                        if stop_hit:
                            exit_reason = 'STOP_LOSS'
                        elif t['bars_held'] >= 4: # 60 mins = 4 bars
                            exit_reason = 'TIME_EXPIRY'
                            
                    elif strategy_id == 3:
                        conv_flip = (bar['long_conv'] < 0 if t['side'] == 'LONG' else bar['short_conv'] < 0)
                        if t['bars_held'] >= 2:
                            exit_reason = 'TIME_EXPIRY'
                        elif conv_flip:
                            exit_reason = 'CONV_FLIP'
                            
                    elif strategy_id == 4:
                        conv_rev = (bar['long_conv'] < bar['long_conv_lag1'] if t['side'] == 'LONG' else bar['short_conv'] < bar['short_conv_lag1'])
                        if conv_rev:
                            exit_reason = 'CONV_REVERSAL'
                        elif t['bars_held'] >= 4:
                            exit_reason = 'TIME_EXPIRY'
                            
                    elif strategy_id == 5:
                        if t['bars_held'] >= 1:
                            exit_reason = 'TIME_EXPIRY'
                            
                    elif strategy_id == 7:
                        regime = t['regime']
                        if regime == 'LOW_VOL':
                            if t['bars_held'] >= 1:
                                exit_reason = 'TIME_EXPIRY'
                        elif regime == 'HIGH_VOL':
                            # Trailing stop -0.4% from peak, TP +0.6%
                            if t['peak_pnl'] - current_pnl >= 0.004:
                                exit_reason = 'TRAILING_STOP'
                            elif current_pnl >= 0.006:
                                exit_reason = 'TAKE_PROFIT'
                            elif t['bars_held'] >= 4:
                                exit_reason = 'TIME_EXPIRY'
                        else: # NORMAL
                            if t['bars_held'] >= 2:
                                exit_reason = 'TIME_EXPIRY'
                                
                    elif strategy_id == 8:
                        # Trailing stop -0.4% from peak, TP +1.0%, max hold 8 bars
                        if t['peak_pnl'] - current_pnl >= 0.004:
                            exit_reason = 'TRAILING_STOP'
                        elif current_pnl >= 0.010:
                            exit_reason = 'TAKE_PROFIT'
                        elif t['bars_held'] >= 8:
                            exit_reason = 'TIME_EXPIRY'
                            
                    elif strategy_id == 9:
                        # Exit when Z-score returns below 1.0 (for long) or above -1.0 (for short)
                        z_exit = (bar['spread_zscore'] < 1.0 if t['side'] == 'LONG' else bar['spread_zscore'] > -1.0)
                        if z_exit:
                            exit_reason = 'ZSCORE_EXIT'
                        elif t['bars_held'] >= 6:
                            exit_reason = 'TIME_EXPIRY'
                            
                    elif strategy_id == 10:
                        # Hold 4 bars, trailing stop -0.5% / TP +1.0%, or conviction flip
                        conv_flip = (bar['long_conv'] < 0 if t['side'] == 'LONG' else bar['short_conv'] < 0)
                        if t['peak_pnl'] - current_pnl >= 0.005:
                            exit_reason = 'TRAILING_STOP'
                        elif current_pnl >= 0.010:
                            exit_reason = 'TAKE_PROFIT'
                        elif conv_flip:
                            exit_reason = 'CONV_FLIP'
                        elif t['bars_held'] >= 4:
                            exit_reason = 'TIME_EXPIRY'
                            
                    elif strategy_id == 11:
                        # Pure 1H model exits after 1 hour (4 bars)
                        if t['bars_held'] >= 4:
                            exit_reason = 'TIME_EXPIRY'
                            
                    # Force close at EOD
                    if is_last_bar:
                        exit_reason = 'FORCE_CLOSE_EOD'
                        
                    if exit_reason:
                        # Exit trade
                        exit_price = bar['close']
                        if exit_reason == 'STOP_LOSS':
                            if strategy_id == 2:
                                exit_price = t['entry_price'] * 1.006
                        elif exit_reason == 'TAKE_PROFIT':
                            if strategy_id == 7:
                                exit_price = t['entry_price'] * (1.006 if t['side'] == 'LONG' else 0.994)
                            elif strategy_id in [8, 10]:
                                exit_price = t['entry_price'] * (1.010 if t['side'] == 'LONG' else 0.990)
                        elif exit_reason == 'TRAILING_STOP':
                            if strategy_id in [7, 8]:
                                exit_price = t['entry_price'] * (1.0 + t['peak_pnl'] - 0.004) if t['side'] == 'LONG' else t['entry_price'] * (1.0 - (t['peak_pnl'] - 0.004))
                            elif strategy_id == 10:
                                exit_price = t['entry_price'] * (1.0 + t['peak_pnl'] - 0.005) if t['side'] == 'LONG' else t['entry_price'] * (1.0 - (t['peak_pnl'] - 0.005))
                                
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
            # Skip if data for this 15M bar is missing
            if T not in dict_15m:
                continue
                
            active_15m = dict_15m[T]
            if len(active_15m) == 0:
                continue
                
            active_tickers = {at['ticker'] for at in active_trades}
                
            # Filter tickers that have daily aligned predictions if strategy requires daily gatekeeper
            has_daily_req = strategy_id in [1, 2, 5, 10]
            if has_daily_req and (not d_daily or d_daily not in dict_daily):
                continue
                
            # Hourly and 30M required signals check
            has_1h_req = strategy_id in [3, 6, 8, 10]
            if has_1h_req and (not t_1h or t_1h not in dict_1h):
                continue
            has_30m_req = strategy_id in [2, 6, 7, 10]
            if has_30m_req and (not t_30m or t_30m not in dict_30m):
                continue
                
            # Sort 15M tickers by ranks
            top_15m_longs = sorted(active_15m, key=lambda x: x['long_rank'])
            top_15m_shorts = sorted(active_15m, key=lambda x: x['short_rank'])
            
            # GLOBAL GATEKEEPER PASS
            if apply_gatekeeper and d_daily and d_daily in dict_daily:
                filtered_longs = []
                for x in top_15m_longs:
                    d_pred = dict_daily[d_daily].get(x['ticker'])
                    if d_pred and d_pred['long_rank'] <= 0.40 * d_pred['count']:
                        filtered_longs.append(x)
                top_15m_longs = filtered_longs
                
                filtered_shorts = []
                for x in top_15m_shorts:
                    d_pred = dict_daily[d_daily].get(x['ticker'])
                    if d_pred and d_pred['short_rank'] <= 0.40 * d_pred['count']:
                        filtered_shorts.append(x)
                top_15m_shorts = filtered_shorts
            
            # Max daily trade limit check
            max_daily = {1: 6, 2: 4, 3: 6, 4: 8, 5: 4, 6: 6, 8: 6, 9: 6, 10: 4}
            # S7 limits depend on regime, handled inside S7 check
            
            # Evaluate entries based on strategy ID
            
            # =================================================================
            # STRATEGY 1: "Daily Macro Gatekeeper"
            # =================================================================
            if strategy_id == 1:
                if 10 * 60 <= t_min <= 14 * 60 + 59:
                    if daily_trade_count[date_str] < max_daily[1]:
                        # Loop through sorted 15M long candidates to find top-1 that matches gates
                        long_cand = None
                        for x in top_15m_longs:
                            try:
                                if x['long_rank'] == 1:
                                    ticker = x['ticker']
                                    d_pred = dict_daily[d_daily].get(ticker)
                                    if d_pred and d_pred['long_rank'] <= 0.40 * d_pred['count']:
                                        if x['long_conv'] > x['p60_long']:
                                            long_cand = x
                                            break
                            except Exception as e:
                                print(f"ERROR inside top_15m_longs loop: {type(e).__name__}: {e}")
                                print(f"x type: {type(x)} | x keys: {list(x.keys()) if isinstance(x, dict) else 'not a dict'}")
                                raise
                                        
                        short_cand = None
                        for x in top_15m_shorts:
                            try:
                                if x['short_rank'] == 1:
                                    ticker = x['ticker']
                                    d_pred = dict_daily[d_daily].get(ticker)
                                    if d_pred and d_pred['short_rank'] <= 0.40 * d_pred['count']:
                                        if x['short_conv'] > x['p60_short']:
                                            short_cand = x
                                            break
                            except Exception as e:
                                print(f"ERROR inside top_15m_shorts loop: {type(e).__name__}: {e}")
                                print(f"x type: {type(x)} | x keys: {list(x.keys()) if isinstance(x, dict) else 'not a dict'}")
                                raise
                                        
                        # Place trades
                        if long_cand and daily_trade_count[date_str] < max_daily[1]:
                            active_trades.append({
                                'ticker': long_cand['ticker'], 'side': 'LONG', 'entry_price': long_cand['close'],
                                'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                            })
                            daily_trade_count[date_str] += 1
                        if short_cand and daily_trade_count[date_str] < max_daily[1]:
                            active_trades.append({
                                'ticker': short_cand['ticker'], 'side': 'SHORT', 'entry_price': short_cand['close'],
                                'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                            })
                            daily_trade_count[date_str] += 1

            # =================================================================
            # STRATEGY 2: "Short-Side Specialist"
            # =================================================================
            elif strategy_id == 2:
                # 30M bar close boundary (T ends in 15 or 45)
                if time_str.endswith("15") or time_str.endswith("45"):
                    if 10 * 60 <= t_min <= 14 * 60 + 30:
                        if daily_trade_count[date_str] < max_daily[2]:
                            # Find candidate that meets conditions
                            for x in top_15m_shorts:
                                ticker = x['ticker']
                                d_pred = dict_daily[d_daily].get(ticker)
                                # Daily short_rank <= 5
                                if d_pred and d_pred['short_rank'] <= 5:
                                    # 30M Top-10 short
                                    m30_pred = dict_30m[t_30m].get(ticker)
                                    if m30_pred and m30_pred['short_rank'] <= 10:
                                        # IBS > 0.55
                                        if x['ibs'] > 0.55:
                                            # Enter SHORT
                                            if ticker not in active_tickers:
                                                active_trades.append({
                                                    'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                                    'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                                })
                                                active_tickers.add(ticker)
                                                daily_trade_count[date_str] += 1
                                                if daily_trade_count[date_str] >= max_daily[2]:
                                                    break

            # =================================================================
            # STRATEGY 3: "Timeframe Divergence Fade"
            # =================================================================
            elif strategy_id == 3:
                if daily_trade_count[date_str] < max_daily[3]:
                    # Long candidates: bearish 1H (bottom 30% long_rank) AND 15M long_rank <= 5 AND conviction > P70
                    for x in top_15m_longs:
                        if x['long_rank'] <= 5:
                            ticker = x['ticker']
                            h1_pred = dict_1h[t_1h].get(ticker)
                            if h1_pred and h1_pred['long_rank'] >= 0.70 * h1_pred['count']:
                                if x['long_conv'] > x['p70_long']:
                                    if ticker not in active_tickers:
                                        active_trades.append({
                                            'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                            'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                        })
                                        active_tickers.add(ticker)
                                        daily_trade_count[date_str] += 1
                                        if daily_trade_count[date_str] >= max_daily[3]:
                                            break
                                            
                    # Short candidates: bullish 1H (top 30% long_rank) AND 15M short_rank <= 5 AND conviction > P70
                    for x in top_15m_shorts:
                        if x['short_rank'] <= 5:
                            ticker = x['ticker']
                            h1_pred = dict_1h[t_1h].get(ticker)
                            if h1_pred and h1_pred['long_rank'] <= 0.30 * h1_pred['count']:
                                if x['short_conv'] > x['p70_short']:
                                    if ticker not in active_tickers and daily_trade_count[date_str] < max_daily[3]:
                                        active_trades.append({
                                            'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                            'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                        })
                                        active_tickers.add(ticker)
                                        daily_trade_count[date_str] += 1
                                        if daily_trade_count[date_str] >= max_daily[3]:
                                            break

            # =================================================================
            # STRATEGY 4: "Score Momentum Scalper"
            # =================================================================
            elif strategy_id == 4:
                if 9 * 60 + 45 <= t_min <= 14 * 60 + 45:
                    if daily_trade_count[date_str] < max_daily[4]:
                        # Long entries: conv[t] > conv[t-1] > conv[t-2] and rank <= 5
                        for x in top_15m_longs:
                            if x['long_rank'] <= 5:
                                if x['long_conv'] > x['long_conv_lag1'] > x['long_conv_lag2']:
                                    ticker = x['ticker']
                                    if ticker not in active_tickers:
                                        active_trades.append({
                                            'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                            'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                        })
                                        active_tickers.add(ticker)
                                        daily_trade_count[date_str] += 1
                                        if daily_trade_count[date_str] >= max_daily[4]:
                                            break
                                            
                        # Short entries
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 5:
                                if x['short_conv'] > x['short_conv_lag1'] > x['short_conv_lag2']:
                                    ticker = x['ticker']
                                    if ticker not in active_tickers and daily_trade_count[date_str] < max_daily[4]:
                                        active_trades.append({
                                            'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                            'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                        })
                                        active_tickers.add(ticker)
                                        daily_trade_count[date_str] += 1
                                        if daily_trade_count[date_str] >= max_daily[4]:
                                            break

            # =================================================================
            # STRATEGY 5: "Power Hour Sniper"
            # =================================================================
            elif strategy_id == 5:
                if 13 * 60 + 45 <= t_min <= 15 * 60 + 0:
                    if daily_trade_count[date_str] < max_daily[5]:
                        # Check daily bias based on median daily conviction of all symbols
                        daily_long_convs = [t_info['long_conv'] for t_info in dict_daily[d_daily].values()]
                        if daily_long_convs:
                            daily_bullish = np.median(daily_long_convs) > 0
                            
                            if daily_bullish:
                                # Enter LONG on 15M Top-3 longs with IBS < 0.40
                                for x in top_15m_longs:
                                    if x['long_rank'] <= 3 and x['ibs'] < 0.40:
                                        ticker = x['ticker']
                                        if ticker not in active_tickers:
                                            active_trades.append({
                                                'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                                'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                            })
                                            active_tickers.add(ticker)
                                            daily_trade_count[date_str] += 1
                                            if daily_trade_count[date_str] >= max_daily[5]:
                                                break
                            else:
                                # Enter SHORT on 15M Top-3 shorts with IBS > 0.60
                                for x in top_15m_shorts:
                                    if x['short_rank'] <= 3 and x['ibs'] > 0.60:
                                        ticker = x['ticker']
                                        if ticker not in active_tickers:
                                            active_trades.append({
                                                'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                                'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                            })
                                            active_tickers.add(ticker)
                                            daily_trade_count[date_str] += 1
                                            if daily_trade_count[date_str] >= max_daily[5]:
                                                break

            # =================================================================
            # STRATEGY 6: "Market-Neutral Pairs"
            # =================================================================
            elif strategy_id == 6:
                # Hourly bars close check
                if time_str in ["10:30", "11:30", "12:30", "13:30", "14:30"]:
                    if len(active_pairs) < 2 and daily_trade_count[date_str] < 6: # max 3 pairs (6 trades)
                        # Compute blended scores for all tickers in active_15m
                        blend_long_scores = {}
                        blend_short_scores = {}
                        for x in active_15m:
                            ticker = x['ticker']
                            h1_pred = dict_1h[t_1h].get(ticker)
                            m30_pred = dict_30m[t_30m].get(ticker)
                            if h1_pred and m30_pred:
                                blend_long_scores[ticker] = 0.2 * h1_pred['long_conv'] + 0.3 * m30_pred['long_conv'] + 0.5 * x['long_conv']
                                blend_short_scores[ticker] = 0.2 * h1_pred['short_conv'] + 0.3 * m30_pred['short_conv'] + 0.5 * x['short_conv']
                                
                        if len(blend_long_scores) > 2:
                            # Calculate medians
                            med_long = np.median(list(blend_long_scores.values()))
                            med_short = np.median(list(blend_short_scores.values()))
                            
                            # Sort tickers
                            sorted_blend_longs = sorted(blend_long_scores.keys(), key=lambda k: blend_long_scores[k], reverse=True)
                            sorted_blend_shorts = sorted(blend_short_scores.keys(), key=lambda k: blend_short_scores[k], reverse=True)
                            
                            long_ticker = sorted_blend_longs[0]
                            short_ticker = sorted_blend_shorts[0]
                            
                            # Distinct tickers check
                            if long_ticker != short_ticker:
                                if blend_long_scores[long_ticker] > med_long and blend_short_scores[short_ticker] > med_short:
                                    # Enter pair
                                    bar_long = ticker_bar_map[long_ticker][T]
                                    bar_short = ticker_bar_map[short_ticker][T]
                                    
                                    active_pairs.append({
                                        'long_ticker': long_ticker, 'short_ticker': short_ticker,
                                        'entry_price_long': bar_long['close'], 'entry_price_short': bar_short['close'],
                                        'entry_time': T, 'bars_held': 0
                                    })
                                    daily_trade_count[date_str] += 2 # Incremented trade counts for both legs

            # =================================================================
            # STRATEGY 7: "Volatility Regime Switcher"
            # =================================================================
            elif strategy_id == 7:
                # Get current volatility regime
                curr_vol = dict_rolling_vol.get(T, 0.0)
                
                # Check regime
                if curr_vol < p30_vol:
                    regime = 'LOW_VOL'
                    limit = 10
                elif curr_vol > p70_vol:
                    regime = 'HIGH_VOL'
                    limit = 4
                else:
                    regime = 'NORMAL'
                    limit = 6
                    
                if daily_trade_count[date_str] < limit:
                    # Place trades based on regime rules
                    if regime == 'LOW_VOL':
                        # 15M Top-3, IBS relaxed (0.45/0.55)
                        for x in top_15m_longs:
                            if x['long_rank'] <= 3 and x['ibs'] < 0.45:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({
                                        'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0, 'regime': regime
                                    })
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break
                                    
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 3 and x['ibs'] > 0.55 and daily_trade_count[date_str] < limit:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({
                                        'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0, 'regime': regime
                                    })
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break
                                    
                    elif regime == 'HIGH_VOL':
                        # 15M Top-1 + 30M Top-5 confluence, strict IBS (0.30/0.70)
                        for x in top_15m_longs:
                            if x['long_rank'] == 1 and x['ibs'] < 0.30:
                                ticker = x['ticker']
                                m30_pred = dict_30m[t_30m].get(ticker)
                                if m30_pred and m30_pred['long_rank'] <= 5:
                                    if ticker not in active_tickers:
                                        active_trades.append({
                                            'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                            'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0, 'regime': regime
                                        })
                                        active_tickers.add(ticker)
                                        daily_trade_count[date_str] += 1
                                        if daily_trade_count[date_str] >= limit: break
                                        
                        for x in top_15m_shorts:
                            if x['short_rank'] == 1 and x['ibs'] > 0.70 and daily_trade_count[date_str] < limit:
                                ticker = x['ticker']
                                m30_pred = dict_30m[t_30m].get(ticker)
                                if m30_pred and m30_pred['short_rank'] <= 5:
                                    if ticker not in active_tickers:
                                        active_trades.append({
                                            'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                            'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0, 'regime': regime
                                        })
                                        active_tickers.add(ticker)
                                        daily_trade_count[date_str] += 1
                                        if daily_trade_count[date_str] >= limit: break
                                        
                    else: # NORMAL
                        # 15M Top-2, IBS (0.38/0.62)
                        for x in top_15m_longs:
                            if x['long_rank'] <= 2 and x['ibs'] < 0.38:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({
                                        'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0, 'regime': regime
                                    })
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break
                                    
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 2 and x['ibs'] > 0.62 and daily_trade_count[date_str] < limit:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({
                                        'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0, 'regime': regime
                                    })
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= limit: break

            # =================================================================
            # STRATEGY 8: "Opening Range Breakout + Confirmation"
            # =================================================================
            elif strategy_id == 8:
                # Evaluation only starts from 10:00 onwards
                if t_min >= 10 * 60:
                    if daily_trade_count[date_str] < max_daily[8]:
                        # Loop candidates
                        for x in top_15m_longs:
                            if x['long_rank'] <= 5:
                                ticker = x['ticker']
                                # Ticker not already traded today
                                if ticker not in daily_tickers_traded[date_str]:
                                    # 1H long_rank <= 10
                                    h1_pred = dict_1h[t_1h].get(ticker)
                                    if h1_pred and h1_pred['long_rank'] <= 10:
                                        # OR Breakout check: Close > OR_High * 1.001
                                        ranges = dict_or.get(date_str, {}).get(ticker)
                                        if ranges:
                                            or_high, or_low = ranges
                                            if x['close'] > or_high * 1.001:
                                                active_trades.append({
                                                    'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                                    'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                                })
                                                daily_trade_count[date_str] += 1
                                                daily_tickers_traded[date_str].add(ticker)
                                                if daily_trade_count[date_str] >= max_daily[8]: break
                                                
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 5 and daily_trade_count[date_str] < max_daily[8]:
                                ticker = x['ticker']
                                if ticker not in daily_tickers_traded[date_str]:
                                    h1_pred = dict_1h[t_1h].get(ticker)
                                    if h1_pred and h1_pred['short_rank'] <= 10:
                                        # OR Breakout check: Close < OR_Low * 0.999
                                        ranges = dict_or.get(date_str, {}).get(ticker)
                                        if ranges:
                                            or_high, or_low = ranges
                                            if x['close'] < or_low * 0.999:
                                                active_trades.append({
                                                    'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                                    'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                                })
                                                daily_trade_count[date_str] += 1
                                                daily_tickers_traded[date_str].add(ticker)
                                                if daily_trade_count[date_str] >= max_daily[8]: break

            # =================================================================
            # STRATEGY 9: "Conviction Spread Z-Score"
            # =================================================================
            elif strategy_id == 9:
                if 9 * 60 + 45 <= t_min <= 14 * 60 + 45:
                    if daily_trade_count[date_str] < max_daily[9]:
                        # Long: Z-score > 2.0 AND 15M long_rank <= 3
                        for x in top_15m_longs:
                            if x['long_rank'] <= 3 and x['spread_zscore'] > 2.0:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({
                                        'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                    })
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= max_daily[9]: break
                                    
                        # Short: Z-score < -2.0 AND 15M short_rank <= 3
                        for x in top_15m_shorts:
                            if x['short_rank'] <= 3 and x['spread_zscore'] < -2.0 and daily_trade_count[date_str] < max_daily[9]:
                                ticker = x['ticker']
                                if ticker not in active_tickers:
                                    active_trades.append({
                                        'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                    })
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= max_daily[9]: break

            # =================================================================
            # STRATEGY 10: "Quad-Timeframe Unanimous"
            # =================================================================
            elif strategy_id == 10:
                if daily_trade_count[date_str] < max_daily[10]:
                    # Long candidates
                    for x in top_15m_longs:
                        if x['long_rank'] <= 3:
                            ticker = x['ticker']
                            # Daily Top 30% long
                            d_pred = dict_daily[d_daily].get(ticker)
                            if d_pred and d_pred['long_rank'] <= 0.30 * d_pred['count']:
                                # 1H Top-5 long
                                h1_pred = dict_1h[t_1h].get(ticker)
                                if h1_pred and h1_pred['long_rank'] <= 5:
                                    # 30M Top-5 long
                                    m30_pred = dict_30m[t_30m].get(ticker)
                                    if m30_pred and m30_pred['long_rank'] <= 5:
                                        if ticker not in active_tickers:
                                            active_trades.append({
                                                'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                                'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                            })
                                            active_tickers.add(ticker)
                                            daily_trade_count[date_str] += 1
                                            if daily_trade_count[date_str] >= max_daily[10]: break
                                            
                    # Short candidates
                    for x in top_15m_shorts:
                        if x['short_rank'] <= 3 and daily_trade_count[date_str] < max_daily[10]:
                            ticker = x['ticker']
                            # Daily Top 30% short
                            d_pred = dict_daily[d_daily].get(ticker)
                            if d_pred and d_pred['short_rank'] <= 0.30 * d_pred['count']:
                                # 1H Top-5 short
                                h1_pred = dict_1h[t_1h].get(ticker)
                                if h1_pred and h1_pred['short_rank'] <= 5:
                                    # 30M Top-5 short
                                    m30_pred = dict_30m[t_30m].get(ticker)
                                    if m30_pred and m30_pred['short_rank'] <= 5:
                                        if ticker not in active_tickers:
                                            active_trades.append({
                                                'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                                'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                            })
                                            active_tickers.add(ticker)
                                            daily_trade_count[date_str] += 1
                                            if daily_trade_count[date_str] >= max_daily[10]: break
        
            # =================================================================
            # STRATEGY 11: "Pure 1-Hour AI Model"
            # =================================================================
            elif strategy_id == 11:
                # Only evaluate entries on the hourly close
                if time_str in ["10:30", "11:30", "12:30", "13:30", "14:30"]:
                    if daily_trade_count[date_str] < 4: # Max 4 trades
                        for x in top_15m_longs:
                            ticker = x['ticker']
                            h1_pred = dict_1h[t_1h].get(ticker)
                            # Top 1 1H Long Rank
                            if h1_pred and h1_pred['long_rank'] == 1:
                                if ticker not in active_tickers:
                                    active_trades.append({
                                        'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                    })
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= 4: break
                                    
                        for x in top_15m_shorts:
                            ticker = x['ticker']
                            h1_pred = dict_1h[t_1h].get(ticker)
                            # Top 1 1H Short Rank
                            if h1_pred and h1_pred['short_rank'] == 1 and daily_trade_count[date_str] < 4:
                                if ticker not in active_tickers:
                                    active_trades.append({
                                        'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                                    })
                                    active_tickers.add(ticker)
                                    daily_trade_count[date_str] += 1
                                    if daily_trade_count[date_str] >= 4: break
        
        # End of simulation for this strategy, evaluate trades
        eval_res = evaluate_strategy_trades(trades, name)
        results[f"strategy_{strategy_id}"] = eval_res
        print(f"  Trades generated: {len(trades)} | Net P&L: {eval_res['total_return']*100:+.2f}%")
        return eval_res

    # Run simulations sequentially
    strategies_info = [
        (1, "Daily Macro Gatekeeper"),
        (2, "Short-Side Specialist"),
        (3, "Timeframe Divergence Fade"),
        (4, "Score Momentum Scalper"),
        (5, "Power Hour Sniper"),
        (6, "Market-Neutral Pairs"),
        (7, "Volatility Regime Switcher"),
        (8, "Opening Range Breakout + Confirmation"),
        (9, "Conviction Spread Z-Score"),
        (10, "Quad-Timeframe Unanimous"),
        (11, "Pure 1-Hour AI Model")
    ]
    
    for s_id, s_name in strategies_info:
        res_ungated = simulate_strategy(s_id, s_name, apply_gatekeeper=False)
        res_gated = simulate_strategy(s_id, s_name, apply_gatekeeper=True)
        results[f"strategy_{s_id}_ungated"] = res_ungated
        results[f"strategy_{s_id}_gated"] = res_gated
        
    # Save the output results json
    output_path = "data/strategy_regime_results.json"
    with open(output_path, "w") as f:
        json.dump({
            'holdout_month': TEST_MONTH,
            'tickers_universe_size': len(common_tickers),
            'strategies': results,
            'backtested_at': datetime.now().isoformat()
        }, f, indent=2)
        
    print(f"\n[SUCCESS] Saved backtest outcomes to {output_path}")
    
    # 5. PRINT SUMMARY ASCII MARKDOWN TABLE
    print("\n" + "=" * 125)
    print("REGIME-AWARE COMPARATIVE BACKTEST SUMMARY (GATED VS UNGATED)")
    print("=" * 125)
    headers = ["ID", "Strategy Name", "Gated WR", "Ungated WR", "Gated Net %", "Ungated Net %", "Verdict"]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-|-'.join(['-' * len(h) for h in headers])}|")
    for s_id, s_name in strategies_info:
        r_ungated = results[f"strategy_{s_id}_ungated"]
        r_gated = results[f"strategy_{s_id}_gated"]
        s_name_trunc = s_name[:38]
        
        g_wr_str = f"{r_gated['win_rate']*100:.1f}% ({r_gated['total_trades']}t)"
        u_wr_str = f"{r_ungated['win_rate']*100:.1f}% ({r_ungated['total_trades']}t)"
        
        g_pnl = r_gated['total_return']*100
        u_pnl = r_ungated['total_return']*100
        g_pnl_str = f"{g_pnl:+.2f}%"
        u_pnl_str = f"{u_pnl:+.2f}%"
        
        # Verdict calculation
        if s_id in [1, 2, 10]:
            verdict = "TREND (Built-in)" # Hardcoded gatekeepers
        elif g_pnl > u_pnl:
            verdict = "TREND"
        elif u_pnl > g_pnl:
            verdict = "REVERSAL"
        else:
            verdict = "NEUTRAL"
            
        print(f"| {s_id:<2} | {s_name_trunc:<38} | {g_wr_str:<15} | {u_wr_str:<15} | {g_pnl_str:<11} | {u_pnl_str:<13} | {verdict:<13} |")
    print("=" * 125)

if __name__ == "__main__":
    main()
