"""
backtest_multi_tf.py — Historical Backtest Simulation of 1H + 30M + 15M Confluence Strategy
For the last month of data (May 2026).

Strategy logic:
1. Hourly Signal (1H model): Picks Top-3 Long & Short symbols at the start of each hour.
2. Trend Confirmation (30M model): Checks if 1H picks are in 30M Top-5 same direction.
3. Entry Timing (15M model): Enters if pick is in 15M Top-5 same direction + IBS pullback confluence:
   - Long: IBS < 0.40
   - Short: IBS > 0.60
4. Exits compared:
   - Strategy 1: Single-Bar (15-min) hold
   - Strategy 2: Dynamic exit (exits on 15M conviction flip, rank drop below top-10, or max 1-hour hold)
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
TRANSACTION_COST_PCT = 0.03  # 0.03% round-trip

# Top-K thresholds
CONV_30M_TOPK = 5
CONV_15M_TOPK = 5

print("=" * 70)
print("MULTI-TIMEFRAME CONFLUENCE BACKTESTER")
print(f"Simulating 1H + 30M + 15M Confluence on Holdout Month: {TEST_MONTH}")
print("=" * 70)

# ========================================
# Helper: Align Timeframes
# ========================================
def align_timeframes(t_str):
    """
    Given a 15-minute bar timestamp (e.g. '2026-05-18 10:30:00+05:30'),
    return the latest closed 1H bar timestamp and latest closed 30M bar timestamp.
    """
    date_part = t_str[:10]
    time_part = t_str[11:16]
    h, m = map(int, time_part.split(':'))
    minutes = h * 60 + m
    
    # 1H Mapping (hourly bars close at H:30)
    if minutes < 10 * 60 + 30:  # before 10:30
        t_1h = None
    elif minutes < 11 * 60 + 30:
        t_1h = f"{date_part} 09:30:00+05:30"
    elif minutes < 12 * 60 + 30:
        t_1h = f"{date_part} 10:30:00+05:30"
    elif minutes < 13 * 60 + 30:
        t_1h = f"{date_part} 11:30:00+05:30"
    elif minutes < 14 * 60 + 30:
        t_1h = f"{date_part} 12:30:00+05:30"
    elif minutes < 15 * 60 + 30:
        t_1h = f"{date_part} 13:30:00+05:30"
    else:
        t_1h = f"{date_part} 14:30:00+05:30"
        
    # 30M Mapping (30m bars close at H:15 and H:45)
    if minutes < 9 * 60 + 45:  # before 09:45
        t_30m = None
    elif minutes < 10 * 60 + 15:
        t_30m = f"{date_part} 09:15:00+05:30"
    elif minutes < 10 * 60 + 45:
        t_30m = f"{date_part} 09:45:00+05:30"
    elif minutes < 11 * 60 + 15:
        t_30m = f"{date_part} 10:15:00+05:30"
    elif minutes < 11 * 60 + 45:
        t_30m = f"{date_part} 10:45:00+05:30"
    elif minutes < 12 * 60 + 15:
        t_30m = f"{date_part} 11:15:00+05:30"
    elif minutes < 12 * 60 + 45:
        t_30m = f"{date_part} 11:45:00+05:30"
    elif minutes < 13 * 60 + 15:
        t_30m = f"{date_part} 12:15:00+05:30"
    elif minutes < 13 * 60 + 45:
        t_30m = f"{date_part} 12:45:00+05:30"
    elif minutes < 14 * 60 + 15:
        t_30m = f"{date_part} 13:15:00+05:30"
    elif minutes < 14 * 60 + 45:
        t_30m = f"{date_part} 13:45:00+05:30"
    elif minutes < 15 * 60 + 15:
        t_30m = f"{date_part} 14:15:00+05:30"
    elif minutes < 15 * 60 + 45:
        t_30m = f"{date_part} 14:45:00+05:30"
    else:
        t_30m = f"{date_part} 15:15:00+05:30"
        
    return t_1h, t_30m

# ========================================
# Helper: Predict Scores for a Timeframe Model
# ========================================
def predict_timeframe_scores(df_tf, model_key, meta_path, long_model_path, short_model_path, scaler_path):
    print(f"\nScoring {model_key} dataset...")
    
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
    bst_short = xgb.Booster()
    bst_short.load_model(short_model_path)
    
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
    df_tf['long_rank'] = df_tf.groupby('Query_ID')['long_conv'].rank(ascending=False)
    df_tf['short_rank'] = df_tf.groupby('Query_ID')['short_conv'].rank(ascending=False)
    
    print(f"  Scoring complete: {df_tf.shape[0]:,} rows.")
    return df_tf

# ========================================
# 1. LOAD DATASETS
# ========================================
print("\nLoading datasets...")
df_1h_all = pd.read_csv("data/ranking_data_upstox_3y.csv")
df_30m_all = pd.read_csv("data/ranking_data_upstox_30min_1y.csv")
df_15m_all = pd.read_csv("data/ranking_data_upstox_15min_1y.csv")

# Filter to the holdout month
df_1h = df_1h_all[df_1h_all['DateTime'].str.startswith(TEST_MONTH)].copy()
df_30m = df_30m_all[df_30m_all['DateTime'].str.startswith(TEST_MONTH)].copy()
df_15m = df_15m_all[df_15m_all['DateTime'].str.startswith(TEST_MONTH)].copy()

print(f"  1H Data:  {df_1h.shape[0]:,} rows")
print(f"  30M Data: {df_30m.shape[0]:,} rows")
print(f"  15M Data: {df_15m.shape[0]:,} rows")

# Intersection of tickers
t_1h = set(df_1h['Ticker'].unique())
t_30m = set(df_30m['Ticker'].unique())
t_15m = set(df_15m['Ticker'].unique())
common_tickers = sorted(list(t_1h.intersection(t_30m).intersection(t_15m)))
print(f"  Tickers universe intersection: {len(common_tickers)} symbols")

df_1h = df_1h[df_1h['Ticker'].isin(common_tickers)].copy()
df_30m = df_30m[df_30m['Ticker'].isin(common_tickers)].copy()
df_15m = df_15m[df_15m['Ticker'].isin(common_tickers)].copy()

# ========================================
# 2. RUN PREDICTIONS
# ========================================
# Hourly Model (v8_upstox_3y)
df_1h = predict_timeframe_scores(
    df_1h, "1H",
    "models/v8_upstox_3y/metadata.json",
    "models/v8_upstox_3y/xgb_long_model.json",
    "models/v8_upstox_3y/xgb_short_model.json",
    "models/scaler.pkl"
)

# 30-Min Model (v1_30min)
df_30m = predict_timeframe_scores(
    df_30m, "30M",
    "models/v1_30min/metadata.json",
    "models/v1_30min/xgb_long_model.json",
    "models/v1_30min/xgb_short_model.json",
    "models/v1_30min/scaler.pkl"
)

# 15-Min Model (v1_15min)
df_15m = predict_timeframe_scores(
    df_15m, "15M",
    "models/v1_15min/metadata.json",
    "models/v1_15min/xgb_long_model.json",
    "models/v1_15min/xgb_short_model.json",
    "models/v1_15min/scaler.pkl"
)

# ========================================
# 3. BUILD LOOKUP DICTS (DateTime -> Ticker -> Data)
# ========================================
print("\nIndexing predictions for O(1) simulation lookups...")

dict_1h = {}
for qid, q_df in df_1h.groupby('DateTime'):
    dict_1h[qid] = {}
    for _, row in q_df.iterrows():
        dict_1h[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv'])
        }

dict_30m = {}
for qid, q_df in df_30m.groupby('DateTime'):
    dict_30m[qid] = {}
    for _, row in q_df.iterrows():
        dict_30m[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv'])
        }

# For 15M, we keep raw prices and returns as well
dict_15m = {}
for qid, q_df in df_15m.groupby('DateTime'):
    dict_15m[qid] = {}
    for _, row in q_df.iterrows():
        # raw IBS computation
        h, l, c, o = row['High'], row['Low'], row['Close'], row['Open']
        ibs_raw = (c - l) / (h - l + 1e-10)
        
        dict_15m[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'ibs_raw': float(ibs_raw),
            'next_return': float(row['Next_15Min_Return'])
        }

# ========================================
# 4. RUN BACKTEST SIMULATION
# ========================================
print("\nRunning backtest simulation...")

trades_s1 = []  # Strategy 1: Single Bar (15m) Hold
trades_s2 = []  # Strategy 2: Dynamic Exit

# Group 15-minute dates by calendar day
df_15m['Date'] = df_15m['DateTime'].str[:10]
unique_dates = sorted(df_15m['Date'].unique())

for date in tqdm(unique_dates, desc="Simulating Days"):
    df_day = df_15m[df_15m['Date'] == date]
    timestamps_15m = sorted(df_day['DateTime'].unique())
    
    # Track daily cooldowns per symbol to prevent double trading same stock on the same bar
    active_trades = {}  # for Strategy 2: symbol -> trade_info
    
    for T in timestamps_15m:
        t_1h, t_30m = align_timeframes(T)
        
        # 1. Fetch 1H picks (requires closed 1H bar at t_1h)
        if not t_1h or t_1h not in dict_1h:
            continue
        
        picks_1h = dict_1h[t_1h]
        # Sort tickers by Long and Short conviction to select Top-3
        sorted_longs = sorted([t for t in picks_1h], key=lambda x: picks_1h[x]['long_conv'], reverse=True)[:3]
        sorted_shorts = sorted([t for t in picks_1h], key=lambda x: picks_1h[x]['short_conv'], reverse=True)[:3]
        
        # 2. Fetch 30M predictions
        if not t_30m or t_30m not in dict_30m:
            continue
        picks_30m = dict_30m[t_30m]
        
        # 3. Fetch 15M predictions
        if T not in dict_15m:
            continue
        picks_15m = dict_15m[T]
        
        # ─── UPDATE ACTIVE TRADES (Strategy 2 Exit Check) ───
        to_remove = []
        for symbol, tr in active_trades.items():
            # Get latest 15m price bar
            if symbol not in picks_15m:
                # Missing data bar, force exit on previous Close
                exit_ret = tr['cumulative_gross']
                trades_s2.append({**tr, 'exit_time': T, 'exit_reason': 'DATA_GAP', 'gross_return': exit_ret, 'net_return': exit_ret - TRANSACTION_COST_PCT/100})
                to_remove.append(symbol)
                continue
                
            curr_bar = picks_15m[symbol]
            tr['bars_held'] += 1
            
            # Gross return of this specific bar
            bar_ret = curr_bar['next_return'] if tr['side'] == 'LONG' else -curr_bar['next_return']
            tr['cumulative_gross'] += bar_ret
            
            # Check exit conditions
            conv_flip = False
            rank_drop = False
            time_expiry = tr['bars_held'] >= 4  # 60 mins max hold
            market_close = T.endswith("15:15:00+05:30")
            
            # Conviction flip and rank drop checks
            if tr['side'] == 'LONG':
                if curr_bar['short_conv'] > curr_bar['long_conv']:
                    conv_flip = True
                if curr_bar['long_rank'] > 10:
                    rank_drop = True
            else:
                if curr_bar['long_conv'] > curr_bar['short_conv']:
                    conv_flip = True
                if curr_bar['short_rank'] > 10:
                    rank_drop = True
            
            if conv_flip or rank_drop or time_expiry or market_close:
                reason = "CONV_FLIP" if conv_flip else ("RANK_DROP" if rank_drop else ("TIME_EXPIRY" if time_expiry else "MARKET_CLOSE"))
                final_price = curr_bar['close']
                gross_ret = (final_price / tr['entry_price'] - 1) if tr['side'] == 'LONG' else (1 - final_price / tr['entry_price'])
                net_ret = gross_ret - TRANSACTION_COST_PCT / 100
                
                trades_s2.append({
                    **tr,
                    'exit_time': T,
                    'exit_reason': reason,
                    'gross_return': gross_ret,
                    'net_return': net_ret,
                    'is_win': net_ret > 0
                })
                to_remove.append(symbol)
                
        for symbol in to_remove:
            del active_trades[symbol]
            
        # ─── EVALUATE NEW ENTRIES (Strategy 1 & Strategy 2) ───
        
        # A. LONGS confluence
        for symbol in sorted_longs:
            # 30M Trend Confirmation
            if symbol not in picks_30m or picks_30m[symbol]['long_rank'] > CONV_30M_TOPK:
                continue
            # 15M Entry Confluence
            if symbol not in picks_15m or picks_15m[symbol]['long_rank'] > CONV_15M_TOPK:
                continue
            # IBS Pullback check
            if picks_15m[symbol]['ibs_raw'] > 0.40:
                continue
                
            # Entry Signal Generated!
            bar_data = picks_15m[symbol]
            
            # Strategy 1 Entry (Immediate exit at the end of this 15-min bar)
            g_ret = bar_data['next_return']
            n_ret = g_ret - TRANSACTION_COST_PCT / 100
            trades_s1.append({
                'date': date, 'entry_time': T, 'ticker': symbol, 'side': 'LONG',
                'entry_price': bar_data['open'], 'exit_price': bar_data['close'],
                'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
            })
            
            # Strategy 2 Entry
            if symbol not in active_trades:
                active_trades[symbol] = {
                    'date': date, 'entry_time': T, 'ticker': symbol, 'side': 'LONG',
                    'entry_price': bar_data['open'], 'bars_held': 0, 'cumulative_gross': 0.0
                }
                
        # B. SHORTS confluence
        for symbol in sorted_shorts:
            # 30M Trend Confirmation
            if symbol not in picks_30m or picks_30m[symbol]['short_rank'] > CONV_30M_TOPK:
                continue
            # 15M Entry Confluence
            if symbol not in picks_15m or picks_15m[symbol]['short_rank'] > CONV_15M_TOPK:
                continue
            # IBS Pullback check (Rally z-score/overbought)
            if picks_15m[symbol]['ibs_raw'] < 0.60:
                continue
                
            # Entry Signal Generated!
            bar_data = picks_15m[symbol]
            
            # Strategy 1 Entry
            g_ret = -bar_data['next_return']  # short captures negative returns
            n_ret = g_ret - TRANSACTION_COST_PCT / 100
            trades_s1.append({
                'date': date, 'entry_time': T, 'ticker': symbol, 'side': 'SHORT',
                'entry_price': bar_data['open'], 'exit_price': bar_data['close'],
                'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
            })
            
            # Strategy 2 Entry
            if symbol not in active_trades:
                active_trades[symbol] = {
                    'date': date, 'entry_time': T, 'ticker': symbol, 'side': 'SHORT',
                    'entry_price': bar_data['open'], 'bars_held': 0, 'cumulative_gross': 0.0
                }
                
    # Force close any remaining trades at the end of the day
    for symbol, tr in active_trades.items():
        if T in picks_15m:
            curr_bar = picks_15m[symbol]
            final_price = curr_bar['close']
            gross_ret = (final_price / tr['entry_price'] - 1) if tr['side'] == 'LONG' else (1 - final_price / tr['entry_price'])
        else:
            gross_ret = tr['cumulative_gross']
        net_ret = gross_ret - TRANSACTION_COST_PCT / 100
        trades_s2.append({
            **tr, 'exit_time': T, 'exit_reason': 'FORCE_CLOSE_EOD',
            'gross_return': gross_ret, 'net_return': net_ret, 'is_win': net_ret > 0
        })

# ========================================
# 5. METRICS CALCULATION AND REPORTING
# ========================================
def evaluate_strategy_trades(trades, name):
    if not trades:
        print(f"\n  Strategy {name}: NO TRADES GENERATED")
        return None
        
    df_t = pd.DataFrame(trades)
    
    # Calculate daily statistics
    daily_stats = []
    for date, group in df_t.groupby('date'):
        wins = group['is_win'].sum()
        total = len(group)
        pnl = group['net_return'].sum()
        daily_stats.append({
            'date': date, 'trades': total, 'wins': wins, 'losses': total - wins,
            'win_rate': wins / total, 'pnl': pnl, 'is_green': pnl > 0
        })
    df_d = pd.DataFrame(daily_stats)
    
    total_trades = len(df_t)
    total_wins = df_t['is_win'].sum()
    total_losses = total_trades - total_wins
    overall_wr = total_wins / total_trades
    
    avg_trades_day = df_d['trades'].mean()
    avg_wins_day = df_d['wins'].mean()
    avg_losses_day = df_d['losses'].mean()
    
    total_net_pnl = df_t['net_return'].sum()
    avg_net_trade = df_t['net_return'].mean()
    
    green_days = df_d['is_green'].sum()
    total_days = len(df_d)
    green_day_rate = green_days / total_days
    
    winners = df_t[df_t['is_win']]
    losers = df_t[~df_t['is_win']]
    avg_win_size = winners['net_return'].mean() if len(winners) > 0 else 0
    avg_loss_size = losers['net_return'].mean() if len(losers) > 0 else 0
    
    profit_factor = abs(winners['net_return'].sum() / losers['net_return'].sum()) if len(losers) > 0 and losers['net_return'].sum() != 0 else float('inf')
    
    # Maximum Drawdown (cumulative sum of trades)
    cum_returns = df_t['net_return'].cumsum()
    max_dd = (cum_returns - cum_returns.cummax()).min()
    
    print(f"\n{'=' * 75}")
    print(f"  STRATEGY RESULTS: {name}")
    print(f"{'=' * 75}")
    print(f"  Total Trades: {total_trades} over {total_days} active days")
    print(f"  Win / Loss  : {total_wins} Wins | {total_losses} Losses")
    print(f"  Overall WR  : {overall_wr:.1%}")
    print(f"")
    print(f"  --- Daily Performance ---")
    print(f"  Avg Trades/Day  : {avg_trades_day:.1f}")
    print(f"  Avg Wins/Day    : {avg_wins_day:.1f}")
    print(f"  Avg Losses/Day  : {avg_losses_day:.1f}")
    print(f"  Green Day Rate  : {green_days}/{total_days} ({green_day_rate:.1%})")
    print(f"")
    print(f"  --- P&L and Magnitudes (Net of {TRANSACTION_COST_PCT:.2f}% Costs) ---")
    print(f"  Total Net Return     : {total_net_pnl * 100:+.2f}%")
    print(f"  Avg Net Return/Trade : {avg_net_trade * 100:+.4f}%")
    print(f"  Avg Win Size         : {avg_win_size * 100:+.4f}%")
    print(f"  Avg Loss Size        : {avg_loss_size * 100:+.4f}%")
    print(f"  Profit Factor        : {profit_factor:.2f}")
    print(f"  Max Drawdown         : {max_dd * 100:.2f}%")
    
    if name.startswith("Strategy 2"):
        # Exit reason counts
        reasons = df_t['exit_reason'].value_counts()
        print(f"\n  --- Exit Reasons ---")
        for reason, count in reasons.items():
            print(f"    {reason:<20} : {count} ({count/total_trades:.1%})")
            
    print(f"{'=' * 75}\n")
    
    return {
        'strategy': name,
        'total_trades': int(total_trades),
        'wins': int(total_wins),
        'losses': int(total_losses),
        'win_rate': float(overall_wr),
        'total_return': float(total_net_pnl),
        'avg_return': float(avg_net_trade),
        'avg_trades_per_day': float(avg_trades_day),
        'green_day_rate': float(green_day_rate),
        'profit_factor': float(profit_factor),
        'max_drawdown': float(max_dd)
    }

eval_s1 = evaluate_strategy_trades(trades_s1, "Strategy 1: Confluence + 15m (Single-Bar) Hold")
eval_s2 = evaluate_strategy_trades(trades_s2, "Strategy 2: Confluence + Dynamic Conviction Exit")

# Save results
out_data = {
    'holdout_month': TEST_MONTH,
    'tickers_universe_size': len(common_tickers),
    'strategies': {
        'strategy_1_single_bar': eval_s1,
        'strategy_2_dynamic': eval_s2
    },
    'backtested_at': datetime.now().isoformat()
}

out_path = "data/strategy_backtest_multi_tf_results.json"
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2)
print(f"[SUCCESS] Multi-Timeframe Confluence backtest complete. Saved results to {out_path}")
