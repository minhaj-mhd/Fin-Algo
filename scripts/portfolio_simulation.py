"""
portfolio_simulation.py — Realistic Portfolio Simulation with Strict Capital Allocation

Simulates a real trading portfolio:
- Initial Capital: Rs 100,000
- Margin Multiplier: 5x (standard intraday margin on NSE cash, meaning Rs 100,000 capital allows Rs 500,000 total exposure)
- Max Trade Slots: 5 concurrent active trades
- Capital Allocation: Rs 100,000 exposure per slot (Rs 20,000 actual capital margin required per slot)
- Strict EOD Close at 15:15 IST
- Strict 0.06% round-trip transaction costs (brokerage, taxes, slippage)
- Inputs: Blended Ensemble (1H: 0.1, 30M: 0.3, 15M: 0.6) top-5 hourly signals
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.getcwd())

TEST_MONTH = "2026-05"
INITIAL_CAPITAL = 100000.0
MAX_SLOTS = 5
MARGIN_MULTIPLIER = 5.0
TRANSACTION_COST_PCT = 0.06 # Strict 0.06% round-trip

print("=" * 80)
print("REALISTIC PORTFOLIO SIMULATION")
print(f"Capital: Rs {INITIAL_CAPITAL:,.2f} | Max Slots: {MAX_SLOTS} | Margin: {MARGIN_MULTIPLIER}x | Cost: {TRANSACTION_COST_PCT}%")
print("=" * 80)

def predict_timeframe_scores(df_tf, model_key, meta_path, long_model_path, short_model_path, scaler_path):
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    missing_cols = [c for c in feature_cols if c not in df_tf.columns]
    for col in missing_cols:
        df_tf[col] = 0.0
    bst_long = xgb.Booster(); bst_long.load_model(long_model_path)
    bst_short = xgb.Booster(); bst_short.load_model(short_model_path)
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as sf:
            scaler = pickle.load(sf)
    X = np.nan_to_num(df_tf[feature_cols].values)
    if scaler and hasattr(scaler, 'scale_') and scaler.scale_ is not None:
        X = scaler.transform(X)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    df_tf['long_score'] = bst_long.predict(dmat)
    df_tf['short_score'] = bst_short.predict(dmat)
    df_tf['long_conv'] = df_tf['long_score'] - df_tf['short_score']
    df_tf['short_conv'] = df_tf['short_score'] - df_tf['long_score']
    df_tf['long_rank'] = df_tf.groupby('Query_ID')['long_conv'].rank(ascending=False)
    df_tf['short_rank'] = df_tf.groupby('Query_ID')['short_conv'].rank(ascending=False)
    return df_tf

print("Loading datasets...")
df_1h_all = pd.read_csv("data/ranking_data_upstox_3y.csv")
df_30m_all = pd.read_csv("data/ranking_data_upstox_30min_1y.csv")
df_15m_all = pd.read_csv("data/ranking_data_upstox_15min_1y.csv")
df_1h = df_1h_all[df_1h_all['DateTime'].str.startswith(TEST_MONTH)].copy()
df_30m = df_30m_all[df_30m_all['DateTime'].str.startswith(TEST_MONTH)].copy()
df_15m = df_15m_all[df_15m_all['DateTime'].str.startswith(TEST_MONTH)].copy()

t_common = sorted(set(df_1h['Ticker'].unique()) & set(df_30m['Ticker'].unique()) & set(df_15m['Ticker'].unique()))
df_1h = df_1h[df_1h['Ticker'].isin(t_common)].copy()
df_30m = df_30m[df_30m['Ticker'].isin(t_common)].copy()
df_15m = df_15m[df_15m['Ticker'].isin(t_common)].copy()

df_1h['Date'] = df_1h['DateTime'].str[:10]
df_30m['Date'] = df_30m['DateTime'].str[:10]
df_15m['Date'] = df_15m['DateTime'].str[:10]
print("Scoring models...")
df_1h = predict_timeframe_scores(df_1h, "1H", "models/v8_upstox_3y/metadata.json",
    "models/v8_upstox_3y/xgb_long_model.json", "models/v8_upstox_3y/xgb_short_model.json", "models/scaler.pkl")
df_30m = predict_timeframe_scores(df_30m, "30M", "models/v1_30min/metadata.json",
    "models/v1_30min/xgb_long_model.json", "models/v1_30min/xgb_short_model.json", "models/v1_30min/scaler.pkl")
df_15m = predict_timeframe_scores(df_15m, "15M", "models/v1_15min/metadata.json",
    "models/v1_15min/xgb_long_model.json", "models/v1_15min/xgb_short_model.json", "models/v1_15min/scaler.pkl")

# Build dictionaries
dict_1h = {}
for qid, q_df in df_1h.groupby('DateTime'):
    dict_1h[qid] = {}
    for _, row in q_df.iterrows():
        dict_1h[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
        }

dict_30m = {}
for qid, q_df in df_30m.groupby('DateTime'):
    dict_30m[qid] = {}
    for _, row in q_df.iterrows():
        dict_30m[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
        }

dict_15m = {}
for qid, q_df in df_15m.groupby('DateTime'):
    dict_15m[qid] = {}
    for _, row in q_df.iterrows():
        dict_15m[qid][row['Ticker']] = {
            'long_rank': int(row['long_rank']),
            'short_rank': int(row['short_rank']),
            'long_conv': float(row['long_conv']),
            'short_conv': float(row['short_conv']),
        }

price_series = {}
df_15m_sorted = df_15m.sort_values('DateTime')
for _, row in df_15m_sorted.iterrows():
    ticker = row['Ticker']
    if ticker not in price_series:
        price_series[ticker] = []
    price_series[ticker].append((
        row['DateTime'],
        float(row['Close'])
    ))

price_idx = {}
for ticker, series in price_series.items():
    price_idx[ticker] = {dt: i for i, (dt, c) in enumerate(series)}

def align_timeframes_at_hour(t_1h_str):
    date_part = t_1h_str[:10]
    h_str, m_str = t_1h_str[11:13], t_1h_str[14:16]
    h, m = int(h_str), int(m_str)
    
    t_30m_start = h * 60 + m + 15
    h_30, m_30 = divmod(t_30m_start, 60)
    t_30m = f"{date_part} {h_30:02d}:{m_30:02d}:00+05:30"
    
    t_15m_start = h * 60 + m + 45
    h_15, m_15 = divmod(t_15m_start, 60)
    t_15m = f"{date_part} {h_15:02d}:{m_15:02d}:00+05:30"
    
    return t_30m, t_15m

# Strategy Parameters: Blended High Return (w1=0.1, w2=0.3, w3=0.6), Top-5 picks, 6H max hold (EOD close)
w_1h, w_30m, w_15m = 0.1, 0.3, 0.6
TOPK = 5
HOLD_BARS = 24 # 6 hours

df_1h['Date'] = df_1h['DateTime'].str[:10]
unique_dates = sorted(df_1h['Date'].unique())

# Portfolio State
current_capital = INITIAL_CAPITAL
active_trades = [] # List of active trade dicts: {ticker, side, entry_price, entry_time, exit_idx, capital_allocated}
trade_ledger = [] # Complete list of finished trades

for date in unique_dates:
    df_day_15m = df_15m[df_15m['Date'] == date]
    day_timestamps_15m = sorted(df_day_15m['DateTime'].unique())
    day_timestamps_1h = sorted(df_1h[df_1h['Date'] == date]['DateTime'].unique())
    
    # Intraday tracking
    active_trades = []
    
    # Calculate per-slot allocation based on daily starting capital
    # We allocate equal slots. E.g. with 5 slots, each slot gets:
    # Allocation = (Capital * Margin Multiplier) / Max Slots
    # E.g. (100,000 * 5) / 5 = 100,000 exposure per slot!
    exposure_per_slot = (current_capital * MARGIN_MULTIPLIER) / MAX_SLOTS
    
    for T_15m in day_timestamps_15m:
        # 1. CHECK AND CLOSE ACTIVE TRADES AT 15-MIN RESOLUTION
        still_active = []
        for trade in active_trades:
            ticker = trade['ticker']
            if T_15m not in price_idx[ticker]:
                # Data gap for this ticker on this bar — skip this bar and hold the trade
                still_active.append(trade)
                continue
            idx = price_idx[ticker][T_15m]
            
            is_expiry = idx >= trade['exit_idx']
            is_eod = T_15m.endswith("15:15:00+05:30")
            
            if is_expiry or is_eod:
                # Close trade!
                exit_price = price_series[ticker][idx][1]
                if trade['side'] == 'LONG':
                    gross_ret = (exit_price / trade['entry_price']) - 1
                else:
                    gross_ret = 1 - (exit_price / trade['entry_price'])
                
                net_ret = gross_ret - TRANSACTION_COST_PCT / 100
                profit_loss = trade['exposure'] * net_ret
                
                # Update portfolio capital
                current_capital += profit_loss
                
                trade_ledger.append({
                    'date': date,
                    'ticker': ticker,
                    'side': trade['side'],
                    'entry_time': trade['entry_time'],
                    'exit_time': T_15m,
                    'net_return': net_ret,
                    'pnl_amount': profit_loss,
                    'is_win': net_ret > 0
                })
            else:
                still_active.append(trade)
        active_trades = still_active
        
        # 2. CHECK FOR NEW ENTRIES ON HOURLY CANDLE CLOSE BOUNDARIES
        # Align: hourly candle closes at T_15m if T_15m is the end of the hourly bar
        # Hourly bars start at 09:30 (closes at 10:30), 10:30 (closes at 11:30) etc.
        # The close times in 15M correspond to timestamps ending with:
        # 10:15:00 (which represents the 10:15-10:30 bar closing at 10:30)
        # 11:15:00 (closes at 11:30), etc.
        # So we align our Hourly Scan strictly at the end of the hour:
        time_part = T_15m[11:19]
        if time_part in ["10:15:00", "11:15:00", "12:15:00", "13:15:00", "14:15:00"]:
            # Find the corresponding 1H start time T_1h
            # E.g. for 10:15 close, the 1H start time was 09:30:00+05:30
            h_part = int(T_15m[11:13])
            t_1h_start_h = h_part - 1
            T_1h = f"{date} {t_1h_start_h:02d}:30:00+05:30"
            t_30m, t_15m_align = align_timeframes_at_hour(T_1h)
            
            if T_1h not in dict_1h or t_30m not in dict_30m or t_15m_align not in dict_15m:
                continue
                
            picks_1h = dict_1h[T_1h]
            picks_30m = dict_30m[t_30m]
            picks_15m = dict_15m[t_15m_align]
            
            common_t = [t for t in picks_1h if t in picks_30m and t in picks_15m]
            if not common_t:
                continue
                
            blend_longs = {}
            blend_shorts = {}
            for sym in common_t:
                blend_longs[sym] = (
                    w_1h * picks_1h[sym]['long_conv'] +
                    w_30m * picks_30m[sym]['long_conv'] +
                    w_15m * picks_15m[sym]['long_conv']
                )
                blend_shorts[sym] = (
                    w_1h * picks_1h[sym]['short_conv'] +
                    w_30m * picks_30m[sym]['short_conv'] +
                    w_15m * picks_15m[sym]['short_conv']
                )
            
            top_longs = sorted(blend_longs.keys(), key=lambda x: blend_longs[x], reverse=True)[:TOPK]
            top_shorts = sorted(blend_shorts.keys(), key=lambda x: blend_shorts[x], reverse=True)[:TOPK]
            
            # Combine signals
            signals = [('LONG', sym) for sym in top_longs] + [('SHORT', sym) for sym in top_shorts]
            
            # Place entries if slots are available
            for side, sym in signals:
                # Check slot availability
                if len(active_trades) >= MAX_SLOTS:
                    break # Portfolio fully occupied
                    
                # Skip if already holding this symbol
                if any(trade['ticker'] == sym for trade in active_trades):
                    continue
                    
                # Place trade!
                if sym not in price_idx or T_15m not in price_idx[sym]:
                    continue
                entry_idx = price_idx[sym][T_15m]
                entry_price = price_series[sym][entry_idx][1]
                
                # Expiry index is entry_idx + HOLD_BARS
                exit_idx = entry_idx + HOLD_BARS
                
                active_trades.append({
                    'ticker': sym,
                    'side': side,
                    'entry_price': entry_price,
                    'entry_time': T_15m,
                    'exit_idx': exit_idx,
                    'exposure': exposure_per_slot
                })

# Portfolio Evaluation
if not trade_ledger:
    print("\nNo trades were generated in the simulation.")
    sys.exit()

df_ledger = pd.DataFrame(trade_ledger)
total_trades = len(df_ledger)
wins = int(df_ledger['is_win'].sum())
losses = total_trades - wins
win_rate = wins / total_trades

total_profit_amt = df_ledger['pnl_amount'].sum()
portfolio_return = (current_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

winners = df_ledger[df_ledger['is_win']]
losers = df_ledger[~df_ledger['is_win']]
profit_factor = abs(winners['pnl_amount'].sum() / losers['pnl_amount'].sum()) if len(losers) > 0 and losers['pnl_amount'].sum() != 0 else float('inf')

# Daily performance
daily_perf = df_ledger.groupby('date')['pnl_amount'].sum()
green_days = (daily_perf > 0).sum()
total_days = len(daily_perf)

print("\n" + "=" * 80)
print("REALISTIC PORTFOLIO SIMULATION RESULTS (NO DOUBLE-COUNTING)")
print("=" * 80)
print(f"  Initial Capital        : Rs {INITIAL_CAPITAL:,.2f}")
print(f"  Ending Capital         : Rs {current_capital:,.2f}")
print(f"  Total Net Profit/Loss  : Rs {total_profit_amt:+,.2f}")
print(f"  REAL PORTFOLIO RETURN  : {portfolio_return:>+7.2%}")
print(f"")
print(f"  Total Trades Placed    : {total_trades}")
print(f"  Wins / Losses          : {wins} Wins | {losses} Losses")
print(f"  Win Rate               : {win_rate:.1%}")
print(f"  Avg. PnL per Trade     : Rs {df_ledger['pnl_amount'].mean():+,.2f} ({df_ledger['net_return'].mean()*100:+.3f}%)")
print(f"  Profit Factor          : {profit_factor:.2f}")
print(f"  Green Days             : {green_days}/{total_days} ({green_days/total_days:.1%})")
print("=" * 80 + "\n")
