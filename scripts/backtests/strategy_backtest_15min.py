"""
strategy_backtest_15min.py -- 15-Min Model Strategy Evaluator & Backtester

Simulates realistic daily intraday trading with the 15-min ranking model.
Tests multiple strategy variants to find the configuration that achieves
~4 winning trades + ~1 losing trade per day (80% win rate target).

Strategies tested:
  A. Baseline: Top-1 long + Top-1 short every 15-min bar (raw model picks)
  B. High-Conviction: Only trade when model score spread is in the top percentile
  C. IBS Confluence: Only enter when IBS confirms the model direction
  D. Time-Filtered: Restrict trading to the highest-edge hours
  E. Combined Best: Confluence of B + C + D for maximum win rate

Usage:
  python scripts/strategy_backtest_15min.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from datetime import datetime

sys.path.append(os.getcwd())

# ============================================================
# CONFIG
# ============================================================
DATA_FILE = "data/ranking_data_upstox_15min_1y.csv"
MODEL_DIR = "models/v1_15min"
LONG_MODEL_PATH = f"{MODEL_DIR}/xgb_long_model.json"
SHORT_MODEL_PATH = f"{MODEL_DIR}/xgb_short_model.json"
META_PATH = f"{MODEL_DIR}/metadata.json"

# Trading Parameters
TRANSACTION_COST_PCT = 0.03  # 0.03% round-trip (brokerage + STT + slippage)
ATR_SL_MULT = 1.5
ATR_TP_MULT = 3.0
TARGET_TRADES_PER_DAY = 5  # User target: 4 wins + 1 loss

print("=" * 70)
print("15-MIN MODEL STRATEGY BACKTESTER")
print("Finding the optimal strategy for 4W + 1L per day")
print("=" * 70)

# ============================================================
# LOAD DATA & MODELS
# ============================================================
if not os.path.exists(DATA_FILE):
    print(f"[FATAL] Data file not found: {DATA_FILE}")
    sys.exit(1)

print(f"\nLoading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
df['YearMonth'] = df['DateTime'].str[:7]
df['Date'] = df['DateTime'].str[:10]
df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
df['Minute'] = pd.to_datetime(df['DateTime']).dt.minute
df['TimeSlot'] = df['Hour'].astype(str).str.zfill(2) + ':' + df['Minute'].astype(str).str.zfill(2)
print(f"  {df.shape[0]:,} rows | {df['Query_ID'].nunique():,} queries | {df['Date'].nunique()} trading days")

with open(META_PATH) as f:
    meta = json.load(f)
feature_cols = meta["features"]
print(f"  Features: {len(feature_cols)}")

# Load models
bst_long = xgb.Booster()
bst_long.load_model(LONG_MODEL_PATH)
bst_short = xgb.Booster()
bst_short.load_model(SHORT_MODEL_PATH)

# ============================================================
# USE WALK-FORWARD TEST MONTHS (Leakage-Free Evaluation)
# ============================================================
# We evaluate on the 3 test months from walk-forward folds
test_months = ['2026-03', '2026-04', '2026-05']
df_test = df[df['YearMonth'].isin(test_months)].copy()
test_dates = sorted(df_test['Date'].unique())

print(f"\n  Test Period: {test_months}")
print(f"  Test Data: {df_test.shape[0]:,} rows | {len(test_dates)} trading days")

# ============================================================
# PREDICT SCORES FOR ALL TEST DATA
# ============================================================
print("\nGenerating predictions...")
X_test = df_test[feature_cols].values
X_test = np.nan_to_num(X_test)
dmatrix = xgb.DMatrix(X_test)

df_test['long_score'] = bst_long.predict(dmatrix)
df_test['short_score'] = bst_short.predict(dmatrix)
df_test['score_spread'] = df_test['long_score'] - df_test['short_score']

# Pre-compute useful columns
# IBS is already in the features, but we need raw IBS for confluence checks
if 'IBS' in df_test.columns:
    # IBS might be Z-scored, check if we have raw OHLC to compute it
    df_test['IBS_raw'] = (df_test['Close'] - df_test['Low']) / (df_test['High'] - df_test['Low'] + 1e-10)
else:
    df_test['IBS_raw'] = 0.5  # fallback

print(f"  Predictions done. Score spread range: [{df_test['score_spread'].min():.4f}, {df_test['score_spread'].max():.4f}]")

# ============================================================
# STRATEGY SIMULATION FRAMEWORK
# ============================================================
def simulate_strategy(df_data, strategy_name, select_fn, max_trades_per_day=8):
    """
    Simulate a day-by-day trading strategy.
    
    select_fn(df_query, side) -> list of ticker indices to trade
        Returns indices into df_query for that query's selections.
    
    Returns a dict of aggregate statistics.
    """
    all_trades = []
    daily_stats = []
    dates = sorted(df_data['Date'].unique())
    
    for date in dates:
        df_day = df_data[df_data['Date'] == date]
        queries = sorted(df_day['Query_ID'].unique())
        day_trades = []
        traded_tickers_today = set()  # cooldown within day
        
        for qid in queries:
            if len(day_trades) >= max_trades_per_day:
                break
                
            q_df = df_day[df_day['Query_ID'] == qid]
            if len(q_df) < 4:
                continue
            
            # Try LONG selection
            if len(day_trades) < max_trades_per_day:
                long_picks = select_fn(q_df, 'LONG', traded_tickers_today)
                for idx in long_picks:
                    if len(day_trades) >= max_trades_per_day:
                        break
                    row = q_df.iloc[idx]
                    ticker = row['Ticker']
                    if ticker in traded_tickers_today:
                        continue
                    ret = row['Next_15Min_Return']
                    cost = TRANSACTION_COST_PCT / 100
                    net_ret = ret - cost
                    trade = {
                        'date': date,
                        'query_id': qid,
                        'ticker': ticker,
                        'side': 'LONG',
                        'gross_return': ret,
                        'net_return': net_ret,
                        'is_win': net_ret > 0,
                        'long_score': row['long_score'],
                        'short_score': row['short_score'],
                        'score_spread': row['score_spread'],
                        'ibs_raw': row.get('IBS_raw', 0.5),
                        'hour': row['Hour'],
                        'time_slot': row['TimeSlot'],
                    }
                    all_trades.append(trade)
                    day_trades.append(trade)
                    traded_tickers_today.add(ticker)
            
            # Try SHORT selection
            if len(day_trades) < max_trades_per_day:
                short_picks = select_fn(q_df, 'SHORT', traded_tickers_today)
                for idx in short_picks:
                    if len(day_trades) >= max_trades_per_day:
                        break
                    row = q_df.iloc[idx]
                    ticker = row['Ticker']
                    if ticker in traded_tickers_today:
                        continue
                    ret = -row['Next_15Min_Return']  # short P&L
                    cost = TRANSACTION_COST_PCT / 100
                    net_ret = ret - cost
                    trade = {
                        'date': date,
                        'query_id': qid,
                        'ticker': ticker,
                        'side': 'SHORT',
                        'gross_return': ret,
                        'net_return': net_ret,
                        'is_win': net_ret > 0,
                        'long_score': row['long_score'],
                        'short_score': row['short_score'],
                        'score_spread': row['score_spread'],
                        'ibs_raw': row.get('IBS_raw', 0.5),
                        'hour': row['Hour'],
                        'time_slot': row['TimeSlot'],
                    }
                    all_trades.append(trade)
                    day_trades.append(trade)
                    traded_tickers_today.add(ticker)
        
        # Daily summary
        if day_trades:
            wins = sum(1 for t in day_trades if t['is_win'])
            losses = len(day_trades) - wins
            daily_pnl = sum(t['net_return'] for t in day_trades)
            daily_stats.append({
                'date': date,
                'trades': len(day_trades),
                'wins': wins,
                'losses': losses,
                'win_rate': wins / len(day_trades) if day_trades else 0,
                'daily_pnl': daily_pnl,
                'is_green_day': daily_pnl > 0,
            })
    
    # Aggregate
    if not all_trades:
        return {'strategy': strategy_name, 'total_trades': 0}
    
    trades_df = pd.DataFrame(all_trades)
    daily_df = pd.DataFrame(daily_stats)
    
    total_trades = len(trades_df)
    total_wins = trades_df['is_win'].sum()
    total_losses = total_trades - total_wins
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    
    avg_trades_per_day = daily_df['trades'].mean()
    avg_wins_per_day = daily_df['wins'].mean()
    avg_losses_per_day = daily_df['losses'].mean()
    avg_daily_wr = daily_df['win_rate'].mean()
    
    avg_gross_ret = trades_df['gross_return'].mean()
    avg_net_ret = trades_df['net_return'].mean()
    avg_daily_pnl = daily_df['daily_pnl'].mean()
    
    green_days = daily_df['is_green_day'].sum()
    total_days = len(daily_df)
    green_day_rate = green_days / total_days if total_days > 0 else 0
    
    # Win/Loss magnitudes
    winners = trades_df[trades_df['is_win']]
    losers = trades_df[~trades_df['is_win']]
    avg_win_size = winners['net_return'].mean() if len(winners) > 0 else 0
    avg_loss_size = losers['net_return'].mean() if len(losers) > 0 else 0
    profit_factor = abs(winners['net_return'].sum() / losers['net_return'].sum()) if len(losers) > 0 and losers['net_return'].sum() != 0 else float('inf')
    
    # Cumulative PnL
    cumulative_pnl = trades_df['net_return'].cumsum()
    max_drawdown = (cumulative_pnl - cumulative_pnl.cummax()).min()
    
    # Best/worst day
    best_day = daily_df.loc[daily_df['daily_pnl'].idxmax()] if not daily_df.empty else None
    worst_day = daily_df.loc[daily_df['daily_pnl'].idxmin()] if not daily_df.empty else None
    
    return {
        'strategy': strategy_name,
        'total_trades': total_trades,
        'total_wins': int(total_wins),
        'total_losses': int(total_losses),
        'overall_win_rate': overall_wr,
        'avg_trades_per_day': avg_trades_per_day,
        'avg_wins_per_day': avg_wins_per_day,
        'avg_losses_per_day': avg_losses_per_day,
        'avg_daily_win_rate': avg_daily_wr,
        'avg_gross_return_per_trade': avg_gross_ret,
        'avg_net_return_per_trade': avg_net_ret,
        'avg_daily_pnl': avg_daily_pnl,
        'green_days': int(green_days),
        'total_days': int(total_days),
        'green_day_rate': green_day_rate,
        'avg_win_size': avg_win_size,
        'avg_loss_size': avg_loss_size,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'best_day': {'date': best_day['date'], 'pnl': best_day['daily_pnl'], 'trades': int(best_day['trades'])} if best_day is not None else None,
        'worst_day': {'date': worst_day['date'], 'pnl': worst_day['daily_pnl'], 'trades': int(worst_day['trades'])} if worst_day is not None else None,
        'trades_df': trades_df,
        'daily_df': daily_df,
    }


def print_strategy_report(result):
    """Pretty-print a strategy evaluation report."""
    s = result
    if s['total_trades'] == 0:
        print(f"  {s['strategy']}: NO TRADES GENERATED")
        return
    
    print(f"\n{'=' * 70}")
    print(f"  STRATEGY: {s['strategy']}")
    print(f"{'=' * 70}")
    print(f"  Total Trades: {s['total_trades']} over {s['total_days']} days")
    print(f"  Total Wins: {s['total_wins']} | Total Losses: {s['total_losses']}")
    print(f"  Overall Win Rate: {s['overall_win_rate']:.1%}")
    print(f"")
    print(f"  --- Daily Averages ---")
    print(f"  Avg Trades/Day   : {s['avg_trades_per_day']:.1f}")
    print(f"  Avg Wins/Day     : {s['avg_wins_per_day']:.1f}")
    print(f"  Avg Losses/Day   : {s['avg_losses_per_day']:.1f}")
    print(f"  Avg Daily Win Rate: {s['avg_daily_win_rate']:.1%}")
    print(f"")
    print(f"  --- Returns (after {TRANSACTION_COST_PCT:.2f}% round-trip cost) ---")
    print(f"  Avg Gross Return/Trade : {s['avg_gross_return_per_trade']*100:+.4f}%")
    print(f"  Avg Net Return/Trade   : {s['avg_net_return_per_trade']*100:+.4f}%")
    print(f"  Avg Daily P&L (summed) : {s['avg_daily_pnl']*100:+.4f}%")
    print(f"")
    print(f"  --- Risk Metrics ---")
    print(f"  Green Days: {s['green_days']}/{s['total_days']} ({s['green_day_rate']:.1%})")
    print(f"  Avg Win Size   : {s['avg_win_size']*100:+.4f}%")
    print(f"  Avg Loss Size  : {s['avg_loss_size']*100:+.4f}%")
    print(f"  Profit Factor  : {s['profit_factor']:.2f}")
    print(f"  Max Drawdown   : {s['max_drawdown']*100:.4f}%")
    if s['best_day']:
        print(f"  Best Day  : {s['best_day']['date']} ({s['best_day']['pnl']*100:+.4f}%, {s['best_day']['trades']} trades)")
    if s['worst_day']:
        print(f"  Worst Day : {s['worst_day']['date']} ({s['worst_day']['pnl']*100:+.4f}%, {s['worst_day']['trades']} trades)")


# ============================================================
# STRATEGY A: BASELINE - Top-1 per query, first 5 trades/day
# ============================================================
print("\n" + "#" * 70)
print("RUNNING STRATEGY SIMULATIONS")
print("#" * 70)

def strategy_a_baseline(q_df, side, traded_tickers):
    """Pick the single best stock per query per side."""
    if side == 'LONG':
        scores = q_df['long_score'].values
    else:
        scores = q_df['short_score'].values
    best_idx = np.argsort(scores)[::-1][0]
    ticker = q_df.iloc[best_idx]['Ticker']
    if ticker in traded_tickers:
        return []
    return [best_idx]

print("\n[RUNNING] Strategy A: Baseline (Top-1 per query, max 5/day)...")
result_a = simulate_strategy(df_test, "A. Baseline (Top-1, Max 5/day)", strategy_a_baseline, max_trades_per_day=5)
print_strategy_report(result_a)


# ============================================================
# STRATEGY B: HIGH-CONVICTION FILTER
# Score spread must exceed a threshold
# ============================================================
# Compute score spread percentiles across the full test data
spread_p70 = df_test['score_spread'].abs().quantile(0.70)
spread_p80 = df_test['score_spread'].abs().quantile(0.80)

def strategy_b_high_conviction(q_df, side, traded_tickers):
    """Only pick the top-1 if its score spread is in the top 30%."""
    if side == 'LONG':
        scores = q_df['long_score'].values
        best_idx = np.argsort(scores)[::-1][0]
        best_row = q_df.iloc[best_idx]
        # Long conviction: long_score - short_score should be positive and high
        if best_row['score_spread'] < spread_p70:
            return []
    else:
        scores = q_df['short_score'].values
        best_idx = np.argsort(scores)[::-1][0]
        best_row = q_df.iloc[best_idx]
        # Short conviction: short_score - long_score should be positive and high
        if -best_row['score_spread'] < spread_p70:
            return []
    
    if best_row['Ticker'] in traded_tickers:
        return []
    return [best_idx]

print("\n[RUNNING] Strategy B: High-Conviction (Score Spread > P70, Max 6/day)...")
result_b = simulate_strategy(df_test, "B. High-Conviction (Spread > P70)", strategy_b_high_conviction, max_trades_per_day=6)
print_strategy_report(result_b)


# ============================================================
# STRATEGY C: IBS CONFLUENCE
# Only go LONG when IBS < 0.3 (oversold), SHORT when IBS > 0.7 (overbought)
# ============================================================
def strategy_c_ibs_confluence(q_df, side, traded_tickers):
    """Only trade when raw IBS confirms the direction."""
    if side == 'LONG':
        scores = q_df['long_score'].values
        ranked = np.argsort(scores)[::-1]
        for idx in ranked[:3]:
            row = q_df.iloc[idx]
            if row['Ticker'] in traded_tickers:
                continue
            if row['IBS_raw'] < 0.30:  # Stock is near low of bar -> mean reversion long
                return [idx]
        return []
    else:
        scores = q_df['short_score'].values
        ranked = np.argsort(scores)[::-1]
        for idx in ranked[:3]:
            row = q_df.iloc[idx]
            if row['Ticker'] in traded_tickers:
                continue
            if row['IBS_raw'] > 0.70:  # Stock is near high of bar -> mean reversion short
                return [idx]
        return []

print("\n[RUNNING] Strategy C: IBS Confluence (Long: IBS<0.30, Short: IBS>0.70, Max 6/day)...")
result_c = simulate_strategy(df_test, "C. IBS Confluence (L<0.30, S>0.70)", strategy_c_ibs_confluence, max_trades_per_day=6)
print_strategy_report(result_c)


# ============================================================
# STRATEGY D: TIME-FILTERED (Best Hours Only)
# Only trade during 10:00-11:30 and 13:30-14:30 (avoiding open noise and close auction)
# ============================================================
BEST_HOURS_LONG = {10, 11, 13, 14}  # Focus on mid-session mean reversion
BEST_HOURS_SHORT = {10, 11, 13, 14}

def strategy_d_time_filtered(q_df, side, traded_tickers):
    """Only trade during historically strongest time windows."""
    best_hours = BEST_HOURS_LONG if side == 'LONG' else BEST_HOURS_SHORT
    
    # Check the hour of this query
    hour = q_df.iloc[0]['Hour']
    if hour not in best_hours:
        return []
    
    if side == 'LONG':
        scores = q_df['long_score'].values
    else:
        scores = q_df['short_score'].values
    
    best_idx = np.argsort(scores)[::-1][0]
    if q_df.iloc[best_idx]['Ticker'] in traded_tickers:
        return []
    return [best_idx]

print("\n[RUNNING] Strategy D: Time-Filtered (10:00-14:59 only, Max 6/day)...")
result_d = simulate_strategy(df_test, "D. Time-Filtered (10-14 only)", strategy_d_time_filtered, max_trades_per_day=6)
print_strategy_report(result_d)


# ============================================================
# STRATEGY E: COMBINED BEST (Conviction + IBS + Time)
# Maximum selectivity for highest win rate
# ============================================================
def strategy_e_combined(q_df, side, traded_tickers):
    """
    Combined confluence: 
    - Time filter (10:00-14:59)
    - IBS confluence (Long: IBS < 0.35, Short: IBS > 0.65)
    - Score spread in top 30% (high conviction)
    """
    # Time filter
    hour = q_df.iloc[0]['Hour']
    if hour < 10 or hour >= 15:
        return []
    
    if side == 'LONG':
        scores = q_df['long_score'].values
        ranked = np.argsort(scores)[::-1]
        for idx in ranked[:3]:
            row = q_df.iloc[idx]
            if row['Ticker'] in traded_tickers:
                continue
            # IBS confluence: stock near low of bar
            if row['IBS_raw'] > 0.35:
                continue
            # Conviction filter: positive score spread
            if row['score_spread'] < spread_p70:
                continue
            return [idx]
        return []
    else:
        scores = q_df['short_score'].values
        ranked = np.argsort(scores)[::-1]
        for idx in ranked[:3]:
            row = q_df.iloc[idx]
            if row['Ticker'] in traded_tickers:
                continue
            # IBS confluence: stock near high of bar
            if row['IBS_raw'] < 0.65:
                continue
            # Conviction filter: negative score spread (short wins)
            if -row['score_spread'] < spread_p70:
                continue
            return [idx]
        return []

print("\n[RUNNING] Strategy E: Combined Best (Time + IBS + Conviction, Max 7/day)...")
result_e = simulate_strategy(df_test, "E. Combined (Time+IBS+Conviction)", strategy_e_combined, max_trades_per_day=7)
print_strategy_report(result_e)


# ============================================================
# STRATEGY F: RELAXED COMBINED (wider IBS + time, no spread filter)
# Targets more trades to hit 5/day while keeping elevated win rate
# ============================================================
def strategy_f_relaxed(q_df, side, traded_tickers):
    """
    Relaxed confluence for more trade volume:
    - Time filter (9:30-14:59)
    - IBS confluence with wider bands (Long: IBS < 0.40, Short: IBS > 0.60)
    """
    hour = q_df.iloc[0]['Hour']
    minute = q_df.iloc[0]['Minute']
    
    # Exclude first 15 min of market and last 30 min
    if hour == 9 and minute < 30:
        return []
    if hour >= 15:
        return []
    
    if side == 'LONG':
        scores = q_df['long_score'].values
        ranked = np.argsort(scores)[::-1]
        for idx in ranked[:3]:
            row = q_df.iloc[idx]
            if row['Ticker'] in traded_tickers:
                continue
            if row['IBS_raw'] > 0.40:
                continue
            return [idx]
        return []
    else:
        scores = q_df['short_score'].values
        ranked = np.argsort(scores)[::-1]
        for idx in ranked[:3]:
            row = q_df.iloc[idx]
            if row['Ticker'] in traded_tickers:
                continue
            if row['IBS_raw'] < 0.60:
                continue
            return [idx]
        return []

print("\n[RUNNING] Strategy F: Relaxed IBS (IBS<0.40/IBS>0.60, 9:30-14:59, Max 7/day)...")
result_f = simulate_strategy(df_test, "F. Relaxed IBS (L<0.40, S>0.60)", strategy_f_relaxed, max_trades_per_day=7)
print_strategy_report(result_f)


# ============================================================
# HOURLY EDGE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("HOURLY EDGE ANALYSIS (Which hours have the best win rate?)")
print("=" * 70)

# Use baseline all-query predictions
for side_label, score_col in [('LONG', 'long_score'), ('SHORT', 'short_score')]:
    print(f"\n  --- {side_label} Model ---")
    print(f"  {'Hour':>6} {'Win Rate':>10} {'Avg Return':>12} {'Queries':>8}")
    print(f"  " + "-" * 42)
    
    for hour in range(9, 16):
        hour_df = df_test[df_test['Hour'] == hour]
        if hour_df.empty:
            continue
        
        hits = 0
        total = 0
        returns = []
        
        for qid in hour_df['Query_ID'].unique():
            q_df = hour_df[hour_df['Query_ID'] == qid]
            if len(q_df) < 4:
                continue
            
            scores = q_df[score_col].values
            actual = q_df['Next_15Min_Return'].values
            median_ret = np.median(actual)
            
            best_idx = np.argsort(scores)[::-1][0]
            
            if side_label == 'LONG':
                if actual[best_idx] > median_ret:
                    hits += 1
                returns.append(actual[best_idx])
            else:
                if actual[best_idx] < median_ret:
                    hits += 1
                returns.append(-actual[best_idx])
            total += 1
        
        if total > 0:
            wr = hits / total
            avg_ret = np.mean(returns) * 100
            print(f"  {hour:>6} {wr:>9.1%} {avg_ret:>+11.4f}% {total:>8}")


# ============================================================
# IBS BAND ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("IBS BAND ANALYSIS (Win rate by IBS level of top pick)")
print("=" * 70)

ibs_bands = [(0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.01)]

for side_label, score_col in [('LONG', 'long_score'), ('SHORT', 'short_score')]:
    print(f"\n  --- {side_label} Model ---")
    print(f"  {'IBS Band':>12} {'Win Rate':>10} {'Avg Return':>12} {'Count':>8}")
    print(f"  " + "-" * 46)
    
    for lo, hi in ibs_bands:
        hits = 0
        total = 0
        returns = []
        
        for qid in df_test['Query_ID'].unique():
            q_df = df_test[df_test['Query_ID'] == qid]
            if len(q_df) < 4:
                continue
            
            scores = q_df[score_col].values
            best_idx = np.argsort(scores)[::-1][0]
            best_row = q_df.iloc[best_idx]
            ibs = best_row['IBS_raw']
            
            if not (lo <= ibs < hi):
                continue
            
            actual = q_df['Next_15Min_Return'].values
            median_ret = np.median(actual)
            
            if side_label == 'LONG':
                if actual[best_idx] > median_ret:
                    hits += 1
                returns.append(actual[best_idx])
            else:
                if actual[best_idx] < median_ret:
                    hits += 1
                returns.append(-actual[best_idx])
            total += 1
        
        if total > 0:
            wr = hits / total
            avg_ret = np.mean(returns) * 100
            print(f"  {lo:.2f}-{hi:.2f} {wr:>9.1%} {avg_ret:>+11.4f}% {total:>8}")


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("FINAL STRATEGY COMPARISON")
print("=" * 70)

all_results = [result_a, result_b, result_c, result_d, result_e, result_f]

print(f"\n  {'Strategy':<40} {'Trades/Day':>10} {'Win Rate':>10} {'W/D':>5} {'L/D':>5} {'Net Ret/Trade':>14} {'PF':>6} {'Green%':>7}")
print(f"  " + "-" * 97)

for r in all_results:
    if r['total_trades'] == 0:
        print(f"  {r['strategy']:<40} {'--':>10}")
        continue
    print(f"  {r['strategy']:<40} "
          f"{r['avg_trades_per_day']:>9.1f} "
          f"{r['overall_win_rate']:>9.1%} "
          f"{r['avg_wins_per_day']:>4.1f} "
          f"{r['avg_losses_per_day']:>4.1f} "
          f"{r['avg_net_return_per_trade']*100:>+13.4f}% "
          f"{r['profit_factor']:>5.2f} "
          f"{r['green_day_rate']:>6.1%}")

print()

# ============================================================
# SAVE RESULTS
# ============================================================
save_results = []
for r in all_results:
    save_entry = {k: v for k, v in r.items() if k not in ['trades_df', 'daily_df']}
    # Convert numpy types for JSON serialization
    for k, v in save_entry.items():
        if isinstance(v, (np.integer,)):
            save_entry[k] = int(v)
        elif isinstance(v, (np.floating,)):
            save_entry[k] = float(v)
        elif isinstance(v, np.bool_):
            save_entry[k] = bool(v)
    save_results.append(save_entry)

output_path = "data/strategy_backtest_15min_results.json"
with open(output_path, "w") as f:
    json.dump(save_results, f, indent=2, default=str)
print(f"[OK] Results saved to: {output_path}")
print("=" * 70)
print()
