"""
optimize_strategy_13.py - Highly Optimized Parameter Sweep for Strategy 13: "Midday Momentum Extension"
"""

import os
import sys
import json
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.getcwd())

# Import helpers from the main backtester to ensure feature calculation consistency
from scripts.strategy_25x_backtest import (
    load_and_filter_csv,
    predict_timeframe_scores,
    align_timeframes,
    find_yesterday_daily_date,
    TRANSACTION_COST_PCT,
    TEST_MONTH
)

def run_s13_simulation(
    unique_15m_times,
    ticker_bar_map,
    dict_15m_longs_sorted,
    dict_15m_shorts_sorted,
    time_alignment,
    is_last_bar_map,
    dict_daily,
    dict_1h,
    dict_30m,
    sl_pct,       # None or float (e.g. 0.005 for 0.5%)
    tp_pct,       # None or float (e.g. 0.010 for 1.0%)
    ts_pct,       # None or float (e.g. 0.004 for 0.4%)
    max_hold,     # int (number of 15m bars, e.g. 4)
    rank_15m,     # int (threshold for 15M rank, e.g. 3)
    rank_1h,      # int (threshold for 1H rank, e.g. 5)
    use_daily,    # bool (if True, requires daily_rank <= 0.30)
    use_30m       # bool (if True, requires m30_rank <= 5)
):
    limit = 6  # standard strategy 13 limit
    trades = []
    active_trades = []
    daily_trade_count = {}

    for T in unique_15m_times:
        date_str = T[:10]
        time_str = T[11:16]
        h, m = map(int, time_str.split(':'))
        t_min = h * 60 + m

        if date_str not in daily_trade_count:
            daily_trade_count[date_str] = 0

        is_last_bar = is_last_bar_map[T]

        # 1. EVALUATE EXITS
        remaining_trades = []
        for t in active_trades:
            t['bars_held'] += 1
            bar = ticker_bar_map[t['ticker']].get(T)
            if not bar:
                # Force close due to data gap
                g_ret = 0.0
                n_ret = g_ret - TRANSACTION_COST_PCT / 100
                trades.append({
                    'date': date_str, 'side': t['side'], 'entry_price': t['entry_price'],
                    'exit_price': t['entry_price'], 'net_return': n_ret, 'is_win': n_ret > 0
                })
                continue

            # P&L tracking
            if t['side'] == 'LONG':
                current_pnl = (bar['close'] / t['entry_price']) - 1.0
                high_pnl = (bar['high'] / t['entry_price']) - 1.0
                t['peak_pnl'] = max(t['peak_pnl'], high_pnl)
            else:
                current_pnl = 1.0 - (bar['close'] / t['entry_price'])
                low_pnl = 1.0 - (bar['low'] / t['entry_price'])
                t['peak_pnl'] = max(t['peak_pnl'], low_pnl)

            exit_reason = None
            exit_price = bar['close']

            # Check Stop Loss
            if sl_pct is not None:
                if t['side'] == 'LONG':
                    if bar['low'] <= t['entry_price'] * (1.0 - sl_pct):
                        exit_reason = 'STOP_LOSS'
                        exit_price = t['entry_price'] * (1.0 - sl_pct)
                else:
                    if bar['high'] >= t['entry_price'] * (1.0 + sl_pct):
                        exit_reason = 'STOP_LOSS'
                        exit_price = t['entry_price'] * (1.0 + sl_pct)

            # Check Take Profit
            if exit_reason is None and tp_pct is not None:
                if t['side'] == 'LONG':
                    if bar['high'] >= t['entry_price'] * (1.0 + tp_pct):
                        exit_reason = 'TAKE_PROFIT'
                        exit_price = t['entry_price'] * (1.0 + tp_pct)
                else:
                    if bar['low'] <= t['entry_price'] * (1.0 - tp_pct):
                        exit_reason = 'TAKE_PROFIT'
                        exit_price = t['entry_price'] * (1.0 - tp_pct)

            # Check Trailing Stop
            if exit_reason is None and ts_pct is not None:
                if t['peak_pnl'] - current_pnl >= ts_pct:
                    exit_reason = 'TRAILING_STOP'
                    if t['side'] == 'LONG':
                        exit_price = t['entry_price'] * (1.0 + t['peak_pnl'] - ts_pct)
                    else:
                        exit_price = t['entry_price'] * (1.0 - (t['peak_pnl'] - ts_pct))

            # Check Time Expiry
            if exit_reason is None and t['bars_held'] >= max_hold:
                exit_reason = 'TIME_EXPIRY'
                exit_price = bar['close']

            # EOD force close
            if exit_reason is None and is_last_bar:
                exit_reason = 'FORCE_CLOSE_EOD'
                exit_price = bar['close']

            if exit_reason is not None:
                if t['side'] == 'LONG':
                    g_ret = (exit_price / t['entry_price']) - 1.0
                else:
                    g_ret = 1.0 - (exit_price / t['entry_price'])
                n_ret = g_ret - TRANSACTION_COST_PCT / 100
                trades.append({
                    'date': date_str, 'side': t['side'], 'entry_price': t['entry_price'],
                    'exit_price': exit_price, 'net_return': n_ret, 'is_win': n_ret > 0
                })
            else:
                remaining_trades.append(t)
        active_trades = remaining_trades

        # 2. EVALUATE NEW ENTRIES
        # Midday window: 11:30 to 13:30
        if not (11 * 60 + 30 <= t_min <= 13 * 60 + 30):
            continue

        if daily_trade_count[date_str] >= limit:
            continue

        # Get precomputed timeframe alignment values
        t_1h, t_30m, d_daily = time_alignment[T]
        active_tickers = {at['ticker'] for at in active_trades}

        # Long Entry Setup
        for x in dict_15m_longs_sorted[T]:
            if x['long_rank'] > rank_15m:
                break
            ticker = x['ticker']
            if ticker in active_tickers:
                continue
            if daily_trade_count[date_str] >= limit:
                break

            # 1H filter
            h1_pred = dict_1h.get(t_1h, {}).get(ticker)
            if not h1_pred or h1_pred['long_rank'] > rank_1h:
                continue

            # Optional 30M filter
            if use_30m:
                m30_pred = dict_30m.get(t_30m, {}).get(ticker)
                if not m30_pred or m30_pred['long_rank'] > 5:
                    continue

            # Optional Daily filter
            if use_daily:
                d_pred = dict_daily.get(d_daily, {}).get(ticker)
                if not d_pred or d_pred['long_rank'] > 0.30 * d_pred['count']:
                    continue

            active_trades.append({
                'ticker': ticker, 'side': 'LONG', 'entry_price': x['close'],
                'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
            })
            active_tickers.add(ticker)
            daily_trade_count[date_str] += 1

        # Short Entry Setup
        for x in dict_15m_shorts_sorted[T]:
            if x['short_rank'] > rank_15m:
                break
            ticker = x['ticker']
            if ticker in active_tickers:
                continue
            if daily_trade_count[date_str] >= limit:
                break

            # 1H filter
            h1_pred = dict_1h.get(t_1h, {}).get(ticker)
            if not h1_pred or h1_pred['short_rank'] > rank_1h:
                continue

            # Optional 30M filter
            if use_30m:
                m30_pred = dict_30m.get(t_30m, {}).get(ticker)
                if not m30_pred or m30_pred['short_rank'] > 5:
                    continue

            # Optional Daily filter
            if use_daily:
                d_pred = dict_daily.get(d_daily, {}).get(ticker)
                if not d_pred or d_pred['short_rank'] > 0.30 * d_pred['count']:
                    continue

            active_trades.append({
                'ticker': ticker, 'side': 'SHORT', 'entry_price': x['close'],
                'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
            })
            active_tickers.add(ticker)
            daily_trade_count[date_str] += 1

    # Evaluate results
    if not trades:
        return 0.0, 0.0, 0.0, 0.0, 0

    df_t = pd.DataFrame(trades)
    total_trades = len(df_t)
    total_ret = df_t['net_return'].sum()
    win_rate = df_t['is_win'].mean()
    
    winners = df_t[df_t['is_win']]
    losers = df_t[~df_t['is_win']]
    profit_factor = abs(winners['net_return'].sum() / losers['net_return'].sum()) if len(losers) > 0 and losers['net_return'].sum() != 0 else float('inf')
    
    # Drawdown
    cum_returns = df_t['net_return'].cumsum()
    max_dd = (cum_returns - cum_returns.cummax()).min() if len(cum_returns) > 0 else 0.0

    return total_ret, win_rate, profit_factor, max_dd, total_trades

def main():
    print("=" * 80)
    print("STRATEGY 13 (MIDDAY MOMENTUM EXTENSION) OPTIMIZATION SWEEP")
    print("=" * 80)

    # 1. LOAD DATASETS
    df_daily_raw = load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", ["2026-04", "2026-05"])
    df_1h_raw = load_and_filter_csv("data/ranking_data_upstox_3y.csv", [TEST_MONTH])
    df_30m_raw = load_and_filter_csv("data/ranking_data_upstox_30min_1y.csv", [TEST_MONTH])
    df_15m_raw = load_and_filter_csv("data/ranking_data_upstox_15min_1y.csv", [TEST_MONTH])
    
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
    
    # Intersection of tickers
    t_daily = set(df_daily['Ticker'].unique())
    t_1h = set(df_1h['Ticker'].unique())
    t_30m = set(df_30m['Ticker'].unique())
    t_15m = set(df_15m['Ticker'].unique())
    common_tickers = sorted(list(t_daily.intersection(t_1h).intersection(t_30m).intersection(t_15m)))
    
    df_daily = df_daily[df_daily['Ticker'].isin(common_tickers)].copy()
    df_1h = df_1h[df_1h['Ticker'].isin(common_tickers)].copy()
    df_30m = df_30m[df_30m['Ticker'].isin(common_tickers)].copy()
    df_15m = df_15m[df_15m['Ticker'].isin(common_tickers)].copy()
    
    sorted_daily_dates = sorted(df_daily['DateTime'].str[:10].unique())
    unique_15m_times = sorted(df_15m['DateTime'].unique())

    # Build O(1) indices
    dict_daily = {}
    for date_str, group in df_daily.groupby(df_daily['DateTime'].str[:10]):
        dict_daily[date_str] = {}
        for _, row in group.iterrows():
            dict_daily[date_str][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'count': len(group)
            }
            
    dict_1h = {}
    for dt, group in df_1h.groupby('DateTime'):
        dict_1h[dt] = {}
        for _, row in group.iterrows():
            dict_1h[dt][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank'])
            }
            
    dict_30m = {}
    for dt, group in df_30m.groupby('DateTime'):
        dict_30m[dt] = {}
        for _, row in group.iterrows():
            dict_30m[dt][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank'])
            }

    # Structured 15M lists
    print("Building structured price lists...")
    ticker_bar_map = {t: {} for t in common_tickers}
    
    # Pre-group and sort the 15M candidates by rank for O(1) search
    dict_15m_longs_sorted = {}
    dict_15m_shorts_sorted = {}
    
    for dt, group in df_15m.groupby('DateTime'):
        bars = []
        for _, row in group.iterrows():
            t = row['Ticker']
            bar_dict = {
                'ticker': t,
                'datetime': row['DateTime'],
                'close': float(row['Close']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank'])
            }
            bars.append(bar_dict)
            ticker_bar_map[t][row['DateTime']] = bar_dict
            
        dict_15m_longs_sorted[dt] = sorted(bars, key=lambda x: x['long_rank'])
        dict_15m_shorts_sorted[dt] = sorted(bars, key=lambda x: x['short_rank'])

    # Precompute time alignments
    print("Precomputing timeframes and alignments...")
    time_alignment = {}
    is_last_bar_map = {}
    for idx, T in enumerate(unique_15m_times):
        date_str = T[:10]
        time_str = T[11:16]
        t_1h, t_30m = align_timeframes(T)
        d_daily = find_yesterday_daily_date(date_str, sorted_daily_dates)
        time_alignment[T] = (t_1h, t_30m, d_daily)
        
        is_last_bar = (idx == len(unique_15m_times) - 1) or (unique_15m_times[idx + 1][:10] != date_str) or (time_str == "15:15")
        is_last_bar_map[T] = is_last_bar

    # Define optimization grid
    sl_options = [None, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
    tp_options = [None, 0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020]
    ts_options = [None, 0.003, 0.004, 0.005, 0.006, 0.008]
    max_hold_options = [2, 3, 4, 6, 8]
    rank_15m_options = [2, 3, 5]
    rank_1h_options = [3, 5, 8]
    use_daily_options = [False, True]
    use_30m_options = [False, True]

    # Total combinations is 7 * 8 * 6 * 5 * 3 * 3 * 2 * 2 = 60,480
    print("\nRunning highly optimized grid search sweep over 60,480 configurations...")
    results = []
    
    grid = list(itertools.product(
        sl_options,
        tp_options,
        ts_options,
        max_hold_options,
        rank_15m_options,
        rank_1h_options,
        use_daily_options,
        use_30m_options
    ))

    # Run grid search
    for sl, tp, ts, max_hold, r15, r1h, ud, u30 in tqdm(grid):
        ret, wr, pf, dd, num_trades = run_s13_simulation(
            unique_15m_times,
            ticker_bar_map,
            dict_15m_longs_sorted,
            dict_15m_shorts_sorted,
            time_alignment,
            is_last_bar_map,
            dict_daily,
            dict_1h,
            dict_30m,
            sl_pct=sl,
            tp_pct=tp,
            ts_pct=ts,
            max_hold=max_hold,
            rank_15m=r15,
            rank_1h=r1h,
            use_daily=ud,
            use_30m=u30
        )
        
        results.append({
            'sl': sl if sl is not None else np.nan,
            'tp': tp if tp is not None else np.nan,
            'ts': ts if ts is not None else np.nan,
            'max_hold': max_hold,
            'rank_15m': r15,
            'rank_1h': r1h,
            'use_daily': ud,
            'use_30m': u30,
            'total_return': ret,
            'win_rate': wr,
            'profit_factor': pf,
            'max_drawdown': dd,
            'total_trades': num_trades
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv("data/strategy_13_optimization_results.csv", index=False)
    print("\nSaved optimization results to data/strategy_13_optimization_results.csv")

    # Display Top 30 configurations by Total Return
    print("\n" + "=" * 120)
    print("TOP 30 OPTIMAL CONFIGURATIONS FOR STRATEGY 13")
    print("=" * 120)
    df_sorted = df_res.sort_values('total_return', ascending=False)
    
    headers = ["SL%", "TP%", "TS%", "Hold", "R15M", "R1H", "Daily", "30M", "Return%", "WR%", "PF", "MaxDD%", "Trades"]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-|-'.join(['-' * len(h) for h in headers])}|")
    
    for _, row in df_sorted.head(30).iterrows():
        sl_s = f"{row['sl']*100:.1f}%" if pd.notna(row['sl']) else "None"
        tp_s = f"{row['tp']*100:.1f}%" if pd.notna(row['tp']) else "None"
        ts_s = f"{row['ts']*100:.1f}%" if pd.notna(row['ts']) else "None"
        hold_s = str(int(row['max_hold']))
        r15_s = str(int(row['rank_15m']))
        r1h_s = str(int(row['rank_1h']))
        ud_s = "Yes" if row['use_daily'] else "No"
        u30_s = "Yes" if row['use_30m'] else "No"
        ret_s = f"{row['total_return']*100:+.2f}%"
        wr_s = f"{row['win_rate']*100:.1f}%"
        pf_s = f"{row['profit_factor']:.2f}"
        dd_s = f"{row['max_drawdown']*100:.2f}%"
        trades_s = str(int(row['total_trades']))
        
        print(f"| {sl_s:<4} | {tp_s:<4} | {ts_s:<4} | {hold_s:<4} | {r15_s:<4} | {r1h_s:<4} | {ud_s:<5} | {u30_s:<3} | {ret_s:<7} | {wr_s:<5} | {pf_s:<4} | {dd_s:<7} | {trades_s:<6} |")

    # Display baseline comparison (SL=None, TP=None, TS=None, Hold=4, R15M=3, R1H=5, Daily=No, 30M=No)
    baseline = df_res[
        df_res['sl'].isna() &
        df_res['tp'].isna() &
        df_res['ts'].isna() &
        (df_res['max_hold'] == 4) &
        (df_res['rank_15m'] == 3) &
        (df_res['rank_1h'] == 5) &
        (~df_res['use_daily']) &
        (~df_res['use_30m'])
    ]
    if not baseline.empty:
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON (CURRENT LOGIC)")
        print("=" * 60)
        row = baseline.iloc[0]
        print(f"Return: {row['total_return']*100:+.2f}% (Expect ~-1.86%)")
        print(f"Win Rate: {row['win_rate']*100:.1f}%")
        print(f"Profit Factor: {row['profit_factor']:.2f}")
        print(f"Max Drawdown: {row['max_drawdown']*100:.2f}% (Expect ~-7.12%)")
        print(f"Total Trades: {row['total_trades']}")

if __name__ == '__main__':
    main()
