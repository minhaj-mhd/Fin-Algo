import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())

from scripts.strategy_25x_backtest import (
    load_and_filter_csv,
    predict_timeframe_scores,
    align_timeframes,
    find_yesterday_daily_date,
    TRANSACTION_COST_PCT,
    TEST_MONTH
)

def run_s24_simulation(unique_15m_times, ticker_bar_map, dict_15m, is_last_bar_map,
                       sl_pct, tp_pct, ts_pct, max_hold, rank_blend):
    limit = 6
    trades = []
    active_trades = []
    daily_trade_count = {}
    
    for T in unique_15m_times:
        date_str = T[:10]
        if date_str not in daily_trade_count:
            daily_trade_count[date_str] = 0
            
        is_last_bar = is_last_bar_map[T]
        remaining_trades = []
        
        # 1. EVALUATE EXITS
        for t in active_trades:
            t['bars_held'] += 1
            bar = ticker_bar_map[t['ticker']].get(T)
            if not bar:
                n_ret = -TRANSACTION_COST_PCT / 100
                trades.append({
                    'side': t['side'], 'net_return': n_ret, 'is_win': n_ret > 0
                })
                continue
                
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
            
            if sl_pct is not None:
                if t['side'] == 'LONG' and bar['low'] <= t['entry_price'] * (1.0 - sl_pct):
                    exit_reason = 'STOP_LOSS'
                    exit_price = t['entry_price'] * (1.0 - sl_pct)
                elif t['side'] == 'SHORT' and bar['high'] >= t['entry_price'] * (1.0 + sl_pct):
                    exit_reason = 'STOP_LOSS'
                    exit_price = t['entry_price'] * (1.0 + sl_pct)
                    
            if exit_reason is None and tp_pct is not None:
                if t['side'] == 'LONG' and bar['high'] >= t['entry_price'] * (1.0 + tp_pct):
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = t['entry_price'] * (1.0 + tp_pct)
                elif t['side'] == 'SHORT' and bar['low'] <= t['entry_price'] * (1.0 - tp_pct):
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = t['entry_price'] * (1.0 - tp_pct)
                    
            if exit_reason is None and ts_pct is not None:
                if t['peak_pnl'] - current_pnl >= ts_pct:
                    exit_reason = 'TRAILING_STOP'
                    if t['side'] == 'LONG':
                        exit_price = t['entry_price'] * (1.0 + (t['peak_pnl'] - ts_pct))
                    else:
                        exit_price = t['entry_price'] * (1.0 - (t['peak_pnl'] - ts_pct))
                    
            if exit_reason is None and t['bars_held'] >= max_hold:
                exit_reason = 'TIME_EXPIRY'
                
            if exit_reason is None and is_last_bar:
                exit_reason = 'FORCE_CLOSE_EOD'

            if exit_reason:
                if t['side'] == 'LONG': g_ret = (exit_price / t['entry_price']) - 1.0
                else: g_ret = 1.0 - (exit_price / t['entry_price'])
                n_ret = g_ret - TRANSACTION_COST_PCT / 100
                trades.append({
                    'side': t['side'], 'net_return': n_ret, 'is_win': n_ret > 0
                })
            else:
                remaining_trades.append(t)
        active_trades = remaining_trades
        
        # 2. ENTRIES
        if daily_trade_count[date_str] >= limit:
            continue
            
        active_tickers = {x['ticker'] for x in active_trades}
        
        # LONG
        for x in dict_15m[T]['longs_blend']:
            if x['blend_long_rank'] > rank_blend: break
            if x['ticker'] not in active_tickers:
                active_trades.append({'ticker': x['ticker'], 'side': 'LONG', 'entry_price': x['close'], 'bars_held': 0, 'peak_pnl': 0.0})
                active_tickers.add(x['ticker'])
                daily_trade_count[date_str] += 1
                if daily_trade_count[date_str] >= limit: break

        if daily_trade_count[date_str] >= limit:
            continue
            
        # SHORT
        for x in dict_15m[T]['shorts_blend']:
            if x['blend_short_rank'] > rank_blend: break
            if x['ticker'] not in active_tickers:
                active_trades.append({'ticker': x['ticker'], 'side': 'SHORT', 'entry_price': x['close'], 'bars_held': 0, 'peak_pnl': 0.0})
                active_tickers.add(x['ticker'])
                daily_trade_count[date_str] += 1
                if daily_trade_count[date_str] >= limit: break
                
    if not trades: return 0.0, 0.0, 0.0, 0.0, 0
    df_t = pd.DataFrame(trades)
    return df_t['net_return'].sum(), df_t['is_win'].mean(), 0, (df_t['net_return'].cumsum() - df_t['net_return'].cumsum().cummax()).min(), len(df_t)


def run_s25_simulation(unique_15m_times, ticker_bar_map, dict_15m, is_last_bar_map,
                       sl_pct, tp_pct, ts_pct, max_hold, rank_15, rank_30, rank_1h):
    limit = 6
    trades = []
    active_trades = []
    daily_trade_count = {}

    for T in unique_15m_times:
        date_str = T[:10]
        if date_str not in daily_trade_count:
            daily_trade_count[date_str] = 0
            
        is_last_bar = is_last_bar_map[T]
        remaining_trades = []
        
        # EXITS
        for t in active_trades:
            t['bars_held'] += 1
            bar = ticker_bar_map[t['ticker']].get(T)
            if not bar:
                n_ret = -TRANSACTION_COST_PCT / 100
                trades.append({'side': t['side'], 'net_return': n_ret, 'is_win': n_ret > 0})
                continue
                
            if t['side'] == 'LONG':
                current_pnl = (bar['close'] / t['entry_price']) - 1.0
                t['peak_pnl'] = max(t['peak_pnl'], (bar['high'] / t['entry_price']) - 1.0)
            else:
                current_pnl = 1.0 - (bar['close'] / t['entry_price'])
                t['peak_pnl'] = max(t['peak_pnl'], 1.0 - (bar['low'] / t['entry_price']))

            exit_reason = None
            exit_price = bar['close']
            
            if sl_pct is not None:
                if t['side'] == 'LONG' and bar['low'] <= t['entry_price'] * (1.0 - sl_pct):
                    exit_reason, exit_price = 'STOP_LOSS', t['entry_price'] * (1.0 - sl_pct)
                elif t['side'] == 'SHORT' and bar['high'] >= t['entry_price'] * (1.0 + sl_pct):
                    exit_reason, exit_price = 'STOP_LOSS', t['entry_price'] * (1.0 + sl_pct)
                    
            if exit_reason is None and tp_pct is not None:
                if t['side'] == 'LONG' and bar['high'] >= t['entry_price'] * (1.0 + tp_pct):
                    exit_reason, exit_price = 'TAKE_PROFIT', t['entry_price'] * (1.0 + tp_pct)
                elif t['side'] == 'SHORT' and bar['low'] <= t['entry_price'] * (1.0 - tp_pct):
                    exit_reason, exit_price = 'TAKE_PROFIT', t['entry_price'] * (1.0 - tp_pct)
                    
            if exit_reason is None and ts_pct is not None:
                if t['peak_pnl'] - current_pnl >= ts_pct:
                    exit_reason = 'TRAILING_STOP'
                    if t['side'] == 'LONG':
                        exit_price = t['entry_price'] * (1.0 + (t['peak_pnl'] - ts_pct))
                    else:
                        exit_price = t['entry_price'] * (1.0 - (t['peak_pnl'] - ts_pct))
                    
            if exit_reason is None and t['bars_held'] >= max_hold:
                exit_reason = 'TIME_EXPIRY'
            if exit_reason is None and is_last_bar:
                exit_reason = 'FORCE_CLOSE_EOD'

            if exit_reason:
                g_ret = (exit_price / t['entry_price'] - 1.0) if t['side'] == 'LONG' else (1.0 - exit_price / t['entry_price'])
                n_ret = g_ret - TRANSACTION_COST_PCT / 100
                trades.append({'side': t['side'], 'net_return': n_ret, 'is_win': n_ret > 0})
            else:
                remaining_trades.append(t)
        active_trades = remaining_trades
        
        # ENTRIES
        if daily_trade_count[date_str] >= limit:
            continue
            
        active_tickers = {x['ticker'] for x in active_trades}
        
        # LONG
        for x in dict_15m[T]['longs']:
            if x['long_rank'] > rank_15: break
            if x['m30_long_rank'] <= rank_30 and x['h1_long_rank'] <= rank_1h:
                if x['long_conv'] > x['long_conv_lag1'] and x['m30_long_conv'] > x['m30_long_conv_lag1'] and x['h1_long_conv'] > x['h1_long_conv_lag1']:
                    if x['ticker'] not in active_tickers:
                        active_trades.append({'ticker': x['ticker'], 'side': 'LONG', 'entry_price': x['close'], 'bars_held': 0, 'peak_pnl': 0.0})
                        active_tickers.add(x['ticker'])
                        daily_trade_count[date_str] += 1
                        if daily_trade_count[date_str] >= limit: break

        if daily_trade_count[date_str] >= limit:
            continue
            
        # SHORT
        for x in dict_15m[T]['shorts']:
            if x['short_rank'] > rank_15: break
            if x['m30_short_rank'] <= rank_30 and x['h1_short_rank'] <= rank_1h:
                if x['short_conv'] > x['short_conv_lag1'] and x['m30_short_conv'] > x['m30_short_conv_lag1'] and x['h1_short_conv'] > x['h1_short_conv_lag1']:
                    if x['ticker'] not in active_tickers:
                        active_trades.append({'ticker': x['ticker'], 'side': 'SHORT', 'entry_price': x['close'], 'bars_held': 0, 'peak_pnl': 0.0})
                        active_tickers.add(x['ticker'])
                        daily_trade_count[date_str] += 1
                        if daily_trade_count[date_str] >= limit: break

    if not trades: return 0.0, 0.0, 0.0, 0.0, 0
    df_t = pd.DataFrame(trades)
    return df_t['net_return'].sum(), df_t['is_win'].mean(), 0, (df_t['net_return'].cumsum() - df_t['net_return'].cumsum().cummax()).min(), len(df_t)


def main():
    print("=" * 80)
    print("STRATEGY 24 & 25 OPTIMIZATION SWEEP")
    print("=" * 80)

    # Loads...
    df_daily = predict_timeframe_scores(
        load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", ["2026-04", "2026-05"]), "Daily",
        "models/daily_xgb/metadata.json", "models/daily_xgb/xgb_long_model.json", "models/daily_xgb/xgb_short_model.json", "models/daily_xgb/scaler.pkl"
    )
    df_1h = predict_timeframe_scores(
        load_and_filter_csv("data/ranking_data_upstox_3y.csv", [TEST_MONTH]), "1H",
        "models/v8_upstox_3y/metadata.json", "models/v8_upstox_3y/xgb_long_model.json", "models/v8_upstox_3y/xgb_short_model.json", "models/scaler.pkl"
    )
    df_30m = predict_timeframe_scores(
        load_and_filter_csv("data/ranking_data_upstox_30min_1y.csv", [TEST_MONTH]), "30M",
        "models/v1_30min/metadata.json", "models/v1_30min/xgb_long_model.json", "models/v1_30min/xgb_short_model.json", "models/v1_30min/scaler.pkl"
    )
    df_15m = predict_timeframe_scores(
        load_and_filter_csv("data/ranking_data_upstox_15min_1y.csv", [TEST_MONTH]), "15M",
        "models/v1_15min/metadata.json", "models/v1_15min/xgb_long_model.json", "models/v1_15min/xgb_short_model.json", "models/v1_15min/scaler.pkl"
    )

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

    dict_daily, dict_1h, dict_30m = {}, {}, {}
    for date_str, group in df_daily.groupby(df_daily['DateTime'].str[:10]):
        dict_daily[date_str] = {row['Ticker']: {'long_rank': int(row['long_rank']), 'short_rank': int(row['short_rank']), 'long_conv': float(row['long_conv']), 'short_conv': float(row['short_conv'])} for _, row in group.iterrows()}
    for dt, group in df_1h.groupby('DateTime'):
        dict_1h[dt] = {row['Ticker']: {'long_rank': int(row['long_rank']), 'short_rank': int(row['short_rank']), 'long_conv': float(row['long_conv']), 'short_conv': float(row['short_conv']), 'long_conv_lag1': float(row['long_conv_lag1']) if pd.notna(row['long_conv_lag1']) else 0.0, 'short_conv_lag1': float(row['short_conv_lag1']) if pd.notna(row['short_conv_lag1']) else 0.0} for _, row in group.iterrows()}
    for dt, group in df_30m.groupby('DateTime'):
        dict_30m[dt] = {row['Ticker']: {'long_rank': int(row['long_rank']), 'short_rank': int(row['short_rank']), 'long_conv': float(row['long_conv']), 'short_conv': float(row['short_conv']), 'long_conv_lag1': float(row['long_conv_lag1']) if pd.notna(row['long_conv_lag1']) else 0.0, 'short_conv_lag1': float(row['short_conv_lag1']) if pd.notna(row['short_conv_lag1']) else 0.0} for _, row in group.iterrows()}

    # Calculate blended scores directly on 15m
    df_15m['Date'] = df_15m['DateTime'].str[:10]
    unique_dates = df_15m['Date'].unique()
    date_to_yesterday = {d: find_yesterday_daily_date(d, sorted_daily_dates) for d in unique_dates}

    aligned_d_l, aligned_d_s = [], []
    aligned_1h_l, aligned_1h_s, aligned_1h_l_lag, aligned_1h_s_lag, aligned_1h_lr, aligned_1h_sr = [], [], [], [], [], []
    aligned_30m_l, aligned_30m_s, aligned_30m_l_lag, aligned_30m_s_lag, aligned_30m_lr, aligned_30m_sr = [], [], [], [], [], []

    for t, dt, d in zip(df_15m['Ticker'], df_15m['DateTime'], df_15m['Date']):
        y_date = date_to_yesterday.get(d)
        d_val = dict_daily.get(y_date, {}).get(t) if y_date else None
        aligned_d_l.append(d_val['long_conv'] if d_val else 0.0)
        aligned_d_s.append(d_val['short_conv'] if d_val else 0.0)
        
        t_1h, t_30m = align_timeframes(dt)
        h_val = dict_1h.get(t_1h, {}).get(t) if t_1h else None
        if h_val:
            aligned_1h_l.append(h_val['long_conv']); aligned_1h_s.append(h_val['short_conv'])
            aligned_1h_l_lag.append(h_val['long_conv_lag1']); aligned_1h_s_lag.append(h_val['short_conv_lag1'])
            aligned_1h_lr.append(h_val['long_rank']); aligned_1h_sr.append(h_val['short_rank'])
        else:
            aligned_1h_l.append(0.0); aligned_1h_s.append(0.0); aligned_1h_l_lag.append(0.0); aligned_1h_s_lag.append(0.0); aligned_1h_lr.append(999); aligned_1h_sr.append(999)
            
        m_val = dict_30m.get(t_30m, {}).get(t) if t_30m else None
        if m_val:
            aligned_30m_l.append(m_val['long_conv']); aligned_30m_s.append(m_val['short_conv'])
            aligned_30m_l_lag.append(m_val['long_conv_lag1']); aligned_30m_s_lag.append(m_val['short_conv_lag1'])
            aligned_30m_lr.append(m_val['long_rank']); aligned_30m_sr.append(m_val['short_rank'])
        else:
            aligned_30m_l.append(0.0); aligned_30m_s.append(0.0); aligned_30m_l_lag.append(0.0); aligned_30m_s_lag.append(0.0); aligned_30m_lr.append(999); aligned_30m_sr.append(999)

    df_15m['blend_long_score'] = 0.1 * pd.Series(aligned_d_l) + 0.2 * pd.Series(aligned_1h_l) + 0.3 * pd.Series(aligned_30m_l) + 0.5 * df_15m['long_conv']
    df_15m['blend_short_score'] = 0.1 * pd.Series(aligned_d_s) + 0.2 * pd.Series(aligned_1h_s) + 0.3 * pd.Series(aligned_30m_s) + 0.5 * df_15m['short_conv']
    df_15m['blend_long_rank'] = df_15m.groupby('DateTime')['blend_long_score'].rank(ascending=False)
    df_15m['blend_short_rank'] = df_15m.groupby('DateTime')['blend_short_score'].rank(ascending=False)
    
    df_15m['m30_long_rank'] = aligned_30m_lr
    df_15m['m30_short_rank'] = aligned_30m_sr
    df_15m['h1_long_rank'] = aligned_1h_lr
    df_15m['h1_short_rank'] = aligned_1h_sr
    
    df_15m['m30_long_conv'] = aligned_30m_l
    df_15m['m30_short_conv'] = aligned_30m_s
    df_15m['m30_long_conv_lag1'] = aligned_30m_l_lag
    df_15m['m30_short_conv_lag1'] = aligned_30m_s_lag
    
    df_15m['h1_long_conv'] = aligned_1h_l
    df_15m['h1_short_conv'] = aligned_1h_s
    df_15m['h1_long_conv_lag1'] = aligned_1h_l_lag
    df_15m['h1_short_conv_lag1'] = aligned_1h_s_lag
    
    df_15m['long_conv_lag1'] = df_15m.groupby('Ticker')['long_conv'].shift(1)
    df_15m['short_conv_lag1'] = df_15m.groupby('Ticker')['short_conv'].shift(1)

    ticker_bar_map = {t: {} for t in common_tickers}
    dict_15m_struct = {}
    
    for dt, group in df_15m.groupby('DateTime'):
        bars = []
        for _, row in group.iterrows():
            b = {
                'ticker': row['Ticker'], 'close': float(row['Close']), 'high': float(row['High']), 'low': float(row['Low']),
                'blend_long_rank': int(row['blend_long_rank']), 'blend_short_rank': int(row['blend_short_rank']),
                'long_rank': int(row['long_rank']), 'short_rank': int(row['short_rank']),
                'm30_long_rank': int(row['m30_long_rank']), 'm30_short_rank': int(row['m30_short_rank']),
                'h1_long_rank': int(row['h1_long_rank']), 'h1_short_rank': int(row['h1_short_rank']),
                'long_conv': float(row['long_conv']), 'short_conv': float(row['short_conv']),
                'long_conv_lag1': float(row['long_conv_lag1']) if pd.notna(row['long_conv_lag1']) else 0.0,
                'short_conv_lag1': float(row['short_conv_lag1']) if pd.notna(row['short_conv_lag1']) else 0.0,
                'm30_long_conv': float(row['m30_long_conv']), 'm30_short_conv': float(row['m30_short_conv']),
                'm30_long_conv_lag1': float(row['m30_long_conv_lag1']), 'm30_short_conv_lag1': float(row['m30_short_conv_lag1']),
                'h1_long_conv': float(row['h1_long_conv']), 'h1_short_conv': float(row['h1_short_conv']),
                'h1_long_conv_lag1': float(row['h1_long_conv_lag1']), 'h1_short_conv_lag1': float(row['h1_short_conv_lag1']),
            }
            bars.append(b)
            ticker_bar_map[row['Ticker']][dt] = b
            
        dict_15m_struct[dt] = {
            'longs_blend': sorted(bars, key=lambda x: x['blend_long_rank']),
            'shorts_blend': sorted(bars, key=lambda x: x['blend_short_rank']),
            'longs': sorted(bars, key=lambda x: x['long_rank']),
            'shorts': sorted(bars, key=lambda x: x['short_rank']),
        }

    unique_15m_times = sorted(df_15m['DateTime'].unique())
    is_last_bar_map = {T: ((idx == len(unique_15m_times) - 1) or (unique_15m_times[idx + 1][:10] != T[:10]) or (T[11:16] == "15:15")) for idx, T in enumerate(unique_15m_times)}

    # Sweep Parameters
    sl_options = [None, 0.003, 0.005]
    tp_options = [None, 0.005, 0.010, 0.015]
    ts_options = [None, 0.003, 0.005]
    max_hold_options = [4, 6, 8, 12]
    
    # S24 specific
    rblend_options = [2, 3, 5, 8]
    
    # S25 specific
    r15_options = [3, 5]
    r30_options = [3, 5, 8]
    r1h_options = [5, 10]

    # Run S24
    print("\nRunning S24 Grid Search...")
    s24_grid = list(itertools.product(sl_options, tp_options, ts_options, max_hold_options, rblend_options))
    s24_res = []
    for sl, tp, ts, hold, rb in tqdm(s24_grid):
        ret, wr, pf, dd, trades = run_s24_simulation(unique_15m_times, ticker_bar_map, dict_15m_struct, is_last_bar_map, sl, tp, ts, hold, rb)
        s24_res.append({'sl': sl, 'tp': tp, 'ts': ts, 'hold': hold, 'rb': rb, 'ret': ret, 'wr': wr, 'dd': dd, 'trades': trades})
        
    s24_df = pd.DataFrame(s24_res).sort_values('ret', ascending=False)
    s24_df.to_csv("data/s24_opt.csv", index=False)
    
    print("\nTop 10 S24 Results:")
    print(s24_df.head(10).to_string())

    # Run S25
    print("\nRunning S25 Grid Search...")
    s25_grid = list(itertools.product(sl_options, tp_options, ts_options, max_hold_options, r15_options, r30_options, r1h_options))
    s25_res = []
    for sl, tp, ts, hold, r15, r30, r1h in tqdm(s25_grid):
        ret, wr, pf, dd, trades = run_s25_simulation(unique_15m_times, ticker_bar_map, dict_15m_struct, is_last_bar_map, sl, tp, ts, hold, r15, r30, r1h)
        s25_res.append({'sl': sl, 'tp': tp, 'ts': ts, 'hold': hold, 'r15': r15, 'r30': r30, 'r1h': r1h, 'ret': ret, 'wr': wr, 'dd': dd, 'trades': trades})
        
    s25_df = pd.DataFrame(s25_res).sort_values('ret', ascending=False)
    s25_df.to_csv("data/s25_opt.csv", index=False)
    
    print("\nTop 10 S25 Results:")
    print(s25_df.head(10).to_string())

if __name__ == "__main__":
    main()
