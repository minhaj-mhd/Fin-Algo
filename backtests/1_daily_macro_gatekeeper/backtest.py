"""
backtest.py — Optimized Daily Macro Gatekeeper Strategy (Strategy 1)

Uses 2 XGBoost models:
1. Daily: models/daily_xgb/
2. 15-Min: models/v1_15min/

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

# Add project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ========================================
# CONFIG
# ========================================
TEST_MONTH = "2026-05"
TRANSACTION_COST_PCT = 0.06  # 0.06% round-trip (including slippage)

# Optimization Parameters
MAX_HOLD_BARS = 3
TAKE_PROFIT_PCT = 0.0120
STOP_LOSS_PCT = -0.0050
DAILY_GATEKEEPER_PCT = 0.40

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
    
    missing_cols = [c for c in feature_cols if c not in df_tf.columns]
    if missing_cols:
        print(f"  [WARN] Missing features in dataset: {missing_cols}. Filling with 0.")
        for col in missing_cols:
            df_tf[col] = 0.0
            
    bst_long = xgb.Booster()
    bst_long.load_model(long_model_path)
    bst_long.set_param({'device': 'cpu'})
    
    bst_short = xgb.Booster()
    bst_short.load_model(short_model_path)
    bst_short.set_param({'device': 'cpu'})
    
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, "rb") as sf:
            scaler = pickle.load(sf)
            
    X = df_tf[feature_cols].values
    X_clean = np.nan_to_num(X)
    
    if scaler is not None and hasattr(scaler, 'scale_') and scaler.scale_ is not None:
        X_final = scaler.transform(X_clean)
        print("  Scaler applied")
    else:
        X_final = X_clean
        
    dmat = xgb.DMatrix(X_final, feature_names=feature_cols)
    df_tf['long_score'] = bst_long.predict(dmat)
    df_tf['short_score'] = bst_short.predict(dmat)
    
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
# Helper: Evaluate Strategy Trades
# ========================================
def evaluate_strategy_trades(trades, name):
    if not trades:
        return {
            'strategy': name, 'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
            'long_win_rate': 0.0, 'short_win_rate': 0.0, 'long_trades': 0, 'short_trades': 0,
            'total_return': 0.0, 'avg_return': 0.0, 'avg_trades_per_day': 0.0, 'green_day_rate': 0.0,
            'profit_factor': 0.0, 'max_drawdown': 0.0, 'avg_bars_held': 0.0
        }
        
    df_t = pd.DataFrame(trades)
    
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
    
    profit_factor = abs(winners['net_return'].sum() / losers['net_return'].sum()) if len(losers) > 0 and losers['net_return'].sum() != 0 else float('inf')
    
    cum_returns = df_t['net_return'].cumsum()
    max_dd = (cum_returns - cum_returns.cummax()).min() if len(cum_returns) > 0 else 0.0
    
    avg_bars = df_t['bars_held'].mean() if 'bars_held' in df_t.columns else 1.0
    
    longs = df_t[df_t['side'] == 'LONG']
    shorts = df_t[df_t['side'] == 'SHORT']
    long_wr = longs['is_win'].mean() if len(longs) > 0 else 0.0
    short_wr = shorts['is_win'].mean() if len(shorts) > 0 else 0.0
    
    return {
        'strategy': name, 'total_trades': int(total_trades), 'wins': int(total_wins), 'losses': int(total_losses),
        'win_rate': float(overall_wr), 'long_win_rate': float(long_wr), 'short_win_rate': float(short_wr),
        'long_trades': int(len(longs)), 'short_trades': int(len(shorts)), 'total_return': float(total_net_pnl),
        'avg_return': float(avg_net_trade), 'avg_trades_per_day': float(avg_trades_day),
        'green_day_rate': float(green_day_rate), 'profit_factor': float(profit_factor),
        'max_drawdown': float(max_dd), 'avg_bars_held': float(avg_bars)
    }

# ========================================
# MAIN SIMULATION SCRIPT
# ========================================
def main():
    print("=" * 70)
    print("OPTIMIZED DAILY MACRO GATEKEEPER BACKTEST")
    print("=" * 70)

    # 1. LOAD DATASETS
    df_daily_raw = load_and_filter_csv("data/ranking_data_upstox_daily_5y.csv", ["2026-04", "2026-05"])
    df_15m_raw = load_and_filter_csv("data/ranking_data_upstox_15min_1y.csv", [TEST_MONTH])
    
    print("\nDatasets loaded:")
    print(f"  Daily: {df_daily_raw.shape[0]:,} rows")
    print(f"  15M:   {df_15m_raw.shape[0]:,} rows")
    
    # 2. RUN ML PREDICTIONS
    df_daily = predict_timeframe_scores(
        df_daily_raw, "Daily",
        "models/daily_xgb/metadata.json",
        "models/daily_xgb/xgb_long_model.json",
        "models/daily_xgb/xgb_short_model.json",
        "models/daily_xgb/scaler.pkl"
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
    t_15m = set(df_15m['Ticker'].unique())
    common_tickers = sorted(list(t_daily.intersection(t_15m)))
    print(f"\nTickers universe intersection: {len(common_tickers)} symbols")
    
    df_daily = df_daily[df_daily['Ticker'].isin(common_tickers)].copy()
    df_15m = df_15m[df_15m['Ticker'].isin(common_tickers)].copy()
    
    # 3. PRE-COMPUTE PERCENTILES ON 15M DATASET
    print("\nPre-computing conviction percentiles...")
    df_15m = df_15m.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)
    df_15m['Date'] = df_15m['DateTime'].str[:10]
    
    p60_long_series = df_15m.groupby('DateTime')['long_conv'].transform(lambda x: x.quantile(0.60))
    p60_short_series = df_15m.groupby('DateTime')['short_conv'].transform(lambda x: x.quantile(0.60))
    df_15m['p60_long'] = p60_long_series
    df_15m['p60_short'] = p60_short_series

    # 4. INDEX DICTIONARIES
    print("\nIndexing predictions...")
    sorted_daily_dates = sorted(df_daily['DateTime'].str[:10].unique())
    
    dict_daily = {}
    for date_str, group in df_daily.groupby(df_daily['DateTime'].str[:10]):
        dict_daily[date_str] = {}
        for _, row in group.iterrows():
            dict_daily[date_str][row['Ticker']] = {
                'long_rank': int(row['long_rank']),
                'short_rank': int(row['short_rank']),
                'count': len(group)
            }

    ticker_bars = {t: [] for t in common_tickers}
    ticker_bar_map = {t: {} for t in common_tickers}
    
    for idx, row in df_15m.iterrows():
        t = row['Ticker']
        if t not in ticker_bars: continue
        bar_dict = {
            'ticker': t, 'datetime': row['DateTime'], 'date': row['Date'],
            'open': float(row['Open']), 'high': float(row['High']),
            'low': float(row['Low']), 'close': float(row['Close']),
            'long_conv': float(row['long_conv']), 'short_conv': float(row['short_conv']),
            'long_rank': int(row['long_rank']), 'short_rank': int(row['short_rank']),
            'p60_long': float(row['p60_long']), 'p60_short': float(row['p60_short'])
        }
        ticker_bars[t].append(bar_dict)
        ticker_bar_map[t][row['DateTime']] = bar_dict
        
    unique_15m_times = sorted(df_15m['DateTime'].unique())
    
    dict_15m = {}
    for dt, group in df_15m.groupby('DateTime'):
        dict_15m[dt] = []
        for _, row in group.iterrows():
            t = row['Ticker']
            if t in ticker_bar_map and dt in ticker_bar_map[t]:
                dict_15m[dt].append(ticker_bar_map[t][dt])

    # =========================================================================
    # SIMULATION
    # =========================================================================
    print(f"\nSimulating Strategy: Daily Macro Gatekeeper (Optimized)...")
    trades = []
    active_trades = [] 
    daily_trade_count = {}
    max_daily = 6
    
    for bar_idx, T in enumerate(unique_15m_times):
        date_str = T[:10]
        time_str = T[11:16]
        h, m = map(int, time_str.split(':'))
        t_min = h * 60 + m
        
        if date_str not in daily_trade_count:
            daily_trade_count[date_str] = 0
            
        d_daily = find_yesterday_daily_date(date_str, sorted_daily_dates)
        is_last_bar = (bar_idx == len(unique_15m_times) - 1) or (unique_15m_times[bar_idx + 1][:10] != date_str) or (time_str == "15:15")
        
        # A. MANAGE ACTIVE TRADE EXITS
        remaining_trades = []
        for t in active_trades:
            t['bars_held'] += 1
            bar = ticker_bar_map[t['ticker']].get(T)
            
            if not bar:
                g_ret = 0.0
                n_ret = g_ret - TRANSACTION_COST_PCT / 100
                trades.append({
                    'date': date_str, 'entry_time': t['entry_time'], 'exit_time': T,
                    'ticker': t['ticker'], 'side': t['side'], 'entry_price': t['entry_price'],
                    'exit_price': t['entry_price'], 'bars_held': t['bars_held'], 'exit_reason': 'DATA_GAP',
                    'gross_return': g_ret, 'net_return': n_ret, 'is_win': n_ret > 0
                })
                continue
                
            # Current P&L calculation
            if t['side'] == 'LONG':
                current_pnl = (bar['close'] / t['entry_price']) - 1.0
                high_pnl = (bar['high'] / t['entry_price']) - 1.0
                low_pnl = (bar['low'] / t['entry_price']) - 1.0
            else:
                current_pnl = 1.0 - (bar['close'] / t['entry_price'])
                low_pnl = 1.0 - (bar['high'] / t['entry_price'])
                high_pnl = 1.0 - (bar['low'] / t['entry_price'])
                
            exit_reason = None
            
            # Stop Loss & Take Profit logic
            if low_pnl <= STOP_LOSS_PCT:
                exit_reason = 'STOP_LOSS'
            elif high_pnl >= TAKE_PROFIT_PCT:
                exit_reason = 'TAKE_PROFIT'
            elif t['bars_held'] >= MAX_HOLD_BARS:
                exit_reason = 'TIME_EXPIRY'
                
            if is_last_bar:
                exit_reason = 'FORCE_CLOSE_EOD'
                
            if exit_reason:
                exit_price = bar['close']
                if exit_reason == 'STOP_LOSS':
                    exit_price = t['entry_price'] * (1.0 + STOP_LOSS_PCT) if t['side'] == 'LONG' else t['entry_price'] * (1.0 - STOP_LOSS_PCT)
                elif exit_reason == 'TAKE_PROFIT':
                    exit_price = t['entry_price'] * (1.0 + TAKE_PROFIT_PCT) if t['side'] == 'LONG' else t['entry_price'] * (1.0 - TAKE_PROFIT_PCT)
                    
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

        # B. EVALUATE NEW ENTRIES
        if T not in dict_15m: continue
        active_15m = dict_15m[T]
        if len(active_15m) == 0: continue
            
        if not d_daily or d_daily not in dict_daily: continue
            
        top_15m_longs = sorted(active_15m, key=lambda x: x['long_rank'])
        top_15m_shorts = sorted(active_15m, key=lambda x: x['short_rank'])
        
        if 10 * 60 <= t_min <= 14 * 60 + 59:
            if daily_trade_count[date_str] < max_daily:
                long_cand = None
                for x in top_15m_longs:
                    if x['long_rank'] == 1:
                        ticker = x['ticker']
                        d_pred = dict_daily[d_daily].get(ticker)
                        # Stricter gatekeeper
                        if d_pred and d_pred['long_rank'] <= DAILY_GATEKEEPER_PCT * d_pred['count']:
                            if x['long_conv'] > x['p60_long']:
                                long_cand = x
                                break
                                
                short_cand = None
                for x in top_15m_shorts:
                    if x['short_rank'] == 1:
                        ticker = x['ticker']
                        d_pred = dict_daily[d_daily].get(ticker)
                        if d_pred and d_pred['short_rank'] <= DAILY_GATEKEEPER_PCT * d_pred['count']:
                            if x['short_conv'] > x['p60_short']:
                                short_cand = x
                                break
                                
                if long_cand and daily_trade_count[date_str] < max_daily:
                    active_trades.append({
                        'ticker': long_cand['ticker'], 'side': 'LONG', 'entry_price': long_cand['close'],
                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                    })
                    daily_trade_count[date_str] += 1
                if short_cand and daily_trade_count[date_str] < max_daily:
                    active_trades.append({
                        'ticker': short_cand['ticker'], 'side': 'SHORT', 'entry_price': short_cand['close'],
                        'entry_time': T, 'bars_held': 0, 'peak_pnl': 0.0
                    })
                    daily_trade_count[date_str] += 1
    
    # End of simulation
    eval_res = evaluate_strategy_trades(trades, "Daily Macro Gatekeeper (Optimized)")
    print(f"  Trades generated: {len(trades)} | Net P&L: {eval_res['total_return']*100:+.2f}%")
    
    # Save output
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), "results", "backtest_results.json")
    with open(output_path, "w") as f:
        json.dump({
            'holdout_month': TEST_MONTH,
            'tickers_universe_size': len(common_tickers),
            'strategy_summary': eval_res,
            'backtested_at': datetime.now().isoformat()
        }, f, indent=2)
        
    print(f"\n[SUCCESS] Saved backtest outcomes to {output_path}")
    
    # Print Summary
    print("\n" + "=" * 80)
    print("BACKTEST SIMULATION SUMMARY TABLE")
    print("=" * 80)
    headers = ["Strategy Name", "Trades", "WR %", "L-WR %", "S-WR %", "Return %", "Profit Factor", "Max DD %", "Avg Hold"]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-|-'.join(['-' * len(h) for h in headers])}|")
    
    res = eval_res
    s_name_trunc = res['strategy'][:30]
    pnl_str = f"{res['total_return']*100:+.2f}%"
    wr_str = f"{res['win_rate']*100:.1f}%"
    l_wr_str = f"{res['long_win_rate']*100:.1f}%" if res['long_trades'] > 0 else "N/A"
    s_wr_str = f"{res['short_win_rate']*100:.1f}%" if res['short_trades'] > 0 else "N/A"
    pf_str = f"{res['profit_factor']:.2f}" if res['profit_factor'] != float('inf') else "inf"
    dd_str = f"{res['max_drawdown']*100:.2f}%"
    print(f"| {s_name_trunc:<30} | {res['total_trades']:<6} | {wr_str:<6} | {l_wr_str:<6} | {s_wr_str:<6} | {pnl_str:<8} | {pf_str:<13} | {dd_str:<8} | {res['avg_bars_held']:<8.1f} |")
    print("=" * 80)

if __name__ == "__main__":
    main()
