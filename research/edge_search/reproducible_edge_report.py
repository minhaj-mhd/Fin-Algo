"""
Self-contained, reproducible Python script to evaluate the "Open GAP-FADE" strategy,
split the data into Development and Holdout sets, and verify the Expected Value (EV) of the edge.

Objective:
- Universe: 110-name liquidity universe (universe.json).
- Signals: At 09:15 open: gap = open0915 / prev_day_close - 1.
- Filter: |gap| <= 3%, skip days with < 60 valid tickers.
- Book construction: SHORT top-5 largest gap-ups, LONG bottom-5 largest gap-downs (equal-weight 50/50).
- Entry price: open0915.
- Exit price: Price at 09:30 (x_0930, close of 09:25 bar or open of 09:30 bar).
- Transaction cost: Subtract 6.0 bps flat round-trip per trade.
- Split: Dev (2023-01-01 to 2025-12-31), Holdout (2026-01-01 to 2026-06-30).
"""

import os
import sys
import glob
import json
import warnings
import datetime
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Set working directory to project root if running from elsewhere
# but assume standard execution is from c:\Users\loq\Desktop\Trading\finalgo
PROJECT_ROOT = r"c:\Users\loq\Desktop\Trading\finalgo"
if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)

UNIV_JSON = os.path.join("data", "research", "v21_rolling_1h", "universe.json")
SRC_DIR = os.path.join("data", "raw_upstox_cache_5min_v3")


def load_data():
    print(f"Loading universe from {UNIV_JSON}...")
    with open(UNIV_JSON) as f:
        univ = set(json.load(f)['tickers'])
        
    print(f"Scanning 5-min cache files in {SRC_DIR}...")
    csv_files = sorted(glob.glob(os.path.join(SRC_DIR, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV cache files found in {SRC_DIR}. Please check path.")
        
    rows = []
    for fp in csv_files:
        tk = os.path.basename(fp)[:-4]
        if tk not in univ:
            continue
            
        # Load necessary columns
        raw = pd.read_csv(fp, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to timezone-naive local IST
        dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        df = pd.DataFrame({
            'dt': dt,
            'o': raw['open'].astype(float),
            'h': raw['high'].astype(float),
            'l': raw['low'].astype(float),
            'c': raw['close'].astype(float),
            'v': raw['volume'].astype(float)
        }).dropna()
        
        # Sort and deduplicate
        df = df.drop_duplicates('dt').sort_values('dt')
        
        # Hygiene checks
        df = df[(df['h'] >= df['l']) & (df['v'] >= 0)]
        
        df['date'] = df['dt'].dt.date
        df['hm'] = df['dt'].dt.strftime('%H:%M')
        
        def at(hm, field):
            return df[df['hm'] == hm].set_index('date')[field]
            
        # Last close of the day
        dclose = df.groupby('date')['c'].last()
        
        # Open price at 09:15
        open0915 = at('09:15', 'o')
        
        # Exit at 09:30: close of 09:25 bar or open of 09:30 bar as fallback
        c_0925 = at('09:25', 'c')
        o_0930 = at('09:30', 'o')
        x_0930 = c_0925.combine_first(o_0930)
        
        w = pd.DataFrame({
            'open0915': open0915,
            'x_0930': x_0930,
            'dclose': dclose
        })
        
        w['ticker'] = tk
        rows.append(w.reset_index())
        
    t = pd.concat(rows, ignore_index=True)
    t = t.sort_values(['ticker', 'date'])
    
    # Calculate prev close and gaps
    t['prev_close'] = t.groupby('ticker')['dclose'].shift(1)
    t['gap'] = t['open0915'] / t['prev_close'] - 1.0
    
    # Drop rows without valid signal/entry
    t = t.dropna(subset=['gap', 'open0915'])
    return t


def run_backtest(df, split_name, cost_bps=6.0):
    # Filter: |gap| <= 3%
    df_filtered = df[df['gap'].abs() <= 0.03].copy()
    
    dates = sorted(df_filtered['date'].unique())
    
    trade_records = []
    daily_book_returns = []
    skipped_days = 0
    total_days = 0
    
    for dt in dates:
        day_q = df_filtered[df_filtered['date'] == dt]
        total_days += 1
        
        # Skip days with fewer than 60 valid tickers
        if len(day_q) < 60:
            skipped_days += 1
            continue
            
        # Sort tickers by gap ascending
        day_q = day_q.sort_values('gap')
        
        # LONG: bottom-5 largest gap-downs (most negative gaps)
        longs = day_q.iloc[:5]
        
        # SHORT: top-5 largest gap-ups (largest positive gaps)
        shorts = day_q.iloc[-5:]
        
        day_longs_pnls = []
        day_shorts_pnls = []
        
        # Process shorts
        for _, r in shorts.iterrows():
            e = r['open0915']
            x = r['x_0930']
            tk = r['ticker']
            gap_val = r['gap']
            
            if not np.isfinite(e) or e <= 0 or not np.isfinite(x) or x <= 0:
                continue
                
            raw_pnl = 1.0 - (x / e)
            net_pnl = raw_pnl - (cost_bps / 10000.0)
            
            trade_records.append({
                'date': dt,
                'ticker': tk,
                'side': 'SHORT',
                'gap': gap_val,
                'entry': e,
                'exit': x,
                'raw_pnl': raw_pnl,
                'net_pnl': net_pnl
            })
            day_shorts_pnls.append(net_pnl)
            
        # Process longs
        for _, r in longs.iterrows():
            e = r['open0915']
            x = r['x_0930']
            tk = r['ticker']
            gap_val = r['gap']
            
            if not np.isfinite(e) or e <= 0 or not np.isfinite(x) or x <= 0:
                continue
                
            raw_pnl = (x / e) - 1.0
            net_pnl = raw_pnl - (cost_bps / 10000.0)
            
            trade_records.append({
                'date': dt,
                'ticker': tk,
                'side': 'LONG',
                'gap': gap_val,
                'entry': e,
                'exit': x,
                'raw_pnl': raw_pnl,
                'net_pnl': net_pnl
            })
            day_longs_pnls.append(net_pnl)
            
        # Book construction (50/50 capital allocation)
        long_leg_ret = np.sum(day_longs_pnls) / 5.0 if day_longs_pnls else 0.0
        short_leg_ret = np.sum(day_shorts_pnls) / 5.0 if day_shorts_pnls else 0.0
        book_ret = 0.5 * long_leg_ret + 0.5 * short_leg_ret
        
        daily_book_returns.append({
            'date': dt,
            'long_ret': long_leg_ret,
            'short_ret': short_leg_ret,
            'book_ret': book_ret
        })
        
    trades_df = pd.DataFrame(trade_records)
    book_df = pd.DataFrame(daily_book_returns)
    
    if trades_df.empty:
        print(f"No trades executed for {split_name}")
        return 0.0
        
    # Calculate trade-level metrics
    total_trades = len(trades_df)
    net_pnls_bps = trades_df['net_pnl'].values * 10000.0
    gross_pnls_bps = trades_df['raw_pnl'].values * 10000.0
    
    win_rate = np.mean(net_pnls_bps > 0)
    gross_ret_mean = np.mean(gross_pnls_bps)
    net_ret_mean = np.mean(net_pnls_bps)
    
    trade_std = np.std(net_pnls_bps, ddof=1)
    trade_t_stat = net_ret_mean / (trade_std / np.sqrt(total_trades)) if trade_std > 0 else np.nan
    
    # Calculate book-level metrics (daily aggregation)
    total_days = len(book_df)
    book_net_pnls_bps = book_df['book_ret'].values * 10000.0
    book_net_mean = np.mean(book_net_pnls_bps)
    book_std = np.std(book_net_pnls_bps, ddof=1)
    book_t_stat = book_net_mean / (book_std / np.sqrt(total_days)) if book_std > 0 else np.nan
    book_sharpe = (book_net_mean / book_std * np.sqrt(247)) if book_std > 0 else np.nan
    
    # Print output formatted for user/auditor
    print(f"\n==================================================")
    print(f" RESULTS FOR {split_name.upper()} SET")
    print(f"==================================================")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total Days in period: {total_days}")
    print(f"Skipped Days (< 60 valid tickers): {skipped_days}")
    print(f"Executed Days: {total_days - skipped_days}")
    print(f"Total Trades (across all days): {total_trades}")
    print(f"Win Rate (percentage of trades with PnL > 0): {win_rate * 100.0:.2f}%")
    print(f"Gross Return per trade (mean bps): {gross_ret_mean:.4f} bps")
    print(f"Net Return per trade (mean bps after 6bps cost) [EV]: {net_ret_mean:.4f} bps")
    print(f"t-statistic (on trade net returns): {trade_t_stat:.4f}")
    print(f"--------------------------------------------------")
    print(f"Daily Book-level Metrics:")
    print(f"  Mean Daily Book Return: {book_net_mean:.4f} bps/day")
    print(f"  t-statistic (book-level): {book_t_stat:.4f}")
    print(f"  Annualized Sharpe Ratio: {book_sharpe:.4f}")
    print(f"==================================================")
    
    return net_ret_mean


def main():
    print("Starting Open GAP-FADE Reproducible Edge Report...")
    t = load_data()
    print(f"Loaded {len(t):,} rows across {t['ticker'].nunique()} tickers and {t['date'].nunique()} days.")
    
    # Convert dates for splitting
    t['date_dt'] = pd.to_datetime(t['date'])
    
    # Splits
    dev_df = t[(t['date_dt'] >= '2023-01-01') & (t['date_dt'] <= '2025-12-31')].copy()
    holdout_df = t[(t['date_dt'] >= '2026-01-01') & (t['date_dt'] <= '2026-06-30')].copy()
    
    print(f"Development Set range: {dev_df['date'].min()} to {dev_df['date'].max()} ({len(dev_df)} records)")
    print(f"Holdout Set range: {holdout_df['date'].min()} to {holdout_df['date'].max()} ({len(holdout_df)} records)")
    
    dev_ev = run_backtest(dev_df, "Development")
    holdout_ev = run_backtest(holdout_df, "Holdout")
    
    print(f"\nVerification:")
    print(f"  Development EV: {dev_ev:.4f} bps")
    print(f"  Holdout EV: {holdout_ev:.4f} bps")
    
    if holdout_ev > 0:
        print("SUCCESS: Expected Value (EV) on the Holdout set is strictly positive (> 0).")
    else:
        print("FAILURE: Expected Value (EV) on the Holdout set is NOT positive (<= 0).")
        sys.exit(1)


if __name__ == '__main__':
    main()
