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
            
        raw = pd.read_csv(fp, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        dt = pd.to_datetime(raw['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        df = pd.DataFrame({
            'dt': dt,
            'o': raw['open'].astype(float),
            'h': raw['high'].astype(float),
            'l': raw['low'].astype(float),
            'c': raw['close'].astype(float),
            'v': raw['volume'].astype(float)
        }).dropna()
        
        df = df.drop_duplicates('dt').sort_values('dt')
        df = df[(df['h'] >= df['l']) & (df['v'] >= 0)]
        df['date'] = df['dt'].dt.date
        df['hm'] = df['dt'].dt.strftime('%H:%M')
        
        def at(hm, field):
            return df[df['hm'] == hm].set_index('date')[field]
            
        dclose = df.groupby('date')['c'].last()
        open0915 = at('09:15', 'o')
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
    t['prev_close'] = t.groupby('ticker')['dclose'].shift(1)
    t['gap'] = t['open0915'] / t['prev_close'] - 1.0
    t = t.dropna(subset=['gap', 'open0915'])
    return t

def run_backtest_flex(df, split_name, cost_bps=6.0, randomize=False, seed=None):
    df_filtered = df[df['gap'].abs() <= 0.03].copy()
    dates = sorted(df_filtered['date'].unique())
    
    trade_records = []
    daily_book_returns = []
    skipped_days = 0
    total_days = 0
    
    rng = np.random.RandomState(seed) if seed is not None else None
    
    for dt in dates:
        day_q = df_filtered[df_filtered['date'] == dt]
        total_days += 1
        
        if len(day_q) < 60:
            skipped_days += 1
            continue
            
        if randomize:
            # Randomly sample 10 tickers
            sampled = day_q.sample(10, random_state=rng)
            longs = sampled.iloc[:5]
            shorts = sampled.iloc[5:]
        else:
            day_q = day_q.sort_values('gap')
            longs = day_q.iloc[:5]
            shorts = day_q.iloc[-5:]
            
        day_longs_pnls = []
        day_shorts_pnls = []
        
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
    if trades_df.empty:
        return 0.0, 0.0
        
    gross_pnls_bps = trades_df['raw_pnl'].values * 10000.0
    net_pnls_bps = trades_df['net_pnl'].values * 10000.0
    
    return np.mean(gross_pnls_bps), np.mean(net_pnls_bps)

def main():
    t = load_data()
    t['date_dt'] = pd.to_datetime(t['date'])
    holdout_df = t[(t['date_dt'] >= '2026-01-01') & (t['date_dt'] <= '2026-06-30')].copy()
    
    print("\n--- STRESS TESTING TRANSACTION COSTS ON HOLDOUT SET ---")
    costs = [0.0, 6.0, 10.0, 15.0]
    for cost in costs:
        gross_ev, net_ev = run_backtest_flex(holdout_df, "Holdout", cost_bps=cost)
        print(f"Cost: {cost:4.1f} bps | Gross EV: {gross_ev:.4f} bps | Net EV: {net_ev:.4f} bps")
        
    # Determine break-even cost where Net EV turns negative
    gross_ev, _ = run_backtest_flex(holdout_df, "Holdout", cost_bps=0.0)
    print(f"Analytical Break-even cost: {gross_ev:.4f} bps")
    
    # Confirm break-even cost empirically
    step = 0.01
    found_break_even = None
    for cost in np.arange(0.0, 30.0, step):
        _, net_ev = run_backtest_flex(holdout_df, "Holdout", cost_bps=cost)
        if net_ev <= 0:
            found_break_even = cost
            break
    print(f"Empirical Break-even cost (precision {step} bps): {found_break_even:.4f} bps")
    
    print("\n--- RUNNING RANDOMIZED NEGATIVE CONTROL (15 SEEDS) ---")
    seeds = list(range(42, 42 + 15))
    neg_control_evs = []
    for s in seeds:
        # At 6.0 bps cost
        _, net_ev = run_backtest_flex(holdout_df, "Holdout", cost_bps=6.0, randomize=True, seed=s)
        neg_control_evs.append(net_ev)
        print(f"Seed: {s:3d} | Net EV (6bps): {net_ev:.4f} bps")
        
    mean_neg_control = np.mean(neg_control_evs)
    std_neg_control = np.std(neg_control_evs, ddof=1)
    print(f"\nRandomized Control Summary (Holdout at 6.0 bps cost, 15 runs):")
    print(f"  Mean Net EV: {mean_neg_control:.4f} bps")
    print(f"  Std Dev:     {std_neg_control:.4f} bps")
    print(f"  Min Net EV:  {np.min(neg_control_evs):.4f} bps")
    print(f"  Max Net EV:  {np.max(neg_control_evs):.4f} bps")

if __name__ == '__main__':
    main()
