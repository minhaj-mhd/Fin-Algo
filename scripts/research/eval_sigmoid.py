import pandas as pd
import numpy as np
import xgboost as xgb

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run():
    print("Loading data...")
    df = pd.read_parquet('data/research/v20_rolling_1h/panel.parquet')
    df['YearMonth'] = pd.to_datetime(df['DateTime']).dt.strftime('%Y-%m')
    unique_months = sorted(df['YearMonth'].unique())
    # Isolate Proxy OOS Month: 2026-06
    proxy_oos = df['YearMonth'] == '2026-06'
    dfte = df[proxy_oos].copy()
    print(f"Proxy OOS Month (2026-06) total rows: {len(dfte)}")
    
    # Get v23 features
    feats = [
        'Dist_Keltner_Lower', 'Relative_Return', 'Keltner_Width', 'Return', 
        'CMF_20', 'Log_Return', 'PPO_Signal', 'Dist_52W_Low', 'RVOL', 
        'Lower_Shadow', 'Dollar_Volume', 'Intraday_Return', 'Dist_HMA_12', 
        'Hour', 'TRIX_15', 'Dist_Donchian_Upper', 'Rolling_Skew', 'IBS', 
        'Donchian_Width', 'VWAP_Dist'
    ]
    Xte = dfte[feats].values.astype(np.float64)
    dte = xgb.DMatrix(Xte)
    
    print("Loading v23 models...")
    bl = xgb.Booster()
    bl.load_model('models/research/v23_rolling_1h/xgb_long_model.json')
    bs = xgb.Booster()
    bs.load_model('models/research/v23_rolling_1h/xgb_short_model.json')
    
    long_raw = bl.predict(dte)
    short_raw = bs.predict(dte)
    
    dfte['long_prob'] = sigmoid(long_raw)
    dfte['short_prob'] = sigmoid(short_raw)
    
    # Filter 70% threshold
    long_trades = dfte[dfte['long_prob'] > 0.7]
    short_trades = dfte[dfte['short_prob'] > 0.7]
    
    print(f"Total Test Rows: {len(dfte)}")
    
    print("\n--- Longs > 0.7 ---")
    print(f"Count: {len(long_trades)}")
    if len(long_trades) > 0:
        lwr = (long_trades['Next_Hour_Return'] > 0).mean()
        ledge = long_trades['Next_Hour_Return'].mean() * 10000
        print(f"Win Rate (>0): {lwr:.1%} | Edge: {ledge:+.2f} bps")
        
    print("\n--- Shorts > 0.7 ---")
    print(f"Count: {len(short_trades)}")
    if len(short_trades) > 0:
        swr = (short_trades['Next_Hour_Return'] < 0).mean()
        sedge = -short_trades['Next_Hour_Return'].mean() * 10000
        print(f"Win Rate (<0): {swr:.1%} | Edge: {sedge:+.2f} bps")

    print(f"Long Raw Score Range: {long_raw.min():.4f} to {long_raw.max():.4f}")
    print(f"Short Raw Score Range: {short_raw.min():.4f} to {short_raw.max():.4f}")
    print(f"Long Prob Range: {dfte['long_prob'].min():.4f} to {dfte['long_prob'].max():.4f}")
    print(f"Short Prob Range: {dfte['short_prob'].min():.4f} to {dfte['short_prob'].max():.4f}")
    
    # Let's check thresholds 0.51, 0.52, 0.53, 0.54
    for thresh in [0.51, 0.52, 0.53, 0.54]:
        print(f"\n--- Threshold {thresh} ---")
        l_tr = dfte[dfte['long_prob'] > thresh]
        s_tr = dfte[dfte['short_prob'] > thresh]
        print(f"Longs Count: {len(l_tr)}")
        if len(l_tr) > 0:
             print(f"  Long WR: {(l_tr['Next_Hour_Return']>0).mean():.1%} | Edge: {l_tr['Next_Hour_Return'].mean()*10000:+.2f} bps")
        print(f"Shorts Count: {len(s_tr)}")
        if len(s_tr) > 0:
             print(f"  Short WR: {(s_tr['Next_Hour_Return']<0).mean():.1%} | Edge: {-s_tr['Next_Hour_Return'].mean()*10000:+.2f} bps")

if __name__ == '__main__':
    run()
