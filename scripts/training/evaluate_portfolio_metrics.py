import pandas as pd
import numpy as np
import xgboost as xgb

# Configuration
DATA_FILE = 'data/ranking_data_upstox_1h_v3_3y.csv'
RET_COL = 'Next_Hour_Return'
COST = 0.0010
TRADE_PROB = 0.52

print("=" * 64)
print("PORTFOLIO METRICS BACKTEST: Logic A1 (Last 12 Months)")
print("=" * 64)

# 1. Load Data
df = pd.read_csv(DATA_FILE)
df['YearMonth'] = df['DateTime'].str[:7]
df_test = df[df['YearMonth'] >= '2025-07'].copy()

exclude_cols = ['DateTime', 'DateTime_15Min', 'DateTime_Hour', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', RET_COL, 'YearMonth']
feature_cols = [c for c in df_test.columns if c not in exclude_cols]

X_test = df_test[feature_cols].values.astype(np.float64)
y_returns = df_test[RET_COL].values.astype(np.float64)
query_ids = df_test['Query_ID'].values

# Handle NaNs
nan_mask = ~np.isfinite(X_test)
if nan_mask.any():
    for ci in range(X_test.shape[1]):
        bad = ~np.isfinite(X_test[:, ci])
        if bad.any():
            good = X_test[~bad, ci]
            X_test[bad, ci] = float(good.mean()) if len(good) else 0.0

dte = xgb.DMatrix(X_test)

# 2. Load Models
v10_l = xgb.Booster(); v10_l.load_model('models/v10_native_1h/xgb_long_model.json')
v10_s = xgb.Booster(); v10_s.load_model('models/v10_native_1h/xgb_short_model.json')
v18_l = xgb.Booster(); v18_l.load_model('models/v18_random_forest_1h/xgb_long_model.json')
v18_s = xgb.Booster(); v18_s.load_model('models/v18_random_forest_1h/xgb_short_model.json')

# 3. Predict
pred_l_v10 = v10_l.predict(dte)
pred_s_v10 = v10_s.predict(dte)
pred_l_v18 = v18_l.predict(dte)
pred_s_v18 = v18_s.predict(dte)

# 4. Collect All Trades for Logic A1
trades = []

for qid in np.unique(query_ids):
    m = query_ids == qid
    if m.sum() < 3: continue
    
    r_l_v10 = pred_l_v10[m]
    r_s_v10 = pred_s_v10[m]
    p_l_v18 = pred_l_v18[m]
    p_s_v18 = pred_s_v18[m]
    actual  = y_returns[m]
    
    q_df = df_test[m].reset_index(drop=True)

    # Top 1 Long
    top1_l_idx = np.argsort(r_l_v10)[-1]
    if p_l_v18[top1_l_idx] > TRADE_PROB:
        trades.append({
            'DateTime': q_df.iloc[top1_l_idx]['DateTime'],
            'Side': 'LONG',
            'Net_Return_bps': (actual[top1_l_idx] - COST) * 10000
        })
            
    # Top 1 Short
    top1_s_idx = np.argsort(r_s_v10)[-1]
    if p_s_v18[top1_s_idx] > TRADE_PROB:
        trades.append({
            'DateTime': q_df.iloc[top1_s_idx]['DateTime'],
            'Side': 'SHORT',
            'Net_Return_bps': (-actual[top1_s_idx] - COST) * 10000
        })

# 5. Calculate True Portfolio Metrics (Realistic Allocation)
trades_df = pd.DataFrame(trades)
trades_df['DateTime'] = pd.to_datetime(trades_df['DateTime'])

# Group by DateTime (concurrent signals)
portfolio_returns = []
dates = []

MAX_CASH_ALLOCATION_PER_TRADE = 0.20 # 20% of portfolio cash per trade
LEVERAGE = 1.0                       # 1x margin leverage

for dt, group in trades_df.groupby('DateTime'):
    n_trades = len(group)
    
    # Cap total cash allocation at 100% of portfolio
    cash_allocation = min(MAX_CASH_ALLOCATION_PER_TRADE, 1.0 / n_trades)
    
    # Effective exposure on the market is the cash allocation multiplied by leverage
    effective_exposure = cash_allocation * LEVERAGE
    
    # Net returns in basis points -> convert to decimal (100 bps = 0.01)
    # Note: Because the 10 bps fee is baked into Net_Return_bps, multiplying by 
    # effective_exposure correctly scales the margin costs as well!
    returns_decimal = group['Net_Return_bps'].values / 10000.0
    
    # Portfolio return for this hour is the sum of leveraged returns
    port_ret = np.sum(returns_decimal * effective_exposure)
    
    portfolio_returns.append(port_ret)
    dates.append(dt)

port_df = pd.DataFrame({'DateTime': dates, 'Port_Return': portfolio_returns})
port_df = port_df.sort_values('DateTime')

# Compounded Cumulative Return
# Portfolio grows by (1 + port_ret)
port_df['Cumulative_Return_Multiplier'] = (1.0 + port_df['Port_Return']).cumprod()
port_df['Cumulative_Return_Pct'] = (port_df['Cumulative_Return_Multiplier'] - 1.0) * 100.0

# Max Drawdown
port_df['Peak_Multiplier'] = port_df['Cumulative_Return_Multiplier'].cummax()
port_df['Drawdown_Pct'] = (port_df['Cumulative_Return_Multiplier'] - port_df['Peak_Multiplier']) / port_df['Peak_Multiplier'] * 100.0
max_drawdown_pct = port_df['Drawdown_Pct'].min()

# Daily Returns for Sharpe
port_df['Date'] = port_df['DateTime'].dt.date
daily_returns = port_df.groupby('Date')['Port_Return'].sum()

mean_daily = daily_returns.mean()
std_daily = daily_returns.std()
if std_daily > 0:
    annualized_sharpe = (mean_daily / std_daily) * np.sqrt(252)
else:
    annualized_sharpe = 0.0

total_trades = len(trades_df)
win_rate = (trades_df['Net_Return_bps'] > 0).mean() * 100
total_cumulative_pct = port_df['Cumulative_Return_Pct'].iloc[-1]

print(f"Allocation Logic: 20% cash per trade with 1x Leverage (Max 20% exposure per trade)")
print(f"Total Trades Executed: {total_trades}")
print(f"Win Rate (Net): {win_rate:.2f}%")
print(f"True Cumulative Return (Compounded): {total_cumulative_pct:+.2f}%")
print(f"True Max Drawdown: {max_drawdown_pct:+.2f}%")
print(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
print("=" * 64)
