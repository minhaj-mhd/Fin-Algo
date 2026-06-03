import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from scripts.strategy_1030.config import (
    DATA_DIR,
    MODEL_DIR,
    LAYER_A_FEATURES,
    LAYER_B_FEATURES,
    SLIPPAGE_PCT,
    TOP_K
)

def run_simulation(prob_threshold=0.55):
    """
    Simulates the two-layer momentum system across all test folds.
    Threshold is a probability threshold (e.g. 0.55 means P(Up) > 0.55 for long, P(Up) < 0.45 for short).
    """
    # Load datasets
    market_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_market.csv"))
    stocks_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_stocks.csv"))
    
    # Drop NaNs on feature sets and target columns
    market_df = market_df.dropna(subset=LAYER_A_FEATURES + ["Nifty_ROD_Return"]).copy()
    stocks_df = stocks_df.dropna(subset=LAYER_B_FEATURES + ["Target_Raw"]).copy()
    
    # Load metadata to get test months per fold
    with open(os.path.join(MODEL_DIR, "market_filter", "metadata.json"), "r") as f:
        meta_a = json.load(f)
        
    trades = []
    daily_returns = {}
    
    # We walk through each fold
    for fold in meta_a["folds"]:
        fold_i = fold["fold"]
        test_months = fold["test_months"]
        
        # Filter test data
        market_df["Month"] = pd.to_datetime(market_df["Date"]).dt.to_period("M").astype(str)
        stocks_df["Month"] = pd.to_datetime(stocks_df["Date"]).dt.to_period("M").astype(str)
        
        mkt_test = market_df[market_df["Month"].isin(test_months)].sort_values("Date")
        stk_test = stocks_df[stocks_df["Month"].isin(test_months)].sort_values("Date")
        
        if mkt_test.empty or stk_test.empty:
            continue
            
        # Load models for this fold
        model_a = XGBClassifier()
        model_a.load_model(os.path.join(MODEL_DIR, "market_filter", f"xgb_market_fold_{fold_i}.json"))
        
        model_b = XGBRegressor()
        model_b.load_model(os.path.join(MODEL_DIR, "stock_selector", f"xgb_residual_fold_{fold_i}.json"))
        
        # Process each date in the test set
        for _, mkt_row in mkt_test.iterrows():
            date_str = mkt_row["Date"]
            
            # Predict Nifty direction (Layer A - Probability of UP)
            X_a = pd.DataFrame([mkt_row[LAYER_A_FEATURES]])
            prob_up = model_a.predict_proba(X_a)[0][1]
            
            # Apply probabilistic market filter
            if prob_up >= prob_threshold:
                direction = "LONG"
            elif prob_up <= (1.0 - prob_threshold):
                direction = "SHORT"
            else:
                direction = "SKIP"
                
            if direction == "SKIP":
                daily_returns[date_str] = 0.0
                continue
                
            # Filter stocks for today
            day_stocks = stk_test[stk_test["Date"] == date_str].copy()
            if day_stocks.empty:
                continue
                
            X_b = day_stocks[LAYER_B_FEATURES]
            
            # Predict stock ranks (Layer B - Expected Vol-Normalized Residual Return)
            day_stocks["Score"] = model_b.predict(X_b)
            
            # Rank stocks
            if direction == "LONG":
                # Want highest expected residual outperformance
                selected = day_stocks.sort_values("Score", ascending=False).head(TOP_K)
            else:
                # Want lowest expected residual outperformance (i.e. most negative residual, best for shorting)
                selected = day_stocks.sort_values("Score", ascending=True).head(TOP_K)
                
            # Execute trades
            day_trade_rets = []
            for _, stk_row in selected.iterrows():
                # We care about raw return for PnL, not the normalized residual target used for ranking
                raw_ret = stk_row["Target_Raw"]
                
                # Net return accounting for slippage
                if direction == "LONG":
                    net_ret = raw_ret - SLIPPAGE_PCT
                else:
                    net_ret = -raw_ret - SLIPPAGE_PCT
                    
                day_trade_rets.append(net_ret)
                
                trades.append({
                    "Date": date_str,
                    "Ticker": stk_row["Ticker"],
                    "Direction": direction,
                    "Prob_Up": prob_up,
                    "Raw_Return": raw_ret,
                    "Net_Return": net_ret,
                    "Score": stk_row["Score"]
                })
                
            daily_returns[date_str] = np.mean(day_trade_rets) if day_trade_rets else 0.0
            
    # Calculate performance metrics
    if not trades:
        return {
            "threshold": prob_threshold,
            "total_trades": 0,
            "win_rate": 0.0,
            "net_return": 0.0,
            "avg_trade_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "trades": pd.DataFrame(),
            "daily_returns": pd.Series()
        }
        
    trades_df = pd.DataFrame(trades)
    daily_ret_series = pd.Series(daily_returns).sort_index()
    
    total_trades = len(trades_df)
    win_rate = (trades_df["Net_Return"] > 0).mean()
    avg_trade_return = trades_df["Net_Return"].mean()
    
    # Capital curve
    cum_returns = (1 + daily_ret_series).cumprod() - 1
    final_return = cum_returns.iloc[-1] if not cum_returns.empty else 0.0
    
    # Sharpe
    mean_daily = daily_ret_series.mean()
    std_daily = daily_ret_series.std()
    sharpe = (mean_daily / (std_daily + 1e-8)) * np.sqrt(252) if std_daily > 0 else 0.0
    
    # Max Drawdown
    cum_equity = (1 + daily_ret_series).cumprod()
    peaks = cum_equity.cummax()
    drawdowns = (cum_equity - peaks) / peaks
    max_dd = drawdowns.min() if not drawdowns.empty else 0.0
    
    # Profit factor
    gains = trades_df.loc[trades_df["Net_Return"] > 0, "Net_Return"].sum()
    losses = abs(trades_df.loc[trades_df["Net_Return"] < 0, "Net_Return"].sum())
    profit_factor = gains / (losses + 1e-8)
    
    return {
        "threshold": prob_threshold,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "net_return": final_return,
        "avg_trade_return": avg_trade_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "trades": trades_df,
        "daily_returns": daily_ret_series
    }

def run_baselines():
    """
    Runs baselines to compare:
    1. Layer A only (Nifty index trades using > 0.5 threshold)
    2. Layer B only (Always LONG Top 3 stocks, no market filter)
    """
    market_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_market.csv"))
    stocks_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_stocks.csv"))
    
    market_df = market_df.dropna(subset=LAYER_A_FEATURES + ["Nifty_ROD_Return"]).copy()
    stocks_df = stocks_df.dropna(subset=LAYER_B_FEATURES + ["Target_Raw"]).copy()
    
    with open(os.path.join(MODEL_DIR, "market_filter", "metadata.json"), "r") as f:
        meta_a = json.load(f)
        
    daily_rets_layer_a = {}
    daily_rets_layer_b = {}
    
    for fold in meta_a["folds"]:
        fold_i = fold["fold"]
        test_months = fold["test_months"]
        
        market_df["Month"] = pd.to_datetime(market_df["Date"]).dt.to_period("M").astype(str)
        stocks_df["Month"] = pd.to_datetime(stocks_df["Date"]).dt.to_period("M").astype(str)
        
        mkt_test = market_df[market_df["Month"].isin(test_months)].sort_values("Date")
        stk_test = stocks_df[stocks_df["Month"].isin(test_months)].sort_values("Date")
        
        if mkt_test.empty or stk_test.empty:
            continue
            
        model_a = XGBClassifier()
        model_a.load_model(os.path.join(MODEL_DIR, "market_filter", f"xgb_market_fold_{fold_i}.json"))
        
        model_b = XGBRegressor()
        model_b.load_model(os.path.join(MODEL_DIR, "stock_selector", f"xgb_residual_fold_{fold_i}.json"))
        
        for _, mkt_row in mkt_test.iterrows():
            date_str = mkt_row["Date"]
            
            # Baseline 1: Layer A only
            X_a = pd.DataFrame([mkt_row[LAYER_A_FEATURES]])
            prob_up = model_a.predict_proba(X_a)[0][1]
            if prob_up > 0.5:
                daily_rets_layer_a[date_str] = mkt_row["Nifty_ROD_Return"] - SLIPPAGE_PCT
            else:
                daily_rets_layer_a[date_str] = -mkt_row["Nifty_ROD_Return"] - SLIPPAGE_PCT
                
            # Baseline 2: Layer B only (Always LONG Top 3 stocks)
            day_stocks = stk_test[stk_test["Date"] == date_str].copy()
            if not day_stocks.empty:
                X_b = day_stocks[LAYER_B_FEATURES]
                day_stocks["Score"] = model_b.predict(X_b)
                selected = day_stocks.sort_values("Score", ascending=False).head(TOP_K)
                daily_rets_layer_b[date_str] = selected["Target_Raw"].mean() - SLIPPAGE_PCT
            else:
                daily_rets_layer_b[date_str] = 0.0
                
    series_a = pd.Series(daily_rets_layer_a).sort_index()
    series_b = pd.Series(daily_rets_layer_b).sort_index()
    
    cum_a = (1 + series_a).cumprod() - 1
    cum_b = (1 + series_b).cumprod() - 1
    
    return {
        "layer_a_only_ret": cum_a.iloc[-1] if not cum_a.empty else 0.0,
        "layer_b_only_ret": cum_b.iloc[-1] if not cum_b.empty else 0.0,
        "layer_a_sharpe": (series_a.mean() / (series_a.std() + 1e-8)) * np.sqrt(252) if not series_a.empty else 0.0,
        "layer_b_sharpe": (series_b.mean() / (series_b.std() + 1e-8)) * np.sqrt(252) if not series_b.empty else 0.0
    }

def print_report():
    print("\n==================================================")
    print("      10:30 AM MOMENTUM BACKTEST REPORT (V3)")
    print("==================================================")
    
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
    sweep_results = []
    
    for t in thresholds:
        print(f"Running simulation with prob_threshold = {t:.2f}..." )
        res = run_simulation(t)
        sweep_results.append(res)
        
    print("\n--- Baseline Comparison ---")
    baselines = run_baselines()
    print(f"Layer A Only (Nifty P(Up)>0.5): Net Return = {baselines['layer_a_only_ret']:.2%}, Sharpe = {baselines['layer_a_sharpe']:.2f}")
    print(f"Layer B Only (Always Long Top 3): Net Return = {baselines['layer_b_only_ret']:.2%}, Sharpe = {baselines['layer_b_sharpe']:.2f}")
    
    print("\n--- Two-Layer Combined Performance (Probability Threshold Sweep) ---")
    print(f"{'Prob Thresh':<12} | {'Trades':<6} | {'Win Rate':<8} | {'Net Return':<10} | {'Sharpe':<6} | {'Max DD':<7} | {'Profit Factor':<12}")
    print("-" * 75)
    
    for r in sweep_results:
        print(f">= {r['threshold']:>10.2f} | {r['total_trades']:>6} | {r['win_rate']:>8.1%} | {r['net_return']:>10.2%} | {r['sharpe_ratio']:>6.2f} | {r['max_drawdown']:>7.1%} | {r['profit_factor']:>12.2f}")
        
    best_res = max(sweep_results, key=lambda x: x["net_return"])
    print(f"\nBest performing combined config: Prob Threshold = {best_res['threshold']:.2f}")
    
    if not best_res["trades"].empty:
        best_trades_df = best_res["trades"]
        best_trades_df.to_csv(os.path.join(DATA_DIR, "backtest_trades_best.csv"), index=False)
        print(f"Saved best backtest trade log to {os.path.join(DATA_DIR, 'backtest_trades_best.csv')}")

if __name__ == "__main__":
    print_report()
