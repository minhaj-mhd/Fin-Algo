import datetime
import os
import numpy as np
import pandas as pd

def generate_synthetic_panel(
    path: str,
    n_tickers: int = 50,
    n_years: int = 3,
    planted_rho: float = 0.05,
    bar_minutes: int = 60,
    label_may_cross_session: bool = False,
    session_close: str = "15:30",
    seed: int = 42,
    plant_overnight_labels: bool = False
) -> pd.DataFrame:
    """
    Generates a synthetic panel of tickers with prices following geometric Brownian motion,
    realistic volatility, and a plantable feature with a specified correlation (rho)
    to the target column (Next_Hour_Return). Used to self-test the Validation Gauntlet.
    """
    rng = np.random.default_rng(seed)
    
    # Generate business days (Monday to Friday)
    start_date = datetime.date(2023, 1, 1)
    days = []
    curr = start_date
    end_date = start_date + datetime.timedelta(days=n_years * 365)
    while curr < end_date:
        if curr.weekday() < 5:
            days.append(curr)
        curr += datetime.timedelta(days=1)
        
    # Bar times (6 bars per day, left-aligned)
    bar_times = ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15"]
    tickers = [f"TKR{i:02d}" for i in range(n_tickers)]
    
    rows = []
    query_id = 0
    for d in days:
        for t_str in bar_times:
            dt_str = f"{d} {t_str}:00"
            for ticker in tickers:
                rows.append({
                    "DateTime": dt_str,
                    "Ticker": ticker,
                    "Query_ID": query_id,
                    "Date": d,
                    "Time": t_str
                })
            query_id += 1
            
    df = pd.DataFrame(rows)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)
    
    # Generate prices using GBM per ticker
    mu = 0.0
    sigma = 0.015 / np.sqrt(len(bar_times))  # rescaled daily volatility
    
    closes = []
    returns = []
    for ticker, group in df.groupby("Ticker"):
        n_bars = len(group)
        shocks = rng.normal(mu, sigma, size=n_bars)
        price = 100.0
        p_series = []
        for s in shocks:
            price *= np.exp(s)
            p_series.append(price)
            
        p_arr = np.array(p_series)
        closes.extend(p_arr)
        
        r_arr = np.zeros(n_bars)
        r_arr[1:] = p_arr[1:] / p_arr[:-1] - 1.0
        returns.extend(r_arr)
        
    df["Close"] = closes
    df["Return"] = returns
    df["Log_Return"] = np.log(df["Close"] / df.groupby("Ticker")["Close"].shift(1).fillna(100.0))
    
    # Generate Open, High, Low, Volume (R2 item 7)
    df["Open"] = df.groupby("Ticker")["Close"].shift(1).fillna(100.0)
    high_mult = 1.0 + rng.uniform(0.0, 0.005, size=len(df))
    low_mult = 1.0 - rng.uniform(0.0, 0.005, size=len(df))
    df["High"] = np.maximum(df["Open"], df["Close"]) * high_mult
    df["Low"] = np.minimum(df["Open"], df["Close"]) * low_mult
    df["Volume"] = rng.lognormal(mean=10.0, sigma=1.0, size=len(df))
    
    # Compute label (forward return)
    df["Next_Hour_Return"] = df.groupby("Ticker")["Close"].shift(-1) / df["Close"] - 1.0
    
    # Apply session mask (overnight guard)
    if not label_may_cross_session:
        next_date = df.groupby("Ticker")["Date"].shift(-1)
        cross_mask = df["Date"] != next_date
        
        if plant_overnight_labels:
            # Recompute next_date_first_close
            df_first = df.groupby(["Ticker", "Date"])["Close"].first().reset_index()
            df_first["next_date_first_close"] = df_first.groupby("Ticker")["Close"].shift(-1)
            df = pd.merge(df, df_first[["Ticker", "Date", "next_date_first_close"]], on=["Ticker", "Date"], how="left")
            
            # Plant overnight return on cross_mask
            overnight_ret = df["next_date_first_close"] / df["Close"] - 1.0
            df.loc[cross_mask, "Next_Hour_Return"] = overnight_ret
            df = df.drop(columns=["next_date_first_close"])
        else:
            df.loc[cross_mask, "Next_Hour_Return"] = np.nan
        
    # Drop rows where label is NaN (replicates compile_dataset dropping last bar)
    df = df.dropna(subset=["Next_Hour_Return"]).copy()
    
    # Re-index Query_IDs to be contiguous
    df["DateTime_Hour"] = df["DateTime"].dt.floor('h')
    df["Query_ID"] = df.groupby("DateTime_Hour").ngroup()
    
    # Ensure min 5 tickers per query group
    query_sizes = df.groupby("Query_ID").size()
    valid_queries = query_sizes[query_sizes >= 5].index
    df = df[df["Query_ID"].isin(valid_queries)].copy()
    df["Query_ID"] = df.groupby("DateTime_Hour").ngroup()
    
    # Plant signal feature targeting exact correlation rho
    y_clean = df["Next_Hour_Return"].values
    y_std = np.std(y_clean)
    y_mean = np.mean(y_clean)
    y_norm = (y_clean - y_mean) / (y_std + 1e-8)
    
    noise = rng.normal(0.0, 1.0, size=len(df))
    # E[signal * y_norm] = planted_rho
    signal = planted_rho * y_norm + np.sqrt(1.0 - planted_rho**2) * noise
    df["signal_feature"] = signal
    
    # Add dummy noise features
    for i in range(1, 10):
        df[f"noise_feature_{i}"] = rng.normal(0.0, 1.0, size=len(df))
        
    # Other indicators required by schema checks
    df["HL_Range"] = rng.uniform(0.005, 0.02, size=len(df))
    df["OC_Range"] = rng.uniform(0.001, 0.01, size=len(df))
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Generated synthetic panel saved to: {path}")
        
    return df
