import pandas as pd
import numpy as np


def compute_daily_technical_indicators(df_daily):
    """
    Computes daily technical indicators including volatility metrics
    for cross-asset normalization in Layer B.
    """
    df = df_daily.copy().sort_values("timestamp")

    # Return
    df["Prev_Day_Return"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility (20-day standard deviation of returns)
    df["Vol_20d"] = df["Prev_Day_Return"].rolling(20).std()

    # Daily range normalized by close
    df["Daily_Range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
    
    # 20-day Average Daily Range (proxy for ATR)
    df["ADR_20d"] = df["Daily_Range"].rolling(20).mean()

    # IBS: (Close - Low) / (High - Low)
    df["Prev_Day_IBS"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

    # Volume Ratio vs 20-day MA
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["Prev_Day_Volume_Ratio"] = df["volume"] / (df["vol_ma20"] + 1e-8)

    # SMA 20 distance — the one daily feature that meaningfully conditions
    # mean-reversion vs momentum for intraday
    df["sma20"] = df["close"].rolling(20).mean()
    df["Dist_SMA_20"] = (df["close"] - df["sma20"]) / (df["sma20"] + 1e-8)

    feature_cols = [
        "timestamp",
        "Prev_Day_Return",
        "Vol_20d",
        "ADR_20d",
        "Daily_Range",
        "Prev_Day_Volume_Ratio",
        "Prev_Day_IBS",
        "Dist_SMA_20",
    ]
    return df[feature_cols].set_index("timestamp")
