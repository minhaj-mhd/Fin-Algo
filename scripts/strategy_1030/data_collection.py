"""
Data collection for the 10:30 AM Momentum Strategy (V3).

PRIMARY DATA SOURCE: 30-min cache (data/raw_upstox_cache/) — ~4.5 years, ~1100 trading days.

CHANGES FROM V2 (V3 Architecture):
- Layer A Target is now a binary classification (Nifty_Up).
- Layer B Target is now a vol-normalized residual return: (Stock_Ret - Nifty_Ret) / Vol_20d.
- Price-based morning features are normalized by Vol_20d to allow cross-sectional 
  apples-to-apples comparison between low-beta and high-beta stocks.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from scripts.tickers import TICKERS
from scripts.strategy_1030.config import (
    GLOBAL_INDICES,
    DATA_DIR,
    CACHE_30MIN_DIR,
    CACHE_DAILY_DIR,
    MORNING_BARS_UTC,
    ENTRY_BAR_UTC,
    EXIT_BAR_UTC,
)
from scripts.strategy_1030.feature_engineering import compute_daily_technical_indicators


def collect_global_indices():
    """
    Downloads global indices from yfinance, computes overnight returns and VIX features.
    All features are shifted by 1 day to prevent look-ahead bias.
    """
    print("=== Step 1: Collecting Global Indices ===")
    merged_df = None

    for name, ticker in GLOBAL_INDICES.items():
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.Ticker(ticker).history(period="5y")
            if df.empty:
                print(f"  Warning: No data returned for {name}")
                continue

            df.index = pd.to_datetime(df.index).tz_localize(None).strftime("%Y-%m-%d")
            df.index.name = "Date"
            df_close = df[["Close"]].rename(columns={"Close": f"{name}_Close"})

            if merged_df is None:
                merged_df = df_close
            else:
                merged_df = merged_df.join(df_close, how="outer")
        except Exception as e:
            print(f"  Error downloading {name}: {e}")

    if merged_df is None:
        raise ValueError("Could not download any global index data.")

    merged_df = merged_df.sort_index().ffill()

    features_df = pd.DataFrame(index=merged_df.index)

    for name in ["SP500", "NASDAQ", "NIKKEI", "HANGSENG", "NIFTY50", "BANKNIFTY"]:
        col = f"{name}_Close"
        if col in merged_df.columns:
            features_df[f"{name}_Return"] = np.log(
                merged_df[col] / merged_df[col].shift(1)
            )
        else:
            features_df[f"{name}_Return"] = 0.0

    if "INDIA_VIX_Close" in merged_df.columns:
        vix = merged_df["INDIA_VIX_Close"]
        features_df["VIX_Level"] = vix
        features_df["VIX_Change"] = np.log(vix / vix.shift(1))
        features_df["VIX_Zscore_20d"] = (vix - vix.rolling(20).mean()) / (
            vix.rolling(20).std() + 1e-8
        )
    else:
        features_df["VIX_Level"] = 15.0
        features_df["VIX_Change"] = 0.0
        features_df["VIX_Zscore_20d"] = 0.0

    output_df = pd.DataFrame(index=features_df.index)
    output_df["SP500_Overnight_Ret"] = features_df["SP500_Return"]
    output_df["Nasdaq_Overnight_Ret"] = features_df["NASDAQ_Return"]
    output_df["Nikkei_Overnight_Ret"] = features_df["NIKKEI_Return"]
    output_df["HangSeng_Overnight_Ret"] = features_df["HANGSENG_Return"]
    output_df["VIX_Level"] = features_df["VIX_Level"]
    output_df["VIX_Change"] = features_df["VIX_Change"]
    output_df["VIX_Zscore_20d"] = features_df["VIX_Zscore_20d"]
    output_df["Prev_Day_Nifty_Ret"] = features_df["NIFTY50_Return"]

    output_df = output_df.shift(1).dropna(subset=["SP500_Overnight_Ret"])

    merged_path = os.path.join(DATA_DIR, "global_indices_merged.csv")
    output_df.to_csv(merged_path)
    print(f"  Saved to {merged_path}. Shape: {output_df.shape}")
    return output_df


def build_morning_datasets():
    """
    Builds Layer A and Layer B datasets using V3 vol-normalized features.
    """
    print("=== Step 2: Building Morning Datasets from 30-min cache ===")

    global_merged_path = os.path.join(DATA_DIR, "global_indices_merged.csv")
    if not os.path.exists(global_merged_path):
        global_df = collect_global_indices()
    else:
        global_df = pd.read_csv(global_merged_path, index_col="Date")

    all_records = []
    skipped = 0

    for ticker_ns in TICKERS:
        ticker = ticker_ns.split(".")[0]
        file_30m = os.path.join(CACHE_30MIN_DIR, f"{ticker}.csv")
        file_daily = os.path.join(CACHE_DAILY_DIR, f"{ticker}.csv")

        if not os.path.exists(file_30m) or not os.path.exists(file_daily):
            skipped += 1
            continue

        df_daily_raw = pd.read_csv(file_daily)
        df_daily_raw["timestamp"] = pd.to_datetime(df_daily_raw["timestamp"]).dt.strftime("%Y-%m-%d")
        df_daily_features = compute_daily_technical_indicators(df_daily_raw)
        daily_close_lookup = df_daily_raw.set_index("timestamp")["close"].to_dict()

        df_30m = pd.read_csv(file_30m)
        df_30m["timestamp"] = pd.to_datetime(df_30m["timestamp"])
        df_30m["date_str"] = df_30m["timestamp"].dt.strftime("%Y-%m-%d")
        df_30m["time_str"] = df_30m["timestamp"].dt.strftime("%H:%M")

        morning_vols_by_date = (
            df_30m[df_30m["time_str"].isin(MORNING_BARS_UTC)]
            .groupby("date_str")["volume"]
            .sum()
        )
        morning_vol_ma20 = morning_vols_by_date.rolling(20, min_periods=10).mean()

        grouped = df_30m.groupby("date_str")

        for date_str, group in grouped:
            morning = group[group["time_str"].isin(MORNING_BARS_UTC)].sort_values("timestamp")
            if len(morning) != 3:
                continue

            exit_candle = group[group["time_str"] == EXIT_BAR_UTC]
            if exit_candle.empty:
                continue

            close_entry = morning.iloc[-1]["close"]
            exit_close = exit_candle.iloc[0]["close"]
            target_raw = (exit_close / close_entry) - 1.0

            daily_rows_before = df_daily_features.loc[df_daily_features.index < date_str]
            if daily_rows_before.empty:
                continue
            prev_day_date = daily_rows_before.index[-1]
            prev_day_features = daily_rows_before.iloc[-1]
            prev_day_close = daily_close_lookup.get(prev_day_date)
            if prev_day_close is None:
                continue

            # Volatility for normalization
            vol_20d = prev_day_features["Vol_20d"]
            if pd.isna(vol_20d) or vol_20d <= 0.001:
                vol_20d = 0.02 # fallback to 2% daily vol if undefined

            open_0915 = morning.iloc[0]["open"]
            high_morning = morning["high"].max()
            low_morning = morning["low"].min()
            close_1015 = morning.iloc[-1]["close"]
            vol_morning = morning["volume"].sum()

            opening_gap = np.log(open_0915 / prev_day_close)
            if opening_gap > 0:
                gap_fill_status = 1 if low_morning <= prev_day_close else 0
            elif opening_gap < 0:
                gap_fill_status = 1 if high_morning >= prev_day_close else 0
            else:
                gap_fill_status = 1

            morning_return = (close_1015 - open_0915) / (open_0915 + 1e-8)
            morning_range = (high_morning - low_morning) / (low_morning + 1e-8)
            orb_position = (close_1015 - low_morning) / (high_morning - low_morning + 1e-8)
            morning_body_direction = np.sum(np.sign(morning["close"].values - morning["open"].values))

            vol_last_1 = morning.iloc[-1:]["volume"].sum()
            vol_first_2 = morning.iloc[:2]["volume"].sum()
            volume_acceleration = vol_last_1 / (vol_first_2 + 1e-8)

            vwap = (morning["close"] * morning["volume"]).sum() / (vol_morning + 1e-8)
            vwap_deviation = (close_1015 - vwap) / (vwap + 1e-8)

            first_candle_return = (morning.iloc[0]["close"] - morning.iloc[0]["open"]) / (morning.iloc[0]["open"] + 1e-8)
            first_candle_range = (morning.iloc[0]["high"] - morning.iloc[0]["low"]) / (morning.iloc[0]["low"] + 1e-8)
            post_open_trend = (close_1015 - morning.iloc[0]["close"]) / (morning.iloc[0]["close"] + 1e-8)
            ibs_morning = (close_1015 - low_morning) / (high_morning - low_morning + 1e-8)

            hist_vol = morning_vol_ma20.get(date_str)
            if hist_vol is None or np.isnan(hist_vol) or hist_vol < 1:
                continue
            morning_volume_ratio = vol_morning / hist_vol

            record = {
                "Ticker": ticker,
                "Date": date_str,
                "Target_Raw": target_raw,
                
                # Vol-Normalized Features
                "Norm_Opening_Gap": opening_gap / vol_20d,
                "Gap_Fill_Status": gap_fill_status,
                "Norm_Morning_Return": morning_return / vol_20d,
                "Norm_Morning_Range": morning_range / vol_20d,
                "ORB_Position": orb_position,
                "Morning_Body_Direction": morning_body_direction,
                "Morning_Volume_Ratio": morning_volume_ratio,
                "Volume_Acceleration": volume_acceleration,
                "VWAP_Deviation": vwap_deviation,
                "Norm_First_Candle_Return": first_candle_return / vol_20d,
                "Norm_First_Candle_Range": first_candle_range / vol_20d,
                "Norm_Post_Open_Trend": post_open_trend / vol_20d,
                "IBS_Morning": ibs_morning,
                
                # Daily Context
                "Prev_Day_Return": prev_day_features["Prev_Day_Return"],
                "Vol_20d": vol_20d,
                "ADR_20d": prev_day_features["ADR_20d"],
                "Prev_Day_Volume_Ratio": prev_day_features["Prev_Day_Volume_Ratio"],
                "Prev_Day_IBS": prev_day_features["Prev_Day_IBS"],
                "Dist_SMA_20": prev_day_features["Dist_SMA_20"],
            }
            all_records.append(record)

        print(f"  {ticker}: {len([r for r in all_records if r['Ticker'] == ticker])} day-records")

    if not all_records:
        raise ValueError("No stock records extracted. Check cache paths.")

    print(f"\n  Total raw records: {len(all_records)} (skipped {skipped} tickers with missing files)")

    df_stocks_raw = pd.DataFrame(all_records)
    df_stocks_raw = df_stocks_raw.join(global_df, on="Date", how="inner")

    # --- Build Market Dataset (Layer A) ---
    market_rows = []
    for date_str, day_group in df_stocks_raw.groupby("Date"):
        nifty_gap = day_group["Norm_Opening_Gap"].mean()
        nifty_morning_ret = day_group["Norm_Morning_Return"].mean()
        nifty_rod_return = day_group["Target_Raw"].mean()  # Proxy for Nifty rest-of-day

        global_row = day_group.iloc[0]

        market_rows.append({
            "Date": date_str,
            "Nifty_ROD_Return": nifty_rod_return,
            "Nifty_Up": 1 if nifty_rod_return > 0 else 0, # V3 Binary Target
            "Nifty_Gap": nifty_gap,
            "Nifty_Morning_Ret": nifty_morning_ret,
            "SP500_Overnight_Ret": global_row["SP500_Overnight_Ret"],
            "Nasdaq_Overnight_Ret": global_row["Nasdaq_Overnight_Ret"],
            "Nikkei_Overnight_Ret": global_row["Nikkei_Overnight_Ret"],
            "HangSeng_Overnight_Ret": global_row["HangSeng_Overnight_Ret"],
            "VIX_Level": global_row["VIX_Level"],
            "VIX_Change": global_row["VIX_Change"],
            "VIX_Zscore_20d": global_row["VIX_Zscore_20d"],
            "Prev_Day_Nifty_Ret": global_row["Prev_Day_Nifty_Ret"],
        })

    df_market = pd.DataFrame(market_rows).sort_values("Date")

    # --- Compute Residual Targets for Stock Dataset (Layer B) ---
    mkt_rod_ret_lookup = df_market.set_index("Date")["Nifty_ROD_Return"].to_dict()
    
    # Target = (Stock Return - Nifty Return) / Vol_20d
    df_stocks_raw["Target"] = df_stocks_raw.apply(
        lambda r: (r["Target_Raw"] - mkt_rod_ret_lookup.get(r["Date"], 0)) / r["Vol_20d"],
        axis=1,
    )
    
    # Relative morning return feature
    mkt_morning_ret_lookup = df_market.set_index("Date")["Nifty_Morning_Ret"].to_dict()
    df_stocks_raw["Relative_Morning_Return"] = df_stocks_raw.apply(
        lambda r: r["Norm_Morning_Return"] - mkt_morning_ret_lookup.get(r["Date"], 0),
        axis=1,
    )

    # Save datasets
    df_market.to_csv(os.path.join(DATA_DIR, "dataset_market.csv"), index=False)
    print(f"\n  Layer A Market Dataset: {df_market.shape[0]} days x {df_market.shape[1]} cols")

    df_stocks_raw.to_csv(os.path.join(DATA_DIR, "dataset_stocks.csv"), index=False)
    n_dates = df_stocks_raw["Date"].nunique()
    n_tickers = df_stocks_raw["Ticker"].nunique()
    print(f"  Layer B Stock Dataset: {df_stocks_raw.shape[0]} rows ({n_dates} dates x {n_tickers} tickers)")


if __name__ == "__main__":
    collect_global_indices()
    build_morning_datasets()
