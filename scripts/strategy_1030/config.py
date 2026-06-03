import os

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "strategy_1030"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "strategy_1030"))
CACHE_30MIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_upstox_cache"))  # 30-min bars, ~4.5 years
CACHE_DAILY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_upstox_daily_cache"))  # Daily bars, ~5 years

# Ensure paths exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "global_indices"), exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, "market_filter"), exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, "stock_selector"), exist_ok=True)

# Global index tickers for yfinance
GLOBAL_INDICES = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "NIKKEI": "^N225",
    "HANGSENG": "^HSI",
    "INDIA_VIX": "^INDIAVIX",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

MORNING_BARS_UTC = ["03:45", "04:15", "04:45"]  # IST 09:15, 09:45, 10:15
ENTRY_BAR_UTC = "04:45"   # Close of this bar = entry price (~10:15 IST close)
EXIT_BAR_UTC = "09:45"    # Close of this bar = exit price (~15:15 IST close)

# Backtest
SLIPPAGE_PCT = 0.0006      # 0.06% round-trip slippage + fees
TOP_K = 3                  # Pick top 3 long or short stocks

# Layer A (Market Filter) — 10 features
LAYER_A_FEATURES = [
    "SP500_Overnight_Ret",
    "Nasdaq_Overnight_Ret",
    "Nikkei_Overnight_Ret",
    "HangSeng_Overnight_Ret",
    "VIX_Level",
    "VIX_Change",
    "VIX_Zscore_20d",
    "Nifty_Gap",
    "Nifty_Morning_Ret",
    "Prev_Day_Nifty_Ret",
]

# Layer B (Stock Selector) — 20 features
# Modified for V3: Using vol-normalized features
LAYER_B_FEATURES = [
    "Norm_Opening_Gap",
    "Gap_Fill_Status",
    "Norm_Morning_Return",
    "Norm_Morning_Range",
    "ORB_Position",
    "Morning_Body_Direction",
    "Morning_Volume_Ratio",
    "Volume_Acceleration",
    "VWAP_Deviation",
    "Norm_First_Candle_Return",
    "Norm_First_Candle_Range",
    "Norm_Post_Open_Trend",
    "IBS_Morning",
    "Prev_Day_Return",
    "Vol_20d",
    "ADR_20d",
    "Prev_Day_Volume_Ratio",
    "Prev_Day_IBS",
    "Dist_SMA_20",
    "Relative_Morning_Return",
]
