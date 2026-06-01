import os
from dotenv import load_dotenv

load_dotenv()

# --- Timezone & Market Hours ---
TIMEZONE = "Asia/Kolkata"
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 15
HARD_CLOSE_HOUR = 15
HARD_CLOSE_MINUTE = 15

# --- Capital Management & Allocation Defaults ---
INITIAL_CAPITAL = 99517.68
MAX_TRADE_SLOTS = 5
MARGIN_MULTIPLIER = 5.0

# --- Brokerage & Statutory Taxes (Indian Market / Upstox Sandbox) ---
BROKERAGE_PER_ORDER = 10.0  # ₹10 Buy + ₹10 Sell = ₹20 Round-trip
STT_RATE = 0.00025          # 0.025% on Selling side
SLIPPAGE_ASSUMPTION_PCT = 0.03 # 0.03% slippage drag per leg

# --- Model Registry & Default Model Configuration ---
MODEL_REGISTRY_FALLBACK_DIR = "models"
DEFAULT_LONG_MODEL_NAME = "xgb_long_model.json"
DEFAULT_SHORT_MODEL_NAME = "xgb_short_model.json"
DEFAULT_MODEL_NAME = "v2_3_production_xgb"

DAILY_MACRO_LONG_PATH = "models/daily_xgb/xgb_long_model.json"
DAILY_MACRO_SHORT_PATH = "models/daily_xgb/xgb_short_model.json"
DAILY_MACRO_META_PATH = "models/daily_xgb/metadata.json"

# --- Gemini API Configuration ---
GEMINI_ENABLED_DEFAULT = True
GEMINI_STATE_FILE = "data/gemini_usage.json"
MAX_GEMINI_KEYS = 3
GEMINI_MAX_REQUESTS_PER_DAY = 20
GEMINI_MODEL_TIERS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]

# --- Features & Filters thresholds ---
MIN_CONVICTION = 0.10
MIN_RAW_SCORE = 0.12

# --- Storage Paths ---
STATS_FILE = "upstox_stats.json"
SQLITE_DB_FILE = "data/vanguard_trades.db"
LATEST_SCORES_FILE = "data/latest_scores.json"

# --- Feature Flags ---
WEBSOCKET_ENABLED = True
SANDBOX_MODE = True  # Always defaults to Safe Sandbox/Paper mode
