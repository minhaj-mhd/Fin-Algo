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

# --- Market Holidays ---
NSE_HOLIDAYS_2026 = {
    "2026-01-26", # Republic Day
    "2026-03-03", # Holi
    "2026-03-20", # Id-ul-Fitr
    "2026-04-02", # Mahavir Jayanti
    "2026-04-03", # Good Friday
    "2026-04-14", # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01", # Maharashtra Day
    "2026-05-27", # Bakri Id
    "2026-06-26", # Muharram
    "2026-08-15", # Independence Day
    "2026-09-14", # Ganesh Chaturthi
    "2026-10-02", # Mahatma Gandhi Jayanti
    "2026-10-18", # Dussehra
    "2026-11-08", # Diwali
    "2026-11-24", # Gurunanak Jayanti
    "2026-12-25", # Christmas
}

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

# 1-hour candle construction for the active 1h ranker.
#   True  -> OVERLAPPING trailing-1h candles (15-min step): fresh 1h signal every 15-min scan.
#            REQUIRED by v20_rolling_1h (trained on the rolling grid).
#   False -> non-overlapping 09:15-anchored 1h bars (v10_native_1h's grid; refreshes hourly).
# ROLLBACK: set this False AND point registry.json active_model back to "v10_native_1h".
ROLLING_1H_CANDLES = True

DAILY_MACRO_LONG_PATH = "models/daily_macro_v2/xgb_long_model.json"
DAILY_MACRO_SHORT_PATH = "models/daily_macro_v2/xgb_short_model.json"
DAILY_MACRO_META_PATH = "models/daily_macro_v2/metadata.json"

# --- Percentile Gating and Recalibration Constants (1H Trades) ---
ENTRY_TOP_K = 3
HOLD_PERCENTILE = 0.95

# --- Violent adverse-thrust entry guard ---
# When the completed look-back candle that failed direction confirmation is also a
# violent bar closing in the extreme quartile AGAINST the trade, cancel instead of
# placing the pending-limit. On a strong rip/dump the pending-limit degrades to an
# instant market fill into the breakout (cf. BALKRISIND.NS 2026-06-15: 3.59% range
# bar closing at pos 0.87, shorted into the breakout, -1.48% stop-loss).
THRUST_VETO_RANGE_PCT = 2.5   # look-back candle high-low range as % of price
THRUST_VETO_POS = 0.75        # close position in range (>= for SHORT, <= 1-x for LONG)

# --- Fade-Entry Quality Guard (post-mortem 2026-06-16: 3 stop-losses in 2h) ---
# When the look-back bar FAILS direction confirmation the engine still tries to fade
# it with a pending-limit toward the bar extreme. That fade is a knife-catch and only
# pays when the adverse move was noise that mean-reverts. The THRUST_VETO above only
# fires on VIOLENT bars (>2.5% range); the three 2026-06-16 stop-losses printed small
# bars (0.35-0.82% range) yet kept running because the move had VOLUME behind it.
# Two volume-gated signatures separated the 3 losers from the 3 winners that day:
#   1. SHORT into a heavy-volume breakout sitting on a fresh 52-week high
#      (BRIGADE.NS rvol 2.42 @ -0.2% from 52wH; CHAMBLFERT.NS rvol 2.79 @ -0.2%).
#   2. Fading a bar that closed in the adverse extreme on non-trivial volume
#      (VBL.NS long bought a bar closing at range-pos 0.10 on rvol 0.64).
# The control that proves it is volume (not bar shape): SUNDARMFIN.NS closed at its
# very low (range-pos 0.00, worst possible) but on rvol 0.25 -> bounced -> +0.36%.
# ⚠️ UNVERIFIED: thresholds are fitted to ONE session (5 fade trades). This WILL also
# block some future winners; backtest over history before trusting. Disable via flag.
FADE_QUALITY_GUARD = True
FADE_BREAKOUT_52H_PROXIMITY = -0.005  # within 0.5% of 52-wk high == fresh breakout
FADE_BREAKOUT_RVOL = 1.5              # heavy participation behind the breakout
FADE_ADVERSE_POS = 0.70               # look-back close in worst 30% of range (against us)
FADE_ADVERSE_MIN_RVOL = 0.5           # below this the adverse close is treated as noise

# --- Gemini API Configuration ---
GEMINI_ENABLED_DEFAULT = True
GEMINI_STATE_FILE = "data/gemini_usage.json"
MAX_GEMINI_KEYS = 3
GEMINI_MAX_REQUESTS_PER_DAY = 20
GEMINI_S1_MODEL_TIERS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
GEMINI_S2_MODEL_TIERS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
# Round-robin the top-N S1 "primary" models per audit so one overloaded model
# isn't always tried first (spreads load across gemini-3.5-flash / 3-flash-preview);
# the -lite fallback tiers keep their fixed order at the tail. 1 = no rotation.
GEMINI_S1_PRIMARY_ROTATE = 2
GEMINI_MODEL_TIERS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]



# --- Storage Paths ---
STATS_FILE = "upstox_stats.json"
SQLITE_DB_FILE = "data/vanguard_trades.db"
LATEST_SCORES_FILE = "data/latest_scores.json"

# --- Feature Flags ---
WEBSOCKET_ENABLED = True
SANDBOX_MODE = True  # Always defaults to Safe Sandbox/Paper mode
GAUNTLET_ENFORCEMENT = "warn"  # "warn" (warn-only rollout) | "enforce" (hard refuse)

# --- Network Monitor ---
NETWORK_MONITOR_ENABLED = True   # Halt the engine and wait when the internet/broker is unreachable
NETWORK_CHECK_INTERVAL = 15      # Seconds between connectivity retries while offline (linear backoff)
NETWORK_PROBE_TIMEOUT = 3.0      # Per-probe TCP connect timeout (seconds)
