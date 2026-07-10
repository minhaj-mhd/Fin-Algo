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

# New-entry window (HH:MM IST, both bounds inclusive). No new trades are opened
# outside this window. LAST_ENTRY_TIME = 14:15 so a full 1-hour hold completes by the
# 15:15 hard close; later entries would be truncated by the forced close (partial hold,
# full round-trip cost) in the choppy pre-close. Single source of truth for both gates.
FIRST_ENTRY_TIME = "10:15"
LAST_ENTRY_TIME = "14:15"

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
# 2026-07-05: concentrate to the single best pick per side. Research (Conv-2026-07-05)
# showed steep K-decay — top-1 net@6 beats top-3 on both sides; rank-2/3 are ~0/negative.
ENTRY_TOP_K = 1
HOLD_PERCENTILE = 0.95

# --- Entry-layer gates (2026-07-05 research; both default-safe to prior behavior) ---
# DAILY_GATE: restrict intraday longs/shorts to the daily macro model's same-side top-40%.
#   DISABLED — the daily same-side rank carries NO intraday information (neg-control matched a
#   real gate; same-day "edge" was a look-ahead leak). It only thinned the universe ~60%.
# ENTRY_15M_GATE: require each candidate in v3_15m's same-side top-10% at entry.
#   DISABLED — same-side 15m "confirmation" is flat-to-backwards (mean reversion); it discarded
#   ~90% of candidates without adding value and is incompatible with top-1 concentration.
# Flip either back to True to restore the old gate; getattr default is True so absence = old behavior.
DAILY_GATE_ENABLED = False
ENTRY_15M_GATE_ENABLED = False

# 2026-07-05: emit ONE top-1 pick per side selected by RAW model score (long_score / short_score),
# not the conviction hybrid (Long_Rank = long−short). Raw beats conviction on longs (+0.6 vs −1.7
# bps net@6), ties on shorts. Drops the AI_Net path so with ENTRY_TOP_K=1 exactly 1 long + 1 short
# emit per scan — the clean per-anchor per-side stream for shadow cross-verification vs the panel.
SIGNAL_RAW_SCORE_ONLY = True

# --- Shadow "record ALL veto layers" mode (2026-07-06) ---
# When ON, the entry loop STOPS short-circuiting at the first veto: every emitted
# signal is scored by EVERY veto layer (15m gate, candle fade-guard, Kronos, Gemini
# S1/S2) and each layer's INDEPENDENT verdict is recorded even when an earlier layer
# would have vetoed. The per-signal × per-layer verdict matrix is appended to
# VETO_LAYERS_LOG (one row/signal) AND stored on the trade row (veto_layers JSON),
# so offline analysis can grade each layer's counterfactual value against the
# realized 1h P&L. This is a pure telemetry enrichment — it does NOT by itself
# change which trades open vs get vetoed (see SHADOW_DECOUPLE_ENFORCEMENT). Note:
# within the Gemini layer an S1 veto still short-circuits the (Google-search-
# grounded) S2 call to protect the daily API budget (S2 logged NOT_RUN); with S1
# already disabled (GEMINI_S1_VETO_ENABLED=0) the S1 layer logs BYPASSED and S2
# always runs. Revert to the sequential first-veto-wins pipeline with
# SHADOW_ALL_LAYERS=0 in the env/.env.
SHADOW_ALL_LAYERS = os.getenv("SHADOW_ALL_LAYERS", "1").strip().lower() in ("1", "true", "yes", "on")
VETO_LAYERS_LOG = "data/veto_layers_live.jsonl"

# Enforcement decoupling (only meaningful when SHADOW_ALL_LAYERS is ON). Two behaviours:
#   OFF (default) — record all layers, but the trade's OPEN vs VETOED status still
#                   follows the current enforcement policy (candle prescreen, Kronos
#                   enforce-mode, Gemini). Vetoed signals are tracked counterfactually
#                   as today (Vetoed tab populated), now carrying the full layer map.
#   ON            — open EVERY signal as a tracked shadow position at the signal price
#                   regardless of any veto (layers observe, they never block); this
#                   suspends the Kronos-enforce (2026-07-03) + candle block and empties
#                   the dashboard's Vetoed tab. The slot cap is bypassed either way in
#                   SHADOW_ALL_LAYERS mode so no per-anchor sample is dropped.
# Flip with SHADOW_DECOUPLE_ENFORCEMENT=1 in the env/.env.
SHADOW_DECOUPLE_ENFORCEMENT = os.getenv("SHADOW_DECOUPLE_ENFORCEMENT", "0").strip().lower() in ("1", "true", "yes", "on")

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

# --- Fade-Limit Expiry Behaviour ---
# When a fade pending-limit does NOT fill within its 15-min window, this controls what
# happens next. ON = market-fill at the then-current price if a new candle confirms
# direction (a momentum chase: you only get the confirming candle once price has already
# run away from your fade level, so you enter late at a worse price — cf. BALKRISIND
# 2026-06-24, filled 1.3% below the limit at EOD for a scratch). OFF = leave it: cancel
# and track counterfactually, so you get the fade price or nothing. DISABLED — re-enable
# with LIMIT_EXPIRY_MARKET_FILL_ENABLED=1 in the environment/.env.
LIMIT_EXPIRY_MARKET_FILL_ENABLED = os.getenv("LIMIT_EXPIRY_MARKET_FILL_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")

# --- Candle Confirmation Layer (master switch) ---
# The candle layer gates every AI-passed trade on the last completed 15m bar: look-back
# direction confirmation, the live-1m reversal veto, the violent-thrust veto, and the
# Fade-Entry Quality Guard (breakout/adverse) above — plus the pending-limit retrace
# toward the bar extreme when the bar fails. When OFF, all of that is skipped and every
# AI-passed trade enters immediately at the signal price.
# ⚠️ Disabling removes the documented knife-catch / live-reversal protections
# (post-mortem 2026-06-16) and turns patient limit fills into immediate market entries.
# ON by default; disable per-session for testing with CANDLE_LAYER_ENABLED=0 in the
# environment/.env (mirrors the GEMINI_S1_VETO_ENABLED bypass pattern).
CANDLE_LAYER_ENABLED = os.getenv("CANDLE_LAYER_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")

# --- Fill-Time Recheck (pending-limit fills) ---
# A fade limit is placed off a look-back bar that is up to 15 minutes stale by the
# time price touches it. This recheck re-validates the live tape at the fill moment:
# the trailing 15 minutes of 1-minute candles must not already form a violent adverse
# thrust (THRUST_VETO_RANGE_PCT / THRUST_VETO_POS applied to that rolling window —
# the same constants at their native 15-minute scale, nothing newly fitted).
# Motivated by the RELAXO 2026-06-29 post-mortem (short filled into the ignition of a
# volume-backed breakout; −2.36% net vs 0.68% budgeted risk). Deliberately NOT the raw
# live-1m reversal test: at a fade fill the current 1m bar is by construction moving
# toward the limit, so that test (which has no magnitude gate) would veto nearly every
# legitimate fill. Part of the candle layer (requires CANDLE_LAYER_ENABLED); disable
# alone with FILL_RECHECK_ENABLED=0 in the environment/.env.
FILL_RECHECK_ENABLED = os.getenv("FILL_RECHECK_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")

# --- 15-min Conviction-Flip Early Exit ---
# When ON, an open trade is force-closed mid-flight if its 15m model conviction drops
# out of the top 33% (re-checked every 15 min). DISABLED for now — leaving positions to
# run to their normal exits (SL / BE / trailing / time-expiry / EOD). Re-enable by
# setting CONVICTION_FLIP_EXIT_ENABLED=1 in the environment/.env or flipping the default.
CONVICTION_FLIP_EXIT_ENABLED = os.getenv("CONVICTION_FLIP_EXIT_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")

# --- Shadow-trade stop-loss CHECKPOINT (2026-07-09) ---
# Marks the counterfactual SHADOW trades — the vetoed / cancelled signals that are tracked
# to their full 1-hour outcome WITHOUT ever executing — with the point at which they WOULD
# have stopped out: sl_hit / sl_hit_time / sl_hit_price / sl_hit_pnl (and symmetric tp_*),
# stamping the FIRST moment P&L breaches the stop (or target) barrier while the shadow
# position keeps running to the 1h close. Makes the "1xATR stop-out rebound replay" (per
# project_1atr_stop_rebound_replay) a live, per-trade measurement: each shadow trade carries
# BOTH its would-be stop-out return AND its full-hold 1h return, so a stop-loss overlay can
# be graded offline. Executed OPEN trades are UNAFFECTED — they still exit at their real
# stop loss (STOP_LOSS status). Pure telemetry; disable with SHADOW_SL_CHECKPOINT=0.
SHADOW_SL_CHECKPOINT = os.getenv("SHADOW_SL_CHECKPOINT", "1").strip().lower() in ("1", "true", "yes", "on")

# --- Kronos zero-shot veto layer (EXPERIMENTAL — shadow-first) ---
# ⚠️ UNCERTIFIED: no Gauntlet run. Deployed 2026-07-03 for live observation only.
# Exploratory OOS backtest (post-2025-09-09 window, 3.6k trades; vault:
# Conv-2026-07-02-Kronos-Zero-Shot-Veto): keep-70% veto uplift +1.73bps/trade on
# longs (t≈2.0) but NOT separable from its timing negative-control; shorts ≈ 0;
# tighter thresholds INVERT. Runs AHEAD of the Gemini S1/S2 audit (cheap local
# GPU vs API quota). Modes: "shadow" = score+log only, never blocks (default);
# "enforce" = would_veto candidates are skipped. Flip via KRONOS_VETO_MODE env.
# Fail-safe: any error in the layer passes the trade through untouched.
KRONOS_VETO_ENABLED = os.getenv("KRONOS_VETO_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
# Default flipped shadow -> enforce per user 2026-07-03 (same day as deploy):
# would-veto candidates are now BLOCKED (tracked as VETOED with counterfactual
# P&L, visible in the dashboard's AI Vetoed tab). Revert to observation-only
# with KRONOS_VETO_MODE=shadow in the environment/.env.
KRONOS_VETO_MODE = os.getenv("KRONOS_VETO_MODE", "enforce").strip().lower()  # shadow | enforce
KRONOS_MODEL_ID = "NeoQuasar/Kronos-base"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_VETO_SAMPLES = 30      # forecast paths per candidate (matches backtest spec)
KRONOS_VETO_LOOKBACK = 480    # 15m bars of context (matches backtest spec)
KRONOS_VETO_MIN_BARS = 240    # below this the layer abstains (pass-through)
# Operating point (thresholds on the side-aligned score, fitted on the
# post-cutoff backtest window). Tightened keep-70% -> keep-50% per user
# 2026-07-03. ⚠️ backtest note: at keep-50% the measured uplift FLIPS NEGATIVE
# (long -0.28bps, short -0.63bps) — shadow-mode observation will arbitrate.
#   keep-70% op point was: LONG 0.30 / SHORT 0.4333
KRONOS_THR_LONG = 0.50        # keep LONG iff p_up >= this
KRONOS_THR_SHORT = 0.70      # keep SHORT iff (1 - p_up) >= this
KRONOS_VETO_LOG = "data/kronos_veto_live.jsonl"

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
# Master switch for the Stage-1 "flash" veto (structural-wall + momentum-trap
# checks). When False, S1 is skipped entirely: trades go straight to the Stage-2
# news/governance audit (which can still veto), and the S1 Gemini call is saved.
# Used to A/B test whether S1 is blocking too many good trades. Flip at runtime
# (no code edit) with GEMINI_S1_VETO_ENABLED=0 in the environment/.env.
GEMINI_S1_VETO_ENABLED = os.getenv("GEMINI_S1_VETO_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
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
