import os
import time
import json
import re
from datetime import datetime, timedelta
import pandas as pd
from google import genai
from google.genai import types
from scripts.vanguard import config
from scripts.terminal_utils import log

class GeminiRateTracker:
    def __init__(self, state_file=config.GEMINI_STATE_FILE, max_keys=3):
        self.state_file = state_file
        self.models = config.GEMINI_MODEL_TIERS
        self.max_requests_per_day = config.GEMINI_MAX_REQUESTS_PER_DAY
        self.max_keys = max_keys
        self.state = self._load_state()

    def _load_state(self):
        today = datetime.now().strftime("%Y-%m-%d")
        default_state = {
            "date": today,
            "usage": {m: {str(i): 0 for i in range(self.max_keys)} for m in self.models}
        }
        
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                if state.get("date") == today:
                    modified = False
                    for m in self.models:
                        if m not in state["usage"]:
                            state["usage"][m] = {str(i): 0 for i in range(self.max_keys)}
                            modified = True
                        else:
                            for i in range(self.max_keys):
                                k_str = str(i)
                                if k_str not in state["usage"][m]:
                                    state["usage"][m][k_str] = 0
                                    modified = True
                            keys_to_remove = [k for k in state["usage"][m] if int(k) >= self.max_keys]
                            if keys_to_remove:
                                for k in keys_to_remove:
                                    del state["usage"][m][k]
                                modified = True
                    if modified:
                        self._save_state(state)
                    return state
            except Exception:
                pass
        
        self._save_state(default_state)
        return default_state

    def _save_state(self, state=None):
        if state is None:
            state = self.state
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=4)

    def get_next_available(self, max_keys=3):
        for model in self.models:
            for i in range(max_keys):
                key_idx = str(i)
                if self.state["usage"][model].get(key_idx, 0) < self.max_requests_per_day:
                    return model, i
        
        log("[ROTATE-RESET] All API keys are exhausted. Resetting all usage statistics to 0 to rotate again.")
        for model in self.models:
            for i in range(max_keys):
                key_idx = str(i)
                self.state["usage"][model][key_idx] = 0
        self._save_state()
        
        if self.models and max_keys > 0:
            return self.models[0], 0
        return None, None

    def increment_usage(self, model, key_idx):
        idx_str = str(key_idx)
        if model in self.state["usage"] and idx_str in self.state["usage"][model]:
            self.state["usage"][model][idx_str] += 1
            self._save_state()

    def mark_exhausted(self, model, key_idx):
        idx_str = str(key_idx)
        if model in self.state["usage"] and idx_str in self.state["usage"][model]:
            self.state["usage"][model][idx_str] = self.max_requests_per_day
            self._save_state()


class AIVetoManager:
    def __init__(self, min_conviction=config.MIN_CONVICTION):
        self.gemini_enabled = config.GEMINI_ENABLED_DEFAULT
        self.api_keys = []
        self.clients = []
        self.min_conviction = min_conviction

        # Rotation State
        self.s1_model_tiers = config.GEMINI_MODEL_TIERS
        self.s1_active_tier_idx = 0
        self.s1_active_key_idx = 0

        self.sentiment_cache = {}

        try:
            keys_env = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
            if keys_env:
                self.api_keys = [k.strip() for k in keys_env.split(",") if k.strip()]

            if self.api_keys:
                self.clients = [genai.Client(api_key=k) for k in self.api_keys]
                log(f"[OK] Gemini AI Veto Manager: ACTIVE ({len(self.api_keys)} Keys Loaded)")
            else:
                self.gemini_enabled = False
                log("[WARN] Gemini API Keys not found. AI Audit Layer: DISABLED.")
        except Exception as e:
            self.gemini_enabled = False
            log(f"[ERROR] Gemini Config Error: {e}")

        self.gemini_tracker = GeminiRateTracker(max_keys=len(self.api_keys) if self.api_keys else 1)

    @staticmethod
    def _compute_sr_levels(features: dict, price: float) -> dict:
        """Derives key support/resistance levels from pre-computed feature values."""
        def pct_to_price(pct_dist):
            try:
                v = float(pct_dist)
                return round(price / (1 - v), 2) if abs(v) < 0.5 else None
            except Exception:
                return None

        bb_upper = pct_to_price(features.get("Dist_BB_Upper", 0))
        bb_lower = pct_to_price(-abs(float(features.get("Dist_BB_Lower", 0))))

        don_upper = pct_to_price(features.get("Dist_Donchian_Upper", 0))
        don_lower = pct_to_price(-abs(float(features.get("Dist_Donchian_Lower", 0))))

        atr_pct  = float(features.get("HL_Range", 0.015))
        atr_abs  = round(price * atr_pct, 2)
        r1_atr   = round(price + atr_abs, 2)
        s1_atr   = round(price - atr_abs, 2)

        sma6  = pct_to_price(features.get("Dist_SMA6", 0))
        sma12 = pct_to_price(features.get("Dist_SMA12", 0))
        sma50 = pct_to_price(features.get("Dist_SMA50", 0))

        return {
            "price": price,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "don_upper": don_upper,
            "don_lower": don_lower,
            "r1_atr": r1_atr,
            "s1_atr": s1_atr,
            "atr_abs": atr_abs,
            "sma6": sma6,
            "sma12": sma12,
            "sma50": sma50,
            "high_52w": pct_to_price(features.get("Dist_52W_High", 0))
        }

    def _extract_response_text(self, resp) -> str:
        try:
            if resp.text:
                return resp.text
        except Exception:
            pass

        try:
            for candidate in resp.candidates:
                for part in candidate.content.parts:
                    t = getattr(part, "text", None)
                    if t:
                        return t
        except Exception:
            pass
        return ""

    def parse_gemini_json(self, text):
        if not text:
            return {}
        text = text.strip()

        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        def fix_control_chars(m):
            return m.group(0).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

        text = re.sub(r'"(?:[^"\\]|\\.)*"', fix_control_chars, text, flags=re.DOTALL)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
            if not cleaned.strip().startswith("{"):
                cleaned = "{" + cleaned
            if not cleaned.strip().endswith("}"):
                cleaned = cleaned + "}"
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            extracted = {}
            keys_to_salvage = [
                "news_found", "chain_of_thought", "structural_bias", "veto_decision",
                "veto_rule_triggered", "final_sentiment", "support_resistance_risk",
                "probability", "risk_factor", "sentiment", "reason"
            ]
            for k in keys_to_salvage:
                m = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.IGNORECASE | re.DOTALL)
                if m:
                    extracted[k] = m.group(1).replace('\\"', '"').replace('\\n', '\n')
            
            if extracted and any(k in extracted for k in ("veto_decision", "final_sentiment", "sentiment")):
                log(f"[JSON-RECOVERY] Salvaged {len(extracted)} keys via regex.")
                return extracted

            log(f"[ERROR] JSON Parse Error: {e} | Raw snippet: {text[:150]}...")
            return {}

    def gemini_audit(self, ticker, side, conviction, features, get_recent_candles_fn):
        if not self.gemini_enabled:
            return "NEUTRAL", "Audit Disabled", "N/A"

        cache_key = f"{ticker}_{side}"
        if cache_key in self.sentiment_cache:
            cached_data = self.sentiment_cache[cache_key]
            if len(cached_data) == 4:
                sent, reason, ts, prob = cached_data
            else:
                sent, reason, ts = cached_data
                prob = "N/A"

            if time.time() - ts < 3600:
                log(f"[CACHE-HIT] Using cached sentiment for {ticker}")
                return sent, f"[CACHED] {reason}", prob

        current_price = float(features.get("Close", 0))
        sr = self._compute_sr_levels(dict(features), current_price)

        strategy_id = features.get("strategy_id") if hasattr(features, "get") else (features["strategy_id"] if "strategy_id" in features else None)
        signal_source = features.get("signal_source") if hasattr(features, "get") else (features["signal_source"] if "signal_source" in features else None)
        is_ensemble = features.get("is_ensemble", False) if hasattr(features, "get") else (features["is_ensemble"] if "is_ensemble" in features else False)

        strategy_descr = {
            2: "Short-Side Specialist [Rules: ML Short_Rank <= 5; Target: Top bearish momentum stocks selected by the machine learning model]",
            8: "Opening Range Breakout Short [Rules: Time >= 10:00 AM, ML Short_Rank <= 5, Close <= Low * 1.002; Target: Strong bearish breakdown candidates selling off heavily near their session lows]",
            10: "Quad-Timeframe Unanimous Long [Rules: ML Long_Rank <= 3; Target: Extremely strong bullish trend unanimity aligned across 15m, 1h, and daily timeframes]",
            18: "Volatility Expansion Short [Rules: ML Short_Rank <= 3, ATR_14_Pct > 1.5%; Target: Bearish setups displaying massive expansion in true range/volatility for high-momentum shorting]",
            19: "Low-Vol Grind Long [Rules: ML Long_Rank <= 5, IBS < 0.15; Target: High-ranked bullish stocks experiencing quiet consolidation or intraday low-volume accumulation near the bottom of their range, building base for breakout]",
            35: "Volatility Contraction Long [Rules: ML Long_Rank <= 5, ATR_14_Pct < 0.2%; Target: Extreme volatility compression / squeeze setups priming for a explosive upward breakout]",
            36: "The Opening Drive Long [Rules: Time <= 10:30 AM, ML Long_Rank <= 10, Gap_Pct > 0.3%; Target: Stocks displaying massive gap-and-go morning momentum drives on strong initial volume]",
            39: "The VWAP Pinch Long [Rules: ML Long_Rank <= 3, ATR_14_Pct < 0.2%, abs(Close - SMA20) / SMA20 < 0.2%; Target: Squeeze setups where the close is pinched tight against the 20-period moving average under low volatility]",
            42: "Trend Exhaustion Trap Short [Rules: ML Short_Rank <= 10, IBS < 0.3; Target: Bearish setups sitting near the daily low, identifying high-probability exhaustion/seller trap setups for a continuation breakdown]"
        }

        strategy_details = []
        if strategy_id is not None and not pd.isna(strategy_id):
            strategy_details.append(f"Strategy Triggered: S{int(strategy_id)} - {strategy_descr.get(int(strategy_id), 'Unknown Strategy')}")
        if signal_source:
            strategy_details.append(f"Signal Pipeline Source: {signal_source}")
        if is_ensemble:
            strategy_details.append("★ ENSEMBLE CONFLUENCE ★: This high-conviction signal is simultaneously matched and confirmed by BOTH the mathematical structural strategy filter AND the machine learning predictive model pipeline.")

        strategy_str = "\n".join(f"  → {d}" for d in strategy_details) if strategy_details else "  → Pure AI Prediction (No structural strategy triggered)"

        price_history_str = "N/A"
        try:
            hist_df = get_recent_candles_fn(ticker, interval='1minute', count=30)
            if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
                recent = hist_df.tail(30)
                history_points = []
                for _, row in recent.iterrows():
                    t_str = row['timestamp'].strftime('%H:%M') if 'timestamp' in row else "??:??"
                    history_points.append(f"{t_str}: ₹{row['close']:.2f}")
                price_history_str = " | ".join(history_points)
        except Exception as e:
            log(f"[WARN] Failed to fetch price history for {ticker}: {e}")

        nearest_wall_label = "nearest RESISTANCE" if side == "LONG" else "nearest SUPPORT"
        bb_upper = sr["bb_upper"]
        bb_lower = sr["bb_lower"]
        don_upper = sr["don_upper"]
        don_lower = sr["don_lower"]
        r1_atr = sr["r1_atr"]
        s1_atr = sr["s1_atr"]
        price = sr["price"]
        sma6 = sr["sma6"]
        sma12 = sr["sma12"]
        sma50 = sr["sma50"]
        high_52w = sr["high_52w"]
        atr_abs = sr["atr_abs"]

        if side == "LONG":
            nearest_wall_price = min([x for x in [bb_upper, don_upper, r1_atr] if x and x > price], default=None)
            sr_context = (
                f"CURRENT PRICE: ₹{sr['price']}\n"
                f"RESISTANCE LEVELS ABOVE (potential reversal zones for LONG):\n"
                f"  • Bollinger Upper Band : ₹{sr['bb_upper']}  "
                f"(dist: {round((sr['bb_upper']/sr['price']-1)*100,2) if sr['bb_upper'] else 'N/A'}%)\n"
                f"  • 20-bar Donchian High : ₹{sr['don_upper']}  "
                f"(dist: {round((sr['don_upper']/sr['price']-1)*100,2) if sr['don_upper'] else 'N/A'}%)\n"
                f"  • 1-ATR Resistance     : ₹{sr['r1_atr']}  (ATR ≈ ₹{sr['atr_abs']})\n"
                f"  • SMA-6                : ₹{sr['sma6']}\n"
                f"  • SMA-12               : ₹{sr['sma12']}\n"
                f"  • SMA-50 (50-day MA)   : ₹{sr['sma50']}\n"
                f"  • 52-Week High         : ₹{sr['high_52w']}\n"
                f"QUESTION: Is there a SIGNIFICANT RESISTANCE ZONE within 0.5% above ₹{sr['price']} "
                f"that could cap this LONG trade and cause a reversal? "
                f"Answer in 'resistance_risk': HIGH / MEDIUM / LOW"
            )
        else:
            nearest_wall_price = max([x for x in [bb_lower, don_lower, s1_atr] if x and x < price], default=None)
            sr_context = (
                f"CURRENT PRICE: ₹{sr['price']}\n"
                f"SUPPORT LEVELS BELOW (potential bounce zones for SHORT):\n"
                f"  • Bollinger Lower Band : ₹{sr['bb_lower']}  "
                f"(dist: {round((1-sr['bb_lower']/sr['price'])*100,2) if sr['bb_lower'] else 'N/A'}%)\n"
                f"  • 20-bar Donchian Low  : ₹{sr['don_lower']}  "
                f"(dist: {round((1-sr['don_lower']/sr['price'])*100,2) if sr['don_lower'] else 'N/A'}%)\n"
                f"  • 1-ATR Support        : ₹{sr['s1_atr']}  (ATR ≈ ₹{sr['atr_abs']})\n"
                f"  • SMA-6                : ₹{sr['sma6']}\n"
                f"  • SMA-12               : ₹{sr['sma12']}\n"
                f"  • SMA-50 (50-day MA)   : ₹{sr['sma50']}\n"
                f"  • 52-Week High         : ₹{sr['high_52w']}\n"
                f"QUESTION: Is there a SIGNIFICANT SUPPORT ZONE within 0.5% below ₹{sr['price']} "
                f"that could bounce this SHORT trade and cause a reversal? "
                f"Answer in 'support_risk': HIGH / MEDIUM / LOW"
            )

        if nearest_wall_price:
            wall_dist = round(abs(price / nearest_wall_price - 1) * 100, 2)
            wall_str = f"₹{nearest_wall_price} ({wall_dist}% away)"
        else:
            wall_str = "N/A"

        def _f(key, default="N/A", pct=False, decimals=2):
            v = features.get(key, default)
            try:
                v = float(v)
                return f"{v * 100:.{decimals}f}%" if pct else f"{v:.{decimals}f}"
            except Exception:
                return str(default)

        def _vs(level):
            if not level: return "N/A"
            diff = round((price / level - 1) * 100, 2)
            return f"{'above' if diff > 0 else 'below'} by {abs(diff)}%"

        percent_b_raw = _f("PercentB_Raw", decimals=3)
        stoch_k = _f("Stoch_K_Raw", decimals=1)
        rvol = _f("rvol_raw", decimals=2)
        dv_cr = features.get("dv_raw", 0)
        try:    dv_str = f"₹{float(dv_cr)/1e7:.1f} Cr"
        except: dv_str = "N/A"

        up_streak = int(features.get("Up_Streak", 0) or 0)
        dn_streak = int(features.get("Down_Streak", 0) or 0)
        green_ratio = _f("Green_Bar_Ratio_5", pct=True, decimals=0)
        bar_pos = _f("Bar_Position", decimals=2)
        accumulation = _f("Accumulation_5", decimals=2)

        nifty_1h = _f("Nifty_1H_Return", pct=True, decimals=2)
        nifty_5h = _f("Nifty_5H_Return", pct=True, decimals=2)
        vix = _f("VIX_Level", decimals=1)
        vix_ma = _f("VIX_5D_MA", decimals=1)
        vix_extreme = features.get("VIX_Extreme", 0)
        regime_raw = int(features.get("Market_Regime", 0) or 0)
        regime_str = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}.get(regime_raw, "NEUTRAL")
        stock_vs_nifty = _f("Stock_vs_Nifty", pct=True, decimals=2)
        vs_sector = _f("Stock_vs_Sector", pct=True, decimals=2)
        sector_breadth = _f("Sector_Breadth", pct=True, decimals=0)

        daily_rsi = _f("Daily_RSI", decimals=1)
        daily_trend = int(features.get("Daily_Trend", 0) or 0)
        daily_trend_str = {1: "UPTREND (5D)", -1: "DOWNTREND (5D)", 0: "SIDEWAYS (5D)"}.get(daily_trend, "N/A")
        daily_sma20 = _f("Daily_SMA20_Dist", pct=True, decimals=2)
        daily_atr = _f("Daily_ATR_Pct", pct=True, decimals=2)

        prompt_flash = f"""You are a professional intraday risk analyst checking if a trade is blocked by an immediate hard structural price wall.
The machine learning model has already approved the trade's technical strength (RSI, volume, trend, conviction). You must NOT evaluate momentum, volume, or trend.
Your ONLY job is a geometric check: Is there an immediate hard structural price wall (Bollinger Bands, Donchian channels, SMAs, 52W High) within 0.2% of the current price that physically blocks this trade?

═══ TRADE PROPOSAL ═══════════════════════════════════════════════
TICKER  : {ticker}
SIDE    : {side}
PRICE   : ₹{price}
ML CONVICTION : {conviction:.4f}

═══ PRICE STRUCTURE & KEY LEVELS ═════════════════════════════════
{sr_context}

═══ RECENT PRICE ACTION (last 30 min, 1-min bars) ════════════════
{price_history_str}

═══ TASK ══════════════════════════════════════════════════════════
Determine if the trade is physically blocked by an immediate structural price wall (within 0.2% of current price).
- If there is a massive resistance/support level within 0.2%, set "veto" to "TRUE" and state the level in "reason".
- Otherwise, set "veto" to "FALSE" and "reason" to "No immediate wall within 0.2%".
- Default bias is PASS (veto = FALSE). The ML model's statistics are preferred.

Output STRICT JSON only — no markdown, no extra text:
{{"veto": "TRUE|FALSE", "reason": "concise explanation"}}
"""

        sent1, reason1, prob1 = "PASS", "N/A", "N/A"
        stage1_success = False

        total_combinations = len(self.s1_model_tiers) * len(self.api_keys)
        attempts = 0
        
        while attempts < total_combinations:
            current_model = self.s1_model_tiers[self.s1_active_tier_idx]
            current_key = self.s1_active_key_idx
            
            try:
                layer1_client = self.clients[current_key]
                log(f"[S1] Attempting {current_model} (Key {current_key}) for {ticker}...")
                resp1 = layer1_client.models.generate_content(
                    model=current_model, contents=prompt_flash,
                    config=types.GenerateContentConfig(
                        temperature=0.1
                    )
                )
                data1 = self.parse_gemini_json(resp1.text)
                veto_s1 = str(data1.get("veto", "FALSE")).upper() == "TRUE"
                sent1 = "VETOED" if veto_s1 else "PASS"
                reason1 = data1.get("reason", "N/A")
                prob1 = "N/A"
                stage1_success = True
                log(f"[S1-OK] {current_model} (Key {current_key}) succeeded for {ticker}.")
                break
                
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "503" in err_str or "quota" in err_str.lower():
                    log(f"[S1-ROTATE] {current_model} Key {current_key} exhausted ({err_str[:80]}). Rotating...")
                    self.s1_active_key_idx += 1
                    if self.s1_active_key_idx >= len(self.api_keys):
                        self.s1_active_key_idx = 0
                        self.s1_active_tier_idx += 1
                        if self.s1_active_tier_idx >= len(self.s1_model_tiers):
                            self.s1_active_tier_idx = 0
                        log(f"[S1-FALLBACK] All keys exhausted. Rotating to {self.s1_model_tiers[self.s1_active_tier_idx]}...")
                    attempts += 1
                    time.sleep(1)
                    continue
                else:
                    log(f"[WARN] Stage 1 non-quota error ({current_model}, Key {current_key}) for {ticker}: {e}")
                    time.sleep(2)
                    break
                    
        if not stage1_success:
            log(f"[S1-FAILED] Both model tiers failed for {ticker}. Skipping trade.")
            return "SYSTEM_ERROR", "Stage 1 Audit Error (All Models/Keys Exhausted)", "N/A"

        if sent1 == "VETOED":
            final_reason = f"[S1-VETO] {reason1}"
            self.sentiment_cache[cache_key] = (sent1, final_reason, time.time(), prob1)
            return sent1, final_reason, prob1

        # STAGE 2: HIERARCHICAL GOVERNANCE AUDIT
        log(f"[STAGE 2] Triggering Hierarchical Super-Veto Audit for {ticker}...")

        company_name = ticker.replace(".NS", "").replace(".BO", "").replace(".", " ").strip()
        current_dt = datetime.now().strftime("%Y-%m-%d %H:%M IST")

        s1_tech_summary = (
            f"RSI={_f('RSI_14_Raw', decimals=1)} | Stoch%K={stoch_k} | %B={percent_b_raw} | "
            f"RVOL={rvol}x | Regime={regime_str} | Nifty1H={nifty_1h} | "
            f"vs Nifty={stock_vs_nifty} | DailyTrend={daily_trend_str} | "
            f"UpStreak={up_streak} | DnStreak={dn_streak} | Conviction={conviction:.4f}"
        )

        # Retrieve rich sector context for Stage 2
        sector = features.get("Sector", "N/A")
        sector_rank = features.get("Sector_Rank", "N/A")
        try:
            sector_rank_str = f"#{int(float(sector_rank))}"
        except Exception:
            sector_rank_str = str(sector_rank)
        sector_mean_ret = _f("Sector_Mean_Return", pct=True, decimals=2)
        sector_breadth = _f("Sector_Breadth", pct=True, decimals=0)

        prompt_search = f"""# ROLE: Institutional Risk Auditor (CRO) — Capital Protection Division
# CURRENT DATE/TIME: {current_dt}
# YOUR ONLY OBJECTIVE: Determine if fundamental/news/corporate reality CONTRADICTS the technical signal.
# You are NOT here to predict price. You are here to VETO trades where news/fundamentals overrides technicals.

════════════════════════════════════════════════════════════════
AUDIT PROFILE
════════════════════════════════════════════════════════════════
Company         : {company_name} (NSE: {ticker})
Sector          : {sector} (Rank: {sector_rank_str} | Mean Return: {sector_mean_ret} | Breadth: {sector_breadth})
Proposed Trade  : {side}
Current Price   : ₹{price}
ML Conviction   : {conviction:.4f}
  → Universe Rank : #{int(features.get("Long_Rank" if side == "LONG" else "Short_Rank", 999))} of ~172 stocks screened this cycle
  → Min gate      : {self.min_conviction:.2f}
RVOL            : {rvol}x
{strategy_str}

S1 TECHNICAL VERDICT  : {sent1}
S1 PROBABILITY        : {prob1}
S1 TECHNICAL SUMMARY  : {s1_tech_summary}

KEY S/R CONTEXT
{sr_context}

RECENT PRICE ACTION (last 30 min, 1-min bars):
{price_history_str}

════════════════════════════════════════════════════════════════
STEP 1 — GROUNDED NEWS SEARCH (INDIAN FINANCIAL MARKETS)
════════════════════════════════════════════════════════════════
Perform highly targeted Google Searches to find localized Indian financial news (specifically targeting NSE/BSE listed entities).
Search for ALL of the following (use Google Search grounding):
  • "{company_name} NSE stock news today"
  • "{ticker.replace('.NS', '')} share price target brokerage"
  • "{company_name} block deal bulk deal today"
  • "{company_name} NSE results earnings dividend buyback corporate action"
  • "{company_name} SEBI regulatory dispute legal news"

Classify findings into two buckets:

BUCKET A — Structural (P0, last 7 days):
  Earnings beat/miss, target rating upgrades/downgrades, SEBI/regulatory action, promotors pledge/stake changes, dividend/split/merger corporate actions, industry shocks.
  → These OVERRIDE technicals. A rating downgrade or regulatory ban for a LONG = mandatory VETO.

BUCKET B — Tactical (P1, last 6 hours):
  Block deals, bulk deals, sudden localized volume spike explanations, AGM, analysts meet.
  → These MODIFY conviction but may not VETO unless strong directional conflict.

If NO material news is found in either bucket, explicitly state "No material catalyst found."

════════════════════════════════════════════════════════════════
STEP 2 — VETO DECISION MATRIX
════════════════════════════════════════════════════════════════
Apply the following rules IN ORDER (first triggered rule wins):

RULE 1 — HARD VETO (Fundamental & Corporate Conflict):
  • LONG + Bucket A shows: earnings miss, rating downgrade, promoter sell/pledge increase, regulatory ban/SEBI fine, tax demand, negative corporate governance issue → VETO=TRUE
  • SHORT + Bucket A shows: earnings beat, rating upgrade, buyback announcement, promoter buy/pledge reduction, major order win, regulatory clearance → VETO=TRUE

RULE 2 — SOFT VETO (Tactical Conflict):
  • LONG + Bucket B shows: large block sell, negative headline → consider VETO if ML conviction < 0.25
  • SHORT + Bucket B shows: large block buy, positive headline → consider VETO if ML conviction < 0.25

RULE 3 — BREAKOUT OVERRIDE (Do NOT veto):
  • If RVOL > 3.0 AND Bucket A/B shows strong POSITIVE catalysts AND {side}=LONG:
    Resistance levels are TARGETS in a genuine breakout, not walls. Do NOT veto.

RULE 4 — NO NEWS = PASS:
  • If no material news/governance issues are found, set veto_decision=FALSE.

════════════════════════════════════════════════════════════════
STEP 3 — FINAL OUTPUT
════════════════════════════════════════════════════════════════
Output STRICT JSON only — absolutely no markdown, no extra text, no commentary:
{{
  "news_found": "Brief description of Bucket A and B news, or 'No material catalyst found'",
  "chain_of_thought": "does the news CONFIRM, CONTRADICT, or is NEUTRAL to the {side} technical signal?",
  "structural_bias": "BULLISH|BEARISH|NEUTRAL",
  "veto_decision": "TRUE|FALSE",
  "veto_rule_triggered": "RULE 1|RULE 2|RULE 3|RULE 4|NONE",
  "final_sentiment": "STRONG BULLISH|BULLISH|NEUTRAL|BEARISH|STRONG BEARISH",
  "support_resistance_risk": "HIGH|MEDIUM|LOW",
  "probability": "XX%",
  "risk_factor": "the one most likely reason this trade fails"
}}
"""

        stage2_retries = 2
        for _ in range(stage2_retries):
            model_name, key_idx = self.gemini_tracker.get_next_available(len(self.api_keys))
            if model_name is None:
                log(f"[L2-FAILED] Stage 2 limits exhausted for {ticker}. Skipping.")
                return "SYSTEM_ERROR", "[L2-FAILED] Layer 2 audit skipped: API limits exhausted.", "N/A"

            try:
                stage2_client = self.clients[key_idx]
                resp2 = stage2_client.models.generate_content(
                    model=model_name,
                    contents=prompt_search,
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}], 
                        temperature=0.1
                    )
                )
                resp2_text = self._extract_response_text(resp2)
                data2 = self.parse_gemini_json(resp2_text)

                if not data2 or not any(k in data2 for k in ("veto_decision", "final_sentiment", "chain_of_thought")):
                    log(f"[L2-EMPTY] Stage 2 returned empty/unparseable JSON for {ticker} (using {model_name}). Skipping.")
                    return "SYSTEM_ERROR", "[L2-EMPTY] Layer 2 returned no usable data.", "N/A"

                self.gemini_tracker.increment_usage(model_name, key_idx)

                news_found = data2.get("news_found", "N/A")
                cot = data2.get("chain_of_thought", "N/A")
                bias = data2.get("structural_bias", "NEUTRAL")
                veto_triggered = str(data2.get("veto_decision", "FALSE")).upper() == "TRUE"
                veto_rule = data2.get("veto_rule_triggered", "NONE")
                sent2 = data2.get("final_sentiment", "NEUTRAL").upper()
                prob2 = data2.get("probability", prob1)
                risk = data2.get("risk_factor", "N/A")
                sr_risk = data2.get("support_resistance_risk", "LOW").upper()

                if veto_triggered:
                    sent2 = "VETOED"
                    final_reason = f"[VETOED by {model_name} | {veto_rule}] News: {news_found} | {cot} | Risk: {risk}"
                    log(f"[HIERARCHICAL VETO] {ticker} {side} VETOED ({veto_rule}) by {model_name}: {cot}")
                else:
                    final_reason = f"[S2-PASS by {model_name} | {veto_rule}] News: {news_found} | {cot} | Bias: {bias} (S1: {sent1})"
                    log(f"[STAGE 2 PASS] {ticker} {side} cleared by {model_name} [{veto_rule}]: {cot}")

                self.sentiment_cache[cache_key] = (sent2, final_reason, time.time(), prob2)
                return sent2, final_reason, prob2

            except Exception as e_search:
                err_str = str(e_search)
                if "429" in err_str or "503" in err_str or "UNAVAILABLE" in err_str or "quota" in err_str.lower():
                    log(f"[RETRY] Stage 2 Rate Limited ({model_name}, Key {key_idx}). Retrying...")
                    self.gemini_tracker.mark_exhausted(model_name, key_idx)
                    time.sleep(1)
                    continue
                else:
                    log(f"[WARN] Stage 2 Search Error for {ticker}: {e_search}")
                    break

        return "SYSTEM_ERROR", "[L2-FAILED] Layer 2 search audit failed (API error/timeout).", "N/A"
