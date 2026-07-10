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

class AIVetoManager:
    def __init__(self, min_conviction=0.0):
        self.gemini_enabled = config.GEMINI_ENABLED_DEFAULT
        self.min_conviction = min_conviction

        # Rotation State
        self.s1_model_tiers = getattr(config, "GEMINI_S1_MODEL_TIERS", config.GEMINI_MODEL_TIERS)
        self.s2_model_tiers = getattr(config, "GEMINI_S2_MODEL_TIERS", config.GEMINI_MODEL_TIERS)
        # Round-robin the top-N S1 "primary" models per audit so one overloaded
        # model isn't always tried first; -lite fallbacks stay fixed at the tail.
        self.s1_primary_rotate = getattr(config, "GEMINI_S1_PRIMARY_ROTATE", 2)
        self.s1_tier_rotation = 0

        self.sentiment_cache = {}

        from scripts.gemini_client_manager import GeminiRotator
        self.rotator = GeminiRotator()
        self.api_keys = self.rotator.main_keys

        if self.rotator.main_keys or self.rotator.backup_key:
            log(f"[OK] Gemini AI Veto Manager: ACTIVE ({len(self.rotator.main_keys)} Main Keys, Backup Key: {'Configured' if self.rotator.backup_key else 'None'})")
        else:
            self.gemini_enabled = False
            log("[WARN] Gemini API Keys not found. AI Audit Layer: DISABLED.")

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

    def _next_s1_tiers(self):
        """S1 model order for this audit, rotating the top-N 'primary' models so a
        single overloaded model isn't always tried first. The -lite fallback tiers
        keep their fixed order at the tail. Advances the rotation each call."""
        tiers = list(self.s1_model_tiers)
        n = min(self.s1_primary_rotate, len(tiers))
        if n <= 1:
            return tiers
        offset = self.s1_tier_rotation % n
        self.s1_tier_rotation = (self.s1_tier_rotation + 1) % n
        primaries = tiers[:n]
        return primaries[offset:] + primaries[:offset] + tiers[n:]

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

        # Support robust plain-text block parsing for Lite models
        if "[VETO_DECISION]" in text or "[FINAL_SENTIMENT]" in text or "[NEWS_FOUND]" in text:
            extracted = {}
            matches = re.finditer(r'\[([A-Z_]+)\]\s*(.*?)(?=\s*\[[A-Z_]+\]|$)', text, re.DOTALL)
            for m in matches:
                extracted[m.group(1).lower()] = m.group(2).strip()
            if extracted and any(k in extracted for k in ("veto_decision", "final_sentiment", "chain_of_thought")):
                log(f"[BLOCK-RECOVERY] Extracted {len(extracted)} keys via robust block parsing.")
                return extracted

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

    def gemini_audit(self, ticker, side, conviction, features, get_recent_candles_fn, use_cache=True):
        if not self.gemini_enabled:
            return "NEUTRAL", "Audit Disabled", "N/A"

        cache_key = f"{ticker}_{side}"
        # use_cache=False forces a fresh audit (SHADOW_ALL_LAYERS: S2 must run on every
        # tracked signal, like the candle/Kronos layers — not reuse a verdict from an
        # earlier anchor within the hour).
        if use_cache and cache_key in self.sentiment_cache:
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

        # ── Momentum / price-action context for the S1 trap veto ─────────────
        # Anchored to the code-level FADE_QUALITY_GUARD thresholds (config.py) so
        # the model's judgment uses the same rvol discriminator the engine does:
        # a knife/breakout only runs us over when VOLUME is behind it.
        knife_rvol    = config.FADE_ADVERSE_MIN_RVOL                 # below this, an adverse move is noise
        breakout_rvol = config.FADE_BREAKOUT_RVOL                    # heavy participation behind a breakout
        breakout_52h  = abs(config.FADE_BREAKOUT_52H_PROXIMITY) * 100  # % from 52W high that counts as a breakout
        dist_52w_pct  = _f("Dist_52W_High", pct=True, decimals=2)
        dist_sma6     = _f("Dist_SMA6", pct=True, decimals=2)
        dist_sma12    = _f("Dist_SMA12", pct=True, decimals=2)

        momentum_block = (
            f"RVOL (relative volume)  : {rvol}x\n"
            f"Up-streak / Down-streak : {up_streak} / {dn_streak} consecutive bars\n"
            f"Green-bar ratio (last 5): {green_ratio}\n"
            f"Close position in bar   : {bar_pos}  (1.0 = top of bar, 0.0 = bottom)\n"
            f"Distance vs SMA-6 / 12  : {dist_sma6} / {dist_sma12}\n"
            f"Distance below 52W High : {dist_52w_pct}\n"
            f"Daily trend             : {daily_trend_str} | Daily RSI {daily_rsi}"
        )

        if side == "LONG":
            trap_check = (
                "This is a LONG. The momentum trap to block is a FALLING KNIFE — buying a stock that is in an\n"
                "active, volume-backed free-fall. Set veto=TRUE only when ALL THREE hold:\n"
                "  1. The last 30 min of 1-min bars show a STEEP, PERSISTENT decline (lower lows, mostly red\n"
                "     bars) that is STILL falling in the final ~5 bars — no basing or stabilization yet.\n"
                "  2. Price is BELOW SMA-6 and SMA-12 and the daily trend is DOWN.\n"
                f"  3. RVOL >= {knife_rvol} — real volume is behind the drop (not a low-volume drift).\n"
                f"Do NOT veto if the decline has STALLED / is basing or turning back up in the last few bars (a\n"
                f"stabilizing knife is the valid mean-reversion entry this LONG wants), or if RVOL < {knife_rvol}\n"
                "(a light-volume dip is noise that tends to bounce)."
            )
        else:
            trap_check = (
                "This is a SHORT. The momentum trap to block is a BREAKOUT — shorting a stock that is in an\n"
                "active, volume-backed breakout (standing in front of a freight train). Set veto=TRUE only when\n"
                "ALL THREE hold:\n"
                "  1. The last 30 min of 1-min bars show a STEEP, PERSISTENT advance (higher highs, mostly green\n"
                "     bars) that is STILL rising in the final ~5 bars — no stalling or rollover yet.\n"
                f"  2. Price is within {breakout_52h:.1f}% of the 52-week high OR breaking the upper Donchian /\n"
                "     Bollinger band (a fresh breakout with little overhead supply).\n"
                f"  3. RVOL >= {breakout_rvol} — heavy participation is confirming the breakout.\n"
                "Do NOT veto if the advance has STALLED / rolled over in the last few bars (exhaustion is the\n"
                f"valid fade this SHORT wants), or if RVOL < {breakout_rvol} (an unconfirmed pop tends to fade)."
            )

        prompt_flash = f"""You are a professional intraday risk analyst running a fast veto on a trade the machine-learning model has ALREADY approved on technical strength (RSI, volume, trend, conviction). You do NOT re-score the model's edge. You run exactly TWO independent veto checks and BLOCK the trade if EITHER one fails.

═══ TRADE PROPOSAL ═══════════════════════════════════════════════
TICKER  : {ticker}
SIDE    : {side}
PRICE   : ₹{price}
ML CONVICTION : {conviction:.4f}

═══ PRICE STRUCTURE & KEY LEVELS ═════════════════════════════════
{sr_context}

═══ MOMENTUM & PRICE-ACTION ══════════════════════════════════════
{momentum_block}

═══ RECENT PRICE ACTION (last 30 min, 1-min bars) ════════════════
{price_history_str}

═══ CHECK A — STRUCTURAL WALL (geometry only) ════════════════════
Is there an immediate HARD structural price wall (Bollinger Bands, Donchian channels, SMAs, 52W High) within 0.2% of ₹{price} that physically blocks this {side}? If yes → veto=TRUE and name the level.

═══ CHECK B — MOMENTUM TRAP (price action) ═══════════════════════
{trap_check}

═══ DECISION ═════════════════════════════════════════════════════
- veto=TRUE if CHECK A finds a hard wall within 0.2% OR CHECK B identifies a live, still-running, volume-backed trap.
- Otherwise veto=FALSE. Default bias is PASS — the ML model's statistics are preferred. A single elevated indicator is NOT enough; only block on a clear wall, or a clear trap where the move is steep, still running, AND volume-backed per the thresholds above.

Output STRICT JSON only — no markdown, no extra text:
{{"veto": "TRUE|FALSE", "reason": "name the CHECK A wall or the CHECK B trap; else 'No wall within 0.2%, no live trap'"}}
"""

        sent1, reason1, prob1 = "PASS", "N/A", "N/A"
        stage1_success = False

        # --- S1 master switch (A/B test: is the flash veto killing good trades?) ---
        # When disabled, skip the Stage-1 model call entirely and fall through to the
        # Stage-2 news/governance audit (which can still veto). S1 verdict is treated
        # as PASS so S2 runs its independent check; the saved call also eases key load.
        if not getattr(config, "GEMINI_S1_VETO_ENABLED", True):
            log(f"[S1-BYPASS] Stage 1 flash veto DISABLED (GEMINI_S1_VETO_ENABLED=0) — {ticker} {side} routed straight to Stage 2.")
            sent1, reason1, prob1 = "PASS", "S1 bypassed (flash veto disabled)", "N/A"
            stage1_success = True
            s1_tiers = []
        else:
            s1_tiers = self._next_s1_tiers()
            log(f"[S1] Model order this audit: {' -> '.join(s1_tiers)}")
        for current_model in s1_tiers:
            try:
                log(f"[S1] Attempting {current_model} via rotator for {ticker}...")
                def run_s1(client):
                    return client.models.generate_content(
                        model=current_model, contents=prompt_flash,
                        config=types.GenerateContentConfig(
                            temperature=0.1
                        )
                    )
                resp1 = self.rotator.execute(run_s1)
                data1 = self.parse_gemini_json(resp1.text)
                veto_s1 = str(data1.get("veto", "FALSE")).upper() == "TRUE"
                sent1 = "VETOED" if veto_s1 else "PASS"
                reason1 = data1.get("reason", "N/A")
                prob1 = "N/A"
                stage1_success = True
                log(f"[S1-OK] {current_model} succeeded for {ticker}.")
                break
            except Exception as e:
                log(f"[S1-WARN] Stage 1 call failed on model {current_model}: {e}")
                time.sleep(1)
                    
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

RULE 2.5 — MOMENTUM TRAP (price action contradicts the trade — applies even with NO news):
  • SHORT into a live BREAKOUT — RECENT PRICE ACTION ripping to higher highs, price within ~{breakout_52h:.1f}% of
    the 52-week high or breaking the upper Donchian/Bollinger band, RVOL >= {breakout_rvol} → VETO=TRUE.
    Do not stand in front of a confirmed, volume-backed breakout.
  • LONG into a live FALLING KNIFE — RECENT PRICE ACTION in a steep persistent decline still making lower lows,
    price below SMA-6/SMA-12 in a daily downtrend, RVOL >= {knife_rvol} → VETO=TRUE.
    Do not catch a volume-backed knife.
  • EXCEPTION: if the move has clearly STALLED / rolled over / based in the last few bars (exhaustion), this rule
    does NOT fire — a stabilizing knife or a stalling breakout is the valid mean-reversion entry.

RULE 3 — GENUINE BREAKOUT / BREAKDOWN OVERRIDE (Do NOT veto on S/R proximity alone):
  • LONG + RVOL > 3.0 + Bucket A/B strong POSITIVE catalyst: resistance levels are TARGETS in a real breakout,
    not walls. Do NOT veto.
  • SHORT + RVOL > 3.0 + Bucket A/B strong NEGATIVE catalyst: support levels are TARGETS in a real breakdown,
    not floors. Do NOT veto.

RULE 4 — NO NEWS = PASS:
  • If no material news/governance issues are found AND RULE 2.5 did not fire, set veto_decision=FALSE.

════════════════════════════════════════════════════════════════
STEP 3 — FINAL OUTPUT
════════════════════════════════════════════════════════════════
Output your response strictly in the following plain-text format (Do NOT use JSON or markdown):
[NEWS_FOUND] Brief description of Bucket A and B news, or 'No material catalyst found'
[CHAIN_OF_THOUGHT] does the news CONFIRM, CONTRADICT, or is NEUTRAL to the technical signal?
[STRUCTURAL_BIAS] BULLISH | BEARISH | NEUTRAL
[VETO_DECISION] TRUE | FALSE
[VETO_RULE_TRIGGERED] RULE 1 | RULE 2 | RULE 3 | RULE 4 | NONE
[FINAL_SENTIMENT] STRONG BULLISH | BULLISH | NEUTRAL | BEARISH | STRONG BEARISH
[SUPPORT_RESISTANCE_RISK] HIGH | MEDIUM | LOW
[PROBABILITY] XX%
[RISK_FACTOR] the one most likely reason this trade fails
"""

        stage2_success = False
        sent2, final_reason, prob2 = "SYSTEM_ERROR", "[L2-FAILED] Layer 2 search audit failed.", "N/A"
        
        for model_name in self.s2_model_tiers:
            try:
                log(f"[S2] Attempting {model_name} via rotator for {ticker}...")
                def run_s2(client):
                    return client.models.generate_content(
                        model=model_name,
                        contents=prompt_search,
                        config=types.GenerateContentConfig(
                            tools=[{"google_search": {}}], 
                            temperature=0.1
                        )
                    )
                resp2 = self.rotator.execute(run_s2)
                resp2_text = self._extract_response_text(resp2)
                data2 = self.parse_gemini_json(resp2_text)

                if not data2 or not any(k in data2 for k in ("veto_decision", "final_sentiment", "chain_of_thought")):
                    log(f"[L2-EMPTY] Stage 2 returned empty/unparseable JSON for {ticker} (using {model_name}).")
                    log(f"[L2-DEBUG] Raw output snippet: {resp2_text[:300]}...")
                    continue

                news_found = data2.get("news_found", "N/A")
                cot = data2.get("chain_of_thought", "N/A")
                bias = data2.get("structural_bias", "NEUTRAL")
                veto_triggered = str(data2.get("veto_decision", "FALSE")).upper() == "TRUE"
                veto_rule = data2.get("veto_rule_triggered", "NONE")
                sent2 = data2.get("final_sentiment", "NEUTRAL").upper()
                prob2 = data2.get("probability", prob1)
                risk = data2.get("risk_factor", "N/A")
                sr_risk = data2.get("support_resistance_risk", "LOW").upper()

                # --- Bias-contradiction hard veto --------------------------------
                # RULE 1 only hard-vetoes on enumerated catalysts (rating UPGRADE,
                # earnings beat, ...), so a standing "Strong Buy / higher targets"
                # -> BULLISH bias can still pass a SHORT. Never trade directly against
                # the audit's own directional read (cf. SBILIFE.NS 2026-06-15: BULLISH
                # bias, shorted into strength, -0.33% SL).
                bias_u = str(bias).upper()
                if not veto_triggered and (
                    (side == "SHORT" and bias_u == "BULLISH") or
                    (side == "LONG" and bias_u == "BEARISH")
                ):
                    veto_triggered = True
                    veto_rule = "RULE 1B (bias-contradiction)"
                    risk = f"S2 structural bias {bias_u} directly contradicts the {side} trade direction."

                if veto_triggered:
                    sent2 = "VETOED"
                    final_reason = f"[VETOED by {model_name} | {veto_rule}] News: {news_found} | {cot} | Risk: {risk}"
                    log(f"[HIERARCHICAL VETO] {ticker} {side} VETOED ({veto_rule}) by {model_name}: {cot}")
                else:
                    final_reason = f"[S2-PASS by {model_name} | {veto_rule}] News: {news_found} | {cot} | Bias: {bias} (S1: {sent1})"
                    log(f"[STAGE 2 PASS] {ticker} {side} cleared by {model_name} [{veto_rule}]: {cot}")

                self.sentiment_cache[cache_key] = (sent2, final_reason, time.time(), prob2)
                stage2_success = True
                return sent2, final_reason, prob2

            except Exception as e_search:
                log(f"[S2-WARN] Stage 2 call failed on model {model_name}: {e_search}")
                time.sleep(1)

        if not stage2_success:
            return "SYSTEM_ERROR", "[L2-FAILED] Layer 2 search audit failed (API error/timeout).", "N/A"
