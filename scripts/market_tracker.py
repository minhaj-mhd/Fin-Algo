"""
market_tracker.py
=================
Provides two snapshot functions for the Vanguard dashboard:

  get_model_snapshot()  — derives current market sentiment from the XGBoost
                          conviction scores already computed by the engine.
                          No external API calls.

  get_market_snapshot() — fetches NIFTY 50 (^NSEI) and India VIX (^INDIAVIX)
                          from yfinance. Caches the result for 5 minutes to
                          avoid excessive external requests.
"""

import os
import json
import time
import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_SCORES_PATH = os.path.join(_BASE_DIR, "data", "latest_scores.json")
_GATEKEEPERS_PATH = os.path.join(_BASE_DIR, "data", "daily_gatekeepers.json")

# ---------------------------------------------------------------------------
# In-memory cache for market data (5-minute TTL)
# ---------------------------------------------------------------------------
_market_cache: dict = {}
_market_cache_time: float = 0.0
_MARKET_CACHE_TTL: float = 300.0  # seconds


# ===========================================================================
# MODEL SNAPSHOT
# ===========================================================================

def get_model_snapshot() -> dict:
    """
    Reads the latest XGBoost conviction scores from data/latest_scores.json
    and derives the current engine sentiment without any external API calls.

    Returns a JSON-serialisable dict:
    {
        "overall_sentiment": "BULLISH" | "BEARISH" | "MIXED" | "UNAVAILABLE",
        "sentiment_strength": float (0-100),
        "bullish_count": int,
        "bearish_count": int,
        "neutral_count": int,
        "total_tickers": int,
        "bullish_pct": float,
        "bearish_pct": float,
        "avg_long_conviction": float,
        "avg_short_conviction": float,
        "top_long_ticker": str,
        "top_long_conviction": float,
        "top_short_ticker": str,
        "top_short_conviction": float,
        "top5_long": [ {"rank": int, "ticker": str, "conviction": float}, ... ],
        "top5_short": [ {"rank": int, "ticker": str, "conviction": float}, ... ],
        "timestamp": str (ISO-8601),
        "scores_age_seconds": float
    }
    """
    empty = {
        "overall_sentiment": "UNAVAILABLE",
        "sentiment_strength": 0.0,
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": 0,
        "total_tickers": 0,
        "bullish_pct": 0.0,
        "bearish_pct": 0.0,
        "avg_long_conviction": 0.0,
        "avg_short_conviction": 0.0,
        "top_long_ticker": "—",
        "top_long_conviction": 0.0,
        "top_short_ticker": "—",
        "top_short_conviction": 0.0,
        "top5_long": [],
        "top5_short": [],
        "timestamp": datetime.datetime.now().isoformat(),
        "scores_age_seconds": -1.0,
        "daily_gatekeepers": {
            "timestamp": datetime.datetime.now().isoformat(),
            "long_eligible": [],
            "short_eligible": [],
            "long_eligible_count": 0,
            "short_eligible_count": 0
        }
    }

    try:
        if not os.path.exists(_SCORES_PATH):
            return empty

        scores_mtime = os.path.getmtime(_SCORES_PATH)
        scores_age = time.time() - scores_mtime

        with open(_SCORES_PATH, "r") as f:
            scores: list = json.load(f)

        if not scores:
            return empty

        total = len(scores)

        # ---------------------------------------------------------------
        # Categorise each ticker
        # Bullish  : Long_Conviction > 0  (model favours long side)
        # Bearish  : Short_Conviction > 0 (model favours short / negative)
        #            Short_Conviction is stored as a negative number when the
        #            model disfavours the short side.  A positive value means
        #            the model *prefers* shorting.
        # Neutral  : Long_Conviction <= 0 AND Short_Conviction <= 0
        # ---------------------------------------------------------------
        bullish_tickers = [s for s in scores if s.get("Long_Conviction", 0) > 0]
        bearish_tickers = [s for s in scores if s.get("Short_Conviction", 0) > 0]
        neutral_count   = total - len(bullish_tickers) - max(0, len(bearish_tickers) - len(bullish_tickers))

        bullish_count = len(bullish_tickers)
        bearish_count = len(bearish_tickers)
        neutral_count = total - bullish_count - bearish_count
        if neutral_count < 0:
            neutral_count = 0

        bullish_pct = round(bullish_count / total * 100, 1)
        bearish_pct = round(bearish_count / total * 100, 1)

        avg_long_conv = (
            round(sum(s["Long_Conviction"] for s in bullish_tickers) / bullish_count, 4)
            if bullish_count else 0.0
        )
        avg_short_conv = (
            round(sum(s["Short_Conviction"] for s in bearish_tickers) / bearish_count, 4)
            if bearish_count else 0.0
        )

        # Overall sentiment
        if bullish_pct >= 60:
            overall = "BULLISH"
        elif bearish_pct >= 60:
            overall = "BEARISH"
        else:
            overall = "MIXED"

        strength = round(abs(bullish_pct - bearish_pct), 1)

        # Top tickers by Long_Rank == 1 / Short_Rank == 1
        long_sorted  = sorted(scores, key=lambda s: s.get("Long_Rank", 9999))
        short_sorted = sorted(scores, key=lambda s: s.get("Short_Rank", 9999))

        top_long = long_sorted[0] if long_sorted else {}
        top_short = short_sorted[0] if short_sorted else {}

        top5_long = [
            {
                "rank": int(s.get("Long_Rank", 0)),
                "ticker": s.get("ticker", "").replace(".NS", ""),
                "conviction": round(s.get("Long_Conviction", 0), 4),
                "long_score": round(s.get("long_score", 0), 4),
            }
            for s in long_sorted[:5]
        ]
        top5_short = [
            {
                "rank": int(s.get("Short_Rank", 0)),
                "ticker": s.get("ticker", "").replace(".NS", ""),
                "conviction": round(abs(s.get("Short_Conviction", 0)), 4),
                "short_score": round(s.get("short_score", 0), 4),
            }
            for s in short_sorted[:5]
        ]

        # Load daily gatekeepers if available
        daily_gatekeepers = {
            "timestamp": datetime.datetime.now().isoformat(),
            "long_eligible": [],
            "short_eligible": [],
            "long_eligible_count": 0,
            "short_eligible_count": 0
        }
        if os.path.exists(_GATEKEEPERS_PATH):
            try:
                with open(_GATEKEEPERS_PATH, "r") as gf:
                    daily_gatekeepers = json.load(gf)
            except Exception:
                pass

        return {
            "overall_sentiment": overall,
            "sentiment_strength": strength,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "total_tickers": total,
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "avg_long_conviction": avg_long_conv,
            "avg_short_conviction": avg_short_conv,
            "top_long_ticker": top_long.get("ticker", "—").replace(".NS", ""),
            "top_long_conviction": round(top_long.get("Long_Conviction", 0), 4),
            "top_short_ticker": top_short.get("ticker", "—").replace(".NS", ""),
            "top_short_conviction": round(abs(top_short.get("Short_Conviction", 0)), 4),
            "top5_long": top5_long,
            "top5_short": top5_short,
            "timestamp": datetime.datetime.fromtimestamp(scores_mtime).isoformat(),
            "scores_age_seconds": round(scores_age, 1),
            "daily_gatekeepers": daily_gatekeepers
        }

    except Exception as e:
        result = empty.copy()
        result["error"] = str(e)
        return result


# ===========================================================================
# MARKET SNAPSHOT (Indices + Universe Analytics)
# ===========================================================================

def get_market_snapshot() -> dict:
    """
    Fetches all major Indian indices (^NSEI, ^BSESN, ^NSEBANK, ^CNXIT, ^INDIAVIX)
    from yfinance and aggregates general Indian stock market breadth metrics
    from data/latest_scores.json.

    Results are cached in memory for 5 minutes.
    """
    global _market_cache, _market_cache_time

    # Serve from cache if fresh
    if _market_cache and (time.time() - _market_cache_time) < _MARKET_CACHE_TTL:
        result = _market_cache.copy()
        result["cached"] = True
        return result

    empty = {
        "nifty_price": None,
        "nifty_prev_close": None,
        "nifty_change_pts": None,
        "nifty_change_pct": None,
        "nifty_weekly_pct": None,
        "nifty_monthly_pct": None,
        "nifty_52w_high": None,
        "nifty_52w_low": None,
        "india_vix": None,
        "market_sentiment": "UNAVAILABLE",
        "vix_regime": "N/A",
        "timestamp": datetime.datetime.now().isoformat(),
        "cached": False,
        "indices": {},
        "analytics": {}
    }

    try:
        import yfinance as yf

        # ---------------------------------------------------------------
        # Fetch multiple indices simultaneously (Nifty 50, Sensex, Bank Nifty, IT, VIX)
        # ---------------------------------------------------------------
        index_symbols = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY IT": "^CNXIT",
            "INDIA VIX": "^INDIAVIX"
        }
        
        tickers_str = " ".join(index_symbols.values())
        df = yf.download(tickers=tickers_str, period="2mo", group_by="ticker", timeout=5)

        indices_data = {}

        def process_df(df_ticker):
            df_clean = df_ticker.dropna(subset=["Close"])
            if df_clean.empty:
                return None
            close = df_clean["Close"]
            current_price   = round(float(close.iloc[-1]), 2)
            prev_close      = round(float(close.iloc[-2]), 2) if len(close) > 1 else current_price
            change_pts      = round(current_price - prev_close, 2)
            change_pct      = round((change_pts / prev_close) * 100, 2) if prev_close else 0.0
            
            weekly_base = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])
            weekly_pct  = round((current_price - weekly_base) / weekly_base * 100, 2)
            
            monthly_base = float(close.iloc[-23]) if len(close) >= 23 else float(close.iloc[0])
            monthly_pct  = round((current_price - monthly_base) / monthly_base * 100, 2)
            
            high52 = round(float(df_clean["High"].max()), 2)
            low52  = round(float(df_clean["Low"].min()), 2)
            
            return {
                "price": current_price,
                "prev_close": prev_close,
                "change_pts": change_pts,
                "change_pct": change_pct,
                "weekly_pct": weekly_pct,
                "monthly_pct": monthly_pct,
                "high_52w": high52,
                "low_52w": low52
            }

        # Process each index
        for name, sym in index_symbols.items():
            if sym in df.columns.levels[0]:
                res = process_df(df[sym])
                if res:
                    indices_data[name] = res

        # NIFTY 50 defaults
        nifty = indices_data.get("NIFTY 50", {})
        nifty_price = nifty.get("price")
        nifty_change_pct = nifty.get("change_pct")

        sentiment = "NEUTRAL"
        if nifty_change_pct is not None:
            if nifty_change_pct >= 0.5:
                sentiment = "BULLISH"
            elif nifty_change_pct <= -0.5:
                sentiment = "BEARISH"

        # India VIX details
        vix = indices_data.get("INDIA VIX", {})
        india_vix = vix.get("price")
        vix_regime = "N/A"
        if india_vix:
            if india_vix > 20:
                vix_regime = "HIGH"
            elif india_vix > 13:
                vix_regime = "NORMAL"
            else:
                vix_regime = "LOW"

        # ---------------------------------------------------------------
        # Total Indian Stock Market Analytics (from latest_scores.json)
        # ---------------------------------------------------------------
        analytics = {
            "total_volume_crores": 0.0,
            "avg_rvol": 0.0,
            "avg_dist_52w_high": 0.0,
            "high_vol_count": 0,
            "near_52w_high_count": 0,
            "universe_size": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "bullish_pct": 0.0,
            "bearish_pct": 0.0
        }

        if os.path.exists(_SCORES_PATH):
            try:
                with open(_SCORES_PATH, "r") as f:
                    scores = json.load(f)
                if scores:
                    size = len(scores)
                    tot_vol = sum(s.get("dv_raw", 0) for s in scores)
                    avg_rv = sum(s.get("rvol_raw", 1.0) for s in scores) / size
                    avg_d52 = sum(s.get("dist_52h_actual", 0.0) for s in scores) / size
                    
                    high_vol = sum(1 for s in scores if s.get("rvol_raw", 0.0) > 1.5)
                    near_52h = sum(1 for s in scores if s.get("dist_52h_actual", 0.0) > -0.05)

                    bullish = sum(1 for s in scores if s.get("Long_Conviction", 0) > 0)
                    bearish = sum(1 for s in scores if s.get("Short_Conviction", 0) > 0)
                    neutral = size - bullish - bearish
                    if neutral < 0:
                        neutral = 0

                    analytics = {
                        "total_volume_crores": round(tot_vol / 10000000.0, 2),
                        "avg_rvol": round(avg_rv, 2),
                        "avg_dist_52w_high": round(avg_d52 * 100, 2),
                        "high_vol_count": high_vol,
                        "near_52w_high_count": near_52h,
                        "universe_size": size,
                        "bullish_count": bullish,
                        "bearish_count": bearish,
                        "neutral_count": neutral,
                        "bullish_pct": round(bullish / size * 100, 1),
                        "bearish_pct": round(bearish / size * 100, 1)
                    }
            except Exception:
                pass

        snapshot = {
            "nifty_price": nifty_price,
            "nifty_prev_close": nifty.get("prev_close"),
            "nifty_change_pts": nifty.get("change_pts"),
            "nifty_change_pct": nifty_change_pct,
            "nifty_weekly_pct": nifty.get("weekly_pct"),
            "nifty_monthly_pct": nifty.get("monthly_pct"),
            "nifty_52w_high": nifty.get("high_52w"),
            "nifty_52w_low": nifty.get("low_52w"),
            "india_vix": india_vix,
            "market_sentiment": sentiment,
            "vix_regime": vix_regime,
            "timestamp": datetime.datetime.now().isoformat(),
            "cached": False,
            "indices": indices_data,
            "analytics": analytics
        }

        # Cache the result
        _market_cache = snapshot.copy()
        _market_cache_time = time.time()

        return snapshot

    except Exception as e:
        result = empty.copy()
        result["error"] = str(e)
        return result
