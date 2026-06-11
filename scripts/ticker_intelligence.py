import os
import json
from google import genai
from google.genai import types
import yfinance as yf
from datetime import datetime
from scripts.upstox_broker import UpstoxSandboxBroker
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from scripts.gemini_client_manager import GeminiRotator

# Load API keys from .env
load_dotenv()

rotator = GeminiRotator()

# Global broker instance for intelligence (uses Analytics token)
broker = UpstoxSandboxBroker()

# Gemini Configuration
DEFAULT_MODEL = "gemini-3.5-flash"

import re

def parse_gemini_json(text):
    """Robustly extracts and parses JSON from Gemini's response string."""
    if not text:
        return {}
    text = text.strip()
    
    # Strip markdown formatting if present
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    
    # Robust JSON extraction: Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
        
    try:
        return json.loads(text)
    except Exception as e:
        print(f"[ERROR] JSON Parse Error: {e} | Raw snippet: {text[:100]}...")
        return {"error": "JSON Parse Error"}

def get_ticker_analysis(ticker, side_preference="NEUTRAL"):
    """
    Fetches a detailed intelligence report for a specific ticker using Gemini Flash.
    """
    if not rotator.main_keys and not rotator.backup_key:
        return {"error": "Gemini API Key not found"}
    
    # Fetch data using Upstox Broker (with fallback)
    curr_price = 0
    try:
        df = broker.get_historical_data(ticker, interval='day', days=1)
        curr_price = broker.get_live_price(ticker)
        
        if not df.empty and curr_price:
            last_row = df.iloc[-1]
            day_high = last_row['high']
            day_low = last_row['low']
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            market_context = f"CURRENT DATE AND TIME: {current_time}\nCURRENT PRICE: {curr_price:.2f}, DAY HIGH: {day_high:.2f}, DAY LOW: {day_low:.2f}"
        else:
            market_context = "Data fetch failed, use latest available knowledge."
        print(f"[DEBUG] Intelligence Context for {ticker}: {market_context}")
    except Exception as e:
        market_context = f"Data fetch failed: {str(e)}"
        
    prompt = f"""
    CRITICAL: You are providing a REAL-TIME intraday report. 
    YOU MUST USE THE FOLLOWING LIVE DATA FOR YOUR ANALYSIS:
    {market_context}
    
    TICKER: {ticker}
    The current bias preference is: {side_preference}.
    
    TASK:
    Provide a comprehensive 1-hour intraday trading intelligence report.
    
    Your report MUST include:
    1. "market_sentiment": Summary of current sentiment for {ticker} (NBFC/Finance sector context if applicable).
    2. "key_levels": Calculated Support/Resistance based on the CURRENT PRICE of {float(curr_price):.2f}.
    3. "catalysts": Recent company-specific catalysts.
    4. "risk_factors": Intraday risks.
    5. "intelligence_verdict": BULLISH/BEARISH/NEUTRAL.
    6. "confidence": A numeric value between 0 and 100 representing your conviction (e.g. "85%"). DO NOT INCLUDE ANY OTHER TEXT.
    7. "one_hour_probability": An estimate of the probability (e.g. "75%") of price moving in favor of {side_preference} bias over the NEXT 1 HOUR.
    
    CRITICAL INSTRUCTION: You MUST return a valid JSON object matching the exact keys above. Do NOT include markdown formatting (like ```json). Do NOT add conversational text. Return ONLY the JSON object starting with {{ and ending with }}.
    """

    def make_call(client):
        try:
            # 1. ATTEMPT WITH SEARCH (Grounded Intelligence)
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    temperature=0.1
                )
            )
            return response
        except Exception as e_search:
            if any(x in str(e_search) for x in ["429", "503", "UNAVAILABLE"]):
                print(f"[WARN] Client limited with Search: {e_search}. Falling back to Technical Analysis on same client...")
                # 2. FALLBACK WITHOUT SEARCH ON SAME CLIENT
                response = client.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=prompt + "\nNOTE: Google Search is unavailable, use technical context and internal knowledge.",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                return response
            else:
                raise e_search

    try:
        response = rotator.execute(make_call)
    except Exception as e:
        print(f"Gemini Intelligence Error: {e}")
        return {
            "error": str(e),
            "intelligence_verdict": "ERROR",
            "verdict_label": "NEUTRAL",
            "market_sentiment": f"High demand or API error. Please try again. ({str(e)})"
        }

    if not response or not hasattr(response, 'text') or not response.text:
        print(f"[WARN] Gemini returned empty response for {ticker} (Safety Filter?)")
        return {
            "intelligence_verdict": "NEUTRAL",
            "verdict_label": "NEUTRAL",
            "market_sentiment": f"Intelligence report blocked by safety filters or empty response.",
            "confidence": "---%",
            "risk_factors": "Response unavailable.",
            "one_hour_probability": "N/A",
            "sources": []
        }
        
    data = parse_gemini_json(response.text)
    if "error" in data:
        return {
            "intelligence_verdict": "ERROR",
            "verdict_label": "NEUTRAL",
            "market_sentiment": f"Gemini returned an invalid format.",
            "confidence": "---%",
            "risk_factors": f"Parse error: {data['error']}",
            "one_hour_probability": "N/A",
            "sources": []
        }
    
    # Normalize intelligence_verdict to string
    verdict = data.get('intelligence_verdict', 'NEUTRAL')
    if isinstance(verdict, dict):
        v_val = verdict.get('verdict') or verdict.get('recommendation') or verdict.get('label')
        data['intelligence_verdict'] = str(v_val or verdict)
    
    # Ensure confidence is formatted as percentage string for UI
    conf = data.get('confidence')
    if conf is not None:
        conf_str = str(conf).strip()
        if not conf_str.endswith('%'):
            conf_str = f"{conf_str}%"
        data['confidence'] = conf_str
    else:
        data['confidence'] = "---%"
    
    # Extract a clean verdict label for CSS classes
    verdict_str = str(data.get('intelligence_verdict', 'NEUTRAL')).upper()
    if 'BULLISH' in verdict_str:
        data['verdict_label'] = 'BULLISH'
    elif 'BEARISH' in verdict_str:
        data['verdict_label'] = 'BEARISH'
    else:
        data['verdict_label'] = 'NEUTRAL'
    
    # Extract Grounding Sources if available
    sources = []
    try:
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            # We'll skip complex extraction for now to keep it fast
            pass
    except Exception as ge:
        print(f"[DEBUG] Grounding Metadata Extraction Error: {ge}")
    
    data['sources'] = sources
    return data

if __name__ == "__main__":
    # Test
    print(get_ticker_analysis("RELIANCE.NS"))
