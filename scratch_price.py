import yfinance as yf
import upstox_client
import os
from scripts.upstox_broker import UpstoxSandboxBroker

def main():
    ticker = "DIVISLAB.NS"
    try:
        broker = UpstoxSandboxBroker()
        instrument_key = broker.get_instrument_key(ticker)
        print(f"Instrument key: {instrument_key}")
        api_instance = upstox_client.MarketQuoteApi(broker.data_api_client)
        api_response = api_instance.get_full_market_quote(instrument_key, '2.0')
        print(f"Response status: {api_response.status}")
        print(f"Response data keys: {list(api_response.data.keys()) if api_response.data else 'None'}")
        
        # Test upstox_broker's get_live_price directly to see if it succeeds or falls back
        live_price = broker.get_live_price(ticker)
        print(f"broker.get_live_price() = {live_price}")
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
