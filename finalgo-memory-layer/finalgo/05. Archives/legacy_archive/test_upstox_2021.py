import os
import sys
import requests
from datetime import datetime

sys.path.append(os.getcwd())
from scripts.upstox_broker import UpstoxSandboxBroker
from scripts.tickers import TICKERS

broker = UpstoxSandboxBroker()
instrument_key = broker.get_instrument_key("RELIANCE.NS")

# Try to fetch Jan 2021 data
from_str = "2021-01-01"
to_str = "2021-01-31"

headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {broker.analytics_token}'
}
url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/15/{to_str}/{from_str}"

print(f"Requesting: {url}")
try:
    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    data = response.json()
    if data.get('status') == 'success' and 'data' in data and data['data'].get('candles'):
        print(f"Success! Received {len(data['data']['candles'])} candles for Jan 2021.")
    else:
        print(f"Failed to get candles: {data}")
except Exception as e:
    print(f"Error: {e}")
