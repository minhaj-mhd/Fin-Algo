import os
import pyotp
import logging
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from NorenRestApiPy.NorenApi import NorenApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ShoonyaApi(NorenApi):
    def __init__(self):
        NorenApi.__init__(self, host='https://api.shoonya.com/NorenWClientTP/', websocket='wss://api.shoonya.com/NorenWSTP/')

def login_shoonya():
    load_dotenv()
    
    user_id = os.getenv('SHOONYA_USER_ID')
    password = os.getenv('SHOONYA_PASSWORD')
    api_key = os.getenv('SHOONYA_API_KEY')
    vendor_code = os.getenv('VENDOR_CODE')
    imei = os.getenv('IMEI')
    totp_secret = os.getenv('SHOONYA_TOTP_SECRET')
    
    if not all([user_id, password, api_key, vendor_code, imei, totp_secret]):
        logging.error("Missing Shoonya credentials in .env file.")
        logging.error("Make sure SHOONYA_USER_ID, SHOONYA_PASSWORD, SHOONYA_API_KEY, VENDOR_CODE, IMEI, SHOONYA_TOTP_SECRET are set.")
        return None

    api = ShoonyaApi()
    
    # Generate TOTP
    try:
        totp = pyotp.TOTP(totp_secret).now()
    except Exception as e:
        logging.error(f"Error generating TOTP: {e}")
        return None

    ret = api.login(userid=user_id, password=password, twoFA=totp, vendor_code=vendor_code, api_secret=api_key, imei=imei)
    
    if ret is not None and ret.get('stat') == 'Ok':
        logging.info("Successfully logged into Shoonya API.")
        return api
    else:
        logging.error(f"Failed to login: {ret}")
        return None

def test_historical_data(api, exchange, token, resolution, days_back_to_test):
    """
    Tests how far back we can fetch historical data for a specific resolution.
    resolution: "1" for 1-minute, "15" for 15-minute, "60" for 1-hour, "1D" for 1-day
    """
    logging.info(f"Testing historical data for {exchange}:{token} at resolution {resolution} (looking back {days_back_to_test} days)")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back_to_test)
    
    # Convert to timestamps
    end_ts = end_time.timestamp()
    start_ts = start_time.timestamp()
    
    try:
        # NorenApi uses get_time_price_series
        # ret = api.get_time_price_series(exchange=exchange, token=token, starttime=start_ts, endtime=end_ts, interval=resolution)
        ret = api.get_time_price_series(exchange=exchange, token=token, starttime=str(int(start_ts)), endtime=str(int(end_ts)), interval=resolution)
        
        if isinstance(ret, list) and len(ret) > 0:
            earliest_candle = ret[-1] # Usually returns latest first, so last element is earliest
            earliest_time = earliest_candle.get('time')
            logging.info(f"SUCCESS: Fetched {len(ret)} candles. Earliest available candle: {earliest_time}")
            return True, earliest_time, len(ret)
        else:
            logging.error(f"FAILED: Could not fetch data or empty list returned. Response: {ret}")
            return False, None, 0
    except Exception as e:
        logging.error(f"Exception during historical data fetch: {e}")
        return False, None, 0

if __name__ == "__main__":
    api = login_shoonya()
    
    if api:
        # Nifty 50 Index (NSE) token is "26000" usually, but let's test with a highly liquid stock like RELIANCE
        # NSE Equity: RELIANCE (Token: 2885)
        exchange = 'NSE'
        token = '2885'
        
        logging.info(f"Starting Historical Data Test for {exchange}:{token}")
        
        # Test 15-minute data (looking back 5 years to see max)
        # We will iterate backwards to find the actual limit if it fails, or just ask for a huge chunk
        logging.info("--- Testing 15 Minute Data ---")
        test_historical_data(api, exchange, token, "15", days_back_to_test=365*5)
        time.sleep(1) # rate limiting
        
        logging.info("--- Testing 60 Minute (Hourly) Data ---")
        test_historical_data(api, exchange, token, "60", days_back_to_test=365*5)
        time.sleep(1)
        
        logging.info("--- Testing 1 Day (Daily) Data ---")
        # For daily data, the interval parameter might need to be 1440 or the API might have another endpoint, but let's try '1D' or '1440'
        # In Shoonya, daily data is usually fetched with interval '1440' or using get_daily_price_series
        # We will try '1440' first which is 1 day in minutes
        test_historical_data(api, exchange, token, "1440", days_back_to_test=365*10)
        
        logging.info("Test complete.")
        
        # Logout
        api.logout()
