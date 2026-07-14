import os, sys, time
sys.path.insert(0, os.getcwd())
import pandas as pd
from datetime import date
from dotenv import load_dotenv
import upstox_client
load_dotenv()

def main():
    tok = os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN")
    cfg = upstox_client.Configuration(sandbox=False)
    cfg.sandbox = False
    cfg.host = "https://api.upstox.com"
    cfg.access_token = tok
    api_client = upstox_client.ApiClient(cfg)
    v3api = upstox_client.HistoryV3Api(api_client)
    cache_path = "data/raw_index_cache/nifty50_15m.csv"
    
    if os.path.exists(cache_path):
        df_existing = pd.read_csv(cache_path)
    else:
        df_existing = None
        
    ik = "NSE_INDEX|Nifty 50"
    
    chunks = [
        (date(2026, 6, 4).strftime("%Y-%m-%d"), date(2026, 6, 30).strftime("%Y-%m-%d")),
        (date(2026, 7, 1).strftime("%Y-%m-%d"), date(2026, 7, 10).strftime("%Y-%m-%d"))
    ]
    
    all_rows = []
    success = True
    
    for frm, to in chunks:
        print(f"Fetching Nifty 50 from {frm} to {to}")
        for attempt in range(3):
            try:
                resp = v3api.get_historical_candle_data1(ik, "minutes", "15", to, frm)
                if resp.status == "success" and resp.data and resp.data.candles:
                    all_rows.extend(resp.data.candles)
                    break
                else:
                    time.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(3)
        else:
            print("Failed to fetch chunk.")
            success = False

    if success and all_rows:
        df_new = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume", "oi"])
        df_new["ts"] = pd.to_datetime(df_new["ts"], utc=True)
        
        if df_existing is not None and not df_existing.empty:
            df_existing["ts"] = pd.to_datetime(df_existing["ts"], utc=True)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        df_combined = df_combined.drop_duplicates(subset=["ts"]).sort_values("ts")
        
        df_combined["ts"] = df_combined["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        df_combined.to_csv(cache_path, index=False)
        print(f"Appended {len(df_new)} new rows to {cache_path}")
    else:
        print("Failed to get Nifty data.")

if __name__ == "__main__":
    main()
