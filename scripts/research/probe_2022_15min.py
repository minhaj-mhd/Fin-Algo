"""Probe whether Upstox has 15-min history back to 2022 (one ticker).
2023 chunk = control (confirms auth works); 2022/2021 = availability test."""
import sys, os
sys.path.append(os.getcwd())
import upstox_client
from scripts.upstox_broker import UpstoxSandboxBroker
from scripts.tickers import TICKERS

broker = UpstoxSandboxBroker()
v3 = upstox_client.HistoryV3Api(broker.data_api_client)
ticker = TICKERS[0]
ik = broker.get_instrument_key(ticker)
print(f"ticker={ticker}  instrument_key={ik}\n")


def probe(frm, to, label):
    try:
        r = v3.get_historical_candle_data1(ik, 'minutes', '15', to, frm)
        ok = (r.status == 'success' and r.data and r.data.candles)
        n = len(r.data.candles) if ok else 0
        if n:
            ts = sorted(c[0] for c in r.data.candles)
            print(f"{label}: {n:5d} candles   {ts[0]}  ..  {ts[-1]}")
        else:
            print(f"{label}: 0 candles (status={r.status})")
    except Exception as e:
        print(f"{label}: ERROR {str(e)[:140]}")


probe('2023-03-01', '2023-03-31', '2023-03 (control)')
probe('2022-06-01', '2022-06-30', '2022-06 (test)   ')
probe('2022-01-01', '2022-01-31', '2022-01 (test)   ')
probe('2021-06-01', '2021-06-30', '2021-06 (test)   ')
