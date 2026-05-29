from tradingview_ta import TA_Handler, Interval, Exchange

def get_tv_sentiment(ticker, interval=Interval.INTERVAL_1_HOUR):
    """
    Fetches the technical analysis summary for a ticker from TradingView.
    Ticker should be in the format 'RELIANCE' (without .NS for TV usually, but depends on exchange).
    For NSE, use exchange='NSE' and ticker like 'RELIANCE'.
    """
    try:
        # Normalize ticker: RELIANCE.NS -> RELIANCE
        symbol = ticker.split('.')[0]
        
        handler = TA_Handler(
            symbol=symbol,
            exchange="NSE",
            screener="india",
            interval=interval
        )
        
        analysis = handler.get_analysis()
        return analysis.summary['RECOMMENDATION']
    except Exception as e:
        print(f"[TV-TA] Error fetching sentiment for {ticker}: {e}")
        return "N/A"

if __name__ == "__main__":
    # Test
    print(f"RELIANCE 1H Sentiment: {get_tv_sentiment('RELIANCE.NS')}")
