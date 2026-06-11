import pytest
import time
import pandas as pd
from unittest.mock import patch, MagicMock

from scripts.vanguard.orchestrator import VanguardOrchestrator

@pytest.fixture
def mock_dependencies(mocker):
    # Mock ModelManager, BrokerAdapter, and other dependencies to allow instantiation
    mocker.patch("scripts.vanguard.orchestrator.ModelManager")
    mocker.patch("scripts.vanguard.orchestrator.BrokerAdapter")
    mocker.patch("scripts.vanguard.orchestrator.SignalGenerator")
    mocker.patch("scripts.vanguard.orchestrator.RiskManager")
    mocker.patch("scripts.vanguard.orchestrator.AIVetoManager")
    
    # Mock yfinance to return empty dataframes to simplify test
    mocker.patch("scripts.vanguard.orchestrator.yf.download", return_value=pd.DataFrame())

def test_calculate_conviction_scores_concurrency(mock_dependencies, mocker):
    """
    Test that the calculate_conviction_scores method executes concurrently.
    We mock the historical data fetching to include a small sleep.
    If executed sequentially, 20 tickers with 0.1s sleep = 2.0 seconds.
    If executed concurrently, it should take ~0.2 seconds.
    """
    orchestrator = VanguardOrchestrator()
    orchestrator.ticker_metadata = {}
    
    # Mock the internal UpstoxSandboxBroker to inject a sleep
    mock_broker_cls = mocker.patch("scripts.upstox_broker.UpstoxSandboxBroker")
    mock_broker_instance = MagicMock()
    
    def fake_get_historical_data(*args, **kwargs):
        time.sleep(0.05) # 50ms sleep
        # Return a dummy dataframe so it enters the compute loop
        df = pd.DataFrame({"close": [100.0] * 30, "open": [100.0] * 30, "high": [100.0] * 30, "low": [100.0] * 30, "volume": [1000] * 30})
        df["timestamp"] = pd.date_range(start="2026-01-01", periods=30, freq="1h")
        return df
        
    mock_broker_instance.get_historical_data.side_effect = fake_get_historical_data
    mock_broker_cls.return_value = mock_broker_instance
    
    # Mock compute_features to also include a small sleep and return the same df
    mocker.patch("scripts.feature_utils.compute_features", side_effect=lambda df, **kwargs: df)
    
    # Mock save_latest_scores so it doesn't write to disk
    mocker.patch("scripts.vanguard.orchestrator.save_latest_scores")

    # Mock only the 0.3s rate-limit sleep inside the orchestrator
    original_sleep = time.sleep
    def filtered_sleep(seconds):
        if seconds == 0.3:
            return
        original_sleep(seconds)
    mocker.patch("time.sleep", side_effect=filtered_sleep)
    
    # 20 tickers
    test_tickers = [f"TICKER_{i}" for i in range(20)]
    
    start_time = time.time()
    # Call the method
    orchestrator.calculate_conviction_scores(test_tickers)
    end_time = time.time()
    
    total_time = end_time - start_time
    
    # Sequential fetch for 20 tickers * 0.05s = 1.0s minimum.
    # Concurrent fetch should be roughly 0.05s - 0.2s depending on overhead and workers (max 10-15).
    # We assert it's less than 0.8s which confirms concurrency.
    assert total_time < 0.8, f"Execution took {total_time}s, which indicates sequential execution!"
