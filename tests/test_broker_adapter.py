import pytest
from unittest.mock import MagicMock
from scripts.vanguard.broker_adapter import BrokerAdapter

def test_place_order_success_dict(mocker):
    # Mock the underlying broker
    mock_upstox = mocker.patch("scripts.vanguard.broker_adapter.UpstoxSandboxBroker")
    
    # Configure it to return a dictionary (simulating sandbox or raw JSON)
    mock_instance = mock_upstox.return_value
    mock_instance.place_order.return_value = {
        "status": "success",
        "data": {"order_id": "DICT-1234"}
    }
    
    adapter = BrokerAdapter()
    result = adapter.place_order("RELIANCE", "LONG", 10, 100.0, 99.0)
    
    assert result["success"] == True
    assert result["order_id"] == "DICT-1234"
    mock_instance.place_order.assert_called_once_with("RELIANCE", "LONG", quantity=10, price=100.0, stop_loss=99.0)

def test_place_order_success_object(mocker):
    # Upstox SDK returns objects, not dicts. We must simulate this.
    mock_upstox = mocker.patch("scripts.vanguard.broker_adapter.UpstoxSandboxBroker")
    
    class MockData:
        order_id = "OBJ-5678"
        
    class MockResponse:
        data = MockData()
        
    mock_instance = mock_upstox.return_value
    mock_instance.place_order.return_value = MockResponse()
    
    adapter = BrokerAdapter()
    result = adapter.place_order("TCS", "SHORT", 5, 200.0, 205.0)
    
    assert result["success"] == True
    assert result["order_id"] == "OBJ-5678"

def test_place_order_failure(mocker):
    mock_upstox = mocker.patch("scripts.vanguard.broker_adapter.UpstoxSandboxBroker")
    
    mock_instance = mock_upstox.return_value
    # Simulate an exception (e.g., Network timeout or Margin Shortfall)
    mock_instance.place_order.side_effect = Exception("Margin Shortfall")
    
    adapter = BrokerAdapter()
    result = adapter.place_order("INFY", "LONG", 10, 150.0, 140.0)
    
    assert result["success"] == False
    assert result["order_id"] == "SANDBOX-ERROR"
    assert "Margin Shortfall" in result["error"]
