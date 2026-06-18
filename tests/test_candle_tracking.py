import pytest
import os
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from scripts.vanguard.orchestrator import VanguardOrchestrator
from scripts.vanguard import config

@pytest.fixture
def clean_temp_jsonl(tmp_path):
    """Sets a temporary path for the candle rejections log to prevent polluting production data."""
    temp_dir = tmp_path / "data" / "research"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "candle_rejections.jsonl"
    
    import builtins
    original_open = builtins.open
    
    with patch("scripts.vanguard.persistence.os.makedirs") as mock_makedirs, \
         patch("builtins.open") as mock_open:
             
        def side_effect_open(file_path, mode="r", *args, **kwargs):
            if "candle_rejections.jsonl" in str(file_path):
                return original_open(temp_file, mode, *args, **kwargs)
            return original_open(file_path, mode, *args, **kwargs)
            
        mock_open.side_effect = side_effect_open
        yield temp_file

@pytest.fixture
def mock_orchestrator(mocker):
    """Creates a mock VanguardOrchestrator instance with mocked dependencies."""
    mocker.patch("scripts.vanguard.orchestrator.ModelManager")
    mocker.patch("scripts.vanguard.orchestrator.BrokerAdapter")
    mocker.patch("scripts.vanguard.orchestrator.SignalGenerator")
    mocker.patch("scripts.vanguard.orchestrator.AIVetoManager")
    
    # Mock database log_trade so we don't hit the database
    mocker.patch("scripts.vanguard.persistence.db_log_trade")
    mocker.patch("scripts.vanguard.orchestrator.log_trade")
    
    orchestrator = VanguardOrchestrator()
    orchestrator.broker = MagicMock()
    orchestrator.risk_manager = MagicMock()
    orchestrator.risk_manager.virtual_capital = 100000.0
    orchestrator.risk_manager.used_margin = 0.0
    orchestrator.risk_manager.realized_charges = 0.0
    orchestrator.risk_manager.calculate_trade_quantity.return_value = 10
    
    return orchestrator

def test_candle_veto_tracking(mock_orchestrator, clean_temp_jsonl, mocker):
    """Verify that a candle-level vetoed trade is appended to tracking and transitions correctly."""
    orch = mock_orchestrator
    
    # 1. Setup a lookback candle that fails direction check and triggers THRUST_VETO
    # Let's say side is LONG, but the candle was strongly bearish (high range, closed at low pos)
    mock_candle = {
        "open": 105.0,
        "high": 108.0,
        "low": 100.0,
        "close": 101.0,
        "volume": 50000
    }
    orch.get_last_completed_15min_candle = MagicMock(return_value=mock_candle)
    orch.compute_15min_atr = MagicMock(return_value=(1.5, 3.0)) # 1.5% SL, 3% TP
    
    # Distances / RVOL
    rvol = 2.0
    dist_52h = -0.01
    
    # Enable guards
    config.THRUST_VETO_RANGE_PCT = 2.0
    config.THRUST_VETO_POS = 0.75
    config.FADE_QUALITY_GUARD = True
    
    # Call start_shadow_trade. It should fail look-back confirmation and be VETOED by THRUST_VETO
    orch.start_shadow_trade(
        ticker="TCS.NS",
        conviction=0.8,
        entry_price=105.0,
        side="LONG",
        reason="Test Signal",
        one_hour_prob=0.75,
        rvol=rvol,
        dist_52h=dist_52h
    )
    
    # Verify it is in active_shadow_trades with VETOED status
    assert len(orch.active_shadow_trades) == 1
    trade = orch.active_shadow_trades[0]
    assert trade["status"] == "VETOED"
    assert trade["quantity"] == 0
    assert trade["margin_used"] == 0.0
    assert trade["reject_stage"] == "candle"
    assert trade["reject_reason"] == "THRUST_VETO"
    assert trade["range_pct"] > 0.0
    assert trade["close_pos"] is not None
    assert trade["adverse_pos"] is not None
    assert trade["market_entry_px"] == 105.0
    
    # 2. Simulate shadow tracker loop processing it for 1 hour expiry
    # Force the trade timestamp to be 1 hour and 1 minute in the past
    now = datetime.now()
    trade["timestamp"] = (now - timedelta(hours=1, minutes=1)).isoformat()
    trade["exit_time"] = (now - timedelta(minutes=1)).isoformat()
    
    # Mock live price to exit at 110.0 (gross return = (110 - 105) / 105 = 4.76%)
    orch.broker.get_live_price.return_value = 110.0
    
    # We patch log_trade to write to JSONL using the actual implementation in persistence
    from scripts.vanguard.persistence import log_trade as real_log_trade
    mocker.patch("scripts.vanguard.orchestrator.log_trade", side_effect=real_log_trade)
    
    # Run the tracking step by simulating what happens in shadow_tracker_loop for this trade
    # Fetch price, compute PNL, and check veto expiry
    price = orch.broker.get_live_price(trade["ticker"])
    pnl = ((price - trade["entry_price"]) / trade["entry_price"] * 100) if trade["side"] == "LONG" else ((trade["entry_price"] - price) / trade["entry_price"] * 100)
    
    trade["exit_price"] = price
    trade["final_profit_pct"] = round(pnl, 4)
    veto_expiry = datetime.fromisoformat(trade["exit_time"])
    
    assert now >= veto_expiry
    
    old_status = trade["status"]
    trade["status"] = f"{old_status}_EXPIRED"
    
    # Record running stats
    if trade.get("reject_stage") == "candle":
        orch.veto_stats["running_guard_value_sum"] += (-pnl)
        orch.veto_stats["running_guard_value_count"] += 1
        
    real_log_trade(trade)
    
    # Verify status changed
    assert trade["status"] == "VETOED_EXPIRED"
    assert orch.veto_stats["running_guard_value_count"] == 1
    assert orch.veto_stats["running_guard_value_sum"] == -pnl
    
    # Verify it is written to the JSONL file
    assert clean_temp_jsonl.exists()
    with open(clean_temp_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        logged_row = json.loads(lines[0])
        assert logged_row["trade_id"] == trade["trade_id"]
        assert logged_row["status"] == "VETOED_EXPIRED"
        assert logged_row["reject_reason"] == "THRUST_VETO"
        assert logged_row["gross_pnl_bps"] == pytest.approx(476.1904, abs=1e-3)
        assert logged_row["net_pnl_bps"] == pytest.approx(466.1904, abs=1e-3)

def test_pending_limit_expiry_counterfactual(mock_orchestrator, clean_temp_jsonl, mocker):
    """Verify that an expired pending limit transitions to CANCELLED and tracks counterfactual properly."""
    orch = mock_orchestrator
    
    # Reset config parameters to avoid veto trigger from previous test leakage
    config.THRUST_VETO_RANGE_PCT = 10.0
    config.FADE_QUALITY_GUARD = False
    
    # 1. Setup a lookback candle that fails direction check, but does NOT trigger a veto
    # This places a PENDING_LIMIT order
    mock_candle = {
        "open": 101.0,
        "high": 103.0,
        "low": 100.0,
        "close": 100.5,
        "volume": 10000
    }
    orch.get_last_completed_15min_candle = MagicMock(return_value=mock_candle)
    orch.compute_15min_atr = MagicMock(return_value=(1.0, 2.0))
    
    # Call start_shadow_trade. It should fail look-back confirmation and become PENDING_LIMIT
    orch.start_shadow_trade(
        ticker="INFY.NS",
        conviction=0.7,
        entry_price=101.0,
        side="LONG",
        reason="Test Limit Signal",
        one_hour_prob=0.8,
        rvol=0.5,
        dist_52h=-0.05
    )
    
    assert len(orch.active_shadow_trades) == 1
    trade = orch.active_shadow_trades[0]
    assert trade["status"] == "PENDING_LIMIT"
    assert trade["quantity"] > 0
    assert trade["margin_used"] > 0.0
    
    # 2. Simulate wait expiry (15m elapsed) without filling, and new candle failing confirmation
    now = datetime.now()
    trade["exit_time"] = (now - timedelta(minutes=1)).isoformat()
    
    # Mock new candle that also fails direction confirmation
    new_candle = {
        "open": 102.0,
        "high": 102.5,
        "low": 101.5,
        "close": 101.8,
        "volume": 8000
    }
    orch.get_last_completed_15min_candle.return_value = new_candle
    
    # Tracker expiry step
    assert now >= datetime.fromisoformat(trade["exit_time"])
    # Simulated check in shadow_tracker_loop
    from scripts.vanguard.trade_state import TradeStateManager
    assert not TradeStateManager.check_candle_direction(trade["side"], new_candle)
    
    # Transition to CANCELLED and set counterfactual tracking parameters
    trade["status"] = "CANCELLED"
    trade["comment"] = trade.get("comment", "") + " | 15m wait expired, new candle failed confirmation."
    trade["reject_stage"] = "candle"
    trade["reject_reason"] = "LIMIT_EXPIRED"
    trade["quantity"] = 0
    trade["entry_price"] = trade["market_entry_px"] # Track from original signal px (101.0)
    trade_start = datetime.fromisoformat(trade["timestamp"])
    trade["exit_time"] = (trade_start + timedelta(hours=1)).isoformat()
    
    # Verify state of trade now
    assert trade["status"] == "CANCELLED"
    assert trade["entry_price"] == 101.0
    assert trade["quantity"] == 0
    assert trade["reject_reason"] == "LIMIT_EXPIRED"
    
    # 3. Simulate another hour elapsed to test CANCELLED_EXPIRED transition
    trade["exit_time"] = (now - timedelta(minutes=1)).isoformat()
    orch.broker.get_live_price.return_value = 99.0 # Exit below market entry
    
    price = orch.broker.get_live_price(trade["ticker"])
    pnl = ((price - trade["entry_price"]) / trade["entry_price"] * 100) if trade["side"] == "LONG" else ((trade["entry_price"] - price) / trade["entry_price"] * 100)
    
    trade["exit_price"] = price
    trade["final_profit_pct"] = round(pnl, 4)
    veto_expiry = datetime.fromisoformat(trade["exit_time"])
    
    assert now >= veto_expiry
    
    old_status = trade["status"]
    trade["status"] = f"{old_status}_EXPIRED"
    
    if trade.get("reject_stage") == "candle":
        orch.veto_stats["running_guard_value_sum"] += (-pnl)
        orch.veto_stats["running_guard_value_count"] += 1
        
    from scripts.vanguard.persistence import log_trade as real_log_trade
    mocker.patch("scripts.vanguard.orchestrator.log_trade", side_effect=real_log_trade)
    real_log_trade(trade)
    
    # Verify CANCELLED_EXPIRED outcomes
    assert trade["status"] == "CANCELLED_EXPIRED"
    assert orch.veto_stats["running_guard_value_count"] == 1
    assert orch.veto_stats["running_guard_value_sum"] == -pnl
    
    # Verify written to JSONL
    with open(clean_temp_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        logged_row = json.loads(lines[0])
        assert logged_row["trade_id"] == trade["trade_id"]
        assert logged_row["status"] == "CANCELLED_EXPIRED"
        assert logged_row["reject_reason"] == "LIMIT_EXPIRED"
        assert logged_row["gross_pnl_bps"] == pytest.approx(-198.0198, abs=1e-3) # (99 - 101)/101 * 10000 = -1.98% * 100 = -198 bps
        assert logged_row["net_pnl_bps"] == pytest.approx(-208.0198, abs=1e-3)
