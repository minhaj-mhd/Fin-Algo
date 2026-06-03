import pytest
from datetime import datetime, timedelta
from scripts.vanguard.trade_state import TradeStateManager

class MockDatetime:
    @classmethod
    def now(cls):
        return datetime(2026, 6, 3, 10, 0, 0)
    
    @classmethod
    def fromisoformat(cls, date_string):
        return datetime.fromisoformat(date_string)

@pytest.fixture(autouse=True)
def patch_datetime(mocker):
    mocker.patch("scripts.vanguard.trade_state.datetime", MockDatetime)

def test_evaluate_open_trade_exit_stop_loss():
    now = MockDatetime.now()
    
    trade = {
        "ticker": "RELIANCE",
        "side": "LONG",
        "stop_loss_pct": 1.0,  # 1% SL
        "take_profit_pct": 3.0,
        "peak_profit_pct": -0.5,
        "exit_time": (now + timedelta(hours=1)).isoformat()
    }
    
    # Not hit yet
    should_exit, status, note = TradeStateManager.evaluate_open_trade_exit(trade, 100, -0.5, now)
    assert not should_exit
    
    # Hit SL exactly
    should_exit, status, note = TradeStateManager.evaluate_open_trade_exit(trade, 100, -1.0, now)
    assert should_exit
    assert status == "STOP_LOSS"

def test_evaluate_open_trade_exit_take_profit():
    now = MockDatetime.now()
    trade = {
        "ticker": "TCS",
        "side": "LONG",
        "stop_loss_pct": 0.5,
        "take_profit_pct": 1.5,
        "peak_profit_pct": 1.0,
        "exit_time": (now + timedelta(hours=1)).isoformat()
    }
    
    # Hit TP exactly
    should_exit, status, note = TradeStateManager.evaluate_open_trade_exit(trade, 100, 1.5, now)
    assert should_exit
    assert status == "TAKE_PROFIT"

def test_evaluate_open_trade_exit_trailing_stop():
    now = MockDatetime.now()
    trade = {
        "ticker": "INFY",
        "side": "SHORT",
        "stop_loss_pct": 0.5,
        "take_profit_pct": 2.0,
        "peak_profit_pct": 1.2, # Profit went up to 1.2%
        "exit_time": (now + timedelta(hours=1)).isoformat()
    }
    
    # Activate trailing stop because PnL (1.2) >= SL*2 (1.0)
    should_exit, status, note = TradeStateManager.evaluate_open_trade_exit(trade, 100, 1.2, now)
    assert trade.get("trailing_active") == True
    assert not should_exit
    
    # Now profit drops. Trailing stop is peak - SL = 1.2 - 0.5 = 0.7
    # If PnL drops to 0.7, it should trigger trailing exit (TAKE_PROFIT)
    should_exit, status, note = TradeStateManager.evaluate_open_trade_exit(trade, 100, 0.7, now)
    assert should_exit
    assert status == "TAKE_PROFIT"
    assert "Trailing Stop" in note

def test_evaluate_open_trade_time_expiry():
    now = MockDatetime.now()
    past_time = now - timedelta(minutes=5)
    
    trade = {
        "ticker": "WIPRO",
        "side": "LONG",
        "stop_loss_pct": 1.0,
        "take_profit_pct": 3.0,
        "peak_profit_pct": 0.5,
        "exit_time": past_time.isoformat(),
        "extension_pending": False
    }
    
    # The expiry time is in the past, so it should exit immediately
    should_exit, status, note = TradeStateManager.evaluate_open_trade_exit(trade, 100, 0.5, now)
    assert should_exit
    assert status == "CLOSED"
    assert "Expiry" in note
