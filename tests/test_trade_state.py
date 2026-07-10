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

def test_check_candle_direction():
    # Test LONG cases
    # 1. Bullish candle (close > open)
    candle_long_bullish = {"open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0}
    assert TradeStateManager.check_candle_direction("LONG", candle_long_bullish) == True

    # 2. Bearish but close is in top 40% (63.3% of total length)
    candle_long_bearish_ok = {"open": 105.0, "high": 110.0, "low": 95.0, "close": 104.5}
    assert TradeStateManager.check_candle_direction("LONG", candle_long_bearish_ok) == True

    # 3. Bearish and close is below 60% of total length (53.3% of total length)
    candle_long_bearish_fail = {"open": 105.0, "high": 110.0, "low": 95.0, "close": 103.0}
    assert TradeStateManager.check_candle_direction("LONG", candle_long_bearish_fail) == False

    # Test SHORT cases
    # 1. Bearish candle (close < open)
    candle_short_bearish = {"open": 100.0, "high": 105.0, "low": 90.0, "close": 95.0}
    assert TradeStateManager.check_candle_direction("SHORT", candle_short_bearish) == True

    # 2. Bullish but close is in bottom 40% (35% of total length)
    candle_short_bullish_ok = {"open": 95.0, "high": 110.0, "low": 90.0, "close": 97.0}
    assert TradeStateManager.check_candle_direction("SHORT", candle_short_bullish_ok) == True

    # 3. Bullish and close is above 40% of total length (60% of total length)
    candle_short_bullish_fail = {"open": 95.0, "high": 110.0, "low": 90.0, "close": 102.0}
    assert TradeStateManager.check_candle_direction("SHORT", candle_short_bullish_fail) == False

    # Edge cases
    # 1. None candle
    assert TradeStateManager.check_candle_direction("LONG", None) == False

    # 2. Invalid high/low (zero range)
    candle_zero_range = {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
    assert TradeStateManager.check_candle_direction("LONG", candle_zero_range) == False


def test_peak_adverse_pct_tracking():
    # Symmetrical simulation of peak tracking in orchestrator tracking loop
    trade = {
        "ticker": "SBIN",
        "side": "LONG",
        "entry_price": 100.0,
        "peak_profit_pct": 0.0,
        "peak_adverse_pct": 0.0,
    }
    
    # 1. Price drops to 99.0 -> PnL = -1.0%
    price = 99.0
    pnl = (price - trade["entry_price"]) / trade["entry_price"] * 100
    trade["peak_profit_pct"] = max(trade["peak_profit_pct"], pnl)
    trade["peak_adverse_pct"] = min(trade.get("peak_adverse_pct", 0.0), pnl)
    
    assert trade["peak_profit_pct"] == 0.0
    assert trade["peak_adverse_pct"] == -1.0
    
    # 2. Price rises to 102.0 -> PnL = +2.0%
    price = 102.0
    pnl = (price - trade["entry_price"]) / trade["entry_price"] * 100
    trade["peak_profit_pct"] = max(trade["peak_profit_pct"], pnl)
    trade["peak_adverse_pct"] = min(trade.get("peak_adverse_pct", 0.0), pnl)
    
    assert trade["peak_profit_pct"] == 2.0
    assert trade["peak_adverse_pct"] == -1.0  # Should remain -1.0% (lowest PnL)

    # 3. SHORT trade simulation
    trade_short = {
        "ticker": "SBIN",
        "side": "SHORT",
        "entry_price": 100.0,
        "peak_profit_pct": 0.0,
        "peak_adverse_pct": 0.0,
    }
    
    # Price rises to 101.5 -> PnL = -1.5%
    price = 101.5
    pnl = (trade_short["entry_price"] - price) / trade_short["entry_price"] * 100
    trade_short["peak_profit_pct"] = max(trade_short["peak_profit_pct"], pnl)
    trade_short["peak_adverse_pct"] = min(trade_short.get("peak_adverse_pct", 0.0), pnl)
    
    assert trade_short["peak_profit_pct"] == 0.0
    assert trade_short["peak_adverse_pct"] == -1.5


def test_record_barrier_checkpoint_first_touch_only():
    # Shadow-trade checkpoint: stamp the first stop-loss touch and hold it; a deeper
    # later dip must not overwrite. Records the would-be stop-out without exiting.
    t = {"side": "LONG", "stop_loss_pct": 0.5, "take_profit_pct": 1.0}
    t0 = datetime(2026, 7, 9, 10, 30)

    TradeStateManager.record_barrier_checkpoint(t, 100.0, 0.2, t0)  # no breach
    assert not t.get("sl_hit") and not t.get("tp_hit")

    TradeStateManager.record_barrier_checkpoint(t, 99.4, -0.6, t0 + timedelta(minutes=5))  # SL touch
    assert t["sl_hit"] == 1
    assert t["sl_hit_pnl"] == -0.6
    first_time, first_px = t["sl_hit_time"], t["sl_hit_price"]

    TradeStateManager.record_barrier_checkpoint(t, 99.0, -1.0, t0 + timedelta(minutes=10))  # deeper
    assert t["sl_hit_time"] == first_time and t["sl_hit_price"] == first_px  # unchanged

    TradeStateManager.record_barrier_checkpoint(t, 101.5, 1.5, t0 + timedelta(minutes=20))  # rebounds thru TP
    assert t["tp_hit"] == 1 and t["tp_hit_pnl"] == 1.5


