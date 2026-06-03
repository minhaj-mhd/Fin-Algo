import pytest
from scripts.vanguard.risk_manager import RiskManager
from scripts.vanguard import config

def test_calculate_exit_charges(monkeypatch):
    monkeypatch.setattr(config, "STATS_FILE", "dummy.json")
    rm = RiskManager(initial_capital=100000)
    
    # Simulate selling 100 shares at Rs. 1000 = 100,000 sell value
    sell_value = 100000.0
    charges = rm.calculate_exit_charges(sell_value)
    
    # Manual calculation expected:
    # brokerage = 10.0
    # stt = 100000 * 0.00025 = 25.0
    # txn = 100000 * 0.0000345 = 3.45
    # gst = (10 + 3.45) * 0.18 = 2.421
    # sebi = 100000 * 0.000001 = 0.1
    # total = 10 + 25 + 3.45 + 2.421 + 0.1 = 40.971
    assert pytest.approx(charges, rel=1e-2) == 40.971

def test_calculate_trade_quantity(monkeypatch):
    monkeypatch.setattr(config, "STATS_FILE", "dummy.json")
    # Capital: 100,000. Risk per trade = 0.5% = Rs. 500
    rm = RiskManager(initial_capital=100000)
    
    price = 100.0
    stop_loss_pct = 1.0 # 1% SL = Rs. 1 distance
    
    # Ideal quantity = 500 / 1 = 500 shares
    # Max slot capital = (100000 / 5 slots) * 5 margin = 100,000 max buying power
    # Max qty = 100000 / 100 = 1000
    # Result should be min(ideal, max) = 500
    qty = rm.calculate_trade_quantity(price, stop_loss_pct)
    assert qty == 500
    
    # Test fallback SL calculation
    qty_zero_sl = rm.calculate_trade_quantity(price, stop_loss_pct=0.0)
    # SL defaults to 0.5% (Rs. 0.5) -> Ideal Qty = 500 / 0.5 = 1000 shares
    assert qty_zero_sl == 1000

def test_recompute_used_margin(monkeypatch):
    monkeypatch.setattr(config, "STATS_FILE", "dummy.json")
    rm = RiskManager(initial_capital=100000)
    
    active_trades = [
        {"status": "OPEN", "quantity": 100, "entry_price": 200, "margin_used": 0},
        {"status": "PENDING_ENTRY", "quantity": 50, "entry_price": 500, "margin_used": 5000},
        {"status": "CLOSED", "quantity": 200, "entry_price": 100}
    ]
    
    # Trade 1: no margin provided, calculates: (100 * 200) / 5 = 4000
    # Trade 2: margin provided = 5000
    # Trade 3: CLOSED, should be ignored
    margin = rm.recompute_used_margin(active_trades)
    
    assert margin == 9000.0
    assert rm.used_margin == 9000.0
