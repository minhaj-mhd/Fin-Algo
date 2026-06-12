---
title: "Upstox Get Brokerage API Integration Plan"
type: archive
status: archived
updated: 2026-06-12
tags: []
---
# Upstox Get Brokerage API Integration Plan

This document serves as a reminder for the future integration of the Upstox `Get Brokerage API` (`/v2/charges/brokerage`).

## Current State: Sandbox / Shadow Tracking Phase
Currently, Vanguard V2.0 operates using the `UpstoxSandboxBroker`. In this paper-trading/shadow-tracking environment, we **do not** use the live Brokerage API.

### Why it is omitted right now:
1. **Network Latency & Rate Limits:** Calling an external API to calculate exact fees every 60 seconds for every active trade inside the `shadow_tracker_loop` would introduce massive latency and exhaust API rate limits unnecessarily.
2. **Local Accuracy is Sufficient:** Hardcoding the major friction costs locally (`brokerage_per_order = 10.0` and `stt_rate = 0.00025`) captures ~98% of the real-world trade cost. This provides excellent baseline accuracy for simulation without the network overhead.
3. **Sandbox Isolation:** Maintaining local virtual calculations keeps the engine fully independent while testing.

---

## Future State: Live Production Phase
When Vanguard V2.0 graduates to executing **Real Money Production Trades**, the `Get Brokerage API` should be integrated to ensure 100% legally accurate financial ledgers.

### Implementation Strategy
The API should **never** be called continuously inside a loop. It should only be triggered at two specific lifecycle events:

1. **Pre-Trade Check (Entry Phase):**
   * **Action:** Right before sending a LIVE Buy/Sell order to the exchange, ping the endpoint with the planned `instrument_token`, `quantity`, `price`, and `product` (MIS).
   * **Purpose:** Extract the exact `breakeven_price` returned by Upstox.
   * **Benefit:** The engine can use this absolute breakeven price to dynamically adjust the algorithmic Take Profit (TP) and Stop Loss (SL) targets.

2. **Post-Trade Ledger (Exit Phase):**
   * **Action:** Once the trade is confirmed CLOSED, ping the endpoint one final time.
   * **Purpose:** Extract the exact `total_charges` (including GST, Stamp Duty, Exchange Fees, and SEBI turnover fees).
   * **Benefit:** Write this exact value into `vanguard_trades.db` and the `VANGUARD_FINANCIAL_AUDIT.md` ledger, ensuring the system's Net P&L matches the broker's official end-of-day contract note exactly.

### Reference Code Snippet
```python
import requests

def get_exact_trade_charges(access_token, token, qty, price, side):
    url = "https://api.upstox.com/v2/charges/brokerage"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "instrument_token": token, 
        "quantity": qty,
        "price": price,
        "transaction_type": side.upper(),
        "product": "MIS"
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json().get("data", {})
        return {
            "total_charges": data.get("total_charges", 0.0),
            "breakeven": data.get("breakeven_price", 0.0)
        }
    return None
```
