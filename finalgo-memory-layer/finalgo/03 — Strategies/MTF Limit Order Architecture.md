---
title: "Multi-Timeframe (MTF) Support & Resistance Limit Order Architecture"
type: spec
status: active
updated: 2026-06-12
tags: []
---
# Multi-Timeframe (MTF) Support & Resistance Limit Order Architecture

This document aggregates the deep research findings on integrating conditional Limit Orders using Multi-Timeframe Support/Resistance levels into the Vanguard Live Trading System.

## 1. Multi-Timeframe Support/Resistance Algorithm
- **Current State:** The system calculates rolling max/min for indicators but lacks explicit S/R extraction logic.
- **Proposed Logic:** Implement a **rolling fractal (pivot) approach** inside `scripts/feature_utils.py`.
    - Identify pivot highs and lows without lookahead bias by validating after `N` bars.
    - Resample 1m data to 5m, 15m, 30m, and 1h intervals.
    - Calculate pivots on higher timeframes and forward-fill them back to the 1m base timeframe.
    - Function: `get_nearest_levels(current_price, timeframes)` extracts the immediate resistance (>= price) and support (<= price) from all combined timeframes.

## 2. Zero-Latency MTF Data Fetching
- **Current State:** The broker adapter falls back to slow REST APIs for intervals not explicitly cached by the WebSocket (`CandleBuilder` only tracks 1m and 15m).
- **Proposed Logic:** Expand `upstox_websocket.py` to process MTF in-memory.
    - Add `5minute`, `30minute`, and `60minute` buckets to `CandleBuilder.on_tick()`.
    - Create `get_current_wip_candle(interval)` to fetch the live, unclosed candle (Work-In-Progress) from the `_wip` dictionary to evaluate real-time candle color (red vs green).
    - This eliminates REST API usage for MTF queries, bypassing rate limit constraints and network latency.

## 3. PENDING_LIMIT Order Execution State
- **Current State:** `orchestrator.py` handles a `PENDING_ENTRY` state (waiting for candle confirmation) and then immediately places a `MARKET` order.
- **Proposed Logic:**
    - **Long Conditional:** If the WIP 15m candle is **red**, place a `LIMIT` order at the nearest MTF support. If **green**, trigger a `MARKET` order.
    - **Short Conditional:** If the WIP 15m candle is **green**, place a `LIMIT` order at the nearest MTF resistance. If **red**, trigger a `MARKET` order.
    - **Broker Updates:** Update `upstox_broker.py` with `get_order_details(order_id)` and `cancel_order(order_id)` to manage the Upstox limit API.
    - **Orchestrator Workflow:** Introduce a `PENDING_LIMIT` state in the `shadow_tracker_loop`. Continuously poll for a fill. If un-filled after expiration (e.g., hard cutoff like 15:00 or a specific timeout), cancel the order in Upstox and transition the trade to `CANCELLED` to release used margin.
