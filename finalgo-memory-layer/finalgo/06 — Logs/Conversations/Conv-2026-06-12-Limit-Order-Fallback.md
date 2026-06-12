---
title: "Conversation Context: Limit Order Fallback Logic"
type: log
status: concluded
updated: 2026-06-12
tags: []
---
# 💬 Conversation Context: Limit Order Fallback Logic

## 📌 Metadata
- **Conversation ID**: 1e4d47ee-b14f-4a08-aceb-e61561f91850
- **Start Date**: 2026-06-12
- **Status**: 🔴 Concluded
- **Focus Area**: Vanguard Engine / Trading Execution

## 🎯 Objectives
- [x] Implement `PENDING_LIMIT` orders when a look-back candle confirmation fails.
- [x] Configure a 15-minute expiry timeout for limit orders.
- [x] Fallback to evaluating the newly completed 15-minute candle if the limit order is not filled.

## 💻 Active Code Files Modified
- [orchestrator.py](file:///C:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py)

## 📝 Compacted Session Log
- **Initial Analysis**: The engine previously hard-cancelled trades if the look-back candle was against the entry direction. Modified this to place a `PENDING_LIMIT` order at the 25% strength line of the failed candle.
- **Execution Engine Update**: Upgraded `shadow_tracker_loop` to monitor `PENDING_LIMIT` trades. If the price retraces to the limit point, it executes the trade. If 15 minutes pass, it waits for the newly completed candle to evaluate direction. If confirmed, it executes a market order; otherwise, it permanently cancels.
- **UI State Hotfix**: Fixed a display bug where `PENDING_LIMIT` trades caused the web dashboard's "Live Price" and "Live P&L" columns to freeze at the entry limit price. Added `exit_price` and `final_profit_pct` live-updates for pending trades in `orchestrator.py`.
- **BrokerAdapter Hotfix**: Fixed an `AttributeError` crash when checking the live forming 1m candle during `start_shadow_trade`. The Orchestrator incorrectly called `self.broker.get_historical_data` which bypassed the Adapter interface. Replaced it with `self.broker.get_recent_candles` to properly utilize the WebSocket cache.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop]]
