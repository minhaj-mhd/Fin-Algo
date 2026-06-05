# 💬 Conversation Context: Fix Upstox Portfolio UI Bugs

## 📌 Metadata
- **Conversation ID**: 1fa60d75-18d4-40f6-bf37-368d1b015500
- **Start Date**: 2026-06-04
- **Status**: 🔴 Concluded
- **Focus Area**: UI/Dashboard & Upstox Integration

## 🎯 Objectives
- [x] Fix bug where closed trades are not reflected in Upstox portfolio.
- [x] Fix bug where intraday tracking cards and in-progress tracking of trades are not reflected in Upstox portfolio.

## 💻 Active Code Files Modified
- [database_manager.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/database_manager.py)

## 📝 Compacted Session Log
- **Initial Analysis**: The UI dashboard needs to properly map or display the upstox portfolio state for closed and active trades. I need to find the files responsible for rendering these components and fetching Upstox data.
- **Root Cause Discovered**: The SQL query fetching performance stats and portfolio summaries hardcoded the `trade_id` prefix matching as `trade_id LIKE 'T-%'`, but the orchestrator actually assigns IDs with the prefix `TRADE-`. As a result, closed trades were entirely invisible to the portfolio summarizer.
- **Open Trades Tracking Added**: The intraday tracking query previously ignored all trades that were not closed. Added the `"OPEN"` status so that open positions and their unrealized P&L now properly reflect in the intraday tracker cards.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[04. Data & Code Map/Upstox Brokerage API Plan]]

- **Step 2**: Fixed a dashboard issue where the 'Expand' button on the Comment column would auto-collapse after a few seconds due to the dashboard data refresh loop. It now preserves expanded state using trade IDs.
