---
title: "Vanguard Dashboard UI: Strategy Integration"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# Vanguard Dashboard UI: Strategy Integration

## Overview
As part of the ongoing evolution of the Vanguard Trading System, the Dashboard UI is being updated to cleanly separate signals generated purely by the ML/AI models from those triggered by defined structural strategies.

## Dashboard Modifications (`vanguard_v2.html`)
The existing Vetoed Trades section is split into two primary views using a nested tab interface:
1. **ML/AI Pipeline**: Displays trades with no associated `strategy_id`. These rely entirely on the high-conviction XGBoost scoring and the AI Gatekeepers.
2. **Strategy-Based Pipeline**: Displays trades with a valid `strategy_id`. The table prominently features a new column displaying a "Strategy X" badge. Clicking this badge navigates the user to a dedicated strategy drill-down page.

## Strategy Details View (`strategy_detail.html` & `vanguard_dashboard.py`)
A new endpoint `/strategy/<int:strategy_id>` has been introduced to the Flask application. This endpoint serves a detailed breakdown of a single strategy's performance and activity.

**Key Features of the Details Page:**
- **Strategy Definition**: Clearly states the core logic, entry conditions, exit conditions, and directional bias (Long/Short).
- **Daily Grouping**: Trades are grouped by date (e.g., "Monday, June 01, 2026") under two main subtabs:
  - **Trades Taken**: Executed trades (OPEN, CLOSED, TAKE_PROFIT, STOP_LOSS).
  - **Trades Vetoed**: Rejected setups (VETOED, VETOED_EXPIRED), complete with the full AI explanation for the veto.

This architectural shift ensures clear visibility into both pure AI recommendations and structural strategy performance, allowing the operator to pinpoint exactly which rules or models are driving profitability.
