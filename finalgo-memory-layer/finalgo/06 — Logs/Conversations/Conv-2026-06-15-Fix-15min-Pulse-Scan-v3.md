---
title: Fix 15-Minute Pulse Scan (Upstox v3 History API)
type: log
status: concluded
updated: 2026-06-15
---

# 💬 Conversation Context: Fix 15-Minute Pulse Scan

## 📌 Metadata
- **Start Date**: 2026-06-15
- **Status**: 🔴 Concluded
- **Focus Area**: Execution & Runtime — live Vanguard data path

## 🎯 Objectives
- [x] Diagnose why the live Pulse Scan failed for all 172 symbols with `UDAPI1020`.
- [x] Fix the 15-minute historical fetch using the Upstox v3 endpoint.
- [x] Verify against the live API.

## 💻 Active Code Files Modified
- [upstox_broker.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/upstox_broker.py) — `get_historical_data`

## 📝 Compacted Session Log
- **Symptom**: Pulse Scan (`orchestrator.py:797`) calls `get_historical_data(interval="15minute", fallback=False)`; every symbol returned `UDAPI1020` "Interval accepts one of (1minute,30minute,day,week,month)" → empty DataFrames (yfinance fallback disabled).
- **Root cause**: The broker passed `15minute` straight to the **v2** `HistoryApi.get_historical_candle_data1(...)`, which has no native sub-hour minute support. (The `*_v3.py` collectors sidestep this by fetching `1minute` and resampling.)
- **Fix**: Added a v3 branch in `get_historical_data` — `5minute`/`15minute` route through `HistoryV3Api` (`unit='minutes'`, `interval=<n>`), merging the v3 historical range + v3 intraday (today's live bars). Same `.status`/`.data.candles` parse. All existing paths (`day`, `60minute`, `1minute`, `30minute`) untouched.
- **Verification**: Live single-symbol call (RELIANCE.NS, 20d) → 329 native 15-min bars, correct 7 columns, modal gap exactly `00:15:00`, includes today's intraday up to 10:00 IST.
- **Open note**: v3 candles are tz-aware (`+05:30`); the scan loop resamples them directly (works). If downstream features need tz-naive IST like the `_v3.py` collectors, normalize in the broker later.

## 🔗 Core Memory Links & Backlinks
- [[01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop]]
