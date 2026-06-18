---
title: "Daily Bars Cache Optimization"
type: log
status: active
updated: 2026-06-16
---
# 💬 Conversation Context: Daily Bars Cache Optimization

## 📌 Metadata
- **Conversation ID**: dd0e090e-f98c-4be3-9957-c46e8dbdedf1
- **Start Date**: 2026-06-16
- **Status**: 🔴 Concluded
- **Focus Area**: Orchestrator Performance / Scan Cycle Efficiency

## 🎯 Objectives
- [x] Cache the universe daily bars (`yf.download` 60d/1d for ~N tickers) per calendar day to avoid re-downloading every 15-min scan cycle.

## 💻 Active Code Files Modified
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py)

## 📝 Compacted Session Log

- **Initial Analysis**: `calculate_conviction_scores()` (line 790) calls `yf.download(tickers, period="60d", interval="1d")` every scan cycle (~26×/day for a 15-min cadence). The data feeds `iloc[-2]` features (Daily_RSI, Daily_SMA20_Dist, Daily_Trend, Daily_ATR_Pct) which are by definition the *previous completed day* — they don't change intraday. Wasted network + latency.
- **Step 1**: Added `self._daily_bars_cache = pd.DataFrame()` and `self._daily_bars_cache_date = None` to `__init__`.
- **Step 2**: In `calculate_conviction_scores()`, replaced the bare `yf.download` call with a cache-check guard: if `self._daily_bars_cache_date == datetime.now().date()` and cache is non-empty, reuse; otherwise fetch fresh and stamp with today's date.
- **Rationale**: Mirrors the existing `self.current_date` / `self._veto_stats_date` per-day guard pattern already in `run()`. No structural changes needed.

## 🔗 Core Memory Links & Backlinks
- [[01 — Architecture/Vanguard Live Engine]] (if it exists)
