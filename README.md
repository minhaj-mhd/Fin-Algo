# 🚀 Vanguard V2.0: Elite Trading Engine (NSE)

Welcome to the **Vanguard V2.0** high-precision trading system. This engine is designed for institutional-grade intraday scanning on the NSE (India) market, combining XGBoost technical ranking with PrimoGPT sentiment analysis.

---

## 🎯 Global Objective
**Project Goal:** Achieve **85% Trade Accuracy** using a "Elite Batch" strategy (only the top 5-7 highest-conviction signals).

---

## 🧠 Core Intelligence System

### 1. Hybrid Ranking Model
- **Technical (XGBoost)**: Features 48 indicators (RSI, MACD, CMF, ATR, etc.) to rank 177+ liquid stocks by 1-hour return probability.
- **Sentiment (PrimoGPT Simulation)**: Simulates deep NLP/RAG scans to filter for "Technical Traps" and negative news.
- **Elite Filter**: Only triggers if `Technical Score > 0.15` and `Sentiment > +0.20`.

### 2. Strategic Execution (Vanguard 1h Logic)
- **Target Exit**: Automatic "Take Profit" (TP) closure at **+0.15% profit** to lock in gains within the momentum window.
- **Safety Window**: Strict **1-hour hard-close** if the target is not hit to prevent overnight exposure.
- **EOD Flush**: Mandatory closure of all open positions at **15:15 IST** (End of Session).

### 3. Financial Portfolio Management
- **Allocation**: 10% of currently redundant (available) liquid pool per trade.
- **Capital Recycling**: Capital is recycled immediately upon trade closure, adding net profit or loss back to the "Available Pool" for subsequent signals.
- **Compounding**: Supports intra-day compounding by reinvesting successful returns into the afternoon session.

---

## 📊 Live Monitoring & Reports

The engine provides real-time transparency across two primary documents:

1.  **[VANGUARD_DEMO_LEDGER.md](file:///c:/Users/Admin/Desktop/finalgo/data/VANGUARD_DEMO_LEDGER.md)**: Human-readable live trade board showing entry, exit, P&L, and result status.
2.  **[VANGUARD_FINANCIAL_AUDIT.md](file:///c:/Users/Admin/Desktop/finalgo/data/VANGUARD_FINANCIAL_AUDIT.md)**: Precise, event-by-event financial audit of every rupee moving through the portfolio.

---

## 🚀 Quick Start (March 2026)

### Prerequisites
- Windows OS (Admin recommended)
- Python 3.10+
- `yfinance`, `xgboost`, `pandas`, `numpy`, `pickle`

### Activation
Run the pre-configured batch file to initialize the environment and the Vanguard Scanner:
```powershell
.\run_vanguard_system.bat
```

### Manual Trigger
To run the Signal Engine directly via Python:
```bash
.\env\Scripts\python.exe scripts/vanguard_signal_engine.py
```

---

## 📅 Session Summary (Audit: March 9)
- **Total Trades**: 48
- **Winning Streak**: High consistency in Afternoon Session (+70% success).
- **Session Net ROI**: **+7.85%** on 1 Lakh Capital.
- **End-of-Month Projection**: ₹4.27 Lakhs (Aggr. Compounding).

---

## 🛡️ Important Safety Notes
- **Scanner Sleep**: The engine suspends new signal generation after **15:15 IST**.
- **Duplicate Prevention**: The engine will not execute a trade on a ticker if a previous position is still being observed by the Shadow Tracker.
- **Timezone Sync**: All timestamps are natively handled in **IST (UTC+5:30)** for NSE market alignment.

---
*Developed for Advanced Agentic Coding - Vanguard Project Phase 2.*
