---
title: "Action Plan: FinalGo \"Vanguard\" Systems (March 2026)"
type: archive
status: archived
updated: 2026-06-12
tags: []
---
# 🎯 Action Plan: FinalGo "Vanguard" Systems (March 2026)

## 📋 Objective
Build and deploy a real-time, high-accuracy (70%+) paper trading system combining our **XGBoost Ranking Model** with **PrimoGPT News Sentiment**. The goal is to achieve 0.1% minimum profit per trade with 85% filtered accuracy through a 4-week "Live Observation" phase.

---

## 🏗️ Phase 1: Real-Time Engine & PrimoGPT Integration (Week 1)
**Goal**: Create a unified backend that scores stocks technically and qualitatively every minute.

- [ ] **Unified Signal Script**: Build `scripts/vanguard_signal_engine.py` to:
    - Fetch 1-minute and 1-hour live data via `yfinance`.
    - Generate 52 technical features for the XGBoost Ranker.
    - Call PrimoGPT (LLM/RAG) for news sentiment on the top 10 technical candidates.
- [ ] **The "Elite Filter" Logic**:
    - `TECH_SCORE > 0.059` (Top 1% technical signals).
    - `SENTIMENT_SCORE > 0.2` (Positive news confirmation).
    - `VOLATILITY_GUARD`: Pause if Market_Mean_Volatility > 0.012.

## 🖥️ Phase 2: Live Testing Dashboard (Week 2)
**Goal**: Develop a Flask-based visual hub to track "Vanguard" trades without risking capital.

- [ ] **Dashboard Enhancements (`templates/dashboard.html`)**:
    - **Live Signal Feed**: Display symbols as they cross the Elite Filter threshold.
    - **Sentiment Insights**: Show PrimoGPT's reasoning for each buy signal (e.g., "Earnings Upgrade").
    - **Virtual P&L Tracker**: Automatic 1-hour "Hold & Exit" simulation to track real-world slippage.
- [ ] **Persistence**: Log every signal and its outcome to `data/vanguard_performance_log.csv`.

## 🧪 Phase 3: The "Observation" Phase (Week 3-4)
**Goal**: Validate the 200% annual profit hypothesis through live forward-testing.

- [ ] **Live Monitoring**: Run the system daily during NSE hours (09:15 - 15:30 IST).
- [ ] **Performance Review**: 
    - Verify if **"Power Hours" (14:15-15:30)** maintain the >60% accuracy.
    - Document **"Technical Traps"** that PrimoGPT correctly identified and blocked.
- [ ] **Slippage Analysis**: Compare the "Signal Price" vs. the "Execution Price" to ensure the 0.05% cost estimate holds.

## 🚀 Phase 4: Production Warm-up (Month 2)
**Goal**: Gradual transition to live brokerage integration.

- [ ] **Broker Bridge**: Implement API connectivity for automated order placement (e.g., Kite Connect).
- [ ] **Capital Allocation**: Start with 10% of target capital (₹10,000) for "Dry Run" live execution.
- [ ] **Compounding Protocol**: Initialize the 0.53% daily compounding spreadsheet.

---

## 📈 Success Metrics for Calibration
The system is ready for live capital **ONLY** if:
1. **Win Rate** > 65% over 20+ live signals.
2. **Profit Factor** (Total Wins / Total Losses) > 1.8.
3. **PrimoGPT Error Rate** < 10% (i.e., news analysis consistently aligns with market reaction).

---

## 🛠️ Immediate Next Steps (Today)
1. Execute `scripts/vanguard_signal_engine.py` prototype.
2. Update the Flask dashboard to include the "NLP_Sentiment" column.
3. Begin logging signals in `data/vanguard_performance_log.csv`.
