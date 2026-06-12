---
title: "Dual-Specialist & Meta-Routing (Proposed Architectural Roadmap)"
type: archive
status: archived
updated: 2026-06-12
tags: []
---
# 🧠 Dual-Specialist & Meta-Routing (Proposed Architectural Roadmap)

> [!NOTE]
> **PROPOSED ARCHITECTURAL ROADMAP**: The specialized dual-model routing (`MR_Model_v2` and `BO_Model_v1` meta-router) detailed below represents our planned phase architecture for future system iterations. 
> 
> The **active production engine (Vanguard Ensemble V2.3)** utilizes a unified, scale-invariant XGBoost hourly ranking Specialist (`v8_upstox_3y` trained on 3 years of hourly candles with microstructure features), governed by a **Daily Macro Trend Scan Gatekeeper** (Daily Macro XGBoost model) and a **hierarchical dual-stage AI Veto**.

---

## 🏗️ Proposed Dual-Specialist Concept

In future phases, the Vanguard system is designed to split execution between two highly specialized sub-models governed by a **Regime-Aware Meta-Decision Layer**. This prevents a single model from trying to learn conflicting technical dynamics (e.g. breakout expansions vs. mean-reversion exhaustion).

```text
                        RAW MARKET DATA
                               │
                       Feature Extraction
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    [Reversal Specialist]             [Breakout Specialist]
       (MR_Model_v2)                     (BO_Model_v1)
               │                               │
       Reversal Probability             Momentum Probability
               └───────────────┬───────────────┘
                               ▼
                    [Meta-Decision Router]
                   (Regime: ADX, VIX, ATR)
                               │
                    LONG / SHORT / NO TRADE
```

---

## 🤖 The Specialized Sub-Models (Roadmap Specs)

### 1. Model A: The Reversal Specialist (`MR_Model_v2`)
Optimized strictly to identify overextended price boundaries and predict clean exhausts and immediate mean-reversions.
*   **Target Labeling**: Trained on binary classification targets evaluating whether a stretched indicator condition reverts cleanly without touching dynamic stop boundaries.
    $$\text{Stretch} = \frac{\text{Close} - \text{EMA}_{20}}{\text{ATR}_{14}}$$
    $$\text{Target} = 1 \text{ if } \left(\text{Stretch} > 2.0 \text{ and } \frac{\text{Future Return}_{1h} - \text{Mean Return}}{\text{Std Return}} < -2.0\right) \text{ else } 0$$
*   **Key Predictors**: Bollinger Bands %B, Keltner Channel Width, Stochastic (%K, %D), CCI overbought/oversold boundaries, and time of session.

### 2. Model B: The Breakout Specialist (`BO_Model_v1`)
Optimized strictly to capture volatility contractions (coils) followed by explosive, high-volume momentum expansion.
*   **Target Labeling**: Classifies whether a structural boundary breakout continues in the breakout direction over multi-hour horizons.
    $$\text{Target} = 1 \text{ if } \left(\text{Close} > \text{Rolling High}_{20} \text{ and } \text{Future Return}_{3h} > 3\%\right) \text{ else } 0$$
*   **Key Predictors**: Relative Volume (RVOL), Bollinger Band Squeeze indices, ADX, EMA slopes, ROC, and PPO momentum acceleration.

---

## 🚦 The Meta-Decision Router

The Meta Layer serves as the system arbiter, routing prediction power based on **volatility and regime indicators**:

```text
                           [Signal Generator]
                       /                        \
            Reversal Prob: 82%              Breakout Prob: 11%
                       \                        /
                     [Regime Condition Routing]
                                  │
                   Is ADX > 25 (Strong Trend)?
                        ├── Yes ──> Route to Breakout Model ──> IGNORE SIGNAL
                        └── No  ──> Route to Reversal Model ──> EXECUTE SHORT
```

### 🛣️ Proposed Heuristic Routing Protocol
1.  **Mean-Reversion / Choppy Range (ADX < 20, Expanding ATR Volatility)**:
    *   Bypasses all breakout signals to protect capital from whipsaws.
    *   Routes all signal processing power to the **Reversal Specialist** (`MR_Model_v2`).
2.  **Trending / Momentum Squeeze (ADX > 25, Compressing ATR)**:
    *   Bypasses all reversal signals (to prevent catching falling knives in strong trend expansions).
    *   Routes all signal processing power to the **Breakout Specialist** (`BO_Model_v1`).

---

## ⚡ Active V2.3 Production Reality

In our active **Vanguard Ensemble V2.3**, this specialized division is effectively achieved through:
1.  **Unified Microstructure Specialist (`v8_upstox_3y`)**: Our core XGBoost hourly model is trained on 3 years of Upstox data, loaded with indicators that capture *both* trend structure (SMAs/EMAs slopes) and intraday liquidity microstructure (Intraday Bar Position [IBS], Buy Pressure, Volume Acceleration), enabling standard scale-invariant trees to optimize entries across all regimes.
2.  **Daily Macro Gatekeeper**: Real-time hourly signals are validated by a startup daily scan, which uses the Daily Macro XGBoost model to filter out tickers that conflict with the macroeconomic daily regime.
3.  **Dynamic Volatility Brackets**: Dynamic ATR calculation at entry automatically switches trade behavior: coiling markets receive tight, high-win-rate brackets, while high-volatility breakouts capture wider, high-profit targets.

---

## 👁️ Key Related Notes
*   See how our active models are configured and selected: [[02 — Models/_Shared/Model Registry & File Structures|Model Registry & File Structures]].
*   Review our dynamic ATR execution rules: [[01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop|Shadow Tracker & Execution Loop]].
*   See the daily gatekeeper lookup logic: [[02 — Models/_Shared/Multi-Timeframe Models|Multi-Timeframe Models]].
