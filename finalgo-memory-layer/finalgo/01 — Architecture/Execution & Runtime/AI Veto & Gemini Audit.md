---
title: "AI Veto & Gemini Audit (Live LLM Validation)"
type: report
status: active
updated: 2026-06-12
tags: []
---
# 🛡️ AI Veto & Gemini Audit (Live LLM Validation)

A defining capability of the Vanguard V2.3 engine is its **Hierarchical Dual-Stage AI Auditing & Veto Layer**. When a numeric signal is generated, the engine executes a real-time logical and fundamental validation sweep. Implemented in `scripts/vanguard/ai_veto.py`, this layer acts as an automated Chief Risk Officer (CRO), validating confluence patterns and shielding our capital from technical traps.

---

## 🏗️ The Auditing Pipeline

The system enforces a multi-layered hierarchical check, rotating API keys to prevent quota exhaustion and applying fallback tiers:

```text
                     [Numeric Signal Generated]
                                 │
                       Extract Live Features
                                 │
                Calculate Dynamic Support & Resistance
                                 │
                    ┌────────────▼────────────┐
                    │ STAGE 1: TECH TRIAGE    │
                    │   gemini-3.5-flash      │ ◄── (Fallback: gemini-3.1-flash-lite)
                    └────────────┬────────────┘
                                 │
                       Is Tech Sentiment OK?
                        ├── No  ──> VETO TRADE (Log Cooldown)
                        └── Yes ──> Proceed to Stage 2
                                 │
                    ┌────────────▼────────────┐
                    │ STAGE 2: CRO NEWS AUDIT │
                    │   gemini-2.5-flash      │ ◄── (Using Google Search Grounding)
                    └────────────┬────────────┘
                                 │
                      Apply 6-Rule Veto Matrix
                        ├── Veto ─> VETO TRADE (Log Cooldown)
                        └── Pass ─> APPROVE ENTRY & START SHADOW TRADE
```

---

## 🛠️ Stage 1: Technical Triage (T1)

Stage 1 evaluates the purely technical and mathematical confluence of the signal. No news search is conducted here; the LLM behaves strictly as an elite chart technician reviewing 1-hour bar features and short-term price history.
*   **The Models**: Primary model is **gemini-3.5-flash** (highly strict schema parser). Fallback is **gemini-3.1-flash-lite** if Tier 1 calls return rate limits (429) or quota errors. Key indexes rotate automatically inside `gemini_audit()`.
*   **Live Context Passed**: RSI-14 (intraday & daily), Stochastic %K, Bollinger %B, RVOL, 1-hour return, streaks, Nifty/Regime returns, VIX, and nearest S/R distance. Also passes a **1-minute bar history string** from the last 30 minutes to capture momentum speed.
*   **Response Schema**: The model must output strict JSON only:
    ```json
    {"sentiment": "LABEL", "reason": "concise 1-sentence technical confluence/conflict rationale", "probability": "XX%"}
    ```
    *   *Labels*: `STRONG BULLISH` | `BULLISH` | `NEUTRAL` | `BEARISH` | `STRONG BEARISH`
*   **Stage 1 Veto Gate**: 
    *   For `LONG` trades: Vetoed if sentiment is `NEUTRAL`, `BEARISH`, or `STRONG BEARISH`.
    *   For `SHORT` trades: Vetoed if sentiment is `NEUTRAL`, `BULLISH`, or `STRONG BULLISH`.
    *   *Result*: If vetoed, returns `[S1-VETO]` with the reason, blocks entry, and places the ticker on a **30-minute veto cooldown**.

---

## 🔍 Stage 2: CRO News Grounding Audit (T2)

Signals that survive Stage 1 are passed to Stage 2, which acts as the ultimate governance check to verify if corporate or market-wide reality contradicts the technical signal.
*   **The Models**: Utilizes **gemini-2.5-flash** (and **gemini-2.5-flash-lite** fallback) from the `GeminiRateTracker` pool, which has access to the **Google Search Grounding Tool**.
*   **Search Queries**: The model programmatically queries:
    1.  `"{Company Name} stock news"`
    2.  `"{Company Name} NSE results earnings"`
    3.  `"{Company Name} block deal bulk deal today"`
*   **News Classification**: Matches news into structural catalysts (Bucket A: earnings, brokerages, policy changes) or tactical catalysts (Bucket B: block deals, volume spikes explanation).
*   **The 6-Rule Veto Decision Matrix** (First matched rule wins):
    *   **RULE 1 — HARD VETO (Fundamental Conflict)**:
        *   `LONG` + Bucket A earnings miss / regulatory bans / rating downgrades $\rightarrow$ `VETO = TRUE`
        *   `SHORT` + Bucket A earnings beat / rating upgrades / buyback $\rightarrow$ `VETO = TRUE`
    *   **RULE 2 — SOFT VETO (Tactical Conflict)**: Vetoes `LONG` if major block sell/negative headline is found AND S1 conviction is low (`< 0.25`). Vetoes `SHORT` under block buy/positive headline if S1 conviction is low (`< 0.25`).
    *   **RULE 3 — BREAKOUT OVERRIDE**: If `RVOL > 3.0` AND strong positive catalysts are present, the trade is marked as a *breakout* and nearest resistance is treated as a target, not a wall $\rightarrow$ `DO NOT VETO`.
    *   **RULE 4 — MAGNET EFFECT**: Vetoes `SHORT` if price action has ground within 0.5% of a resistance level for $>15$ minutes (absorption $\rightarrow$ breakout imminent). Vetoes `LONG` if price grinds within 0.5% of support for $>15$ minutes (imminent breakdown).
    *   **RULE 5 — S/R PROXIMITY RISK**: Vetoes if the nearest dynamic resistance/support wall is within **0.3%** of the current entry price, unless a powerful catalyst (Rule 3) supports pushing through.
    *   **RULE 6 — NO NEWS = DEFER TO S1**: If no news is found, it skips the veto and anchors its final score to Stage 1 technical conviction.

---

## ⏱️ Asynchronous Time-Stop Extensions

Intraday predictions decay over time. The Shadow Tracker enforces a strict **1-hour hard close (4 bars of 15M)**. However, if a trade is in consolidation right near expiry, the system can override this close:
*   **Condition**: The trade is open, at a loss (`pnl < 0`), has been held for 1 hour, has an extension count $< 2$, and the time is before `15:15 IST`.
*   **Background Sweep**: Wakes up an asynchronous background daemon thread (`_gemini_check_extension()`) to prevent locking the execution loop.
*   **The Prompt**: Sends live entry price, current price, P&L, and 15-minute candlesticks. Asks: *"EXTEND or CLOSE?"*
*   **Action**:
    *   If Gemini returns `EXTEND`: Extends the hard-close time by **15 minutes** (up to 2 times, total 30 mins max).
    *   If Gemini returns `CLOSE` (or errors out): The trade exits immediately at market value.

---

## 📈 Veto Alpha Saved Analytics

The effectiveness of the AI Veto layer is logged in the `trades` database:
*   **Vetoed Trade Logging**: When a trade is vetoed, it is entered in the SQLite DB with status `VETOED` and tracked live by `shadow_tracker_loop` (exits only at 1-hour expiry `VETOED_EXPIRED`).
*   **The Alpha Metric**: `database_manager.py` audits what would have happened if those vetoed trades were taken:
    $$\text{Veto Alpha Saved} = \sum (\text{Final Profit \% of Vetoed Trades} - 0.06\% \text{ Friction})$$
*   If vetoed trades hit their stop losses, **Veto Alpha is positive** (the LLM successfully saved us from losses). If they hit targets, **Veto Alpha is negative** (the LLM is too conservative), triggering prompt-tuning loops.

---

## 👁️ Key Related Notes
*   See how vetoed trades are registered and updated in SQLite: [[01 — Architecture/Data & Code/Database Architecture|Database Architecture]].
*   See how shadow trades and cooldowns are processed: [[01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop|Shadow Tracker & Execution Loop]].
