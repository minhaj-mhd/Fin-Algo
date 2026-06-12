# 🗃️ Outdated Vanguard V1 Specification (Archive)

> [!WARNING]
> **ARCHIVAL MATERIAL**: The specifications detailed below represent the retired baseline configurations of **Vanguard V1.0** and intermediate V2 testing iterations. These rules have been retired and are preserved here strictly for historical backtest auditing, prompt regression testing, and comparative analysis. Do **NOT** apply these configurations to active live engines.

---

## 🚦 Vanguard V1.0 Execution Rules (Retired May 2026)

*   **Static Take Profit (TP)**: Set at a strict, rigid **+0.15%** target closure. 
    *   *Retirement Reason*: High vulnerability to slippage and transaction charges. Standard round-trip friction of 0.06% consumed **40%** of the gross trade edge, making it mathematically unviable under realistic market conditions.
*   **Static Stop Loss (SL)**: Set at a strict, fixed **-0.50%**.
    *   *Retirement Reason*: Replaced by dynamic volatility-adjusted ATR-SL to prevent premature exits during high-volatility spikes and capture tighter stops during low-volatility coils.
*   **Time Stop**: Strict 1-hour hard-close with no asynchronous LLM extensions.
*   **Model Horizon**: Single regularity XGBoost model trained on hourly returns, which did not support daily trend gatekeepers or temporal Transformers.

---

## 🛡️ V1.0 Single-Stage AI Auditing (Retired Schema)

During early testing, the AI veto was managed by a single-stage model calling `gemini-3.5-flash` with a strict JSON format evaluating only technical traps. 
*   **V1 JSON Response Format**:
    ```json
    {
      "decision": "APPROVED" | "VETOED",
      "confidence": 0.85,
      "reason": "Stock is trading at a 5-day resistance pivot. Volume is exhaustive, suggesting institutional distribution.",
      "trap_type": "Resistance Closeness / Distribution Trap"
    }
    ```
    *   *Retirement Reason*: Bypassed news and block deals entirely, making the system vulnerable to fundamental shocks, corporate earnings reports, and regulatory announcements.

---

## 📈 Strategic Evolution: V1.0 vs. Active V2.3

| Parameter | Vanguard V1.0 (Retired) | Vanguard Ensemble V2.3 (Active) |
| :--- | :--- | :--- |
| **Model Scope** | Single Hourly Model | **XGBoost Daily Macro Gatekeeper** + **Unified v8 Hourly Ranker** with microstructure features. |
| **Take Profit (TP)** | Static **+0.15%** | **Dynamic 15M ATR-based** (3.0x ATR multiplier, clamped `[+0.75%, +2.50%]`) |
| **Stop Loss (SL)** | Static **-0.50%** | **Dynamic 15M ATR-based** (1.5x ATR multiplier, clamped `[0.30%, 1.50%]`) |
| **AI Auditing Layer**| None / Single-stage tech traps | **Hierarchical Dual-Stage Veto** (S1 `gemini-3.5-flash` tech triage + S2 `gemini-2.5-flash` Google Search grounded CRO audit). |
| **Active Risk Rules**| None | **15-Min Conviction Flip**, **Breakeven Locking**, **Trailing Stop pegging**, and **charge leakage refunds**. |
| **Expiry Extensions**| None | Asynchronous background daemon thread check for up to **two 15-minute extensions** on consolidating trades at a loss. |

---

## 🔄 Historical V1 Inversion Workaround (Retired May 2026)

During early system development (V1.0), the engine utilized a single XGBoost regressor/ranker to predict future returns. Out-of-sample testing revealed a massive performance anomaly: the model's standard predictions exhibited a strong **negative Spearman correlation (-0.55)** with future returns. 

When the model recommended a buy with absolute confidence, the stock almost always collapsed. Conversely, when it predicted a crash, the stock consistently rallied.

### The Underlying Anomaly: High-Volume Distribution vs. Accumulation
This negative correlation was driven by XGBoost over-weighting **volume-based features** (which occupied the top 5 spots in feature importance):
*   **High-Volume Days (FOMO Distribution)**: Retail momentum buying created massive volume spikes, but institutional smart money used this deep liquidity to distribute (sell) their holdings, leading to immediate intraday price declines.
*   **Low-Volume Days (Quiet Accumulation)**: Smart money quietly accumulated shares without shifting price significantly. The lack of retail interest allowed the stock to coil tight, leading to a massive upward breakout in subsequent sessions.

### The Legacy V1.0 Solution
Because standard libraries ranked outputs based on positive mathematical coefficients, standard raw predictions led the engine to buy high-volume FOMO traps. To circumvent this, a **one-line negative multiplier** was implemented:
```python
# Legacy V1.0 negating workaround:
inverted_predictions = -raw_predictions
```

### Strategic V2.3 Replacement
This raw negating multiplier was a fragile workaround. In **Vanguard V2.3**, the Inversion Factor has been completely retired. 

Instead, the system utilizes **Dual-Stage Classification**, training separate, dedicated Long and Short models (`xgb_long_model.json` and `xgb_short_model.json`). Conviction is natively calculated at runtime as:
$$\text{Conviction} = \text{long\_score} - \text{short\_score}$$
This ensures the model behaves intuitively and robustly, removing the need for manual multiplier hacks.
