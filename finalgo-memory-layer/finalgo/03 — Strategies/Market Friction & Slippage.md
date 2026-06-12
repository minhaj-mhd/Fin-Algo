# ⚠️ Market Friction & Slippage (The Cost Drag Analysis)

In retail and institutional trading, **transaction costs and market friction are the ultimate strategy killers**. During the Vanguard engine development, we performed a deep friction sweep, shifting our round-trip costs from a lenient **0.03%** to a realistic **0.06%** (which accounts for NSE brokerage, STT, stamp duty, Exchange transaction charges, GST, SEBI turnover fees, and bid-ask slippage).

This single adjustment completely reshaped our strategy rankings, proving that **high-frequency scalpers face a massive mathematical disadvantage compared to low-frequency trend riders**.

---

## 📈 The Catastrophic Impact of Friction

When round-trip transaction costs were doubled to **0.06%**, high-frequency strategies experienced a dramatic collapse in profitability due to their high trade count:

### 📉 Case Study 1: Strategy 4 (Score Momentum Scalper)
*   **Lenient Cost Model (0.03%)**: **+1.81% Total Return** (Looking highly profitable!)
*   **Realistic Cost Model (0.06%)**: **-2.51% Total Return** (Catastrophic loss!)
*   **Why?**: Strategy 4 executed **144 trades** in May 2026. Doubling the friction cost by an extra 0.03% per trade created a massive **4.32% absolute drag** on the account's capital, turning a viable strategy into a bankruptcy trap.

### 📉 Case Study 2: Strategy 6 (Market-Neutral Pairs)
*   **Lenient Cost Model (0.03%)**: **+1.24% Total Return**
*   **Realistic Cost Model (0.06%)**: **-2.00% Total Return**
*   **Why?**: Executing **108 trades** with narrow profit margins meant that exchange charges and bid-ask spreads ate the entire profit edge.

---

## 🛡️ The Survival of Low-Frequency Strategies

In contrast, strategies that traded less frequently and held positions longer to ride structural trends proved highly resilient to the increased friction:

### 🏆 Strategy 8: Opening Range Breakout (ORB)
*   **Trades**: **14**
*   **Net Return**: **+2.31%**
*   **Profit Factor**: **3.92**
*   **Why it Survived**: By only entering breakouts with strong ML validation, Strategy 8 traded less than once a day on average. Its high average hold time allowed trades to hit larger profit targets (+1.00%), making the 0.06% transaction fee an insignificant cost of doing business.

### 🏆 Strategy 10: Quad-Timeframe Unanimous
*   **Trades**: **24**
*   **Net Return**: **+1.86%**
*   **Profit Factor**: **1.60**
*   **Why it Survived**: Complete timeframe agreement filtered out noisy entries, reducing unnecessary trades. It only executed high-conviction setups, preserving capital from fee erosion.

---

## 🧬 Understanding the Drag Equation

The mathematical drag of transaction fees on account capital is expressed as:

$$\text{Capital Drag} = \text{Trade Count} \times \text{Round-Trip Friction \%}$$

*   **For a 100-trade strategy at 0.06% friction**: **6.00%** of total capital is lost strictly to trading friction. The model's edge must exceed 6.00% just to break even!
*   **For a 14-trade strategy at 0.06% friction**: Only **0.84%** of capital is lost to friction. The required edge to achieve profitability is incredibly low.

---

## 🛡️ Best Practices for Combatting Slippage

1.  **Block High-Frequency Signals**: De-prioritize or retire momentum scalpers with high trade counts unless their win rate exceeds 65% with average gains over 0.50%.
2.  **Trade High ATR Stocks**: Focus on tickers with high Average True Range (ATR) relative to their stock price. Higher volatility allows trades to easily hit 1.00% TP targets, minimizing the impact of the fixed 0.06% transaction cost.
3.  **Optimize Limit Order Fills**: Utilize passive limit orders where possible to capture the bid-ask spread instead of crossing it with market orders.

---

## 👁️ Key Related Notes
*   See the strategy parameter configurations: [[03 — Strategies/Strategy Catalog|Strategy Catalog]].
*   See how exit charges are computed dynamically: [[01 — Architecture/Execution & Runtime/Shadow Tracker & Execution Loop|Shadow Tracker & Execution Loop]].
*   Review our database schema for fee auditing: [[01 — Architecture/Data & Code/Database Architecture|Database Architecture]].
