---
title: "Upstox Fees & Statutory Taxes Research (2024-2025)"
type: archive
status: archived
updated: 2026-06-12
tags: []
---
# Upstox Fees & Statutory Taxes Research (2024-2025)
**Project**: Vanguard Trading Engine
**Segment**: Equity Intraday (NSE)

As you transition Vanguard to live execution via the Upstox API, understanding the "Friction" (costs) is critical. These charges will be deducted from every trade, impacting your **Net Profit**.

---

## 1. Upstox Brokerage Structure
Upstox follows a flat-fee model for intraday trading to keep costs predictable for high-volume traders.

| Charge Type | Rate |
| :--- | :--- |
| **Intraday Brokerage** | **₹20 per executed order** OR **0.05%** (whichever is lower) |

> [!NOTE] 
> "Per executed order" means if Vanguard buys 100 shares of RELIANCE and later sells them, that is **two orders** (₹20 + ₹20 = ₹40 total brokerage).

---

## 2. Statutory Charges (Taxes & Regulatory Fees)
These are mandated by the Government of India and SEBI. Upstox collects them and passes them to the authorities.

| Fee Name | Rate | Applied On |
| :--- | :--- | :--- |
| **STT (Securities Transaction Tax)** | **0.025%** | **Sell Side only** |
| **Exchange Transaction Charges** | **0.00297%** | Both Buy & Sell sides |
| **GST** | **18%** | (Brokerage + Trans. Charges + SEBI Fee) |
| **SEBI Turnover Fee** | **₹10 per Crore** (0.0001%) | Both Buy & Sell sides |
| **Stamp Duty** | **0.003%** | **Buy Side only** |

---

## 3. Scaling Efficiency (The "Whale" Advantage)
As Vanguard's position size increases, the fixed ₹20 brokerage becomes negligible. This significantly lowers your "Breakeven" point.

| Capital Per Trade | Brokerage (RT) | STT + Trans + Stamp | **Total Friction %** |
| :--- | :--- | :--- | :--- |
| ₹10,000 | ₹10 (0.10%) | ~0.04% | **~0.14%** |
| ₹50,000 | ₹40 (0.08%) | ~0.04% | **~0.12%** |
| **₹1,00,000** | **₹40 (0.04%)** | **~0.04%** | **~0.08%** |
| ₹5,00,000 | ₹40 (0.008%) | ~0.04% | **~0.05%** |

> [!TIP]
> At ₹1 Lakh+ per trade, your friction is roughly **0.06% to 0.08%**. You only need the stock to move **0.1%** to be profitable.

---

## 4. Equity Futures (F&O) Integration
Trading Futures allows Vanguard to use **Leverage** and reduces the STT burden on high-turnover days.

### F&O Fee Structure
| Fee Name | Rate | Applied On |
| :--- | :--- | :--- |
| **Brokerage** | **₹20 per order** | Flat |
| **STT (Futures)** | **0.02%** (to be 0.05% in 2026) | **Sell Side only** |
| **Exchange Charges** | **0.0019%** | Both Sides |
| **Stamp Duty** | **0.002%** | **Buy Side only** |

### Leverage & Margin (10x Goal)
- **Standard Margin**: For most Nifty 200 stocks, the exchange requires ~15-20% margin (roughly **5x to 6x leverage**).
- **Intraday Leverage**: Some brokers/products allow up to **10x leverage** for intraday futures, meaning you can control a **₹10 Lakh contract with only ₹1 Lakh capital.**

**The "Leverage" Multiplier**:
If you use 10x leverage:
- Stock moves **+1%**.
- Your capital grows by **+10%**.
- Friction (0.06%) only eats **0.6% of your capital**, leaving you with **9.4% net profit.**

---

## 5. Income Tax Treatment (Speculative Business)
Intraday and Futures trading are classified as **Speculative Business Income**.
- **Taxation**: Profits are added to your total income and taxed at your **Slab Rate**.
- **Losses**: Can be carried forward for 4 years but can only be set off against other speculative profits.

---

## 5. Engineering Recommendation for Vanguard
To account for these fees in your code:
1. **Minimum Move**: Vanguard should not exit a trade unless the profit is at least **0.15% - 0.20%** to ensure the "Friction" is covered.
2. **Slippage Buffer**: Always assume an additional **0.02% slippage** (the difference between the price you see and the price you get) when calculating your "expected net profit."
