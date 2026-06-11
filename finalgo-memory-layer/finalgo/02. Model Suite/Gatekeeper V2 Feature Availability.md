# 🏛️ Gatekeeper V2 Feature Availability & Point-in-Time Contract

To ensure that the Daily Gatekeeper V2 model (`daily_macro_v2`) is completely free of lookahead bias, we establish a strict **Decision-Time Contract** governing when data is considered finalized and available.

---

## 1. The Decision-Time Contract

> [!IMPORTANT]
> All trading decisions for **Trade Day T** are made at **09:00 IST on Trade Day T** (prior to the market pre-open). 
> Any data point used in the feature calculation for Trade Day T must be fully finalized and public before **09:00 IST on Trade Day T**.

---

## 2. Feature Availability Table

The following table defines the exact data availability, timestamps, and lag rules:

| Feature Block / Asset | Source | Finalized Timestamp (Local Time) | Available at (IST) | Applied Lag / Join Rule |
| :--- | :--- | :--- | :--- | :--- |
| **India Equities (Universe)** | Upstox Daily | Day T-1 15:30 IST | Day T-1 ~16:00 IST | **No Lag**: Joined directly on row `T-1`. |
| **India Indices & VIX** | Upstox Daily | Day T-1 15:30 IST | Day T-1 ~16:00 IST | **No Lag**: Joined directly on row `T-1`. |
| **US Markets (S&P 500, Nasdaq, DXY, US10Y)** | `yfinance` | Day T-1 16:00 EST | Day T 01:30 / 02:30 IST | **No Lag**: Joined on row `T-1` (since US session T-1 is completed before 09:00 IST T). |
| **Commodities (Brent Crude, Gold)** | `yfinance` | Day T-1 17:00 EST | Day T 02:30 / 03:30 IST | **No Lag**: Joined on row `T-1` (US session T-1 ends before 09:00 IST T). |
| **Asia Morning Indices (Nikkei, HSI)** | `yfinance` | Day T-1 Closes | Day T-1 14:00 IST | **1-Day Lag**: Use **T-1 closes only** (Asian markets are active on morning T, so same-day T values are incomplete at 09:00 IST). |

---

## 3. Row Indexing & Join Convention

- The dataset is indexed by the last India trading day, i.e., `DateTime = T-1`.
- A row with `DateTime = 2026-06-09` (Tuesday) represents the state of the world available at **09:00 IST on 2026-06-10** (Wednesday, Trade Day T).
- **US joins**: S&P 500 (`^GSPC`) close for calendar date `2026-06-09` is joined to `DateTime = 2026-06-09`.
- **Asia joins**: Nikkei (`^N225`) close for calendar date `2026-06-09` is joined to `DateTime = 2026-06-09`.
- **India joins**: Upstox close for calendar date `2026-06-09` is joined to `DateTime = 2026-06-09`.
- **Target Label**: `label = Close(T+2) / Close(T-1) - 1` (a 3-bar forward return, where `Close(T+2)` represents the close of the 3rd daily bar starting from the current bar). Since row is `T-1`, `Close(T-1)` is the current close, and `Close(T+2)` is the close of trade day `T+2` (e.g., if `T-1` is Tuesday, trade day T is Wednesday, T+1 is Thursday, T+2 is Friday. The label measures return from Tuesday close to Friday close).

---

## 4. Point-in-Time Join Verification Assertion

In `build_daily_macro_dataset.py`, we enforce a programmatic check:
```python
# Programmatic assertion during dataset compilation:
assert (df['US_Close_Timestamp'] < Trade_Day_T_0900_IST).all(), "US close contains lookahead!"
assert (df['Asia_Close_Timestamp'] < Trade_Day_T_0900_IST).all(), "Asia close contains lookahead!"
assert (df['India_Close_Timestamp'] < Trade_Day_T_0900_IST).all(), "India close contains lookahead!"
```
Any column failing this check will be automatically excluded from features.
