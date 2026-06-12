# Feature Engineering for 10:30 AM Intraday Trading Model (Indian Stock Market)

I have completed the research on feature engineering for an intraday trading model targeting the Indian stock market, specifically focused on entering trades after the initial volatility subsides at 10:30 AM IST. 

Below is the comprehensive, highly detailed markdown report on recommended features and their mathematical/logical transformations.

---

## 1. Capturing Overnight News & Global Sentiment
Overnight news, US market performance, and early morning Asian market data heavily influence the opening sentiment and initial trend of the Indian stock market. The model should capture these as exogenous variables that dictate the "expected baseline" for the day.

### Recommended Features & Transformations:
*   **Gift Nifty (formerly SGX Nifty) Premium/Discount:** Captures the overnight global sentiment priced into the Indian index before the 9:15 AM NSE open.
    *   *Mathematical Transformation:* Percentage spread between Gift Nifty's 9:00 AM IST price and the Nifty 50's previous day close. 
        `Spread_Gift = (Gift_Nifty_9AM - Nifty_Close_T1) / Nifty_Close_T1`
*   **US Markets Overnight Returns:** Captures risk-on or risk-off sentiment from Wall Street (S&P 500, Nasdaq, Dow Jones).
    *   *Mathematical Transformation:* Log returns of the US indices from open to close. 
        `LogRet_US = ln(Close_US / Open_US)`
*   **Asian Markets Morning Momentum:** Captures concurrent regional sentiment (e.g., Nikkei 225, Hang Seng) up to the Indian market open.
    *   *Mathematical Transformation:* Intraday return from the Asian market open to 10:15 AM IST. 
        `Ret_Asia = (Price_Asia_10:15IST - Open_Asia) / Open_Asia`

---

## 2. Utilizing 9:15 AM to 10:30 AM Market Dynamics
The first 75 minutes of the NSE/BSE open are characterized by high volatility, gap closures, and the establishment of initial ranges. By 10:30 AM, the market generally reveals whether it will continue the morning trend or reverse.

### Recommended Features & Transformations:
*   **Opening Gap Magnitude:** Quantifies the shock of the open.
    *   *Mathematical Transformation:* Log difference between today's open and yesterday's close.
        `Gap_Nifty = ln(Open_9:15 / Close_T1)`
*   **Gap Fill Status (9:15 - 10:30):** Identifies if the initial overnight shock was rejected by the market.
    *   *Logical Transformation (Boolean):* 
        If Gap Up: `1 if Min(Low_9:15_to_10:30) <= Close_T1 else 0`
        If Gap Down: `1 if Max(High_9:15_to_10:30) >= Close_T1 else 0`
*   **Opening Range Breakout (ORB) Position:** Measures where the 10:30 AM price sits relative to the first hour's extremes.
    *   *Mathematical Transformation:* Distance from the 9:15-10:15 High/Low, normalized by the 1-hour True Range.
        `ORB_Pos = (Price_10:30 - Min_Low_1hr) / (Max_High_1hr - Min_Low_1hr)`
*   **Volume Weighted Average Price (VWAP) Deviation:** Institutional traders track VWAP closely. Price relative to morning VWAP dictates trend confirmation.
    *   *Mathematical Transformation:* Percentage deviation of the 10:30 AM price from the cumulative morning VWAP.
        `Dev_VWAP = (Price_10:30 - VWAP_9:15_to_10:30) / VWAP_9:15_to_10:30`
*   **Volume Profile - Point of Control (POC):** The price level with the highest traded volume between 9:15 and 10:30 AM (High Volume Node).
    *   *Mathematical Transformation:* Log return of the current 10:30 AM price relative to the morning POC.
        `Ret_POC = ln(Price_10:30 / POC_Morning)`

---

## 3. Volatility and Macroeconomic Features
Incorporating market breadth, sector dynamics, and local volatility metrics (India VIX) contextualizes the breakout/reversal potential at 10:30 AM.

### Recommended Features & Transformations:
*   **India VIX Dynamics:** India VIX has a strong inverse correlation with the Nifty 50. High VIX dictates wider expected moves and gap downs.
    *   *Mathematical Transformation:* Z-score of the current VIX relative to a 20-day rolling window to detect volatility regimes.
        `VIX_Z = (VIX_10:30 - Mean(VIX_20d)) / StdDev(VIX_20d)`
    *   *Mathematical Transformation:* Daily VIX momentum.
        `VIX_Delta = ln(VIX_10:30 / VIX_Close_T1)`
*   **Sector Momentum & Divergence:** Often, the Nifty 50 might show a false signal while heavily weighted sectors (like Bank Nifty or Nifty IT) diverge.
    *   *Mathematical Transformation:* Relative Strength (RS) of a sector versus the Nifty 50 at 10:30 AM.
        `RS_BankNifty = ln(BankNifty_10:30 / BankNifty_Close_T1) - ln(Nifty_10:30 / Nifty_Close_T1)`
*   **Market Cap Weighting / Breadth (Advance-Decline Ratio):** Highlights participation strength. A Nifty breakout driven by only 2 large-cap stocks while the rest decline is likely a trap.
    *   *Mathematical Transformation:* Ratio of advancing stocks to declining stocks in the Nifty 500 or Nifty 50 at 10:30 AM.
        `ADR_Ratio = Advances_10:30 / Declines_10:30` (Can apply a log transform `ln(ADR_Ratio)` to center the parity around 0).

---

## 4. Machine Learning Implementation Guidelines
To feed these features into ML algorithms (like XGBoost, LightGBM, or LSTMs), the following preprocessing steps are required:

1.  **Stationarity Checks:** Raw prices are non-stationary. Always use **Logarithmic Returns** `ln(Pt/Pt-1)` or fractional differentiation rather than raw index points.
2.  **Standardization:** Apply **Z-Score standardization** (using rolling windows, e.g., 20-day or 50-day) so features like volume ratios and price deviations operate on comparable scales, minimizing weight bias.
3.  **Handling Skewness:** Volume features and Market Breadth ratios often exhibit skewed distributions. Apply **Power Transforms** (e.g., Yeo-Johnson transform) to stabilize variance and create a more Gaussian distribution, which improves neural network convergence.
4.  **Signal Smoothing:** For 1-minute or 5-minute momentum features leading up to 10:30 AM, apply a **Savitzky-Golay filter** or an Exponential Moving Average (EMA). This reduces market microstructure noise while preserving turning points.
5.  **Strict Avoidance of Look-Ahead Bias:** When calculating moving windows for 10:30 AM features, strictly use data up to 10:29:59 AM.
