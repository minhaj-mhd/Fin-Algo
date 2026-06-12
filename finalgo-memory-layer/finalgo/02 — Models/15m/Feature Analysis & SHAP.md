# Feature Analysis & SHAP: v2_15min_3y

**Date:** June 7, 2026
**Subject:** Feature importance by gain, SHAP beeswarm analysis, and SHAP dependence plots for `IBS` and `Buy_Pressure` — the two dominant features in the 15-minute model.
**Method:** TreeExplainer SHAP on 3,000 OOS samples (Apr–Jun 2026). Importance type: `gain`.

---

## Visual Summary

![[assets/01_feature_importance.png]]
![[assets/02_shap_summary.png]]
![[assets/03_shap_dependence.png]]

---

## 1. Top-20 Feature Importance (Gain)

| Rank | Feature | Long Gain | Short Gain | Description |
|---|---|---|---|---|
| 1 | **IBS** | 52.67 | 57.56 | `(Close - Low) / (High - Low)` — bar position |
| 2 | **Buy_Pressure** | 47.23 | — | `Vol × (Close - Open) / Range` — order flow |
| 3 | **Log_Return** | 33.33 | 16.45 | `log(Close / Prev_Close)` — bar momentum |
| 4 | **Return** | 6.34 | 31.10 | `(Close - Prev_Close) / Prev_Close` |
| 5 | **Relative_Return** | 9.51 | 24.93 | Stock return minus cross-sectional mean |
| 6 | **Lower_Shadow** | 15.40 | 8.88 | `(Open - Low) / Range` — bullish rejection |
| 7 | **Is_Open_Hour** | 8.57 | 8.64 | Binary: first hour of trading (09:15–10:15) |
| 8 | **Time_To_Close** | 7.56 | 8.11 | Minutes remaining until 15:30 market close |
| 9 | **OC_Range** | 6.03 | 9.42 | `(Open - Close) / Close` |
| 10 | **Dollar_Volume** | 3.77 | 9.24 | `Close × Volume` — liquidity proxy |
| 11 | **IBS_3** | — | 11.85 | 3-bar rolling IBS average |
| 12 | **Dist_Donchian_Lower** | — | 11.77 | Distance from 20-bar Donchian channel low |
| 13 | **Hour** | 5.76 | 5.94 | Hour of day as numeric (9, 10, ..., 15) |
| 14 | **Upper_Shadow** | — | 11.10 | `(High - Close) / Range` — bearish rejection |
| 15 | **Stoch_K** | 6.67 | — | Stochastic %K oscillator |
| 16 | **Dist_SMA_6** | 3.61 | 5.61 | Distance from 6-bar simple moving average |
| 17 | **Stoch_D** | — | 8.36 | Stochastic %D (smoothed K) |
| 18 | **Dist_Keltner_Lower** | — | 7.66 | Distance from lower Keltner channel band |
| 19 | **Keltner_Width** | — | 6.32 | Keltner channel width (volatility measure) |
| 20 | **Is_Close_Hour** | 5.90 | — | Binary: last 30 minutes before close |

**A dash (—) means the feature did not appear in the top 20 for that direction.**

---

## 2. SHAP Beeswarm Analysis (Long Model)

The SHAP beeswarm plot shows 3,000 OOS samples, where each dot is one stock-bar observation. The x-axis is the SHAP value (impact on model output in score units), and color represents the feature value (red = high, blue = low).

### Key Insights from Beeswarm

**IBS (SHAP value: highest spread)**
- Low IBS (blue dots) → strongly positive SHAP values → model pushes the stock UP in ranking.
- High IBS (red dots) → negative SHAP values → model pushes the stock DOWN.
- Interpretation: Bars closing near their low (low IBS) signal imminent mean-reversion upward. This is the primary alpha mechanism in the 15-min regime.

**Buy_Pressure**
- High Buy_Pressure (red) → consistently positive SHAP.
- Low Buy_Pressure (blue) → consistently negative SHAP.
- Interpretation: Aggressive buying within the bar (volume-weighted net buying) predicts continuation into the next bar.

**Log_Return**
- Wide bi-modal distribution: extreme negative returns (blue) have high positive SHAP (mean reversion expected), extreme positive returns (red) have negative SHAP.
- The model partially fades very large up-moves (momentum reversal) and lifts stocks with extreme down-moves.

**Time features (Is_Open_Hour, Time_To_Close, Hour)**
- Moderate SHAP magnitude but consistent direction — the model has learned that signal reliability varies strongly by time of day.

---

## 3. SHAP Dependence Analysis

### IBS Dependence Plot

- **X-axis:** Raw IBS value (0 = bar closes at low, 1 = bar closes at high)
- **Y-axis:** SHAP(IBS) — contribution of IBS to this stock's ranking score
- **Color:** Buy_Pressure (red = high buying pressure, blue = low)

**Shape:** Strongly monotonically decreasing. IBS = 0.0–0.2 → SHAP ≈ +0.03 to +0.05 (strong upward push). IBS = 0.8–1.0 → SHAP ≈ −0.03 to −0.05 (strong downward push).
**Interaction:** At low IBS, high Buy_Pressure (red) adds further upward contribution. The combination of low IBS + high Buy_Pressure is the model's strongest long signal. At high IBS, low Buy_Pressure (blue) amplifies the bearish signal.

### Buy_Pressure Dependence Plot

- **X-axis:** Raw Buy_Pressure value
- **Y-axis:** SHAP(Buy_Pressure)
- **Color:** IBS (red = high, blue = low)

**Shape:** Monotonically increasing — higher Buy_Pressure → stronger positive SHAP contribution.
**Interaction:** The Buy_Pressure effect is amplified when IBS is also low (blue coloring at moderate-high Buy_Pressure). This confirms the models' two dominant features reinforce each other.

---

## 4. Feature Groups and Their Roles

### Group 1: Microstructure (Primary Alpha Drivers)
| Feature | Role |
|---|---|
| IBS | Bar position mean-reversion anchor. Most important feature in both models. |
| Buy_Pressure | Net order flow volume-weighted signal. Confirms or contradicts IBS. |
| Lower_Shadow / Upper_Shadow | Candle-body rejection wicks — corroborate IBS direction. |
| IBS_3 | 3-bar smoothed IBS — detects multi-bar mean-reversion setup. |

### Group 2: Return Momentum
| Feature | Role |
|---|---|
| Log_Return / Return | Short-term momentum. Extreme values trigger mean-reversion prediction. |
| Relative_Return | Return relative to cross-sectional average — pure alpha/beta separation. |
| OC_Range | Open-to-close within-bar momentum. |

### Group 3: Time Context
| Feature | Role |
|---|---|
| Is_Open_Hour | Opening hour dynamics — different volatility regime. |
| Is_Close_Hour | EOD squeeze and institutional book-squaring. |
| Time_To_Close | Continuous version — model prices in decreasing time-to-close. |
| Hour | Coarse time-of-day regime marker. |

### Group 4: Volatility Channels
| Feature | Role |
|---|---|
| Dist_Keltner_Lower / Upper | Distance from Keltner channel bounds — volatility-adjusted extremes. |
| Dist_Donchian_Lower | Distance from breakout channel — supports short signal at lower extremes. |
| Keltner_Width | Channel width = current volatility regime. |

### Group 5: Liquidity
| Feature | Role |
|---|---|
| Dollar_Volume | Stock liquidity filter — higher dollar volume stocks are more predictable. |
| RVOL | Relative volume — unusual volume signals potential trending day. |
| Volume_Zscore | Volume spike detection. |

---

## 5. Structural Comparison vs 1-Hour Model (v8_upstox_3y)

| Feature | 15-Min (v2) Rank | 1-Hour (v8) Rank | Interpretation |
|---|---|---|---|
| IBS | #1 (52.7 gain) | #1 | IBS dominates across timeframes — mean-reversion is universal |
| Buy_Pressure | #2 (47.2 gain) | #2 | Microstructure order flow equally important |
| Log_Return | #3 | #3 | Short-term momentum fade is consistent |
| Return_Accel | Not in top 5 | #4 | Return acceleration matters more at hourly timeframe |
| Lower_Shadow | #6 | #4 | Shadow patterns equally relevant |

Both models are anchored to the same two primary signals (IBS + Buy_Pressure), confirming that NSE intraday microstructure mean-reversion is the dominant alpha source regardless of bar frequency.

---

## 6. Backlinks

- [[Complete Edge Catalog]] — Walk-forward performance tables.
- [[OOS Calibration & Thresholds]] — How IBS drives the calibration monotonicity.
- [[Model Diagnostics & Visualizations]] — Feature importance PNG and SHAP plot assets.
