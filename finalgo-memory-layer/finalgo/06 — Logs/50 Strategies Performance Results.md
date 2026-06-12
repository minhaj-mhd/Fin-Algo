---
title: "50-Strategy Regime-Aware Backtest Results"
type: report
status: active
updated: 2026-06-12
tags: []
---
# 50-Strategy Regime-Aware Backtest Results

**Test Period:** May 2026
**Tickers Universe:** 172 symbols
**Data Regimes Evaluated:** Gated (Daily Gatekeeper applied) vs Ungated (Intraday only)

The following table summarizes the dual-pass regime backtest over all 50 strategies:

| ID | Strategy Name                        | Gated WR          | Ungated WR        | Gated Net % | Ungated Net % | Verdict                    |
|---|--------------------------------------|-------------------|-------------------|-------------|---------------|----------------------------|
| 1  | Daily Macro Gatekeeper              | 50.9% (108t)     | 50.9% (108t)     | +2.20%      | +2.20%       | ROBUST: Profitable in Both |
| 2  | Short-Side Specialist               | 58.8% (17t)      | 58.8% (17t)      | +0.69%      | +0.69%       | ROBUST: Profitable in Both |
| 3  | Timeframe Divergence Fade           | 58.3% (108t)     | 53.7% (108t)     | +4.67%      | -0.17%       | TREND: Gatekeeper Required |
| 4  | Score Momentum Scalper              | 57.6% (144t)     | 49.3% (144t)     | +4.70%      | -2.51%       | TREND: Gatekeeper Required |
| 5  | Power Hour Sniper                   | 52.5% (61t)      | 41.7% (72t)      | +3.02%      | -1.62%       | TREND: Gatekeeper Required |
| 6  | Market-Neutral Pairs                | 43.5% (108t)     | 43.5% (108t)     | -2.00%      | -2.00%       | FAIL: Unprofitable         |
| 7  | Volatility Regime Switcher          | 49.4% (160t)     | 48.1% (160t)     | -1.21%      | -2.39%       | FAIL: Unprofitable         |
| 8  | Opening Range Breakout (ORB) + Confirmation | 40.0% (10t)      | 50.0% (14t)      | -1.20%      | -1.06%       | FAIL: Unprofitable         |
| 9  | Conviction Spread Z-Score           | 43.9% (82t)      | 46.3% (108t)     | -3.94%      | -9.63%       | FAIL: Unprofitable         |
| 10 | Quad-Timeframe Unanimous            | 50.0% (20t)      | 50.0% (20t)      | +2.46%      | +2.46%       | ROBUST: Profitable in Both |
| 11 | Hourly Trend Rider                  | 50.0% (108t)     | 50.0% (108t)     | -0.52%      | -0.52%       | FAIL: Unprofitable         |
| 12 | Pre-Noon Reversal Scalp             | 42.9% (7t)       | 57.1% (14t)      | +1.12%      | +2.38%       | TACTICAL: Run Ungated      |
| 13 | Microstructure Exhaustion Fade      | 59.1% (22t)      | 45.2% (42t)      | +3.81%      | +0.63%       | TREND: Gatekeeper Required |
| 14 | Volume Shock Breakout               | 42.9% (42t)      | 44.9% (69t)      | +0.11%      | +1.42%       | TACTICAL: Run Ungated      |
| 15 | Triple Moving Conviction Fade       | 0.0% (0t)        | 0.0% (0t)        | +0.00%      | +0.00%       | FAIL: Unprofitable         |
| 16 | Overnight Gap Fade                  | 0.0% (0t)        | 30.8% (26t)      | +0.00%      | -10.98%      | FAIL: Unprofitable         |
| 17 | Micro-Volatility Breakout           | 0.0% (0t)        | 38.0% (108t)     | +0.00%      | -27.39%      | FAIL: Unprofitable         |
| 18 | Regime-Aware Trend Follower         | 0.0% (0t)        | 100.0% (1t)      | +0.00%      | +0.89%       | TACTICAL: Run Ungated      |
| 19 | Low-Vol Squeeze Scalper             | 50.0% (80t)      | 59.0% (83t)      | +6.72%      | +15.93%      | TACTICAL: Run Ungated      |
| 20 | High-Vol Knife Catcher              | 36.0% (50t)      | 33.8% (71t)      | -2.49%      | -5.05%       | FAIL: Unprofitable         |
| 21 | Intraday Trend Exhaustion           | 53.3% (15t)      | 51.5% (33t)      | +3.41%      | -0.07%       | TREND: Gatekeeper Required |
| 22 | Conviction Peak Divergence          | 43.1% (72t)      | 44.4% (72t)      | -0.28%      | -5.78%       | FAIL: Unprofitable         |
| 23 | Opening Range Trend Runner          | 76.2% (21t)      | 58.3% (72t)      | +5.04%      | +7.35%       | TACTICAL: Run Ungated      |
| 24 | Rolling Z-Score Momentum            | 47.2% (108t)     | 47.2% (108t)     | +2.31%      | +7.43%       | TACTICAL: Run Ungated      |
| 25 | Triple-Timeframe Momentum Grid      | 50.0% (38t)      | 46.5% (71t)      | +3.17%      | +2.68%       | TREND: Gatekeeper Required |
| 26 | The Morning Gap Reversal            | 0.0% (0t)        | 0.0% (0t)        | +0.00%      | +0.00%       | FAIL: Unprofitable         |
| 27 | Consecutive Conviction Acceleration | 42.6% (108t)     | 46.3% (108t)     | -11.81%     | -3.06%       | FAIL: Unprofitable         |
| 28 | Midday Volatility Squeeze           | 0.0% (1t)        | 0.0% (1t)        | -0.41%      | -0.41%       | FAIL: Unprofitable         |
| 29 | Contrarian High-Vol Fade            | 40.0% (5t)       | 26.7% (15t)      | -1.42%      | -5.47%       | FAIL: Unprofitable         |
| 30 | Macro Alignment Scalper             | 0.0% (0t)        | 0.0% (2t)        | +0.00%      | -1.45%       | FAIL: Unprofitable         |
| 31 | Extreme Z-Score Reversion           | 31.2% (48t)      | 36.4% (99t)      | -8.61%      | -10.99%      | FAIL: Unprofitable         |
| 32 | The Persistent Anchor               | 0.0% (0t)        | 0.0% (0t)        | +0.00%      | +0.00%       | FAIL: Unprofitable         |
| 33 | EOD Retail Liquidity Trap           | 57.5% (73t)      | 57.4% (108t)     | +8.05%      | +4.92%       | TREND: Gatekeeper Required |
| 34 | Triple-Timeframe Laggard            | 33.3% (27t)      | 40.3% (62t)      | -3.75%      | -3.95%       | FAIL: Unprofitable         |
| 35 | Volatility Contraction Breakout     | 100.0% (2t)      | 75.0% (4t)       | +0.91%      | +1.69%       | TACTICAL: Run Ungated      |
| 36 | The Opening Drive                   | 66.7% (6t)       | 66.7% (6t)       | -0.19%      | -0.19%       | FAIL: Unprofitable         |
| 37 | Mid-Morning Reversal                | 12.5% (8t)       | 21.4% (14t)      | -6.85%      | -13.07%      | FAIL: Unprofitable         |
| 38 | Dead-Cat Bounce                     | 0.0% (0t)        | 0.0% (0t)        | +0.00%      | +0.00%       | FAIL: Unprofitable         |
| 39 | The VWAP Pinch                      | 100.0% (2t)      | 100.0% (3t)      | +1.22%      | +2.06%       | TACTICAL: Run Ungated      |
| 40 | Multi-Timeframe Alignment           | 45.0% (20t)      | 52.5% (40t)      | -4.38%      | -4.06%       | FAIL: Unprofitable         |
| 41 | Late-Day Liquidation                | 0.0% (0t)        | 0.0% (0t)        | +0.00%      | +0.00%       | FAIL: Unprofitable         |
| 42 | Trend Exhaustion Trap               | 0.0% (0t)        | 100.0% (1t)      | +0.00%      | +0.46%       | TACTICAL: Run Ungated      |
| 43 | Relative Strength Breakout          | 0.0% (1t)        | 0.0% (4t)        | -0.03%      | -1.53%       | FAIL: Unprofitable         |
| 44 | Mean Reversion Grind                | 0.0% (0t)        | 0.0% (0t)        | +0.00%      | +0.00%       | FAIL: Unprofitable         |
| 45 | Volatility Contraction Short        | 33.3% (3t)       | 25.0% (4t)       | -0.71%      | -0.96%       | FAIL: Unprofitable         |
| 46 | 1H Pullback Buy                     | 45.2% (31t)      | 44.4% (72t)      | +3.90%      | +4.68%       | TACTICAL: Run Ungated      |
| 47 | EOD Squeeze Long                    | 0.0% (0t)        | 0.0% (0t)        | +0.00%      | +0.00%       | FAIL: Unprofitable         |
| 48 | The Gap and Crap                    | 33.3% (6t)       | 40.0% (10t)      | -3.04%      | -0.18%       | FAIL: Unprofitable         |
| 49 | The Persistent Dip                  | 75.0% (4t)       | 55.6% (9t)       | +0.94%      | -1.21%       | TREND: Gatekeeper Required |
| 50 | The Perfect Storm                   | 0.0% (0t)        | 0.0% (0t)        | +0.00%      | +0.00%       | FAIL: Unprofitable         |

## Analysis

By analyzing the dual outputs, we can deduce which strategies organically manage risk intraday, and which implicitly depend on daily macro factors to align momentum. Strategies labeled as **"TREND: Gatekeeper Required"** suffer significantly when executing against the broader 24-hour macro tide, whereas **"TACTICAL: Run Ungated"** strategies rely predominantly on very short-term structural anomalies that require zero macro input.
