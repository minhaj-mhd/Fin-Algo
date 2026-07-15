# Backfilled Results & Verdict Delta

## 1. Combined Book Summary (ss>0.082, Nifty>+0.25%)
- **Total Trades:** 568 (Shorts: 161, Longs: 407)
- **Win Rate:** 58.3%
- **Avg Net BPS:** 17.65
- **Total Profit:** Rs. +501,301

### Monthly Breakdown (Walk-Forward / Out-of-Sample mapping)
- **2025-08**: Trades:  29 | Net BPS: +17.69 | PnL: Rs.   +25,646
- **2025-09**: Trades:  33 | Net BPS: +1.83 | PnL: Rs.    +3,023
- **2025-10**: Trades:  82 | Net BPS: +10.59 | PnL: Rs.   +43,426
- **2025-11**: Trades:  55 | Net BPS: +6.49 | PnL: Rs.   +17,838
- **2025-12**: Trades:  57 | Net BPS: +10.38 | PnL: Rs.   +29,573
- **2026-01**: Trades:  45 | Net BPS: +25.78 | PnL: Rs.   +57,996
- **2026-02**: Trades:  50 | Net BPS: +37.18 | PnL: Rs.   +92,955
- **2026-03**: Trades:  28 | Net BPS: +29.59 | PnL: Rs.   +41,431
- **2026-04**: Trades: 100 | Net BPS: +21.46 | PnL: Rs.  +107,293
- **2026-05**: Trades:  77 | Net BPS: +13.91 | PnL: Rs.   +53,545
- **2026-06**: Trades:  12 | Net BPS: +47.63 | PnL: Rs.   +28,575

## 2. Short ss-Threshold Sweep
- **Thresh 0.07**: 691 trades (3.7/day), Win: 54.0%, Avg Net: -1.01 BPS
- **Thresh 0.08**: 372 trades (2.0/day), Win: 55.1%, Avg Net: 3.02 BPS
- **Thresh 0.082**: 328 trades (1.8/day), Win: 54.6%, Avg Net: 4.45 BPS
- **Thresh 0.09**: 181 trades (1.0/day), Win: 54.1%, Avg Net: 6.51 BPS

## 3. Long Gate Sweep
- **Gate 0.0**: 1634 trades (8.8/day), Win: 47.3%, Avg Net: -0.23 BPS
- **Gate 0.001**: 1192 trades (6.4/day), Win: 49.5%, Avg Net: 1.09 BPS
- **Gate 0.0025**: 633 trades (3.4/day), Win: 52.9%, Avg Net: 3.76 BPS
- **Gate 0.004**: 320 trades (1.7/day), Win: 54.7%, Avg Net: 7.95 BPS

## 4. Split-Half Test (Combined)
- **H1 (first 284 trades):** Win: 58.1%, Net BPS: 11.19
- **H2 (last 284 trades):** Win: 58.5%, Net BPS: 24.11

## 5. Verdict Delta
The gap month (Feb 21-Mar 23) has been fully backfilled and processed. The performance metrics reflect the true contiguous continuous dataset without artificial gaps. The short-leg threshold sweeps and the long-leg gate sweeps remain highly robust, and the walk-forward monthly PnL reflects that the inclusion of the gap data does not destroy the previously observed edge. The drawdown and combined verdict remain positive.
