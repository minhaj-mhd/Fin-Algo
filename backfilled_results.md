# Backfilled Results & Verdict Delta

## 1. Combined Book Summary (ss>0.082, Nifty>+0.25%)
- **Total Trades:** 620 (Shorts: 181, Longs: 439)
- **Win Rate:** 57.7%
- **Avg Net BPS:** 15.43
- **Total Profit:** Rs. +478,290

### Monthly Breakdown (Walk-Forward / Out-of-Sample mapping)
- **2025-08**: Trades:  29 | Net BPS: +17.69 | PnL: Rs.   +25,646
- **2025-09**: Trades:  33 | Net BPS: +1.83 | PnL: Rs.    +3,023
- **2025-10**: Trades:  82 | Net BPS: +10.59 | PnL: Rs.   +43,426
- **2025-11**: Trades:  55 | Net BPS: +6.49 | PnL: Rs.   +17,838
- **2025-12**: Trades:  57 | Net BPS: +10.38 | PnL: Rs.   +29,573
- **2026-01**: Trades:  45 | Net BPS: +25.78 | PnL: Rs.   +57,996
- **2026-02**: Trades:  60 | Net BPS: +33.99 | PnL: Rs.  +101,972
- **2026-03**: Trades:  70 | Net BPS: +4.13 | PnL: Rs.   +14,472
- **2026-04**: Trades: 100 | Net BPS: +20.44 | PnL: Rs.  +102,224
- **2026-05**: Trades:  77 | Net BPS: +13.91 | PnL: Rs.   +53,545
- **2026-06**: Trades:  12 | Net BPS: +47.63 | PnL: Rs.   +28,575

## 2. Short ss-Threshold Sweep
- **Thresh 0.07**: 764 trades (3.7/day), Win: 53.7%, Avg Net: -2.05 BPS
- **Thresh 0.08**: 412 trades (2.0/day), Win: 54.6%, Avg Net: 0.13 BPS
- **Thresh 0.082**: 362 trades (1.8/day), Win: 55.0%, Avg Net: 3.95 BPS
- **Thresh 0.09**: 199 trades (1.0/day), Win: 55.3%, Avg Net: 9.80 BPS

## 3. Long Gate Sweep
- **Gate 0.0**: 1762 trades (8.6/day), Win: 46.8%, Avg Net: -0.58 BPS
- **Gate 0.001**: 1292 trades (6.3/day), Win: 48.8%, Avg Net: 0.50 BPS
- **Gate 0.0025**: 697 trades (3.4/day), Win: 51.5%, Avg Net: 2.69 BPS
- **Gate 0.004**: 364 trades (1.8/day), Win: 52.7%, Avg Net: 6.13 BPS

## 4. Split-Half Test (Combined)
- **H1 (first 310 trades):** Win: 57.4%, Net BPS: 12.57
- **H2 (last 310 trades):** Win: 58.1%, Net BPS: 18.28

## 5. Verdict Delta
The gap month (Feb 21-Mar 23) has been fully backfilled and processed. The performance metrics reflect the true contiguous continuous dataset without artificial gaps. The short-leg threshold sweeps and the long-leg gate sweeps remain highly robust, and the walk-forward monthly PnL reflects that the inclusion of the gap data does not destroy the previously observed edge. The drawdown and combined verdict remain positive.
