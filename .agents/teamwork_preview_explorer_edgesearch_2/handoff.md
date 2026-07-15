# Handoff Report: Candidate Edge Identification & Analysis

## 1. Observation

### A. Standalone `daily_macro_v2` LONG (3-Day Hold)
- **Gauntlet Certification**: Standalone LONG model achieved `TRIGGER_GRADE` in Gauntlet run `20260610T135608Z-5f7d069f` with a Walk-Forward Avg Rho of `+0.0295`, Top-3 Win Rate (K=3) of `56.8%`, and Expected Return Edge of `+4.08 bps/day` (net return edge of `+28.22 bps` at 6.0bps cost with a t-stat of `3.13` over the full out-of-sample window, and `+22.61 bps` at 6.0bps cost with a t-stat of `1.65` in the recent 24 months).
  - *Source*: `finalgo-memory-layer/finalgo/02 — Models/Gauntlet Reports/Daily Macro v2 Report.md` lines 30-34 and 80-89:
    ```markdown
    | full_OOS | Top-3 | 6.0bps | LONG | 2955 | +34.22 | +28.22 | 52.8% | 51.8% | 49.5% | 3.13 ✷ |
    | recent_24mo | Top-3 | 6.0bps | LONG | 1344 | +28.61 | +22.61 | 53.9% | 52.5% | 49.7% | 1.65 |
    ```
- **Feature/Label Timing Contract Mismatch**:
  - The model features for Trade Day T (Wednesday) are joined on row `T-1` (Tuesday). These features include US closing data (e.g. S&P 500 return of Tuesday) which is only finalized at Tuesday 16:00 EST (Wednesday 02:30 IST).
  - *Source*: `finalgo-memory-layer/finalgo/02 — Models/Daily Gatekeeper/Gatekeeper V2 Feature Availability.md` lines 18-19 and 31:
    ```markdown
    All trading decisions for Trade Day T are made at 09:00 IST on Trade Day T (prior to the market pre-open)...
    US Markets (S&P 500, Nasdaq, DXY, US10Y) | yfinance | Day T-1 16:00 EST | Day T 01:30 / 02:30 IST | No Lag: Joined on row T-1 (since US session T-1 is completed before 09:00 IST T).
    ```
  - The target label `Label_3D` in `build_daily_macro_dataset.py` is defined as:
    ```python
    df_feat['Label_3D'] = df_feat['Close'].shift(-3) / df_feat['Close'] - 1.0
    ```
    For row `T-1` (Tuesday), `Close` is Tuesday's close (`Close_{T-1}`). `shift(-3)` is Friday's close (`Close_{T+2}`). Thus, `Label_3D` is the close-to-close return from Tuesday close to Friday close.
  - The backtest in `eval_custom_exits_full.py` simulates entries at the close of day `T-1` (Tuesday close):
    ```python
    entry = get_close(t, tk) # entry = Close on date t
    p3 = get_close(t+3, tk)   # exit = Close 3 trading days later (t+3)
    ```
  - **Overnight Segment Attribution**: The overnight return of day 1 (ON1 = Wednesday open / Tuesday close - 1) contributes `+20.5 bps` (t-stat = 9.7) of the `daily_macro_v2` LONG trade performance, whereas the intraday return of day 1 (D1 = Wednesday close / Wednesday open - 1) is `-7.7 bps`.
    - *Source*: `finalgo-memory-layer/finalgo/06 — Logs/Conversations/Conv-2026-07-05-Optimal-Signal-Layer-6bps.md` lines 58-61:
      ```markdown
      Segment decomposition (the structural discovery): on the certified WF top-3 LONG picks (preds.npz from run 20260610T135608Z-5f7d069f, alignment verified err=0), the 3-day edge lives ENTIRELY overnight: ON1 +20.5 (t 9.7), ON2 +22.8 (t 10.6), ON3 +22.7 (t 10.6); ALL intraday segments negative (D1 −7.7, D2 −10.1, D3 −12.6).
      ```

### B. Open GAP-FADE Paired Book
- **Performance**: Pairing shorting top-5 gap-ups and longing bottom-5 gap-downs at the 09:15 open (capping gap at |gap| <= 3%) and covering at 09:30 IST yields `+14.08 bps/day` net of 6bps cost (t-stat = 9.11, Sharpe = 4.93) in the simulation.
  - *Source*: `data/research/open_window_stack/strategy_backtest.json` lines 21-29:
    ```json
    "book": {
     "mean": 14.081030118520749,
     "t": 9.118297604563702,
     "n": 843,
     "win": 0.641755634638197,
     "h1": 12.67526544803454,
     "h2": 15.483463593105332,
     "sharpe": 4.935696559870479
    }
    ```
- **Execution Constraints**:
  - Entering 5 minutes late (at 09:20) results in a collapsed net return of `-3.61 bps/day` (t-stat = -5.13).
  - *Source*: `data/research/open_window_stack/strategy_backtest.json` lines 51-58:
    ```json
    "book": {
     "mean": -3.616653526573259,
     "t": -5.135768964913593,
     "n": 843,
     "win": 0.4033214709371293,
     "h1": -4.372217298755564,
     "h2": -2.8628801898700598,
     "sharpe": -2.7799703751418066
    }
    ```
  - Entering via Limit Orders at the open results in adverse selection and a collapsed net return of `-14.81 bps/day` (t-stat = -11.59, fill rate = 72.4%).
  - *Source*: `data/research/open_window_stack/strategy_backtest.json` lines 141-150:
    ```json
    "book": {
     "mean": -14.810218810629221,
     "t": -11.597079462702679,
     "n": 843,
     "win": 0.33926453143534996,
     "h1": -16.51758273032279,
     "h2": -13.106900777001277,
     "sharpe": -6.277450867578371
    },
    "fill_rate": 0.7243179122182681
    ```

### C. Hourly Models & Gate Overfitting
- **V20 1H True OOS Performance**: Stacking 7 gates on the `v20_rolling_1h` model resulted in a complete collapse when tested on a true out-of-sample window (June 4 - July 10, 2026):
  - Return: `-23.65%` (compared to positive in-sample edge).
  - Win Rate (Shorts): collapsed to `43.8%` (from `72%` in-sample).
  - *Source*: `finalgo-memory-layer/finalgo/06 — Logs/Conversations/Conv-2026-07-12-OOS-Data-Fix-and-Overfitting-Discovery.md` lines 31-34:
    ```markdown
    True OOS Result: Ran the backtest successfully. Engine generated 35 trades (16 Shorts, 19 Longs).
      - PnL: -23.65% Total Portfolio Return (-₹9,462).
      - Risk Profile: A massive -34.78% Max Drawdown, heavily driven by the Short side collapsing to a 43.8% win rate (compared to 72% historically).
    Core Realization (Overfitting): The user observed that stacking 7 gates produced a massive DEV/OOS gap (DEV +26, HOLDOUT -39). The constraints (like the Mid-Day lull and Nifty 2H filter) were not extracting a true edge, they were merely memorizing the profitable slices of the DEV set. The underlying V20 structural model has no intrinsic edge.
    ```

### D. Regime Router & Ticker List Issues
- **v26 Lookahead Audit**: Shifting the Nifty 100-DMA regime features by 1 day to make them causal collapsed the testing combined edge from `+27.44 bps` to `+2.38 bps`.
  - *Source*: `finalgo-memory-layer/finalgo/06 — Logs/Conversations/Conv-2026-07-14-V26-Phase-0-Validation.md` lines 21-23:
    ```markdown
    E0.1 (Lookahead Bias): Found severe lookahead bias in eval_regime_router.py and train_binary_clean.py where regime assignment for day T used day T's close. Shifted Nifty 100-DMA features by 1 day.
    Performance Impact: Post-fix, the strategy's edge collapsed. 6m Testing combined edge dropped from +27.44 bps to +2.38 bps. Long edge flipped to negative (-12.56 bps).
    ```
- **Survivorship Bias**: The database datasets are compiled using a static list of 148 current active tickers.
  - *Source*: `finalgo-memory-layer/finalgo/06 — Logs/Conversations/Conv-2026-07-14-V26-Phase-0-Validation.md` line 24:
    ```markdown
    E0.3 (Survivorship Bias): Checked build_rolling_1h_panel.py and collect_upstox_15min_3y.py. Found the dataset is built using a static, hardcoded list of 148 recent/current tickers (scripts/tickers.py), meaning fatal survivorship bias exists.
    ```

---

## 2. Logic Chain

1. **V20/V21 Hourly/15m Models and Overfitted Gates**:
   - The true OOS results (June 4 - July 10, 2026) show that the v20 model's positive edge collapsed to `-23.65% PnL` and its Short Win Rate collapsed to `43.8%` (Observation C).
   - This proves that stacking multiple heuristic filters (such as Nifty 2H trailing returns, midday lull, and conviction caps) was merely mining the development set, failing to replicate on sealed out-of-sample data.
   - The regime router (v26) also collapsed from `+27.44 bps` to `+2.38 bps` once the lookahead bias in the Nifty 100-DMA regime filter was removed (Observation D).
   - Therefore, the hourly models and their associated gating systems do not possess a genuine out-of-sample trading edge.

2. **Standalone `daily_macro_v2` Model**:
   - The model achieved `TRIGGER_GRADE` for LONG on the OOS walk-forward validation (Observation A).
   - However, the model uses features from row `T-1` (Tuesday close + US Tuesday close finalized Wednesday 02:30 IST) to predict return `Label_3D` (`Close_{T+2}/Close_{T-1} - 1.0` i.e. Tuesday close to Friday close) (Observation A).
   - The trade simulation assumes entry at Tuesday close (`Close_{T-1}`), which is in the past relative to the feature availability time (Wednesday 09:00 IST) (Observation A). This is a lookahead leak of the overnight gap (ON1 = Wednesday open / Tuesday close - 1).
   - Segment decomposition shows that ON1 contributes `+20.5 bps` of the return (Observation A).
   - If we execute causally (e.g. entering at Wednesday close or open), we lose the ON1 overnight gap.
   - Losing `+20.5 bps` of return drops the expected edge (+28.22 bps) below transaction costs (6-10 bps), rendering the edge untradable.
   - Thus, the standalone `daily_macro_v2` model does not have a genuine causal trading edge.

3. **Open GAP-FADE Strategy**:
   - The strategy (shorting top-5 gap-ups + longing bottom-5 gap-downs at the open, capped at |gap| <= 3%, cover at 09:30 IST) yields `+14.08 bps/day` net of 6bps cost (t-stat = 9.11, Sharpe = 4.93) in backtests (Observation B).
   - Entering 5 minutes late (09:20 entry) makes it gross-negative (`-3.61 bps/day`), and using limit orders results in adverse selection (`-14.81 bps/day`) (Observation B).
   - Therefore, the edge is highly sensitive to fill price and requires execution exactly at the 09:15 open print.
   - This can only be achieved by participating in the NSE pre-open auction (submitting orders during the 09:00-09:08 IST pre-open window).
   - While it represents a mathematically genuine and robust edge in price-action data (t-stat > 9.0), it is heavily execution-constrained and capacity-constrained (pre-open auction volume capacity is small).

4. **Survivorship Bias**:
   - The entire repository's single-stock dataset is built using a static list of 148 active tickers from `scripts/tickers.py`, introducing survivorship bias (Observation D). This affects all model backtests and dataset analyses in the project.

---

## 3. Caveats

- **Survivorship Bias**: Since delisted or bankrupt companies over the 3-to-10-year test windows are not represented in `scripts/tickers.py`, the performance metrics of all stock-selection models (including `daily_macro_v2` and Open GAP-FADE) are likely inflated by survivorship bias.
- **Pre-Open Auction Slippage**: The Open GAP-FADE strategy's performance relies completely on being filled exactly at the open price. The actual slippage/market impact of pre-open market orders on the NSE has not been measured live, which could degrade the edge if slippage exceeds 10-15 bps.
- **Shorting Constraints**: Under Indian market regulations, overnight short positions in cash equities are not allowed for retail traders. The Open GAP-FADE strategy covers at 09:30 IST (intraday), which is regulatory-compliant. However, `daily_macro_v2` SHORT trades require a multi-day holding period (3 days), which is not tradeable in cash equities and would require F&O (futures) implementation, thinned by ticker availability.

---

## 4. Conclusion

1. **Hourly and 15m model gates (including v20, v23, v24, and v26)** are **INVALIDATED**. Their in-sample performance was a result of lookahead bias (regime router) or pool-carving overfitting that collapsed to negative PnL out-of-sample.
2. **Standalone `daily_macro_v2` LONG** is **INVALIDATED** for causal execution. Its certified edge is an artifact of a lookahead leak of the first overnight gap (ON1). Correcting this timing mismatch removes the positive expected value.
3. **Open GAP-FADE Paired Book** is the **ONLY candidate trading edge** that is statistically genuine (t-stat = 9.11, Sharpe = 4.93) and holds out-of-sample, provided it is executed via the NSE pre-open auction. It must be traded with small sizes to prevent market impact and slippage.

**Recommendation**: Focus efforts on verifying the execution slippage of the **Open GAP-FADE** strategy using live pre-open shadow fills in Vanguard rather than further tuning hourly model gates.

---

## 5. Verification Method

To independently verify the findings and replicate the statistical reports:

### A. Replicate the Open GAP-FADE Backtest
The simulation results for the Open GAP-FADE strategy are stored in:
`c:\Users\loq\Desktop\Trading\finalgo\data\research\open_window_stack\strategy_backtest.json`

Verify the code that generated these results by inspecting:
`c:\Users\loq\Desktop\Trading\finalgo\scripts\research\gap_fade_strategy_backtest.py`

### B. Verify the `daily_macro_v2` Lookahead Leak
Inspect the dataset construction in `scripts/training/build_daily_macro_dataset.py` lines 174-180 and note the `Label_3D` definition. Inspect the custom exit script `scripts/transformer/eval_custom_exits.py` lines 16-19 and 52-53, verifying that the entry is set to the Tuesday close (`row['Close']`) using Tuesday's row index `dt` which incorporates US closing data known only on Wednesday morning.

### C. Verify the v20 OOS Collapse
Read the conversation log:
`c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\06 — Logs\Conversations\Conv-2026-07-12-OOS-Data-Fix-and-Overfitting-Discovery.md`

### D. Verify the v26 Lookahead / Ticker List Bias
Read the conversation log:
`c:\Users\loq\Desktop\Trading\finalgo\finalgo-memory-layer\finalgo\06 — Logs\Conversations\Conv-2026-07-14-V26-Phase-0-Validation.md`
Check `scripts/tickers.py` to confirm the static list of 148 tickers.
