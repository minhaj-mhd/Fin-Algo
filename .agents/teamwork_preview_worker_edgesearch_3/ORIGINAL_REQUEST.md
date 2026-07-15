## 2026-07-15T08:20:35Z

### Objective
Create a self-contained, reproducible python script `research/edge_search/reproducible_edge_report.py` that loads finalgo system data, evaluates the "Open GAP-FADE" strategy, splits the data into Development and Holdout sets, and prints the Expected Value (EV) of the edge on the Holdout set.

### Strategy Rules (Open GAP-FADE)
1. Universe: The 110-name liquidity universe in `data/research/v21_rolling_1h/universe.json`.
2. Signals: At the 09:15 open of day T: `gap_i = open_i / prev_day_close_i - 1.0`.
3. Filter: `|gap| <= 3%` (circuit-breaker guard). Skip days with fewer than 60 valid tickers.
4. Book construction: SHORT top-5 largest gap-ups (largest positive gaps), LONG bottom-5 largest gap-downs (most negative gaps). Equal-weighted capital (50% long book, 50% short book).
5. Entry price: Open price (`open0915`).
6. Exit price: Price at 09:30 (`x_0930` which corresponds to the close of the 09:25 bar in left-labeled 15m/5m data, or the open of the 09:30 bar).
7. Transaction cost: Subtract 6.0 bps flat round-trip per trade.

### Dataset Split
- Development Set: 2023-01-01 to 2025-12-31
- Holdout Set: 2026-01-01 to 2026-06-30

### Script Requirements
- The script must be written to: `c:\Users\loq\Desktop\Trading\finalgo\research\edge_search\reproducible_edge_report.py` (ensure parent directories are created).
- The script must be completely self-contained and run end-to-end without errors in the existing environment.
- It must load the data from `data/raw_upstox_cache_5min_v3` (or other appropriate cache if necessary, using similar logic to `scripts/research/gap_fade_strategy_backtest.py`).
- It must calculate and output to stdout:
  1. For the Development Set (2023-01-01 to 2025-12-31):
     - Total Trades (across all days)
     - Win Rate (percentage of trades with PnL > 0)
     - Gross Return per trade (mean bps)
     - Net Return per trade (mean bps after 6bps cost) - this is the Expected Value (EV) per trade
     - t-statistic
  2. For the Holdout Set (2026-01-01 to 2026-06-30):
     - Total Trades (across all days)
     - Win Rate
     - Gross Return per trade (mean bps)
     - Net Return per trade (mean bps after 6bps cost) - this is the Expected Value (EV) per trade
     - t-statistic
- The script must verify that the calculated Net Return (EV) on the Holdout set is strictly positive (> 0).

### Execution and Verification
- Run the python script end-to-end to verify it runs without errors.
- Capture the stdout of the script run and document it in your handoff report `handoff.md` in your working directory.
- Verify that the resulting EV on the Holdout set is positive.

### Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
