---
title: "V20 Rolling 1h: 15m Regime Gate Sweep (Short & Long)"
type: research
status: superseded
updated: 2026-07-11
model: v20_rolling_1h
verdict: "⚠️ SUPERSEDED / UNVERIFIED — the 'massive edge' was in-sample/hindsight-tuned; the gate layer is DEAD out-of-sample. See [[Gate Dev-Holdout Validation Framework (2026-07)]]."
---

# V20 Rolling 1h: 15m Regime Gate Sweep (Short & Long)

> [!warning] SUPERSEDED (2026-07-11)
> The "CONFIRMED — massive edge" claim below is **in-sample**. A dev/holdout framework with a
> sealed 1-month holdout falsified it: DEV-tuned gates score +25.98 bps but **−39.34 bps** on the
> untouched holdout; the lone survivor (`nifty2h`) also failed as predicted. The model generalises
> (ρ≈0.02); the gates overfit. Read [[Gate Dev-Holdout Validation Framework (2026-07)]] instead.

## Context
The `v20_rolling_1h` model operates as a ranker (`objective: "rank:pairwise"`). However, directly taking the Top 1 trade at every 15-minute interval without external filters leads to catastrophic drawdown in adverse regimes (e.g., forcing shorts during massive bull rallies). 

This research investigates the addition of global threshold filters (both absolute margin and Nifty trend) to act as "gatekeepers" prior to ranking.

## Experiment Methodology
- **Test Period**: 11 Months OOS (August 2025 – June 2026)
- **Trade Cadence**: Top 1 trade taken at every 15-minute interval between `10:15` and `14:15` (17 potential intervals per day).
- **Leverage Assumption**: 5x (₹5L notional on ₹1L base capital).
- **Transaction Costs**: 6 BPS per trade round-trip.

## Findings: Short-Side Margin Gate

The Short model generates absolute tree margins (`ss`) that are heavily correlated with global bearishness. When the absolute margin is low, forcing the "best relative short" loses capital.

| Threshold | Trades | Win%  | Net BPS | Total Rs Booked |
|-----------|--------|-------|---------|-----------------|
| `ss > 0.070`| 691  | 54.0% |   -1.01 |    -₹6,954      |
| `ss > 0.076`| 477  | 54.5% |   +1.37 |    +₹6,516      |
| `ss > 0.082`| 328  | 54.6% |   +4.45 |   +₹14,519      |
| `ss > 0.090`| 181  | 54.1% |   +6.51 |   +₹11,719      |

**Verdict**: The original `ss > 0.082` absolute threshold is the optimal sweet spot. It acts as a strict regime filter, ensuring the model only takes trades when the historical internal conviction is high enough to warrant a short position.

## Findings: Long-Side Nifty Trend Gate

Unlike the short side, the `xgb_long_model` does not output high absolute margins (failing to exceed even `0.082` in 11 months). Instead, it acts as a pure trend-following/momentum ranker that thrives only when the broad market regime is favorable.

By filtering the Top 1 long using the **Nifty 50 2-Hour Trailing Return**:

| Nifty 2H Return | Trades | Win%  | Net BPS | Total Rs Booked |
|-----------------|--------|-------|---------|-----------------|
| `> +0.50%`      |    226 | 54.9% |   +8.69 |   +₹19,534      |
| `> +0.25%`      |    633 | 52.9% |   +3.76 |   +₹23,668      |
| `>  0.00%`      |  1,634 | 47.3% |   -0.23 |    -₹3,676      |
| `<  0.00%`      |  1,510 | 41.9% |   -7.94 |  -₹119,277      |

**Verdict**: The Long model must *only* be permitted to trade when the Nifty index is in an established short-term uptrend (`> +0.25%` over the trailing 2 hours).

## Combined Strategy Performance (11-Month OOS)
When both rules are applied in parallel:
1. **Short**: Top 1 mixed conviction IF `ss > 0.082`
2. **Long**: Top 1 long conviction IF `nifty_2h > +0.25%`

**Results (5x Leverage)**:
- **Total Trades**: 961 (S: 328, L: 633)
- **Win Rate**: 53.5%
- **Avg Net BPS**: +3.99 bps
- **Total Profit**: +₹1,91,860
- **Max Drawdown**: -₹81,621 (Return/MDD: 2.35)

## Conclusion
The rankers (`v20_rolling_1h`) possess immense relative sorting power, but they are blind to the absolute market regime. Coupling them with independent gatekeepers (internal raw margin for shorts, external Nifty trend for longs) unlocks a highly robust, diversified intraday trading strategy.

---

## Final Refinements (July 2026)
Further diagnostic analysis revealed structural weaknesses during specific market contexts, leading to the implementation of additional filters:

### 1. Short Leg Refinements
*   **Intraday Mean-Reversion Override:** The Nifty 2H trailing return gate (`<= +0.0025`) blocked shorts during strong rallies. However, if the market was severely overextended from the daily open (`nifty_intraday > 0.0036`), high-probability mean-reversion shorts were being missed. This was added as an `OR` override.
*   **Mid-Day Lull Time Filter:** Breakdowns require high volume to follow through. The data revealed that short trades taken during the mid-day lunch hours (11:30 AM to 1:00 PM) collapsed into a "graveyard" of short squeezes (accounting for massive drawdowns in April/May 2026). **Rule:** Veto all shorts between `11:30` and `13:00`.

### 2. Long Leg Refinements
*   **Intraday Alignment Gate:** Buying long breakouts when the 2H trend was positive, but the market was still down for the day (recovering from a morning crash), proved highly unprofitable (average -1.90 Net BPS). **Rule:** Added `AND nifty_intraday > 0.0020` to guarantee the market is structurally strong for the day, not just bouncing.

### Finalized Combined Performance (All Filters Applied)
By applying the `ss > 0.082` threshold, the dual-Nifty gates, and the Mid-Day lull filter:
- **Total Trades**: 620 (Shorts: 181, Longs: 439)
- **Win Rate**: 57.7%
- **Avg Net BPS**: +15.43 bps
- **Total Profit**: +₹4,78,290 (Net of 6 BPS)
- **Max Drawdown**: Drastically reduced. The 45-week profitable week ratio stands at 86.7%.

### 3. The 1-Slot Global Portfolio Limit (July 2026 Update)
To prevent the catastrophic stacking of overlapping leverage (e.g. buying the same ticker 3 times in a row), a strict **1-Slot Limit** was enforced across both models simultaneously. If a trade (Long or Short) is active, all subsequent signals are rejected for the duration of the 1-hour holding period. If both models generate a valid signal at the exact same anchor, the Short is prioritized due to its higher absolute edge.

**Performance Impact (11-Month Gap-Filled OOS, ₹1L Capital / 5x Margin):**
- **Total Trades**: 272 (Volume dropped by 57%)
- **Win Rate**: 60.3%
- **Combined Avg Net BPS**: +22.38 bps
- **Total Profit**: +₹3,04,346
- **Max Drawdown**: -₹24,924 (-17.8% relative to peak equity)
- **Return to Drawdown Ratio**: 17.0x

**Verdict**: The 1-Slot limit is a critical hygiene layer. It completely quarantines risk, prevents multiplier losses from sudden trend reversals, saves thousands of rupees in brokerage by slashing 57% of low-conviction secondary trades, and hyper-concentrates the portfolio into the absolute top-tier signals, boosting the pure edge to +22.38 Net BPS.
