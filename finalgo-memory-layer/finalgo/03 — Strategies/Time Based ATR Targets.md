# Time-Based ATR Targeting (1-Hour Hold Constraints)

**Concept:** Dynamic Stop Loss (SL) and Take Profit (TP) must be scaled to realistically match the holding period of the trade using the square root of time rule.

## The Mathematical Constraint
When holding a trade for 1 hour, that equates to **four 15-minute bars**. 
According to the random walk theory of volatility, the expected maximum price move over time $t$ scales roughly by $\sqrt{t}$. 
For 4 bars:
$\sqrt{4} = 2$
Therefore, the expected maximum price move within a 1-hour window is approximately **`2.0 x (15-min ATR)`**.

## The Flaw in Previous Logic
Previously, Vanguard was computing:
- **SL:** `1.5 x ATR`
- **TP:** `3.0 x ATR`

By setting the TP to `3.0x ATR`, the strategy was targeting an extreme 1.5 standard deviation tail-event that required the price to move in a straight line for the full hour. Since the normal expected chop is around `2.0x ATR`, the price would naturally oscillate and trigger the tight `1.5x ATR` Stop Loss long before reaching the impossible `3.0x ATR` target.

## Current Corrected Scaling (Vanguard Orchestrator)
To align the Risk/Reward ratio with the realistic 1-hour expected move, the targets have been adjusted to stay within the `2.0x` probability cone:

```python
# Realistic levels for a 4-bar (1-hour) hold
sl_pct = max(0.25, min(1.20, atr_pct * 1.0))
tp_pct = max(0.50, min(2.00, atr_pct * 1.8))
```

- **Take Profit (`1.8x ATR`):** Sits just below the `2.0x` maximum expected move limit, making it highly achievable.
- **Stop Loss (`1.0x ATR`):** Tightened proportionately to maintain a favorable **1 : 1.8** Risk/Reward ratio.
- **Caps:** Maximum TP is strictly capped at `2.00%` to prevent chasing mathematically improbable intraday tail events.
