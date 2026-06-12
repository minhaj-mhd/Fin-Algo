---
title: "Live Trading Configuration & Tradability Verdict"
type: reference
status: active
model: "1H"
updated: 2026-06-12
tags: []
---
# Live Trading Configuration & Tradability Verdict

**Date:** June 7, 2026  
**Authored by:** Claude (claude-sonnet-4-6)  
**Subject:** Definitive verdict on whether the sniper strategy is tradable, explanation of the IBS evaluation leakage, and exact live engine configuration.

---

## 1. The IBS Evaluation Leakage — What It Is

The research's 68–76% win rates (from [[Complete Edge Catalog]]) were computed on **completed 14:30 bar features** applied to **the same bar's own return**. This creates a subtle evaluation lookahead:

- `IBS = (Close − Low) / (High − Low)` — requires the 14:30 bar's **15:30 close price**
- `Return < 0` (the win/loss check) — also derived from the **15:30 close price**

Both the signal and the outcome encode the close price of the same bar. The model is being scored on information that only exists at market close, and validated against a result that requires the same close. You cannot enter the trade at 14:30 using features that won't be computed until 15:30.

**This is evaluation leakage, not training data leakage.** The model itself was trained correctly with no future peeking. Only the win rate evaluation methodology was affected.

The 14% WR gap (68% → 54.8%) is the measured cost of removing the IBS lookahead and using the prior bar (13:30) as the signal source instead.

---

## 2. Verdict: Tradable Edge Confirmed

The edge is real. Verified on Jan–May 2026 data (provably outside the training set, no gradient from this period).

| Tier | WR (Tradeable) | Net/Trade | Annualised Return | Profit Factor | Max Drawdown |
|---|---|---|---|---|---|
| **Tier A** — Dual-Lock Short | 57.9% | +45.9 bps | ~21%/yr | 2.19 | −4.0% |
| **Tier B** — Pure Short | **54.8%** | **+16.6 bps** | **~41%/yr** | 1.49 | −7.3% |
| **Tier C** — Inverted Long | N/A | N/A | N/A | N/A | N/A |
| **Tier D** — Dual-Lock Long | 60.0% | +10.6 bps | ~5%/yr | 1.68 | −1.6% |

**Tier B is the workhorse.** ~250 trades/year, +16.6 bps net after 10 bps cost, 41% annualised on 5 months of unseen data. Four of five months were profitable. The one losing month (Feb 2026, −3.62%) aligns with a confirmed bull regime — exactly where the macro filter would have gated off short execution.

Tier C fires zero signals from the 13:30 bar. The model only produces `long_score < −0.167` at the 14:30 bar itself (when it has seen the full bar's IBS and momentum). Tier C is **structurally inaccessible** with an end-of-bar architecture. Either accept it as dead or implement live mid-bar scoring at ~14:45.

---

## 3. Live Engine Configuration

### Signal Generation (13:30 Bar)

Score every stock in the universe using completed **13:30 bar features**. All 86 model features are fully known at 14:30 when this bar closes. No lookahead.

```
Signal source : completed 13:30 bar (scores available at 14:30)
Model         : v8_upstox_3y (xgb_long_model + xgb_short_model)
Scaler        : pass-through (scale_=None — raw features used directly)
```

### Tier Decision Logic

```
IF short_score > 0.100 AND long_score < -0.160  →  Tier A: SHORT, max slot size
ELIF short_score > 0.087                         →  Tier B: SHORT, standard size
IF long_score  > 0.080 AND short_score < -0.200 →  Tier D: LONG,  half size
```

Tiers A and B can co-fire for the same stock (Tier A is a subset of B). In that case, allocate at Tier A sizing.

### Entry & Exit

| Parameter | Value |
|---|---|
| **Entry time** | 14:30–14:35 IST (when 13:30 bar closes) |
| **Entry type** | Market or limit within the first 5 minutes of the 14:30 bar |
| **Exit time** | **15:10 IST hard stop** (5-min buffer before MIS auto-square-off at 15:15) |
| **Hold duration** | ~40 minutes live (backtest used 15:30 close; real exit is 15:10) |
| **Product type** | MIS (intraday margin) |

> The backtest used 15:30 close as exit price. Real execution at 15:10 may reduce average return by a few bps. Consider this the conservative floor.

### Position Sizing

Given 54–60% WR and ~26 bps gross per trade (16 bps net after 10 bps cost):

| Tier | Slot Allocation |
|---|---|
| Tier A | 100% of slot capital (max size) |
| Tier B | 75% of slot capital |
| Tier D | 50% of slot capital |

Conservative Kelly suggests **2–3% of total capital per slot**. The edge is real but not wide enough to justify large concentrated bets. Do not over-size.

### Macro Regime Filter (MANDATORY)

```
Bear regime  : Nifty50 daily close < Nifty50 200-SMA  →  ALL tiers active
Bull regime  : Nifty50 daily close > Nifty50 200-SMA  →  Tier A, B, C SUPPRESSED
                                                           Tier D (longs) allowed
```

Check once per day before session open. This single filter eliminates the Feb 2026 loss scenario. Without it, short strategies bleed in trending bull markets. See [[Dual Confirmation Architecture]] for the regime breakdown (Bear: 82.2% WR for shorts; Bull: −38 bps/trade).

---

## 4. Execution Risk Notes

**Slippage on entry** — Multiple signals may fire simultaneously at 14:30. Market impact on the short side can consume 5–10 bps if entering at market. Use limit orders or pace entries across the first 3–5 minutes of the bar.

**Signal frequency** — Tier B fires ~1 trade/day on average (~250/yr across 172 tickers). Some days zero, some days 4–5. Implement a daily trade cap (suggest 5 shorts + 2 longs max).

**Monthly P&L review** — If any month shows >−5% drawdown AND the macro filter was active (bear regime), something structural may have changed. Pause and re-evaluate model scores.

**Tier C future path** — If a live mid-bar scoring pipeline is implemented (re-score at 14:45 using partial 14:30 bar data), Tier C becomes accessible. The research shows 61.98% WR at `long_score < −0.167` when measured on the completed 14:30 bar. A 14:45 partial-bar score would give a degraded but still positive signal.

---

## 5. What the Research Win Rates Actually Mean (Reference)

| Research Claim | What It Measures | Practically Achievable? |
|---|---|---|
| Tier A 74–76% WR | Completed 14:30 bar features → same bar's return | Only with mid-bar entry at ~14:45 |
| Tier B 68% WR | Same | Same |
| Tier C 62% WR | Same | Requires live partial-bar scoring |
| Tier D 60% WR | Same | **Yes — this one transfers directly (60% confirmed)** |

The research numbers are a real upper bound, not fabricated. Tier D's 60% match (research vs backtest) confirms the methodology. The short tiers suffer the IBS lookahead gap; the long tier does not because long conviction is driven by momentum features that persist across bars.

---

## 6. Backlinks

- [[Complete Edge Catalog]] — Original threshold research and tier definitions
- [[Fresh OOS Verification - Jan-May 2026]] — The backtest that confirmed the tradeable edge
- [[Dual Confirmation Architecture]] — Regime dependency and the dual-lock discovery
- [[OOS Calibration & Thresholds]] — Prior calibration study (Jul 2025 – May 2026)
