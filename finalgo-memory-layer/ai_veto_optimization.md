# AI Veto Prompts & Optimizations for 30-60 Min Intraday Timeframes

## 1. Optimization Plan

Since the holding period for trades is extremely short (30 mins to 1 hour), the current Layer 2 (S2) Prompt places too much emphasis on medium-term structural news ("last 7 days") and not enough on immediate intraday catalysts or liquidity traps.

### Core Changes Needed
1. **Change the Time Horizon**:
   - Redefine "Bucket A" from "last 7 days" to **"last 24 hours (Overnight/Pre-market/Today)"**. If news is from 3 days ago, the market has already digested it; a 30-min setup is just exploiting technical momentum.
   - Redefine "Bucket B" to focus heavily on **Live Intraday Flow**: "last 1-2 hours (Block deals, management live commentary, sudden volume spikes)".

2. **Rewrite the Veto Decision Matrix for Intraday Reality**:
   - **Old RULE 1 (Hard Veto)** triggered on *any* downgrade.
   - **New RULE 1 (Intraday Catalyst Veto)**: Only veto if there is an *immediate* negative headline today (e.g. "SEBI halts trading", "Block deal selling 5M shares today", "Disastrous earnings released *during market hours*").
   - **New Rule (Liquidity & Spread Risk)**: Ask the LLM to veto if the volume is extremely thin, making a 30-min round trip impossible without slippage.

3. **Tune S1 (Technical Prompt)**:
   - Make it explicit to the LLM that this is a **short-term 30-60 min trade**.
   - Tell it to weigh the **Recent Price Action (last 30 min, 1-min bars)** much heavier than the Daily Trend, because we only need the trade to work for the next 30 minutes.

---

## 2. Current Prompt References

### Layer 1: Technical & Momentum (S1)

```python
prompt_flash = f"""You are a professional intraday technical analyst reviewing a real-time trade signal.
All data below is LIVE from the current 1-hour bar. No external data is available — evaluate ONLY what is provided.

═══ TRADE PROPOSAL ═══════════════════════════════════════════════
TICKER  : {ticker}
SIDE    : {side}
PRICE   : ₹{price}
ML CONVICTION : {conviction:.4f}
  → Universe Rank : #{int(features.get("Long_Rank" if side == "LONG" else "Short_Rank", 999))} of ~172 stocks screened this cycle
  → Min gate      : {self.min_conviction:.2f}
  → Scale context : 0.15=gate | 0.25=moderate | 0.35+=STRONG | 0.50+=very strong
  → IMPORTANT: This signal survived ML pre-filtering and ranks in the TOP candidates. Do NOT label it "low conviction".
{strategy_str}

═══ MOMENTUM & OSCILLATORS ═══════════════════════════════════════
RSI-14 (intraday)  : {_f("RSI_14_Raw", decimals=1)}  {'→ OVERBOUGHT' if float(_f("RSI_14_Raw", "50")) > 70 else ('→ OVERSOLD' if float(_f("RSI_14_Raw", "50")) < 30 else '→ NEUTRAL ZONE')}
RSI-14 (daily)     : {daily_rsi}
Stochastic %K      : {stoch_k}
Bollinger %B       : {percent_b_raw}
1H Return          : {_f("Return_Raw", pct=True, decimals=2)}
Up Streak          : {up_streak} consecutive green bars
Down Streak        : {dn_streak} consecutive red bars
Green Bar Ratio(5) : {green_ratio}
Bar Close Position : {bar_pos}
Accumulation(5)    : {accumulation}

═══ PRICE STRUCTURE & KEY LEVELS ═════════════════════════════════
vs SMA-6  (fast)   : {_vs(sma6)}   [₹{sma6}]
vs SMA-12 (medium) : {_vs(sma12)}  [₹{sma12}]
vs SMA-50 (daily)  : {_vs(sma50)}  [₹{sma50}]
BB Upper           : ₹{bb_upper}  ({round((bb_upper/price-1)*100,2) if bb_upper else 'N/A'}% above)
BB Lower           : ₹{bb_lower}  ({round((1-bb_lower/price)*100,2) if bb_lower else 'N/A'}% below)
Donchian High(20)  : ₹{don_upper}
Donchian Low(20)   : ₹{don_lower}
1-ATR Resistance   : ₹{r1_atr}    ATR = ₹{atr_abs}
1-ATR Support      : ₹{s1_atr}
52-Week High       : ₹{high_52w}  ({_f("dist_52h_actual", pct=True, decimals=1)} from 52W high)
{nearest_wall_label.upper()} : {wall_str}

═══ VOLUME ════════════════════════════════════════════════════════
RVOL (Relative Vol): {rvol}x  {'→ HIGH ACTIVITY' if float(rvol) > 2 else ('→ MODERATE' if float(rvol) > 1 else '→ THIN')}
Dollar Volume      : {dv_str}

═══ MARKET & SECTOR CONTEXT ══════════════════════════════════════
Market Regime      : {regime_str}
Nifty 1H Return    : {nifty_1h}
Nifty 5H Return    : {nifty_5h}
Stock vs Nifty     : {stock_vs_nifty}
Stock vs Sector    : {vs_sector}
Sector Breadth     : {sector_breadth}
VIX Level          : {vix}  (5D MA: {vix_ma}){'  ← EXTREME FEAR' if vix_extreme else ''}

═══ DAILY TREND ALIGNMENT ════════════════════════════════════════
Daily Trend (5D)   : {daily_trend_str}
Daily RSI          : {daily_rsi}
Daily vs SMA-20    : {daily_sma20}
Daily ATR %        : {daily_atr}

═══ RECENT PRICE ACTION (last 30 min, 1-min bars) ════════════════
{price_history_str}

═══ TASK ══════════════════════════════════════════════════════════
1. Evaluate if ALL of the following align for a {side} trade:
   - Momentum direction (RSI, Stoch, streak)
   - Price structure (price vs SMAs, band position)
   - Volume confirmation (RVOL > 1.5 preferred)
   - Market regime support (is Nifty + Regime aligned with {side}?)
   - Daily trend alignment (intraday trade WITH the daily trend is safer)
   - Proximity risk: Is {nearest_wall_label} too close (< 0.3%) to allow a clean move?

2. Output STRICT JSON only — no markdown, no extra text:
{{"sentiment": "LABEL", "reason": "concise 1-sentence rationale covering the confluences or conflicts", "probability": "XX%"}}

Labels: STRONG BULLISH | BULLISH | NEUTRAL | BEARISH | STRONG BEARISH
"""
```

### Layer 2: Fundamental & Governance Audit (S2)

```python
prompt_search = f"""# ROLE: Institutional Risk Auditor (CRO) — Capital Protection Division
# CURRENT DATE/TIME: {current_dt}
# YOUR ONLY OBJECTIVE: Determine if fundamental/news/corporate reality CONTRADICTS the technical signal.
# You are NOT here to predict price. You are here to VETO trades where news/fundamentals overrides technicals.

════════════════════════════════════════════════════════════════
AUDIT PROFILE
════════════════════════════════════════════════════════════════
Company         : {company_name} (NSE: {ticker})
Sector          : {sector} (Rank: {sector_rank_str} | Mean Return: {sector_mean_ret} | Breadth: {sector_breadth})
Proposed Trade  : {side}
Current Price   : ₹{price}
ML Conviction   : {conviction:.4f}
  → Universe Rank : #{int(features.get("Long_Rank" if side == "LONG" else "Short_Rank", 999))} of ~172 stocks screened this cycle
  → Min gate      : {self.min_conviction:.2f}
RVOL            : {rvol}x
{strategy_str}

S1 TECHNICAL VERDICT  : {sent1}
S1 PROBABILITY        : {prob1}
S1 TECHNICAL SUMMARY  : {s1_tech_summary}

KEY S/R CONTEXT
{sr_context}

RECENT PRICE ACTION (last 30 min, 1-min bars):
{price_history_str}

════════════════════════════════════════════════════════════════
STEP 1 — GROUNDED NEWS SEARCH (INDIAN FINANCIAL MARKETS)
════════════════════════════════════════════════════════════════
Perform highly targeted Google Searches to find localized Indian financial news (specifically targeting NSE/BSE listed entities).
Search for ALL of the following (use Google Search grounding):
  • "{company_name} NSE stock news today"
  • "{ticker.replace('.NS', '')} share price target brokerage"
  • "{company_name} block deal bulk deal today"
  • "{company_name} NSE results earnings dividend buyback corporate action"
  • "{company_name} SEBI regulatory dispute legal news"

Classify findings into two buckets:

BUCKET A — Structural (P0, last 7 days):
  Earnings beat/miss, target rating upgrades/downgrades, SEBI/regulatory action, promotors pledge/stake changes, dividend/split/merger corporate actions, industry shocks.
  → These OVERRIDE technicals. A rating downgrade or regulatory ban for a LONG = mandatory VETO.

BUCKET B — Tactical (P1, last 6 hours):
  Block deals, bulk deals, sudden localized volume spike explanations, AGM, analysts meet.
  → These MODIFY conviction but may not VETO unless strong directional conflict.

If NO material news is found in either bucket, explicitly state "No material catalyst found."

════════════════════════════════════════════════════════════════
STEP 2 — VETO DECISION MATRIX
════════════════════════════════════════════════════════════════
Apply the following rules IN ORDER (first triggered rule wins):

RULE 1 — HARD VETO (Fundamental & Corporate Conflict):
  • LONG + Bucket A shows: earnings miss, rating downgrade, promoter sell/pledge increase, regulatory ban/SEBI fine, tax demand, negative corporate governance issue → VETO=TRUE
  • SHORT + Bucket A shows: earnings beat, rating upgrade, buyback announcement, promoter buy/pledge reduction, major order win, regulatory clearance → VETO=TRUE

RULE 2 — SOFT VETO (Tactical Conflict):
  • LONG + Bucket B shows: large block sell, negative headline → consider VETO if S1 conviction < 0.25
  • SHORT + Bucket B shows: large block buy, positive headline → consider VETO if S1 conviction < 0.25

RULE 3 — BREAKOUT OVERRIDE (Do NOT veto):
  • If RVOL > 3.0 AND Bucket A/B shows strong POSITIVE catalysts AND {side}=LONG:
    Resistance levels are TARGETS in a genuine breakout, not walls. Do NOT veto.

RULE 4 — MAGNET EFFECT (Price Action):
  • VETO SHORT if price has been grinding within 0.5% of a resistance level for >15 minutes.
  • VETO LONG if price is sitting on a support level without a bounce for >15 minutes.

RULE 5 — S/R PROXIMITY RISK:
  • If {nearest_wall_label} is within 0.3% of current price AND no strong catalyst supports pushing through → VETO=TRUE

RULE 6 — SECTOR TREND DIVERGENCE (Macro Group Risk):
  • VETO LONG if {side}=LONG and Sector Mean Return is highly negative (< -0.5%) or Sector Breadth is extremely weak (< 20%), indicating no group participation, unless a strong stock-specific positive news catalyst overrides it.
  • VETO SHORT if {side}=SHORT and Sector Mean Return is highly positive (> 0.5%) or Sector Breadth is extremely strong (> 80%), indicating high sector momentum.

RULE 7 — NO NEWS = DEFER TO S1:
  • If no material news/governance issues are found, set veto_decision=FALSE and anchor to S1's probability ({prob1}).

════════════════════════════════════════════════════════════════
STEP 3 — FINAL OUTPUT
════════════════════════════════════════════════════════════════
Output STRICT JSON only — absolutely no markdown, no extra text, no commentary:
{{
  "news_found": "Brief description of Bucket A and B news, or 'No material catalyst found'",
  "chain_of_thought": "does the news CONFIRM, CONTRADICT, or is NEUTRAL to the {side} technical signal?",
  "structural_bias": "BULLISH|BEARISH|NEUTRAL",
  "veto_decision": "TRUE|FALSE",
  "veto_rule_triggered": "RULE 1|RULE 2|RULE 3|RULE 4|RULE 5|RULE 6|RULE 7|NONE",
  "final_sentiment": "STRONG BULLISH|BULLISH|NEUTRAL|BEARISH|STRONG BEARISH",
  "support_resistance_risk": "HIGH|MEDIUM|LOW",
  "probability": "XX%",
  "risk_factor": "the one most likely reason this trade fails"
}}
"""
```
