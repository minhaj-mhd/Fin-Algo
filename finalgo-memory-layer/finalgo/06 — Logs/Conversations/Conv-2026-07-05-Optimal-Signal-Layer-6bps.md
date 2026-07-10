---
title: "Conv 2026-07-05 — Optimal signal-generation layer vs 6bps cost"
type: log
status: concluded
updated: 2026-07-05
verdict: 1h stack CANNOT beat 6bps (certified); the layer that does = daily_macro_v2 LONG (+28.2bps net@6, TRIGGER_GRADE) + open gap-fade (+14.1bps/day net@6, exploratory); S2 live veto value FALSIFIED on the 1h stream. NEW (part 2) — E3 OVERNIGHT-EXIT LAYER: daily_macro_v2 edge is 100% overnight (all 3 ON segments +20..+23bps t≈10, all intraday segments NEGATIVE); exiting at open[t+4] instead of close[t+3] nets +49.4bps@6 (t 5.2) vs +29.0, paired uplift +20.3bps t 9.3, better EVERY year 2018-26, flips 2025/2026 positive. ⚠️ UNVERIFIED (exploratory, WF preds from run 20260610T135608Z-5f7d069f); needs one pre-registered Gauntlet run
gauntlet_run_id: 20260615T175149Z-5f7d069f, 20260610T135608Z-5f7d069f, 20260610T183618Z-d795438c
---
# 💬 Conversation Context: Optimal signal-generation layer that beats 6bps cost

## 📌 Metadata
- **Start Date**: 2026-07-05
- **Status**: 🔴 Concluded
- **Focus Area**: Signal-generation strategy / veto-layer verification / cost line

## 🎯 Objective
User goal: with rule engine + Kronos + candle vetoes + v10–v20 models + S2 Gemini
search-grounding veto in place, find the optimal SIGNAL GENERATION layer that beats 6bps cost
(premise: "S2 filtered out very much of the bad trades last week").

## 📝 Compacted Session Log
- **S2 premise VERIFIED AGAINST LIVE DATA — FALSIFIED.** `vanguard_trades.db`, full S2 window
  (first `[S2-VETO]` 2026-05-21 → 2026-07-03): EXECUTED book n=146, **−4.2bps gross avg**
  (−Rs 6,608 total); S2-VETOED shadow n=295, **+3.1bps gross avg** (rawWR 53%); S2-vetoed LONGs
  n=72 **+9.1bps gross → +3.1bps NET @6bps**. Shadow `final_profit_pct` is side-adjusted
  (orchestrator.py:1631), so the veto killed trades *better* than what it passed. Last week
  specifically: executed net −Rs 946; vetoed-shadow avg positive on 3 of 5 days. S2 is not
  selecting against bad trades on the 1h stream (neither is S1: n=1058, +1.0bps gross ≈ book).
  Caveat: shadow = gross, fixed +1h mark, no SL path — but the sign of the comparison is unambiguous.
- **Certified 1h stack at 6bps (Gauntlet reports, costs_bps included 6.0 — no new runs needed):**
  - v20 (run `20260615T175149Z-5f7d069f`): best cell Top-1 SHORT **+0.71bps net@6, t=0.27** (zero);
    K3 long −2.16 (t −1.9), K3 short −3.83 (t −2.58). Recent-12mo top-1 short +0.98, t 0.33.
  - v10 (run `20260610T183618Z-d795438c`): ALL cells negative @6 (best K1 long −0.84, t −0.64).
  - Time-of-day cells are `tod_diagnostic_only`; panel probe (:30 anchors, partially in-sample
    for the saved booster) showed top-1 short +5.5 t 2.05 and 14:30-short +4.9 t 1.5, but the
    honest walk-forward equivalent collapses to ~0; adjacent-anchor sign flips (14:15 long + /
    14:30 long −) ⇒ noise. No hour pocket crosses 6bps credibly.
- **Ensembling cannot fix it**: v10/v20 are certified peers (ρ≈0.026–0.033, highly correlated);
  15+ dead research lines (CST, DualRes, sided-transformer, co-sign, meta-gate, retrieval,
  graph/GCN, SMC, horizon sweep, HP tuning) all hit the same info-ceiling. Crossing 6bps at K3
  needs ~+6bps gross ⇒ roughly 2× the rank-IC (~0.06); nothing in price/volume exceeded ~0.035.
- **What DOES beat 6bps (already in repo):**
  1. **daily_macro_v2 LONG, K3, 3-day hold** — **+28.2bps net@6/trade, t=3.13, n=2955 full-OOS**
     (recent-24mo +22.6, t 1.65) — run `20260610T135608Z-5f7d069f`, the ledger's ONLY TRIGGER_GRADE.
  2. **Open gap-fade paired book** — **+14.1bps/day net@6** at auction fills, t=9.1, Sharpe 4.9,
     every year positive (`data/research/open_window_stack/strategy_backtest.json`) —
     ⚠️ UNVERIFIED by Gauntlet, ~90% of edge is the open print; pre-open module + slippage
     measurement is the gating step (Active Board item stands).
- **Recommendation delivered**: signal layer = daily_macro_v2-LONG + gap-fade pre-open; keep the
  1h stack + vetoes as risk infrastructure, not generators. Best S2 fit is the GAP-FADE stream
  (overnight news is exactly what search grounding sees: veto news-justified gaps, fade
  mechanical ones) — a testable hypothesis, not yet tested.

## 📝 Part 2 — Forward levers (user directive: build, don't just falsify)
- **Entry-timing overlay on daily_macro_v2 — NEGATIVE**: every delayed/limit entry loses vs the
  certified close[t] entry (next-open −20bps, limit-fallback variants −3…−14). Edge is FRONT-LOADED;
  same-day close entry is a binding design constraint for the live engine.
- **Segment decomposition (the structural discovery)**: on the certified WF top-3 LONG picks
  (preds.npz from run `20260610T135608Z-5f7d069f`, alignment verified err=0), the 3-day edge lives
  ENTIRELY overnight: ON1 +20.5 (t 9.7), ON2 +22.8 (t 10.6), ON3 +22.7 (t 10.6); ALL intraday
  segments negative (D1 −7.7, D2 −10.1, D3 −12.6). Explains the 1h-long headwind AND the gap-fade:
  this universe drifts up overnight and bleeds intraday.
- **E3 exit layer (parameter-free): exit open[t+4] instead of close[t+3]** — net@6 **+49.35bps/trade
  (t 5.19)** vs +29.01 baseline; net@20 (cash-delivery STT reality) +35.4; **paired uplift +20.34bps
  t 9.32 full-OOS, +16.77 t 4.26 recent-24mo; better in EVERY year 2018–26 (2025: −0.5→+9.7,
  2026: −5.8→+12.4); holds at K=1/3/5**. Attribution: ~17bps of the extra overnight is universe-wide
  overnight drift — irrelevant to tradability (harvested inside an existing position at zero extra
  cost), relevant to hedging/beta. Exit at open = auction participation (same mechanics the gap-fade
  execution study validated). E5 (gap≥1% early exit at open[t+1], else open[t+4]) adds ~+1.7bps —
  fitted threshold, second-order, parked. E1 overnight-only (close→open daily recycle) +14.5@6/day —
  futures-only option (cash delivery 20bps kills it).
- **⚠️ All Part-2 numbers UNVERIFIED (exploratory)**: single pre-registered Gauntlet run with an
  open[t+4]-exit label variant is the certifying step (dataset-family t-deflation applies; user
  approval required per protocol).

## 📝 Part 3 — Mandate constraint + intraday-compatible layer (user directives)
- **Operating mandate declared by user**: NO holds > 1 hour; entries ONLY 10:15–14:15
  (matches `scripts/vanguard/config.py` FIRST/LAST_ENTRY_TIME, ENTRY_TOP_K=3). All layers must
  augment the HOURLY models. E3/daily/gap-fade registered in [[00 — Start Here/Ray of Hope]] but
  parked as outside-mandate.
- **New rule from user: EVERY uplift (even +1–2bps) gets registered in the hope document**
  ([[00 — Start Here/Ray of Hope]]) — done for today's finds.
- **G\* hour×side gate @6bps (mandate-compatible, from v20 WF preds run `20260615T175149Z-5f7d069f`)**:
  10:15 LONG + 11:15 SHORT + 14:15 both, skip 12:15/13:15, **K=1** → **+5.17bps/book (t 1.93),
  H1 +6.3 / H2 +4.5, ≈+20.7bps/day**; K=3 dies in H2. Cells: 11:15-short K1 +11.0 (t 1.96) strongest;
  midday uniformly negative. ⚠️ Honest caveat: H1→H2 cell re-picking FAILS (−1.2) — gate is only
  defensible as pre-declared (4 prior independent looks agree on the shape); NO further cell mining.
  Config delta vs live: skip 12:15/13:15 books, side-tilt 10:15/11:15, ENTRY_TOP_K 3→1.
- Next validation: 2–4wk live shadow A/B (gated vs current book) or one pre-registered Gauntlet
  variant; plus execution-slippage audit (live entry_price vs signal-bar price) to make 6bps real.

## 📝 Part 4 — 11:15-short deep-dive + 10:30–11:30 zone scan (user request)
- **11:15-short anatomy (certified WF preds)**: +10.98 net@6 t 1.96, WR 58% net (60.7% gross);
  11/16 fold-months positive (worst −36.4 2025-11); K-decay steep (K2 +6.05, K3 +0.34 → TOP-1 ONLY);
  104 unique names, top-10 = 29% of picks (PAYTM 10, KEC 9, PRESTIGE/METROPOLIS/THERMAX/BRIGADE 8).
- **Zone scan** (rolling panel, pseudo-OOS 2025-01→2026-06, top-1 short; calibration bridge:
  panel 11:15 +9.23 vs certified +10.98 → panel ≈ honest here): **10:30 +6.88 (WR 60.9%,
  H1 +6.87 / H2 +6.89 — the most stable cell in the study)**, 10:45 −9.41, 11:00 +5.68,
  11:15 +9.23, 11:30 +6.29.
- **What FAILED honest validation (all registered as refinement dead-ends, do NOT deploy):**
  conviction-quartile gate (pooled Q4 +24.6 but H1-threshold→H2 = −2.64 vs +2.05 ungated);
  5-anchor ladder (28–40% consecutive same-ticker overlap, per-day t 0.95 — zone ≈ ONE
  independent bet); oracle max-conv anchor +20.7 t 3.44 both halves is an UPPER BOUND —
  implementable timing rules all fail (secretary R1 −10.1 / R2 −6.2; trailing-threshold R3
  H1 +1.2 / H2 +29.7 unstable). Matches the exclusion-engine lesson: within-day micro-timing
  doesn't transfer.
- **Actionable**: keep G\* 11:15-short as pre-declared; **10:30-short is the one candidate addition**
  (2nd short slot) for the live shadow A/B — decided there, not by more mining here.

## 📝 Part 5 — 20-signal/day funnel for the veto layer (user spec)
- Full grid: 17 anchors (10:15–14:15) × side × rank1-3 on the panel pseudo-OOS (856k rows).
  **Mining proof: top-20 cells picked on H1 (+11.2) collapse to −3.3 (t −2.25) on H2.**
  Pre-declared structured 21-slot menu: H1 +5.3 → H2 −1.6. **v20's information budget is
  ~6-8 above-cost signals/day, not 20**; rank-2 ≈ 0, rank-3 negative everywhere.
- **Cross-validated Tier-A slate (positive in BOTH the certified-WF view and the panel-H2 view):**
  10:30 S (H1 +6.9/H2 +6.9, WR ~59-61% — the rock), 11:00 S, 11:15 S, 11:30 S, 13:45 S,
  14:15 L, 10:30 L (panel-only, WR 49% — probational). ⚠️ G\*'s 10:15-L leg WEAKENED: certified-WF
  +4.25 but panel-H2 −15.9 (recent period bad) — keep in shadow, expect it may drop.
- To reach 20 signals: add Tier-B filler (zone rank-2 shorts + remaining non-midday anchors,
  expected −1…−3bps/signal pre-veto, 36% same-day duplicate share). **The veto-funnel experiment**:
  S2 selects ~5 from Tier-A+B daily in SHADOW; success criterion = S2's picks beat plain Tier-A.
  That is the clean test of the user's "generator → veto picks" thesis.

## 📝 Part 6 — Rule-engine pre-filter implemented + measured (user request)
- **Implemented** `scripts/analysis/rule_engine_prefilter.py` (reusable harness: declared exclusion
  rules per side, applied BEFORE ranking, top-1 re-taken from filtered pool, paired Δ vs baseline
  AND vs matched-count random-exclusion control, H1/H2 split, Tier-A slots, 6bps).
- **Result: NO uplift — every rule subtracts.** Baseline Tier-A top-1: short +6.26, long +4.34.
  Alphas (rule−random): up_thrust −1.39 (H2 −4.75); **live FADE_QUALITY_GUARD analog −0.42/book
  (touches 8% of books — "cheap insurance" framing is the only defense)**; vol_extreme −3.18 S /
  −3.47 L (both halves negative); combined risk_gate −3.45/−3.99 (≈ half the edge);
  **illiquidity rule INERT at top-1 (0% books changed — the ranker never picks illiquid names)**.
- Confirms + extends the exclusion-engine dead-end to the top-1@6bps regime: the model's best
  top-1 picks ARE the extreme movers (fades); universe pre-filters remove exactly them. RELAXO-class
  protection belongs at FILL time (candle fill-recheck), not as a universe pre-filter.
- **INVERTED keep-lists re-tested at top-1@6bps (user recall of the probe4 "+2bps")**: conditional
  info REAL on shorts — alpha vs random-same-count **+4.8…+5.4bps, WR +5.6pp (54.1% vs 48.5%),
  t 2.05, both halves +**; LONG side FRAGILE (alpha +0.3…+2.6 seed-sensitive, WR +0.7pp ≈ noise) —
  but absolute book net still LOSES to full-universe baseline (**shorts −2.6bps / WR −2.0pp**
  [54.1% vs 56.1%]; longs −1.2bps / WR flat; pool-shrinkage > condition value; ranker already holds
  MOM/RSI as features). Baseline reference: full-universe Tier-A top-1 = short +6.26 net@6 /
  WR 56.1% net (59.3% gross), long +4.34 / WR 50.1% (54.2% gross). Usable form = candidate METADATA
  (reversion-favorable flag, ~5pp short-side selection signal) in the 20-signal funnel as a selector
  tiebreak, zero pool shrinkage; measure in the shadow A/B. Registered in [[00 — Start Here/Ray of Hope]].

## 📝 Part 7 — v3 15m top-33% same-side confirmation gate (user request) → DEAD (artifact)
- **Literal question**: require the v20/host pick to sit in v3's top-33% on the SAME side.
  On `dualtf_trade_panel.csv` (13,020 WF trades 2024-09→2026-06), same-side v3-HOT picks
  (rk≥.67) net **−20 to −32 bps, WR 32–37%**; the confirmation intuition is BACKWARDS.
  v3's same-side rank is a CONTRA-indicator (mean reversion): high 15m rank = exhaustion.
  Sign corroborated 3× (README verdict, monotone quintile decline, inverted keep-list).
- **The tempting inverse LOOKED huge** — requiring v3-COLD (rk<.67) on top-1 host cells:
  10:15L +22.4, 11:15S +23.3 (WR 69%), long +9.9 / short +11.7 pooled, both halves +.
  **KILLED as an ENTRY-PRINT ARTIFACT via a lag test**: the rank only predicts when read at
  the entry bar dt1+60 (shares price P60 with nhr's start). Measured ONE bar earlier
  (dt1+45, clean) the same gate flips to **−7.9 long / −8.0 short** (≤ baseline); dt1+30
  also negative. Adjacent-bar 15m ranks are ~uncorrelated (corr 0.02–0.10) → the "edge"
  is mechanically tied to the shared entry print (bid-ask bounce), not a capturable state.
  Same un-fillable-print trap as [[project_intraday_overnight_reversal_edge]] /
  [[project_stop_loss_research]] / [[project_news_shock_study]].
- **Panel caveat noted**: `dualtf_trade_panel.csv` sub-period returns do NOT reconcile to
  `nhr` (corr 0.73, mean |Δ| 35 bps; compounded-sub gate even flips sign) — do not use its
  `sub_*` columns for magnitude claims; nhr-only + the lag diagnostic are the trustworthy reads.
- **Reusable diagnostic delivered**: the "rank-lag collapse" test (does a gate survive reading
  its trigger one bar before the shared entry price?) is now the standard artifact screen for
  ANY entry-timing gate. Nothing registered in Ray of Hope (nothing survived).
- **"Can a LIMIT order capture the print artifact?" — NO (adverse selection), answered with
  existing gap-fade evidence** (`open_window_stack/strategy_backtest.json`): same-universe
  print-reversal edge, limit-at-open = **−14.8 bps book (t −11.6), fill 72%** — INVERTED the
  +14.1 auction edge. Limit fills are adversely selected: the ~28% unfilled are the winners
  (bounced away), the filled are the falling knives; the "90% will fill" assumption IS the trap
  (the missing tail is your best trades). Limit orders remain a REAL lever for CUTTING COST on
  already-clean edges (G* +5.2, 10:30-short) — measure passive fill-vs-signal slip live in the
  shadow A/B — but they do NOT resurrect a shared-print artifact.

## 📝 Part 8 — LIVE v3 gate audit ("is v3 harming us?")
- **Code fact**: the live engine ALREADY uses v3 as a binding same-side entry gate —
  `orchestrator.py:2051` requires each candidate's `score_15m` in the **TOP 10% same side**
  or it is SKIPPED ("Entry demands high conviction"). The 15m conviction-flip EXIT
  (`orchestrator.py:1787`, top-33% maintenance) is **DISABLED by default**
  (`CONVICTION_FLIP_EXIT_ENABLED=False`) — not active.
- **So the deployed gate is exactly the same-side-confirmation shape Part 7 found backwards,
  at an even more extreme threshold.** Kept vs skipped (dualtf panel, top-1 host, entry-bar rank):
  LONG kept −34.9 (WR 26%) vs skipped +1.9; SHORT kept −49.9 (WR 31%) vs skipped +7.0 — the gate
  keeps the losers, drops the winners.
- **Artifact discipline applied (do NOT quote −50 as live damage)**: the live gate keys off the
  ~entry 15m score → shares the entry print → the −35/−50 is inflated by the same bid-ask artifact.
  On the CLEAN pre-entry rank the true marginal effect collapses to a small MIXED drag: longs kept
  −1.5 vs skipped −4.2 (marginally helps), shorts kept −9.5 vs skipped −7.4 (marginally hurts),
  both near the negative baseline; panel tie/reconciliation issues cap magnitude confidence.
- **Assessment**: v3 is NOT a −50bps catastrophe (mostly artifact) but is NOT delivering the
  "high-conviction confirmation" it's coded for — same-side 15m confirmation is flat-to-backwards
  (mean reversion). Real non-artifact cost = it demands top-10% ⇒ discards ~90% of candidates,
  a major pool-thinner biasing the book toward extended/exhausted names. Net = dead-weight/mild
  drag + heavy thinning.
- **Recommendation (NOT executed — live config change, needs the user)**: don't rip out blind;
  settle it via (a) fresh v3 scoring on the v20 panel at anchors (no shared-print, real ties) A/B'ing
  the top-10% gate, or (b) flip it to SHADOW in the live A/B (log what it would skip, compare
  kept-vs-skipped on realized fills 2–4 wks). If confirmed neutral-to-negative, demote to a logged
  diagnostic. Config lever: the gate is the `_check_15m_percentile(..., top_percent=0.10)` call at
  orchestrator.py:2051.

## 📝 Part 9 — CORRECTION: the stable cell set is broader than G* (10 cells, short-dominated)
- Earlier framing over-indexed on the narrow 4-cell G* (10:15L, 11:15S, 14:15L+S). Full top-1
  per-side scan of ALL 15-min anchors 10:15–14:15 (panel pseudo-OOS 2025–26), keeping only
  BOTH-HALVES-POSITIVE cells, gives **10 stable cells** — and G*'s short cell (14:15S +0.8) is
  among the WEAKEST. Leaders (net@6 / WR / H1/H2):
  - **13:30 S +9.90 (52.5%, +10.3/+9.5)** — strongest, NOT in G*
  - 11:15 S +9.23 (54.8%, +15.7/+2.8) — G*, but DECAYING H1→H2
  - **10:30 S +6.88 (60.9%, +7.0/+6.7)** — most balanced short, NOT in G*
  - 11:30 S +6.29, **10:30 L +6.23 (+7.0/+5.5)**, 11:00 S +5.68, 13:15 S +5.13, 13:45 S +3.25,
    14:15 L +2.45 (G*), 14:15 S +0.81 (G*).
- Set is SHORT-dominated (8 of 10). Requires the 15-min scan cadence (config supports overlapping
  rolling mode; user confirms 10:30 trades fire live). CAVEATS: panel = partially in-sample
  (individual t ~1–1.8, none >2; book-level aggregate carries significance); prefer BALANCED cells
  (10:30 S, 13:30 S, 10:30 L: H1≈H2) over decaying ones (11:15 S, 11:30 S: H1≫H2); menu-mining
  over-assembly still applies (both-halves filter mitigates, doesn't eliminate). Certified
  cross-check: panel 11:15 S +9.23 vs certified +10.98 → panel was conservative there.
- **Corrected signal layer** = top-1 per side at the stable cells (led by 13:30 S / 10:30 S / 10:30 L),
  ~6–8 trades/day, NOT the narrow G* four. Supersedes the "G* is the mandate-compatible layer" framing.

## 📝 Part 10 — v21 CROSS-CHECK: the stable-cell slate mostly does NOT survive a model swap
- Re-ran the identical top-1/side stable-cell scan on **v21** (lean clean rebuild,
  `models/research/v21_rolling_1h`, 88 feats, 540k-row leaner universe) alongside v20.
- **Of v20's 10 both-halves-stable cells, only 13:15 S is ALSO both-halves-stable on v21.**
  v20's short leaders flip on v21: 10:30 S +6.9→**−2.7**, 11:15 S +9.2→**−4.7**, 11:00 S +5.7→−4.7,
  11:30 S +6.3→−3.1; 13:30 S (my "gold standard") +9.9→+2.2 but H2 flips −7.2. **v21's side-tilt is
  REVERSED** — it favors LONGS (10:15 L +7.7, 11:15 L +6.5, 12:00 L +6.0) with broadly negative shorts.
- **Conclusion: the per-cell hour×side timing edges are largely v20-specific OVERFIT, not robust market
  structure; even the short-tilt isn't robust.** Info-ceiling from the timing angle (cf. v21 rebuild:
  cleanliness ≠ edge). Caveat: v21 uses a leaner universe + 88 vs 86 feats (different-but-comparable
  model, not a re-seed) — some divergence expected, but near-total sign disagreement exceeds that.
- **Robust survivors (cross-model): 13:15 S is the ONE both-halves-stable-on-BOTH cell** (v20 +5.1,
  v21 +8.7 t2.1). Softer "positive on both models" set (half-unstable on one): 10:30 L, 13:30 S,
  14:15 L, 13:45 L, 12:00 L — tilts LONG.
- **New selection rule going forward: require CROSS-MODEL AGREEMENT (positive on v20 AND v21) — a much
  stronger bar than single-model both-halves.** Supersedes Part 9's short-tilted stable slate. The
  shadow A/B should track only cross-model-agreed cells.

## 📝 Part 11 — Daily-macro (v2) gating of intraday signals → LEAK (no robust edge)
- Tested: gate intraday v20 signals on daily_macro_v2's same-side per-ticker ranking (WF-OOS
  preds, run 20260610T135608Z). SAME-DAY gate looked spectacular & CONTRARIAN — fade the daily
  model intraday: intraday-long on daily-top-long −8.1 (t −6.9); **short on daily-top-LONG +15.9
  (t 11.3)**, long on daily-top-SHORT +11.2 (t 11.9); monotone gradient, both halves. t 11–14 is
  10× anything in project history → suspect.
- **LEAK CONFIRMED via lag test** (same discipline as the v3 rank-lag test): the daily score for
  day D uses D's CLOSE, not known during D's 10:15–14:15 session. Re-run with PRIOR-day daily score
  (no look-ahead): intraday LONG goes FLAT vs daily rank (bottom −0.6 / top +1.2); intraday SHORT
  mildly + but SAME both buckets (+3.9 / +3.2 — daily rank doesn't differentiate, it's just the
  baseline short edge); contrarian long-on-daily-short DEAD (+0.03); contrarian short-on-daily-long
  +5.9 t7.4 but **H1 +10.7 → H2 +0.8 (decays to ~0)**. → No robust daily-gating edge; re-confirms
  memory "daily-macro gate anti-additive" ([[project_v20_v3_oos_overnight_artifact]]).
- CAVEAT: daily preds are sparse (WF fold months) → merge_asof gave median 7-DAY lag, unfairly
  stale for a real 1-day-lag gate. The ONE clean follow-up = score daily_macro_v2 continuously
  (every day) and test a true T-1 gate. Priors against it (leak collapse + H2 decay + anti-additive
  memory), but it's the honest remaining test. **Does NOT help the 20-quality-signal goal**: gating
  is a filter (cuts count), and adds no quality once de-leaked.
- **INVERTED gate — DEFINITIVE TEST (continuous in-sample daily scoring → true T-1 lag=1d + NEG-CONTROL): DEAD.**
  Inverted short (short daily-top-LONG) T-1 = +4.47 (t5.72, H2 +0.3); **neg-control (SHUFFLED daily
  rank) = +4.56** ≥ real; baseline all-shorts = +3.96. Daily rank carries NO information — the +4.5 is
  entirely the baseline short-side tilt any same-size pick captures; inverted long +1.37 ≈ noise; both
  decay H2→0. Ran with IN-SAMPLE (optimistic) daily scores and STILL can't beat a shuffle → conclusive.
  **Summary: same-side gate dead (untradeable reversion); same-day inverted +16 t14 = 100% look-ahead
  leak; clean T-1 inverted = random. Daily gating (any polarity) provides NO intraday edge.**

## 📝 Part 12 — LIVE CONFIG CHANGES APPLIED (user-directed, reversible)
Three reversible changes to the live entry layer (defaults preserve old behavior via getattr):
- **`ENTRY_TOP_K = 3 → 1`** (config.py:70) — concentrate to the single best pick/side (steep K-decay;
  top-1 > top-3 both sides). NOTE: signal gen still emits top-1 net + top-1 raw = up to 2/side
  (AI_Raw path); disable AI_Raw for strictly-1 if desired.
- **`DAILY_GATE_ENABLED = False`** (config.py; gate at orchestrator.py:633) — daily same-side top-40%
  whitelist OFF (daily rank carries no intraday info, neg-control-verified; was pure ~60% thinning).
  When off, long/short_eligible = full evaluated universe.
- **`ENTRY_15M_GATE_ENABLED = False`** (config.py; gate at orchestrator.py:2051) — v3 15m top-10%
  same-side entry gate OFF (flat-to-backwards confirmation; discarded ~90% of candidates).
- Verified: both files compile; flags load; `risk_manager.entry_top_k` reads config; downstream
  `score_15m` still sourced from `full_feature_row` (no NameError); synthetic signal-gen test = top-1/side.
- **Honest framing**: these gates weren't actively bleeding — they weren't earning their keep + thinned
  the book. Removal CLEARS THE PATH for top-1 + veto-funnel; it does NOT itself make the book net-positive
  (info-ceiling stands). Watch shadow book 2–4 wks: trade count should rise, question = does executed
  per-trade net improve. Rollback = flip any flag True / ENTRY_TOP_K=3.

## 📝 Part 13 — AI_Net (conviction rank) vs AI_Raw (raw score): raw is BETTER on longs
- Live `Long_Rank` = rank of CONVICTION = centered(long_score) − centered(short_score)
  (model_inference.py:229); AI_Net picks by that, AI_Raw picks by raw long_score (excl. net pick).
- Measured (v20 panel, top-1/side, net@6): **LONG AI_Net −1.71 (WR 46.4%) vs AI_Raw +0.58 (WR 47.9%)
  — raw beats conviction by ~2.3bps**; SHORT AI_Net +5.02 vs AI_Raw +5.33 (≈ equal). Hourly cadence
  same pattern (LONG net −1.10 vs raw +0.38; SHORT +4.27 vs +4.70). Both decay H2 (longs → negative).
- **Implication: the conviction (long−short) ranking HURTS long selection** — subtracting short
  penalizes good longs that also score short-ish. The raw long_score IS the certified v20 ranker;
  conviction is an unvalidated hybrid. **Do NOT disable AI_Raw** (user's strictly-1 idea) — if
  concentrating, make RAW SCORE the primary long selector, not Long_Rank. Shorts: wash.
- Within-model comparison (same booster) → relative result trustworthy; absolute still panel/H2-fragile.

## 📝 Part 14 — PER-ANCHOR BASELINE (v20 raw-score top-1) + live shadow cross-verification plan
**Baseline to cross-verify against** (v20 top-1 by RAW score/side, net@6, canonical OOS 2025-01→2026-06,
295 days). ✓ = both-halves-positive on v20; ✓✓ = also survives v21 (cross-model robust).

| Time | LONG net | WR | H1/H2 | SHORT net | WR | H1/H2 |
|---|---|---|---|---|---|---|
| 10:15 | −2.95 | 46% | +10.1/−15.9 | −4.89 | 51% | −0.8/−9.0 |
| 10:30 | +6.23 ✓ | 52% | +7.0/+5.5 | +6.88 ✓ | 61% | +7.0/+6.7 |
| 10:45 | −5.33 | 46% | −6.2/−4.5 | −9.41 | 51% | −17.2/−1.7 |
| 11:00 | +2.28 | 46% | +5.2/−0.6 | +5.68 ✓ | 59% | +8.2/+3.2 |
| 11:15 | −2.54 | 49% | +3.7/−8.7 | +9.23 ✓ | 55% | +15.7/+2.8 |
| 11:30 | −3.22 | 44% | +0.1/−6.5 | +6.29 ✓ | 52% | +10.5/+2.1 |
| 11:45 | −1.24 | 46% | +3.5/−6.0 | +0.60 | 57% | +11.5/−10.2 |
| 12:00 | +1.55 | 49% | +7.6/−4.5 | −2.58 | 53% | +4.3/−9.4 |
| 12:15 | −1.02 | 43% | −1.9/−0.2 | −4.20 | 52% | +5.8/−14.1 |
| 12:30 | −6.82 | 42% | −5.8/−7.9 | −6.74 | 52% | +0.1/−13.5 |
| 12:45 | −2.57 | 45% | −0.4/−4.7 | −12.00 | 50% | −3.8/−20.2 |
| 13:00 | −2.28 | 47% | +7.4/−11.9 | −7.37 | 48% | −1.7/−13.0 |
| 13:15 | −8.51 | 45% | +2.3/−19.2 | +5.13 ✓✓ | 52% | +7.1/+3.1 |
| 13:30 | +0.84 | 48% | +4.4/−2.7 | +9.90 ✓ | 53% | +10.3/+9.5 |
| 13:45 | +2.40 | 47% | +9.6/−4.8 | +3.25 ✓ | 54% | +3.9/+2.6 |
| 14:00 | −3.73 | 50% | −0.1/−7.3 | +1.49 | 51% | +9.3/−6.2 |
| 14:15 | +2.45 ✓ | 48% | +0.9/+4.0 | +0.81 ✓ | 50% | +0.9/+0.7 |

Patterns: SHORT side is the engine (10/17 cells +, longs mostly −, worse in H2); midday 12:00–13:00
is a dead zone; 9:15–10:00 = NO v20 signal (needs a full same-day trailing hour → 10:15 is the
earliest anchor; the open window is gap-fade territory, execution-bound).

**Live shadow cross-verification (from 2026-07-06):** `SIGNAL_RAW_SCORE_ONLY=True` + `ENTRY_TOP_K=1`
→ engine emits exactly ONE long + ONE short per 15-min scan, selected by raw long_score/short_score.
These flow through the pipeline and are shadow-tracked (executed OR vetoed-shadow counterfactual), so
each anchor×side accumulates a LIVE 1h outcome to compare against the panel cells above. Purpose: does
live per-anchor net/WR match the panel? (esp. confirm 10:30 S/L, 13:30 S, 13:15 S; watch H2-decayers).
Caveats for the comparison: (a) 1h-hold dedup skips a name already open at a later anchor; (b) live
fills/slippage vs panel close-to-close; (c) veto layer still intervenes on executed (shadow captures
the counterfactual regardless).

## 💻 Active Code Files Modified
- [signal_generation.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/signal_generation.py) — SIGNAL_RAW_SCORE_ONLY branch (single raw-score pick/side)
- [config.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/config.py) — ENTRY_TOP_K=1, DAILY_GATE_ENABLED=False, ENTRY_15M_GATE_ENABLED=False, SIGNAL_RAW_SCORE_ONLY=True (reversible flags)
- [orchestrator.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/vanguard/orchestrator.py) — daily-gate + 15m-gate wrapped in the flags
- [rule_engine_prefilter.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/rule_engine_prefilter.py) (NEW — exploratory harness, no verdict authority)
- Otherwise read-only research; probes in session scratchpad: signal_layer_6bps.py,
  daily_v2_entry_timing.py, daily_v2_overnight_decomp.py, daily_v2_e3_robustness.py,
  hourly_gate_6bps.py, short_1115_deepdive.py, late_morning_short_ladder.py).

## 🔗 Core Memory Links
- [[06 — Logs/Conversations/Conv-2026-07-02-Open-Window-Trade-Stacking|Gap-fade research]] ·
  [[06 — Logs/Conversations/Conv-2026-07-05-Short-Long-Exclusion-Rule-Engine|Exclusion-engine dead-end]]
- [[01 — Architecture/Execution & Runtime/AI Veto & Gemini Audit|AI Veto & Gemini Audit]] — S2 live
  value now has a measured (negative) baseline; recheck after more live data.
