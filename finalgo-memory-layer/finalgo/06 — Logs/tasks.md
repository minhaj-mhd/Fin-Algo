## 

Look at the pattern across your own vault. v8's 0.046 Spearman → early-stopping-on-test leakage, real value ~0.025. TBM short's +5.18 bps → cost-sign bug, real value negative. Sniper's 68–76% WR → same-bar IBS lookahead, real value 55–60%. Hybrid depth-4's +7.82 bps → contradicted by the full walk-forward (−2.5 bps). The 15m "dual-TF edge" → overnight-return artifact.

Two things are true about that list:

1. **Every metric was produced by a different, hand-rolled evaluation script**, each with its own fresh opportunity for a leak or sign error. That's why "every model has one or other issue" — the issues are mostly in the harnesses, not the models. No LLM (or human) will reliably write a bespoke bug-free backtest from scratch every time; that's not how this problem gets solved anywhere in the industry.
2. **Every correction moved the number down, never up.** That's the statistical signature of the underlying truth: 1-hour OHLCV features on NSE equities carry roughly 2–3 bps of gross edge against a 6–10 bps cost hurdle. You've now proven this across XGBoost (depths 4 & 5), CatBoost, Random Forest, monotonic constraints, TBM labels, and ensembles. A perfectly bug-free model trained on the same data will be bug-free **and still unprofitable**. The architecture search is exhausted — stop spending effort there.

## The solution: three changes, in this order

**1. Build one canonical, self-testing validation gauntlet — and make it the only source of truth.** A single module (e.g. `scripts/validation/gauntlet.py`) that every candidate model must pass through, with the known bug classes converted into automated assertions so they can never recur silently:

- Purged + embargoed walk-forward, early stopping on a validation fold that is never the test fold (kills the v8 bug class).
- **Feature-timing self-test**: recompute features with future rows truncated and assert outputs are identical — mechanically catches same-bar/IBS-style lookahead (the sniper bug class).
- **Cost invariant**: assert `median(net − gross) == −cost` per side, both sides (the TBM bug class).
- **Label sanity**: assert the label horizon never silently crosses the overnight boundary (the 15m bug class).
- Standard output: per-fold raw and net Spearman/WR, decay trend across folds, t-stats, with a pre-registered pass threshold (e.g. net ≥ +2 bps with t ≥ 2 in the last-12-month folds). Dataset hash + config recorded so every number is reproducible.

This is maybe a few days of work and it ends the whack-a-mole permanently: a model either passes the gauntlet or it's dead, in minutes, before it ever touches the live engine. You already have most of the pieces in `scripts/analysis/v8_walkforward.py`.

**2. Change the information, not the algorithm.** The feature set is the binding constraint. New alpha at 1h has to come from data the price chart doesn't already contain, and you have realistic access to: NSE option-chain OI/PCR/IV skew, India VIX term structure, Nifty futures basis / GIFT Nifty, advance-decline breadth, USDINR, Upstox market-depth snapshots (order-imbalance is the classic genuine intraday signal), and your Gemini news layer repurposed as an upstream _feature_ rather than only a downstream veto. Each new feature family goes through the same gauntlet — no special-case scripts ever again.

**3. Reframe what the 1h layer has to be.** Your pipeline doesn't actually need the 1h model to be a standalone profitable strategy — it needs it to _rank_. A ranker with a real-but-tiny ρ≈0.025 (and your short-side score is genuinely real, z=5–7) is useless as a trade trigger but perfectly useful for shortlisting the gatekeeper-approved universe down to a handful of names. Then the **entry edge lives at the trigger layer, where you already have verified OOS winners**: Sniper Tier B (+17.2% in 5 months of clean out-of-sample), Strategy 8 ORB, and the time-of-day pockets. Daily gatekeeper picks the day → 1h ranker shortlists symbols (filter, not trigger) → 15m/structural triggers fire only in proven windows → Gemini vetoes. That architecture works with the edge you've actually proven, instead of waiting on an edge that ten model generations say isn't in the data.

**One immediate action:** per your own walk-forward audit, v8 is net-negative and decaying, and it's currently generating live signals. It should be demoted today from trade-trigger to shortlist-filter duty (or the live engine should lean on the structural strategies until the gauntlet re-baselines everything).

My recommendation for the first concrete step is the gauntlet — everything else (re-auditing `v2_15min_3y`, testing new features, deciding what's live-worthy) depends on having one trustworthy measuring stick. Say the word and I'll build it.