# BRIEFING — 2026-07-15T08:19:16Z

## Mission
Analyze existing finalgo features, data, and models to identify a candidate trading edge that is small, genuine, tradable, not invalidated, and supports a reproducible statistical report with positive Expected Value.

## 🔒 My Identity
- Archetype: Teamwork explorer
- Roles: Read-only investigation, analyze problems, synthesize findings, produce structured reports
- Working directory: c:\Users\loq\Desktop\Trading\finalgo\.agents\teamwork_preview_explorer_edgesearch_2
- Original parent: 40002285-cf43-43d2-bfec-8fb0b0a1ea97
- Milestone: Candidate Edge Identification

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Code-only network mode: no external web access/requests

## Current Parent
- Conversation ID: 40002285-cf43-43d2-bfec-8fb0b0a1ea97
- Updated: 2026-07-15T08:19:16Z

## Investigation State
- **Explored paths**: Memory vault specs and reports, recent conversation logs, daily/hourly dataset scripts, custom exits analysis scripts, and backtest results.
- **Key findings**:
  1. Standalone `daily_macro_v2` model long leg is certified (`TRIGGER_GRADE`) but its edge (+28 bps) is lookahead-leaked because it uses Tuesday night US close data to decide to enter at Tuesday close. Entering at Wednesday close or open destroys the edge (losing the ON1 overnight drift of ~20.5 bps).
  2. "Open GAP-FADE" paired book has a strong simulated edge (+14.1 bps/day net@6), but it is heavily execution-constrained (requires pre-open auction fills; 5-min delay makes it gross-negative).
  3. Stacking gates on v20 and Nifty 2H filters suffered from extreme overfitting and collapsed out-of-sample (-23.65% PnL on true OOS).
  4. Regime router (v26) lookahead bias fix (shifting 100-DMA by 1 day) collapsed its edge from +27.44 bps to +2.38 bps.
  5. The entire dataset has survivorship bias due to using a static ticker list from `scripts/tickers.py`.
- **Unexplored areas**: None. We have scanned all relevant paths to evaluate candidate edges.

## Key Decisions Made
- Discarded overfitted gates on v20 and regime router due to true OOS failures.
- Identified and detailed lookahead leak in `daily_macro_v2` (due to feature/label timing mismatch) and execution constraints in the "Open GAP-FADE" edge.
- Set up a clear verification plan for the most viable candidate (Open GAP-FADE).

## Artifact Index
- `c:\Users\loq\Desktop\Trading\finalgo\.agents\teamwork_preview_explorer_edgesearch_2\handoff.md` — Structured findings, logic chain, caveats, and verification plan.
