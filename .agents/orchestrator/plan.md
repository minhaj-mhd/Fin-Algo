# Project Plan: Edge Search and Verification

## Architecture
- Working directory: `research/edge_search/`
- Data Sources: Existing finalgo historical files (parquet/csv/db)
- Metrics: Expected Value (EV) on Holdout set after transaction costs/slippage
- Integrity Mode: `development`

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Workspace Setup | Initialize memory logs and active board | None | DONE |
| 2 | Broad Exploration | Analyze existing features/models & Ray of Hope for candidate edges | M1 | DONE |
| 3 | Edge Selection | Select candidate edge & define holdout split | M2 | DONE |
| 4 | Implementation | Write reproducible Python/Jupyter script to calculate EV | M3 | DONE |
| 5 | Review & Stress Test | Review code, check for leakage, stress-test the edge | M4 | DONE |
| 6 | Forensic Audit | Verify integrity via Forensic Auditor | M5 | DONE |
| 7 | Handoff & Memory | Update memory vault and finalize report | M6 | DONE |

## Interface Contracts
- The reproducible script must run in `research/edge_search/` environment.
- Input: Designated holdout dataset (e.g. parquet file or SQL database query).
- Output: Print to stdout of:
  - Total Trades
  - Win Rate
  - Gross Return
  - Net Return (after transaction fees/slippage)
  - Calculated Expected Value (EV) per trade
