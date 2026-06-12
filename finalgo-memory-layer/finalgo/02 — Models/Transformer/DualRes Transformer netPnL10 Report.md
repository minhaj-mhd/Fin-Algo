# 📊 DualRes Transformer — netPnL@10 Consolidated Report

> Model: `artifacts/dualres_transformer_netpnl.pt` (cost-aware net-PnL objective, cost_bps=10, 509,569 params).
> Architecture: [[02 — Models/Transformer/DualRes Transformer Flowchart]] · Build log: [[06 — Logs/Conversations/Conv-2026-06-12-Sophisticated-Transformer]]
> ⚠️ EXPLORATORY — no Gauntlet verdict. Date: 2026-06-12.

## 1 · Standalone (own predictions, held-out TEST, 90,590 samples, OOS last 15%)
| Metric | Value |
|---|---|
| AUC | 0.5127 |
| Accuracy | 51.4% (base 49.3%) |
| deploy (avg \|position\|) | **0.0025** → abstains as a standalone trader |
| netPnL@10 | −0.022 bps (flat) |

Own Top-K self-ranked net (bps):
| | K1 long | K1 short | K3 long | K3 short | K5 long | K5 short |
|---|---|---|---|---|---|---|
| @6bps  | −3.6 | +0.8 | −1.6 | −2.8 | −1.3 | −3.7 |
| @10bps | −7.6 | −3.2 | −5.6 | −6.8 | −5.3 | −7.7 |

Sub-cost everywhere → a FILTER, not a standalone trader.

## 2 · As a veto on v10 XGBoost — walk-forward (OOS 2025-08..2026-05, 104,272 rows, veto th=0.5)
Net bps (t-stat). v10 picks Top-K, transformer vetoes picks it disagrees with.
| K / cost | LONG | SHORT |
|---|---|---|
| K1 @6   | n=186  −5.9 (−1.3)  | n=493  **+5.4 (1.45)** |
| K1 @10  |        −9.9 (−2.1) |        +1.4 (0.37) |
| K1 @20  |       −19.9 (−4.3) |        −8.6 (−2.3) |
| K3 @6   | n=502  −4.5 (−1.6) | n=1491 −1.2 (−0.5) |
| K3 @10  |        −8.5 (−3.1) |        −5.2 (−2.4) |
| K5 @6   | n=787  −2.3 (−1.1) | n=2466 −1.4 (−0.9) |
| K5 @10  |        −6.3 (−2.9) |        −5.4 (−3.4) |

vs v10-alone: improved net in EVERY cell (loss reduction), strongest on SHORT; cuts ~70% of (dead) longs.
Comparison baseline v10-alone @10: K1 SHORT −1.1, K3 SHORT −8.1, K5 SHORT −7.3.

## 3 · Threshold-sweep fragility (⚠️ NOT a result)
Tightening short veto (P_up<0.45, Top-3) → 15 trades/9 days, +73 raw / +53 net@20, t=2.44 —
but 89% of return from 5 trades, 48% from one day (2026-05-29 crash). Fat-tail luck, cherry-picked
from a 10-config sweep. Untradeable.

## Verdict
- **Filter-grade = yes**, SHORT side only, as a loss-reducer on v10. Best operating point: K1 short, low cost (+5.4 @6bps).
- **NOT significantly net-positive @10bps** (best +1.4, t=0.37 = noise). Longs dead regardless.
- The BCE transformer (`dualres_transformer.pt`) is NOT filter-grade (probs ~0.5, vetoes ~nothing);
  the netPnL@20 model abstains entirely (deploy=0) → too aggressive.
- To deploy: pre-register the veto threshold (do NOT tune on test) and validate via the Validation Gauntlet.
