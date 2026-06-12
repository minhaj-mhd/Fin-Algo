---
title: "Final Conclusion: Unsupervised Clustering for Veto Override — Research Closed"
type: reference
status: dead
updated: 2026-06-12
tags: []
---
# Final Conclusion: Unsupervised Clustering for Veto Override — Research Closed

**Status**: ❌ **NEGATIVE RESULT — DO NOT IMPLEMENT**  
**Research Period**: May 12 – June 3, 2026  
**Dataset**: 1,466 trades across 15 trading days (~48 trades/day)  
**Verdict**: The unsupervised clustering approach does not produce a reliable, deployable edge for overriding AI veto decisions. This research line is closed.

---

## 1. What We Were Trying To Do

The Vanguard trading system uses a two-stage Gemini AI audit (`ai_veto.py`) to filter incoming trade signals. The veto rate is high — roughly 90% of signals are blocked. The hypothesis was:

> *Some vetoed trades would have been profitable. If we cluster historically winning trades by their mathematical features, we can identify "sweet spots" — regions in feature space where trades consistently win. When a new vetoed trade falls geometrically close to a sweet spot centroid, we override the veto and take the trade.*

This would turn the veto system from a binary gate into a probabilistic filter, rescuing high-conviction winners that the AI conservatively blocks.

---

## 2. What We Built and Tested

### Phase 1: Initial Clustering (In-Sample Only)
- Extracted 138 high-return winners (`final_profit_pct > 0.5%`) from all 1,466 trades.
- Features: `tech_score`, `nlp_sentiment`, `tv_sentiment`, `one_hour_prob`.
- Fit KMeans (K=4) on StandardScaler-normalized winners.
- Hierarchically sub-clustered each primary cluster (K=3 for clusters 0,1,2; K=5 for cluster 3).
- Swept distance thresholds (0.50, 0.75, 1.00, 1.50) and applied quality gates (WR ≥ 60%, Avg PnL ≥ 0.10%, min 5 trades).
- **In-sample results looked spectacular**: Multiple sub-clusters showed 60-83% win rates with +0.10% to +0.50% average PnL. The "Master Veto Override Table" identified ~221 historical trades with aggregate WR safely above 60%.

### Phase 2: Out-of-Sample Validation (Chronological Split)
- **Train**: First 1,000 trades (May 12 – May 25).
- **Test**: Last 466 trades (May 25 – June 3).
- Two evaluation paths:
  - **Path 1 (Fixed Researched Sweet Spots)**: Rebuild the exact sub-cluster configurations from Phase 1 on the training set.
  - **Path 2 (Dynamically Discovered Sweet Spots)**: Let the algorithm discover sweet spots fresh from training data using quality gates.

| Path | OOS Trades Matched | OOS Win Rate | OOS Avg PnL | OOS Leveraged Return |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (All Test Trades)** | 466 | 46.57% | -0.0294% | — |
| **Path 1: Fixed** | 51 | 43.14% | **-0.1545%** | **-42.47%** |
| **Path 2: Dynamic** | 23 | 52.17% | **-0.0040%** | **-1.84%** |

Both paths failed. Path 1 was catastrophic. Path 2 was flat-to-negative.

The failure was traced to **Primary Cluster 3**, which mixed LONG contrarian trades (profitable archetype) with SHORT momentum trades (losing archetype) under the same centroid. In-sample, the winners dominated by volume. Out-of-sample, the losers showed up and the cluster collapsed: 37 trades caught at 37.8% WR and -0.2121% Avg PnL, producing -41.46% leveraged return from that single cluster alone.

Two individual sub-clusters survived:
- **Cluster 1 Sub 1**: 11 trades, 63.6% WR, +0.0881% PnL, +4.18% return
- **Cluster 2 Sub 0**: 8 trades, 62.5% WR, +0.1342% PnL, +4.89% return

But 11 and 8 trades over 9 days is **not** a statistically significant sample. These could easily be noise.

### Phase 3: Walk-Forward Validation (Rolling Window)
- **Design**: 8-day rolling training window, test on next 1 day. 7 independent test splits.
- This simulates the intended production deployment: daily retraining on a rolling 30-day (here 8-day) lookback.

| Path | WF Trades Matched | WF Win Rate | WF Avg PnL | WF Leveraged Return |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** | 620 | 48.06% | -0.0284% | — |
| **Path 1: Fixed** | 54 | 53.70% | +0.0401% | **+7.58%** |
| **Path 2: Dynamic** | 78 | 51.28% | -0.0496% | **-24.03%** |

The +7.58% from Path 1 appears positive, but decomposition reveals it is not robust:

| Split | Test Date | Fixed Overrides | Fixed WR | Fixed Net Return |
| :---: | :--- | :---: | :---: | :---: |
| 1 | 2026-05-25 | 41 | 58.5% | **+7.41%** |
| 2 | 2026-05-26 | 11 | 36.4% | -2.29% |
| 3 | 2026-05-27 | 1 | 0.0% | -1.29% |
| 4 | 2026-05-29 | 1 | 100.0% | +3.76% |
| 5 | 2026-06-01 | 0 | — | 0.00% |
| 6 | 2026-06-02 | 0 | — | 0.00% |
| 7 | 2026-06-03 | 0 | — | 0.00% |

- The entire cumulative return is driven by **Split 1** (+7.41%) and a **single lucky trade** in Split 4 (+3.76%).
- **3 out of 7 splits (43%) produced zero overrides** — the model was completely inert.
- On the 4 splits where overrides fired, 2 were net negative.

### Phase 4: Root Cause — Data Discontinuity
Investigation revealed that `nlp_sentiment` and `tv_sentiment` are **100% missing** in the database for June 2nd and 3rd. When the scaler (fit on enriched data) encounters missing-filled-with-zero values, it produces extreme z-scores that push those trades far from any centroid. The model silently stops functioning whenever a data feed drops — which is an unacceptable fragility for a live system.

---

## 3. Why This Approach Fundamentally Does Not Work

### 3.1 The Feature Space is Too Thin
Four features (`tech_score`, `nlp_sentiment`, `tv_sentiment`, `one_hour_prob`) do not contain enough information to separate future winners from future losers. These features describe the *state at entry* — but whether a trade wins or loses depends overwhelmingly on what happens *after* entry: subsequent price action, news, sector rotation, market microstructure. Entry-time features are a poor predictor of exit-time outcomes, especially for a system that already uses them in its signal generation pipeline.

### 3.2 KMeans Creates Arbitrary Boundaries in Continuous Space
KMeans partitions feature space using Voronoi cells — hard geometric boundaries that have no relationship to the underlying market dynamics. Two trades 0.01 apart in Euclidean distance can land in different clusters with completely different performance profiles. The centroids shift unpredictably every time the training window moves by even a single day, because the clusters aren't capturing a real, stable structure — they're fitting to noise.

### 3.3 The "Sweet Spots" Are Survivorship Artifacts
The quality gates (WR ≥ 60%, PnL ≥ 0.10%, ≥ 5 trades) sound rigorous but are actually insufficient. With 14 sub-clusters × 4 distance thresholds = 56 combinations tested per training window, finding a few that pass WR ≥ 60% by chance is statistically expected even under a pure random process. This is a textbook case of **multiple comparisons bias**: test enough configurations and some will look good in-sample purely by luck. The out-of-sample collapse confirms this.

### 3.4 The Veto System Already Uses These Features
The Gemini AI audit in `ai_veto.py` already considers technical scores, sentiment, and probability estimates — at a far deeper level than a 4-dimensional Euclidean distance check. Attempting to "override" a sophisticated multi-factor LLM-based evaluation with a simple geometric proximity test is fundamentally mismatched. If the AI vetoed a trade, it did so having already considered these features in a richer context (hourly returns, RVOL, dollar volume, 52-week positioning, etc.). A clustering model that uses a strict subset of the same inputs cannot reliably second-guess a model that uses a superset.

### 3.5 Data Infrastructure Cannot Support It
Even if the methodology were sound, the data pipeline is too fragile. Two out of four clustering features (`nlp_sentiment`, `tv_sentiment`) have fill rates below 90%, and both dropped to 0% for the most recent trading days. A live override engine that silently stops working when a data feed goes down — without alerting anyone — is a production hazard.

### 3.6 Sample Sizes Are Insufficient for Confidence
With ~48 trades per day and 15 trading days, we have 1,466 total observations. After filtering for winners (>0.5%), we have only ~80-100 per training window. After primary clustering into 4 groups and sub-clustering into 3-5 each, individual sub-clusters contain 3-20 winners. At these sample sizes, any performance metric has enormous standard error. A sub-cluster with 8 trades at 62.5% WR has a 95% confidence interval of roughly [25%, 92%]. We cannot distinguish signal from noise.

---

## 4. What Would Need to Be True for This to Work

For a clustering-based veto override to be viable, we would need:

1. **A much larger dataset**: At minimum 10,000+ trades across 6+ months to get statistically meaningful sub-cluster sizes.
2. **Features the veto system doesn't already use**: Novel inputs like intraday order flow, options market signals, cross-sector momentum, or macro regime indicators — information orthogonal to what the AI already evaluates.
3. **A non-Euclidean similarity metric**: Markets don't respect Euclidean distance in feature space. A learned similarity function (e.g., from a supervised model) would be needed, but that becomes a supervised learning problem, not unsupervised clustering.
4. **Stable data infrastructure**: All features must have 100% fill rates with redundant data sources and alerting for feed failures.

None of these conditions are met today.

---

## 5. Broader Implications for the Vanguard System

This negative result carries an important lesson: **the edge is in the veto system, not in overriding it**.

The AI veto exists because taking every signal is net negative. The veto improves performance by filtering out bad trades. Trying to re-admit filtered trades using a weaker model is working against the system's own intelligence. The productive research directions are:

1. **Improve the veto system itself**: Make the AI audit smarter, not try to circumvent it.
2. **Improve trade management**: Better exits (dynamic stop-losses, trailing stops, time-based exits) can improve the profitability of trades that do pass the veto.
3. **Improve signal quality upstream**: Better entry signals mean fewer vetoes needed and higher base win rates.
4. **Regime-aware filtering**: Rather than overriding vetoes trade-by-trade, adjust the veto sensitivity based on broad market regime (trending vs. mean-reverting, high vs. low volatility).

---

## 6. Research Artifacts Produced

All files in this directory are preserved for reference:

| Document | Purpose |
| :--- | :--- |
| [Deep_Dive_High_Return_Clusters.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Deep_Dive_High_Return_Clusters.md) | Initial in-sample clustering with hierarchical sub-clustering |
| [Master_Veto_Override_Table.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Master_Veto_Override_Table.md) | Consolidated sweet spot table (in-sample only — do not use) |
| [Feature_Definitions.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Feature_Definitions.md) | Feature definitions and mappings |
| [Feature_Set_Analysis.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Feature_Set_Analysis.md) | Full database schema and fill rate analysis |
| [Unsupervised_Vetoed_Trades_Analysis.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Unsupervised_Vetoed_Trades_Analysis.md) | Early-stage clustering of vetoed trades by archetype |
| [Out_of_Sample_Validation_Report.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Out_of_Sample_Validation_Report.md) | Chronological train/test split results |
| [Walk_Forward_Validation_Report.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Walk_Forward_Validation_Report.md) | Rolling walk-forward validation results |
| [Live_Implementation_Mechanics.md](file:///c:/Users/loq/Desktop/Trading/finalgo/finalgo-memory-layer/finalgo/07.%20Cluster%20Research/Live_Implementation_Mechanics.md) | Proposed implementation mechanics (not deployed) |

Scripts produced (preserved in `scripts/`):
- `out_of_sample_validation.py` — Chronological validation framework
- `walk_forward_validation.py` — Rolling walk-forward backtest
- `cluster_deep_dive.py`, `subcluster_all.py`, `aggregate_sweet_spots.py`, `optimize_cluster3.py` — Research scripts
- `backtest_today.py` — Daily backtest runner

Models saved (should be deleted or archived):
- `models/high_return_clusters/scaler.pkl` — StandardScaler fit on all data (leaks future)
- `models/high_return_clusters/kmeans.pkl` — KMeans fit on all data (leaks future)

---

## 7. Final Statement

We invested significant research effort into this approach. The in-sample results were genuinely exciting — 60-83% win rates, clear cluster archetypes, elegant geometric intuition. But rigorous out-of-sample testing revealed what in-sample metrics always hide: the patterns were noise dressed up as signal.

This is not a failure. This is science working correctly. We formed a hypothesis, designed experiments, tested rigorously, and obtained a clear negative result. The system is better for knowing definitively that this path doesn't work, rather than deploying an untested override engine that would have lost real money.

**Research line: CLOSED.**  
**Recommendation: Do not revisit unless dataset grows to 10,000+ trades with novel orthogonal features.**
