# 💬 Conversation Context: Open GAP-FADE Edge Verification

## 📌 Metadata
- **Conversation ID**: e7bebb94-8396-4c05-b1d6-36c4f73874f7
- **Start Date**: 2026-07-15
- **Status**: 🔴 Concluded
- **Focus Area**: Trading Strategies

## 🎯 Objectives
- [x] Calculate Holdout EV at 10.0 bps cost
- [x] Calculate Holdout EV at 15.0 bps cost
- [x] Determine break-even cost where Holdout EV turns negative
- [x] Run randomized negative control (10-20 runs with different seeds) to compute mean Holdout EV at 6.0 bps cost

## 💻 Active Code Files Modified
- [stress_test_edge.py](file:///c:/Users/loq/Desktop/Trading/finalgo/research/edge_search/stress_test_edge.py)

## 📝 Compacted Session Log
- **Initial Analysis**: Started the Open GAP-FADE edge verification task. Initialized ORIGINAL_REQUEST.md and BRIEFING.md. Ran the baseline reproducible edge report.
- **Cost Sensitivity**: Verified that Net EV on Holdout is +8.9625 bps at 10.0 bps cost, +3.9625 bps at 15.0 bps cost, and the break-even cost is exactly 18.9625 bps.
- **Negative Control**: Confirmed that random selection results in expected Net EV of -6.0 bps after 6.0 bps cost, proving that random ticker selection does not yield a positive edge.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[03 — Strategies/Market Friction & Slippage]]
