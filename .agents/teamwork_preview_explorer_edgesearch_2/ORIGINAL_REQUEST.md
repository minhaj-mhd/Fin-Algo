## 2026-07-15T08:15:24Z

You are a teamwork_preview_explorer.
Your working directory is: c:\Users\loq\Desktop\Trading\finalgo\.agents\teamwork_preview_explorer_edgesearch_2\
Your conversation ID is your own subagent ID.

### Objective
Conduct a broad analysis of the existing features, data, and models in finalgo to identify a candidate trading edge that:
1. Is small but genuine.
2. Is tradable.
3. Has not been invalidated (cross-check against Dead-Ends Register and recent project findings).
4. Outputs a reproducible statistical report showing positive Expected Value (EV) on a holdout set.

### Detailed Steps
1. Scan the memory layer files to see what edges are promising or active, and which ones are dead ends:
   - Read `finalgo-memory-layer/finalgo/00 — Start Here/Ray of Hope.md` to see positive/promising edges already identified.
   - Read `finalgo-memory-layer/finalgo/00 — Start Here/Dead-Ends Register.md` to see what has been invalidated.
   - Read recent conversations (e.g. `finalgo-memory-layer/finalgo/06 — Logs/Conversations/Conv-2026-07-12-OOS-Data-Fix-and-Overfitting-Discovery.md` or similar files) to understand recent overfitting findings.

2. Explore the available datasets under `data/`:
   - Inspect files such as `ranking_data_upstox_1h_3y_clean.csv`, `vanguard_trades.db`, `ranking_data_daily_macro_v2.csv`, and any other parquet or csv datasets.
   - Use the SQLite MCP server (or appropriate queries) to look at `data/vanguard_trades.db` structure and summary stats if helpful.

3. Identify a promising candidate edge. For example:
   - "G* HOURxSIDE GATE @6bps" or "LONG NIFTY index gate" mentioned in Active Board.
   - "OPEN GAP-FADE" edge mentioned in Active Board (shorting top-K gap-ups at 09:15 open and longing bot-K gap-downs).
   - "OVERNIGHT-EXIT layer on daily_macro_v2" edge.
   - Determine which candidate has the cleanest data, is least prone to overfitting/mining, and can be evaluated on a holdout set.

4. Formulate the verification plan:
   - Specify what data we will use for the development set vs the holdout set.
   - For example: if we use an 11-month panel, which part is development (e.g. Aug 2025 - May 2026) and which part is holdout/OOS (e.g. June 2026 - July 2026, or a fresh slice of July 2026 data).
   - Define the transaction fee and slippage assumptions (e.g. 10bps cost per trade, or dynamic slippage based on Upstox fees).

5. Document all candidate edges and your recommendations in `handoff.md` inside your working directory.

### Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
