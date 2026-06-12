---
title: "SQLite Database MCP Integration"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# SQLite Database MCP Integration

**Status**: Active Setup  
**Target File**: `c:/Users/loq/Desktop/Trading/finalgo/data/vanguard_trades.db`

## Configuration

To enable the Vanguard AI agent to query the trade database directly, add the following configuration block to your MCP client config (e.g., `mcp.json` or equivalent configuration file for your IDE):

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": [
        "mcp-server-sqlite",
        "--db-path",
        "c:/Users/loq/Desktop/Trading/finalgo/data/vanguard_trades.db"
      ]
    }
  }
}
```

## Agent Capabilities Enabled
Once active, the agent gains the ability to:
1. Instantly query the `trades`, `performance`, and `state` tables.
2. Audit Upstox P&L and slipped execution prices natively without writing Python wrappers.
3. Validate backend architecture changes immediately via SQL.

## Maintenance Notes
- **CRITICAL PREREQUISITE**: The official SQLite MCP server runs on Node.js. You MUST have Node.js installed on your Windows machine so the `npx` command works in your terminal. You can download it from [nodejs.org](https://nodejs.org/).
- If the `vanguard_trades.db` file is locked during live trading, the SQLite MCP will open it in read-only mode by default, which is safe and optimal for querying.
