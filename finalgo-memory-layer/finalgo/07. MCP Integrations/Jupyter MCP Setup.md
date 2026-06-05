# Jupyter / Python Data Sandbox MCP Integration

**Status**: Active Setup  
**Target Environment**: Local Python / Jupyter Environment

## Configuration

To enable the Vanguard AI agent to natively execute Python code, run machine learning backtests, and generate plots within the chat, add the `jupyter` block to your existing `mcp_config.json`.

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@berthojoris/mcp-sqlite-server",
        "sqlite:////c:/Users/loq/Desktop/Trading/finalgo/data/vanguard_trades.db"
      ]
    },
    "jupyter": {
      "command": "uvx",
      "args": [
        "mcp-jupyter"
      ]
    }
  }
}
```

## Agent Capabilities Enabled
Once active, the agent gains the ability to:
1. Natively spin up a secure Python kernel.
2. Load XGBoost models and `pandas` dataframes into memory without flooding standard output.
3. Generate and return interactive plots (e.g., drawdown curves for the 50-strategy regime backtests) directly in the UI.

## Maintenance Notes
- **CRITICAL PREREQUISITE**: The `uvx` command requires you to have the **`uv` package manager** installed. It is the modern, lightning-fast Python toolchain.
- If you don't have `uv` installed, you can install it via PowerShell: `irm https://astral.sh/uv/install.ps1 | iex`
