---
title: "Obsidian MCP Integration"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# Obsidian MCP Integration

**Status**: Active Setup  
**Target Environment**: `finalgo-memory-layer/finalgo` (Obsidian Vault)

## Configuration Steps

Unlike the SQLite or Jupyter MCPs, the Obsidian MCP requires a secure bridge to your Obsidian application. 

### Step 1: Install the Local REST API Plugin in Obsidian
1. Open your Obsidian vault (`finalgo-memory-layer/finalgo`).
2. Go to **Settings > Community Plugins** and disable "Safe Mode" if it's on.
3. Click **Browse**, search for **"Local REST API"** (by coddingtonbear), and install it.
4. Enable the plugin.
5. In the plugin settings, you will see an **API Key** (a long string of characters). Copy this key!

### Step 2: Update your MCP Config
Once you have the API key, add the `obsidian` block to your `mcp_config.json`. **Be sure to replace `<YOUR_API_KEY>` with the key you just copied.**

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
    },
    "obsidian": {
      "command": "npx",
      "args": [
        "-y",
        "obsidian-mcp-server@latest"
      ],
      "env": {
        "OBSIDIAN_API_KEY": "<YOUR_API_KEY>"
      }
    }
  }
}
```

## Agent Capabilities Enabled
Once active, the agent gains the ability to:
1. Semantically search your entire vault for concepts, tags, and strategies.
2. Safely read and update markdown files without breaking YAML frontmatter or Obsidian graph links.
3. Automatically log conversation summaries to your Daily Logs using Obsidian's native API.

---

> [!IMPORTANT]
> **Implementation Deferred (June 2026)**
> This integration is currently on hold. Because the Obsidian vault resides directly within the `finalgo` workspace, the Vanguard AI agent can natively read and search markdown files using `grep_search` and `view_file`. 
> 
> Relying on native tools is significantly faster (zero HTTP overhead) and far more token-efficient than routing through the Local REST API plugin. If the vault is ever moved outside the workspace, or if advanced vector-backed semantic search is required (e.g., via `mcp-server-memory` or Smart Connections), this setup should be resumed.
