---
title: "MCP Servers Registry"
type: reference
status: active
updated: 2026-06-12
tags: []
---
# MCP Servers Registry

This folder tracks the research, configuration, and integration status of Model Context Protocol (MCP) servers connected to the Vanguard AI agent. 

By integrating external MCPs, we enhance the agent's capabilities in securely querying databases, managing version control, analyzing data in isolated kernels, and deploying infrastructure.

## Integration Roadmap

### Phase 1: Local & Data Infrastructure
* **Target 1**: **Database Connector (SQLite)** - To query trade logs and historical tick data directly. *(Status: 🟢 Connected - [[05 — Operations/MCP/SQLite MCP Setup|Docs]])*
* **Target 2**: **Jupyter / Python Data Sandbox** - To securely backtest and plot model performance directly in the workspace. *(Status: Setting Up - [[05 — Operations/MCP/Jupyter MCP Setup|Docs]])*

### Phase 2: Workflow & Deployment
* **Target 3**: **Obsidian Memory Engine** - For semantic search and safe markdown edits. *(Status: ⏸️ Deferred (Using Native Filesystem) - [[05 — Operations/MCP/Obsidian MCP Setup|Research Docs]])*
* **Target 4**: **GitHub / Gitlab** - To manage branches, issues, and PRs natively. *(Status: Pending)*

## Current Active MCPs

| MCP Name | Primary Use Case | Configuration Link | Status |
| :--- | :--- | :--- | :--- |
| **TradingView** | Read and control live TradingView Desktop charts via CDP. | [[07. MCP Integrations/TradingView MCP Setup\|Docs]] | 🟡 Pending Setup |

---

> [!TIP]
> When integrating a new MCP, create a separate documentation file in this folder (e.g., `SQLite MCP Setup.md`) detailing the exact connection strings, required environment variables, and testing procedures. Link it back to this registry.
