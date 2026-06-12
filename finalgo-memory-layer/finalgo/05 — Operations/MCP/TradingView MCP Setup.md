# TradingView MCP Setup

This document outlines the setup and connection procedures for integrating the **TradingView MCP** (`tradesdontlie/tradingview-mcp`) with our AI agent workspace. This MCP leverages the Chrome DevTools Protocol (CDP) to interface directly with the TradingView Desktop application, allowing the agent to read live chart data, indicators, and control the UI.

## Setup Instructions

### 1. Download the Source Code
The repository has been cloned locally into `finalgo/mcp_servers/tradingview-mcp`.

### 2. Install Dependencies
Navigate to the directory and install the necessary Node.js packages:
```powershell
cd C:\Users\loq\Desktop\Trading\finalgo\mcp_servers\tradingview-mcp
npm install
```

### 3. Launch TradingView in Debug Mode
For the MCP server to interact with TradingView Desktop, the application must be launched with remote debugging enabled. Run the following command in PowerShell:
```powershell
Start-Process "TradingView.exe" -ArgumentList "--remote-debugging-port=9222"
```

### 4. Configure Your AI Assistant's MCP Settings
Depending on which AI client you are using (Claude Desktop, Cursor, etc.), add the following to your MCP configuration file (`claude_desktop_config.json` or equivalent):

```json
{
  "mcpServers": {
    "tradingview": {
      "command": "node",
      "args": ["C:/Users/loq/Desktop/Trading/finalgo/mcp_servers/tradingview-mcp/build/index.js"]
    }
  }
}
```
*Note: Make sure to run `npm run build` if the project uses TypeScript to generate the `build/index.js` file, or adjust the path to the correct entry script (`src/server.js` etc).*

## Security & Constraints
- **Warning**: Using CDP gives the MCP server root-level control over the TradingView application interface.
- Ensure the TradingView subscription plan is active to receive real-time intraday data.
- The use of this MCP might conflict with TradingView's official Terms of Service if used for high-frequency or automated scraping purposes.

## Linked Resources
- Part of the [[05 — Operations/MCP/MCP Registry]]
