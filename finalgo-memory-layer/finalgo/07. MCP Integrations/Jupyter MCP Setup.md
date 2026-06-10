# Jupyter / Python Data Sandbox MCP Integration

**Status**: Active Setup  
**Target Environment**: Local Python / Jupyter Environment

## Configuration

We have removed all wrapper scripts and environment variable hacks. The clean configuration uses standard `uvx` to launch the MCP bridge, and we disable token authentication on the local Jupyter server to allow seamless connection.

### Step 1: Update your MCP Config
Add this clean block to your `mcp_config.json` (located at `C:\Users\loq\.gemini\config\mcp_config.json`):

```json
{
  "mcpServers": {
    "jupyter": {
      "command": "uvx",
      "args": [
        "mcp-jupyter"
      ]
    }
  }
}
```

### Step 2: Launch the Jupyter Server Locally
Before asking the AI to run any notebook tasks, you must have the local Jupyter backend running on port 8888. Because the MCP bridge connects over localhost, we can safely disable the token to bypass authentication errors.

Run this command in a separate terminal from your project root:
```powershell
env\Scripts\jupyter.exe notebook --port=8888 --no-browser --ServerApp.token="" --ServerApp.password="" --ServerApp.disable_check_xsrf=True --ServerApp.allow_origin="*"
```

*(If you prefer to use your global `uvx`, you can run `uvx --from jupyter-core jupyter notebook --port=8888 --no-browser --ServerApp.token="" --ServerApp.password="" --ServerApp.disable_check_xsrf=True --ServerApp.allow_origin="*"` instead).*

### Step 3: Agent Initialization
Once your config is updated, **reload the IDE**. 
When the IDE connects, the AI will now natively have the Jupyter tools available (`setup_notebook`, `execute_notebook_code`, etc.). The agent will call `setup_notebook` targeting `http://127.0.0.1:8888` to verify the connection.
