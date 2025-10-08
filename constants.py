import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).parents[1]

CONFIG_PATH = BASE_DIR

ENV_PATH = CONFIG_PATH / ".env"

SYSTEM_PROMPT_PATH = CONFIG_PATH / "system_prompt.txt"

ASSETS_PATH = BASE_DIR / "assets"

TERADATA_LOGO_PATH = ASSETS_PATH / "td_new_trans.png"

CHARTS_PATH = BASE_DIR / "charts"

MCP_CONFIG = {
    "mcpServers": {
        "teradata": {
            "command": "uvx",
            "args": ["teradata-mcp-server"],
            "env": {
                "MCP_TRANSPORT": "stdio"
            }
        },
    }
}