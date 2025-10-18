from pathlib import Path
from dotenv import load_dotenv

import os

BASE_DIR = Path(__file__).parents[1]

CONFIG_PATH = BASE_DIR / "config"

ENV_PATH = CONFIG_PATH / ".env"

DEFAULT_SYSTEM_PROMPT = CONFIG_PATH / "system_prompt_banking.txt"

PLOT_AGENT_SYSTEM_PROMPT_PATH = CONFIG_PATH / "plot_agent_system_prompt.txt"

TERADATA_AGENT_SYSTEM_PROMPT_PATH = CONFIG_PATH / "teradata_agent_system_prompt.txt"

MANAGER_AGENT_SYSTEM_PROMPT_PATH = CONFIG_PATH / "manager_agent_system_prompt.txt"

ASSETS_PATH = BASE_DIR / "assets"

TERADATA_LOGO_PATH = ASSETS_PATH / "td_new_trans.png"

LANGGRAPH_GRPAH_IMAGE_PATH = ASSETS_PATH / "langgraph_graph.png"

CHARTS_PATH = BASE_DIR / "charts"

load_dotenv(ENV_PATH)

MCP_CONFIG = {
    "mcpServers": {
        "teradata": {
            "command": "uvx",
            "args": ["teradata-mcp-server"],
            "env": {
                "TD_NAME": os.getenv("TD_NAME"),
                "TD_HOST": os.getenv("TD_HOST"),
                "TD_USER": os.getenv("TD_USER"),
                "TD_PASSWORD": os.getenv("TD_PASSWORD"),
                "TD_PORT": os.getenv("TD_PORT"),
                "MCP_TRANSPORT": os.getenv("MCP_TRANSPORT"),
                "DATABASE_URI": os.getenv("DATABASE_URI"),
            }
        }
    }
}