import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).parents[1]
CONFIG_PATH = BASE_DIR / "config"
MCP_TOOLS_YML_PATH = CONFIG_PATH / "mcp_tools.yml"
MCP_TOOLS_TOML_PATH = CONFIG_PATH / "mcp_tools.toml"
ENV_PATH = CONFIG_PATH / ".env"
ASSETS_PATH = BASE_DIR / "assets"
TERADATA_LOGO_PATH = ASSETS_PATH / "td_new_trans.png"
SYSTEM_PROMPT = """Use the database called {database_name}"""

MCP_CONFIG = {
    "mcpServers": {
        "teradata": {
        "command": "uvx",
        "args": ["teradata-mcp-server"],
        "env": {
            # "DATABASE_URI": "",
            "MCP_TRANSPORT": "stdio"
        }
        }
    }
}

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}