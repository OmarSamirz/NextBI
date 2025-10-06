import torch
from pathlib import Path

from string import Template

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).parents[1]

CONFIG_PATH = BASE_DIR / "config"

MCP_TOOLS_YML_PATH = CONFIG_PATH / "mcp_tools.yml"

MCP_TOOLS_TOML_PATH = CONFIG_PATH / "mcp_tools.toml"

ENV_PATH = CONFIG_PATH / ".env"

ASSETS_PATH = BASE_DIR / "assets"

TERADATA_LOGO_PATH = ASSETS_PATH / "td_new_trans.png"

CHARTS_PATH = BASE_DIR / "charts"

SYSTEM_PROMPT = Template("""
You must use the database called $database_name in every query. 

NEVER use or mention the table named `user_query` in any context — not in outputs, not in intermediate steps, and not for reasoning. 
It is a log table and must be completely ignored. 
If a query or visualization might involve `user_query`, you must exclude it entirely and focus on other relevant tables instead.

When creating visualizations with matplotlib:
1. Always use matplotlib.use('Agg') before importing pyplot
2. Save plots using plt.savefig() instead of plt.show()
3. Save the plot in the following directory $charts_path
4. Close figures with plt.close() after saving
5. Never print or return the file path of the saved image.
6. After saving, provide a clear, concise explanation of what the chart represents.
""")

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

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}