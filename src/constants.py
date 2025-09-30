import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).parents[1]

CONFIG_PATH = BASE_DIR / "config"

ENV_PATH = CONFIG_PATH / ".env"

DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}