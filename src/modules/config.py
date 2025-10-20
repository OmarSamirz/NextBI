from pathlib import Path
from dotenv import load_dotenv

import os
from typing import Optional

from constants import ENV_PATH


class Config:
    """Strict loader for environment-backed settings from ``config/.env``.

    - Requires config/.env to exist, else raises FileNotFoundError.
    - Uses python-dotenv to load variables into the process environment.
    """

    def __init__(self, env_path: Path, log_enabled: bool, log_file: Path) -> None:
        self.env_path = env_path
        self.log_enabled = log_enabled
        self.log_file = log_file

    @staticmethod
    def _env_bool(name: str, default: str = "false") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> "Config":
        if not ENV_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {ENV_PATH}")
        # Load .env, but do not override existing environment variables by default
        # so that OS/envvars (e.g., pytest monkeypatch) take precedence.
        load_dotenv(dotenv_path=ENV_PATH, override=False)

        log_enabled = cls._env_bool("LOG_ENABLED", "false")
        log_file = Path(os.getenv("LOG_FILE", "logs/log.txt").strip())
        return cls(env_path=ENV_PATH, log_enabled=log_enabled, log_file=log_file)