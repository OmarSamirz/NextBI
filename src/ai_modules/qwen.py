import torch
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain_ollama import ChatOllama

import os
from typing import Any

from ai_modules.base import AI, Message
from constants import MCP_CONFIG, ENV_PATH, SYSTEM_PROMPT

load_dotenv(ENV_PATH)


class AIQwen(AI):
    """Concrete AI implementation using OpenAI Chat Completions models.

    Configuration is sourced from :func:`config.get_openai_config` which returns
    the API key, model name (e.g., ``gpt-4o``), and a configured OpenAI client.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self.mcp_config = MCP_CONFIG.copy()
        self.system_prompt = SYSTEM_PROMPT.format(database_name=os.getenv("TD_NAME"))
        self.mcp_config["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = os.getenv("DATABASE_URI")
        self.llm = ChatOllama(model="qwen3:8b")
        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.agent = MCPAgent(llm=self.llm, client=self.client, max_steps=30, system_prompt=self.system_prompt)

    async def generate_reply(self, messages: list[Message], context: dict | None = None) -> str:
        """Generate an assistant reply using OpenAI Chat Completions.

        Notes:
        - Unknown roles are coerced to "user".
        - App's internal role "ai" is translated to OpenAI's "assistant".
        - Empty/whitespace-only contents are skipped.
        - Retries up to 3 times on exceptions with backoff (0.5s, 1s).
        """
        msg = messages[-1].get("content", "")

        return await self.agent.run(msg)