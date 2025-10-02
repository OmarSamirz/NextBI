"""OpenAI GPT backend.

Responsibilities:
- Initialize an OpenAI client from env via config.get_openai_config().
- Map app roles ("user"/"ai"/"system") to OpenAI roles ("user"/"assistant"/"system").
- Call chat.completions.create and return the assistant's content.
- Emit lightweight events for diagnostics (init, call, call.error).
- Retry transient failures with simple exponential backoff.
"""
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from langchain_openai import ChatOpenAI

import os
from typing import Any

from constants import MCP_CONFIG
from ai_modules.base import AI, Message
from modules.config import get_openai_config

load_dotenv()


class AIGPT(AI):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = get_openai_config()
        self.api_key = cfg["api_key"]
        self.model = cfg["model"]
        self.mcp_config = MCP_CONFIG.copy()
        self.mcp_config["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = os.getenv("DATABASE_URI")
        self.default_database = os.getenv("DEFAULT_DATABASE", "BANK_DB")
        self.llm = ChatOpenAI(model=self.model)
        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.agent = MCPAgent(llm=self.llm, client=self.client, max_steps=30)

    async def generate_reply(self, messages: list[Message], context: dict | None = None) -> str:
        user_msg = messages[-1].get("content", "")
        enhanced_prompt = f"Use the database '{self.default_database}' for all queries. {user_msg}"
        all_steps = []
        final_output = ""
        async for item in self.agent.stream(enhanced_prompt):
            if isinstance(item, tuple) and len(item) == 2:
                action, observation = item
                all_steps.append({
                    "action": action,
                    "observation": observation
                })
            elif isinstance(item, str):
                final_output = item
        formatted_response = ""
        if all_steps:
            formatted_response += "**Execution Steps:**\n\n"
            for i, step in enumerate(all_steps, 1):
                action = step["action"]
                observation = step["observation"]
                formatted_response += f"**Step {i}:**\n"
                formatted_response += f"- **Tool:** `{action.tool}`\n"
                formatted_response += f"- **Input:** `{action.tool_input}`\n"
                formatted_response += f"- **Observation:** {observation}\n\n"
            formatted_response += "---\n\n"
        formatted_response += "**Final Result:**\n" + (final_output or "[No output received]")
        return formatted_response
