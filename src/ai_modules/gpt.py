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

from ai_modules.base import AI, Message
from modules.config import get_openai_config
from constants import MCP_CONFIG, SYSTEM_PROMPT

load_dotenv()


class AIGPT(AI):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = get_openai_config()
        self.api_key = cfg["api_key"]
        self.model = cfg["model"]
        self.mcp_config = MCP_CONFIG.copy()
        self.system_prompt = SYSTEM_PROMPT.format(database_name=os.getenv("TD_NAME"))
        self.mcp_config["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = os.getenv("DATABASE_URI")
        self.llm = ChatOpenAI(model=self.model)
        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.agent = MCPAgent(llm=self.llm, client=self.client, system_prompt=self.system_prompt, max_steps=30)

    async def generate_reply(self, messages: list[Message], context: dict | None = None) -> str:
        msg = messages[-1].get("content", "")
        all_steps = []
        final_output = ""
        async for item in self.agent.stream(msg):
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
