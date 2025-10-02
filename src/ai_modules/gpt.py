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
import json
from typing import Any

from modules.logger import logger
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
        self.max_steps = int(os.getenv("MAX_STEPS"))
        self.mcp_config = MCP_CONFIG.copy()
        self.system_prompt = SYSTEM_PROMPT.format(database_name=os.getenv("TD_NAME"))
        self.mcp_config["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = os.getenv("DATABASE_URI")

        self.llm = ChatOpenAI(model=self.model)
        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.agent = MCPAgent(llm=self.llm, client=self.client, system_prompt=self.system_prompt, max_steps=self.max_steps)

    async def generate_reply(self, messages: list[Message]) -> str:
        msg = messages[-1].get("content", "")
        final_output = ""
        async for step, item in enumerate(self.agent.stream(msg)):
            logger.log(f"[Step {step+1}/{self.max_steps}]", "")
            if isinstance(item, tuple) and len(item) == 2:
                action, observation = item

                logger.log("[LLM -> MCP Tool Call]", action.tool)
                logger.log("[LLM -> MCP Tool Input]", str(action.tool_input))

                observation = json.loads(observation) if observation != "" else {}
                if "status" in observation:
                    status = observation["status"]
                    logger.log("[MCP Status]", status)
                if status == "success":
                    results = observation["results"]
                    logger.log("[MCP Results]", str(results))
                    sql_query = observation.get("metadata", {}).get("sql", "[No SQL query found]")
                    logger.log("[MCP SQL Query]", sql_query)
            elif isinstance(item, str):
                final_output = item

        return final_output
