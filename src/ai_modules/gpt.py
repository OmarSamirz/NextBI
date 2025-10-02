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
import re
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
        self.default_database = os.getenv("TD_NAME", "BANK_DB")
        self.llm = ChatOpenAI(model=self.model)
        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.agent = MCPAgent(llm=self.llm, client=self.client, max_steps=30)

    async def generate_reply(self, messages: list[Message], context: dict | None = None) -> str:
        user_msg = messages[-1].get("content", "")
        enhanced_prompt = f"{user_msg}. Use the'{self.default_database}' database"
        response = await self.agent.run(enhanced_prompt)
        sql_queries = self._extract_sql_queries(response)
        if sql_queries:
            formatted_response = "**SQL Query Executed:**\n```sql\n"
            formatted_response += "\n\n".join(sql_queries)
            formatted_response += "\n```\n\n**Results:**\n"
            formatted_response += response
        else:
            formatted_response = response
        return formatted_response

    def _extract_sql_queries(self, text: str) -> list[str]:
        queries = []
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, text, re.DOTALL | re.IGNORECASE)
        queries.extend(matches)
        if not queries:
            statement_pattern = r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.*?;"
            matches = re.findall(statement_pattern, text, re.DOTALL | re.IGNORECASE)
            queries.extend(matches)
        return queries
