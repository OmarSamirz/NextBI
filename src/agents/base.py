from mcp_use import MCPClient
from mcp_use.adapters import LangChainAdapter
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
import json
import textwrap
from abc import ABC, abstractmethod
from typing import Any, TypedDict, Literal, Optional

from modules.logger import logger
from constants import MCP_CONFIG, CHARTS_PATH, SYSTEM_PROMPT


class Message(TypedDict, total=False):
    role: Literal["user", "ai", "assistant", "system"]
    content: str
    ts: Optional[str]


class Agent(ABC):

    def __init__(self, config: Any = None) -> None:
        """Initialize the backend with an optional configuration object."""
        self.config = config
        self.mcp_config = MCP_CONFIG.copy()
        self.system_prompt = SYSTEM_PROMPT.substitute(
            database_name=os.getenv("TD_NAME"),
            charts_path=CHARTS_PATH
        )
        self.mcp_config["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = os.getenv("DATABASE_URI")

        self.chat_histories = {}

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.adapter = LangChainAdapter()

        self.llm = None
        self.tools = None
        self.agent_executor = None

    @abstractmethod
    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = InMemoryChatMessageHistory()

        return self.chat_histories[session_id]

    @classmethod
    @abstractmethod
    async def create(cls, config: Any = None):
        self = cls(config)
        self.tools = await self.adapter.create_tools(self.client)
        self.tools.append(PythonREPLTool())

        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        base_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True
        )
        self.agent_executor = RunnableWithMessageHistory(
            base_executor,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        return self

    @abstractmethod
    def _process_intermediate_logs(self, result):
        tools_used = []
        found_sql = False
        found_plot = False
        sql_message = ["\n\n**SQL Commands Used:**\n"]

        intermediate_steps = result.get("intermediate_steps", [])
        for i, (action, observation) in enumerate(intermediate_steps, start=1):
            logger.log(f"[Step {i}/{self.max_steps}]", "")
            tools_used.append(action.tool)

            try:
                obs_json = json.loads(observation) if observation else {}
            except (json.JSONDecodeError, TypeError):
                obs_json = {}

            status = obs_json.get("status")
            if status:
                logger.log("[MCP Status]", status)
                if status == "success":
                    results = obs_json.get("results")
                    logger.log("[MCP Results]", str(results))
                    sql_query = obs_json.get("metadata", {}).get("sql")
                    if sql_query:
                        found_sql = True
                        logger.log("[MCP SQL Query]", sql_query)
                        wrapped_sql = textwrap.fill(sql_query, width=100)
                        sql_message.append(f"\n```sql\n{wrapped_sql}\n```\n")

        found_plot = "Python_REPL" in tools_used
        final_sql_message = "\n".join(sql_message) if found_sql else ""

        return final_sql_message, found_plot

    @abstractmethod
    async def generate_reply(self, messages: list[Message], session_id: str = "default") -> str:
        msg = messages[-1].get("content", "")
        result = await self.agent_executor.ainvoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}}
        )
        sql_message, is_plot = self._process_intermediate_logs(result)

        if sql_message is not None:
            result["output"] += sql_message

        return result["output"], is_plot