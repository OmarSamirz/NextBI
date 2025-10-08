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
from string import Template
from abc import ABC, abstractmethod
from typing import Any, TypedDict, Literal, Optional

from modules.logger import logger
from constants import MCP_CONFIG, CHARTS_PATH, SYSTEM_PROMPT_PATH


class Message(TypedDict, total=False):
    role: Literal["user", "ai", "assistant", "system"]
    content: str
    ts: Optional[str]


class Agent(ABC):

    def __init__(self, config: Any = None) -> None:
        """Initialize the backend with an optional configuration object."""
        self.config = config
        self.mcp_config = MCP_CONFIG.copy()
        with open(str(SYSTEM_PROMPT_PATH), "r") as f:
            content = Template(f.read())
            
        self.system_prompt = content.safe_substitute(
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

        # Optional debug: print tool schemas when DEBUG_TOOL_SCHEMA is enabled
        if os.getenv("DEBUG_TOOL_SCHEMA", "").lower() in {"1", "true", "yes"}:
            for t in self.tools:
                name = getattr(t, 'name', 'unknown')
                args_schema = getattr(t, 'args_schema', None)
                if args_schema:
                    try:
                        logger.log("[SCHEMA]", f"{name}: {args_schema.schema_json()}")
                    except Exception as e:
                        logger.log("[SCHEMA]", f"{name}: <error serializing schema> {e}")
                else:
                    logger.log("[SCHEMA]", f"{name}: <no schema>")

        # Wrap specific tools for parameter normalization and output truncation
        self._wrap_mcp_tools()
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
        debug_enabled = os.getenv("AGENT_DEBUG", "").lower() in {"1", "true", "yes"}

        intermediate_steps = result.get("intermediate_steps", [])
        if debug_enabled:
            logger.log("[DEBUG]", f"Intermediate steps count: {len(intermediate_steps)}")

        for i, (action, observation) in enumerate(intermediate_steps, start=1):
            tools_used.append(action.tool)
            if debug_enabled:
                snippet = (observation or "")[:300].replace("\n", " ")
                logger.log("[DEBUG]", f"Step {i} tool={action.tool} input={getattr(action, 'tool_input', None)} obs_snippet={snippet}")

            # Try JSON parse; on failure treat as plain text
            obs_json = {}
            if observation:
                if isinstance(observation, str):
                    try:
                        obs_json = json.loads(observation)
                    except (json.JSONDecodeError, TypeError):
                        if debug_enabled:
                            logger.log("[DEBUG]", f"Non-JSON observation at step {i}")
                        # Heuristic: detect SQL keyword in plain text
                        lowered = observation.lower()
                        if any(k in lowered for k in ["select ", " update ", " delete ", " insert "]):
                            found_sql = True
                            wrapped_sql = textwrap.fill(observation, width=100)
                            sql_message.append(f"\n```sql\n{wrapped_sql}\n```\n")
                elif isinstance(observation, (dict, list)):
                    obs_json = observation  # Already structured

            status = obs_json.get("status") if isinstance(obs_json, dict) else None
            if status:
                logger.log("[MCP Status]", status)
                if status == "success":
                    results = obs_json.get("results")
                    logger.log("[MCP Results]", str(results))
                    sql_query = obs_json.get("metadata", {}).get("sql") if isinstance(obs_json.get("metadata"), dict) else None
                    if sql_query:
                        found_sql = True
                        logger.log("[MCP SQL Query]", sql_query)
                        wrapped_sql = textwrap.fill(sql_query, width=100)
                        sql_message.append(f"\n```sql\n{wrapped_sql}\n```\n")

        found_plot = "Python_REPL" in tools_used
        final_sql_message = "\n".join(sql_message) if found_sql else ""
        return final_sql_message, found_plot

    # ------------------------- INTERNAL HELPERS ------------------------- #
    def _wrap_mcp_tools(self):
        """Wrap MCP tools to normalize params & truncate large outputs.

        Normalizations:
          - base_tableList: rename database_name -> database; ensure database present from TD_NAME.
          - base_readQuery: accept sql/sql_text/query -> unified 'query'.

        Truncation:
          For base_tableList results, limit rows to TABLE_LIST_LIMIT (default 25) and append summary.
        """
        limit = int(os.getenv("TABLE_LIST_LIMIT", 25))
        td_name = os.getenv("TD_NAME")

        wrapped = []
        for tool in self.tools:
            name = getattr(tool, 'name', None)
            if name not in {"base_tableList", "base_readQuery"}:
                wrapped.append(tool)
                continue

            original_tool = tool

            async def _ainvoke_proxy(input, _orig=original_tool, _name=name, **kwargs):  # type: ignore
                # Normalize input into dict
                if isinstance(input, str):
                    payload = {"query": input} if _name == "base_readQuery" else {}
                elif isinstance(input, dict):
                    payload = dict(input)
                else:
                    payload = {}

                # Param normalization
                if _name == "base_tableList":
                    if 'database' not in payload:
                        # Accept alternative keys
                        for alt in ('database_name', 'db'):  # prefer existing value order
                            if alt in payload and payload[alt]:
                                payload['database'] = payload[alt]
                                break
                    if 'database' not in payload and td_name:
                        payload['database'] = td_name
                elif _name == "base_readQuery":
                    # unify sql-related keys
                    if 'query' not in payload:
                        for alt in ('sql', 'sql_text', 'query_text'):
                            if alt in payload and payload[alt]:
                                payload['query'] = payload[alt]
                                break

                debug = os.getenv("AGENT_DEBUG", "").lower() in {"1", "true", "yes"}
                if debug:
                    logger.log("[DEBUG]", f"Calling {_name} with normalized payload keys={list(payload.keys())}")

                # Call underlying tool (prefer ainvoke/arun)
                if hasattr(_orig, 'ainvoke'):
                    result = await _orig.ainvoke(payload)
                elif hasattr(_orig, 'arun'):
                    result = await _orig.arun(payload)
                else:
                    # Fallback sync path
                    if hasattr(_orig, 'invoke'):
                        result = _orig.invoke(payload)
                    elif hasattr(_orig, 'run'):
                        result = _orig.run(payload)
                    else:
                        return {"status": "error", "error": f"Underlying tool {_name} has no callable interface"}

                # Truncation for table list
                if _name == "base_tableList" and isinstance(result, dict):
                    rows = result.get('results')
                    if isinstance(rows, list) and len(rows) > limit:
                        truncated = rows[:limit]
                        result['results'] = truncated
                        result.setdefault('metadata', {})
                        meta = result['metadata']
                        meta['truncated'] = True
                        meta['original_count'] = len(rows)
                        meta['returned_count'] = len(truncated)
                return result

            # Create a lightweight proxy object preserving name & description
            class _ProxyTool:
                name = name  # type: ignore
                description = getattr(original_tool, 'description', '')
                args_schema = getattr(original_tool, 'args_schema', None)
                # Provide ainvoke for LangChain compatibility
                async def ainvoke(self, input, **kwargs):  # type: ignore
                    return await _ainvoke_proxy(input, **kwargs)
                # Provide arun fallback
                async def arun(self, tool_input, **kwargs):  # type: ignore
                    return await _ainvoke_proxy(tool_input, **kwargs)
            wrapped.append(_ProxyTool())

        self.tools = wrapped

    @abstractmethod
    async def generate_reply(self, messages: list[Message], session_id: str = "default") -> str:
        msg = messages[-1].get("content", "")
        result = await self.agent_executor.ainvoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}}
        )
        sql_message, is_plot = self._process_intermediate_logs(result)
        output = result.get("output", "")
        if sql_message:
            output += sql_message
        if not output:
            output = "[no output produced by agent]"
        return output, is_plot