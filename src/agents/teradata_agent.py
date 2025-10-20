from mcp_use import MCPClient
from mcp_use.adapters import LangChainAdapter
from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os
import re
import json
import textwrap
from string import Template
from typing import Self, Union
from typing_extensions import override

from agents import BaseAgent
from modules.logger import logger
from states import MultiAgentState
from constants import MCP_CONFIG, TERADATA_AGENT_SYSTEM_PROMPT_PATH, CHARTS_PATH


class TeradataAgent(BaseAgent):

    def __init__(self, llm: BaseLanguageModel, memory: BaseChatMemory) -> None:
        super().__init__(llm, memory)
        self.mcp_config = MCP_CONFIG.copy()
        with open(str(TERADATA_AGENT_SYSTEM_PROMPT_PATH), "r", encoding="utf-8") as f:
            content = Template(f.read())

        self.system_prompt = content.safe_substitute(
            database_name=os.getenv("TD_NAME"),
            charts_path=CHARTS_PATH
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.adapter = LangChainAdapter()

    @override
    @classmethod
    async def create(cls: type[Self], llm: BaseLanguageModel, memory: BaseChatMemory) -> Self:
        self = cls(llm, memory)
        self.tools = await self.adapter.create_tools(self.client)

        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=self.return_intermediate_steps,
        )

        return self

    def _process_intermediate_logs(self, response) -> Union[str, None]:
        found_sql = False
        sql_message = ["\n\n**SQL Commands:**\n"]

        intermediate_steps = response.get("intermediate_steps", [])
        intermediate_steps_len = len(intermediate_steps)
        for i, (action, observation) in enumerate(intermediate_steps, start=1):
            logger.log(f"[Step {i}/{intermediate_steps_len}]", "")
            logger.log("[Used Tool]", action.tool)

            if not isinstance(observation, list):
                match = re.search(r"text='(.*)'", observation)
                if match:
                    observation = match.group(1).encode("utf-8").decode("unicode_escape")

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

        final_sql_messages = "\n".join(sql_message) if found_sql else None

        return final_sql_messages

    @override
    async def __call__(self, state: MultiAgentState) -> MultiAgentState:
        logger.log("[Agent]", "teradata")
        explanation = state.get("explanation", None)
        input_message = f"Manager Request:\n{explanation}"

        try:
            response = await self.agent_executor.ainvoke(
                {"input": input_message},
            )
        except ValueError as e:
            logger.log("[Error]", f"{e}")
            state["td_agent_response"] = e
            return state

        state["td_agent_response"] = response["output"]
        sql_messages = self._process_intermediate_logs(response)
        state["sql_queries"] = sql_messages

        logger.log("[Teradata Agent Output]", response["output"])

        return state