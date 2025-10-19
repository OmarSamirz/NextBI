from mcp_use import MCPClient
from mcp_use.adapters import LangChainAdapter
from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os
from typing import Self
from string import Template
from typing_extensions import override

from agents.base import BaseAgent
from states.multi_agent_state import MultiAgentState
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

    @override
    async def __call__(self, state: MultiAgentState) -> MultiAgentState:
        explanation = state.get("explanation", None)
        input_message = f"Manager Request:\n{explanation}"

        response = await self.agent_executor.ainvoke(
            {"input": input_message},
        )

        state["td_agent_response"] = response["output"]

        return state