from mcp_use import MCPClient
from mcp_use.adapters import LangChainAdapter
from langchain.base_language import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from typing import Self
from string import Template
from typing_extensions import override

from graph_agents.base import GraphAgent
from graph_agents.state import MultiAgentState
from constants import MCP_CONFIG, TERADATA_AGENT_SYSTEM_PROMPT_PATH, CHARTS_PATH


class TeradataAgent(GraphAgent):

    def __init__(self, llm: BaseLanguageModel) -> None:
        super().__init__(llm)
        self.mcp_config = MCP_CONFIG.copy()
        self.mcp_config["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = os.getenv("DATABASE_URI")
        with open(str(TERADATA_AGENT_SYSTEM_PROMPT_PATH), "r") as f:
            content = Template(f.read())

        self.system_prompt = content.safe_substitute(
            database_name=os.getenv("TD_NAME"),
            charts_path=CHARTS_PATH
        )

        self.chat_histories = {}
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
    async def create(cls: type[Self], llm: BaseLanguageModel) -> Self:
        self = cls(llm)
        self.tools = await self.adapter.create_tools(self.client)

        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        base_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=self.return_intermediate_steps,
        )
        self.agent_executor = RunnableWithMessageHistory(
            base_executor,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        return self

    @override
    async def __call__(self, state: MultiAgentState):
        user_query = state.get("user_query")
        explanation = state.get("explanation", None)
        input_message = user_query + f"\n\nExplanation: {explanation}.\n" if explanation is not None else user_query

        response = await self.agent_executor.ainvoke(
            {"input": input_message},
            config={"configurable": {"session_id": self.session_id}}
        )

        state["response"] = response["output"]

        return state