from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

import json
from typing import Self
from typing_extensions import override

from graph_agents.base import GraphAgent
from graph_agents.state import MultiAgentState
from constants import MANAGER_AGENT_SYSTEM_PROMPT_PATH


class ManagerAgent(GraphAgent):

    def __init__(self, llm: BaseLanguageModel, memory: BaseChatMemory) -> None:
        super().__init__(llm, memory)
        with open(str(MANAGER_AGENT_SYSTEM_PROMPT_PATH), "r") as f:
            self.system_prompt = f.read()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    @override
    @classmethod
    async def create(cls: type[Self], llm: BaseLanguageModel, memory: BaseChatMemory) -> Self:
        self = cls(llm, memory)
        self.tools = []

        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=self.return_intermediate_steps
        )

        return self

    @override
    async def __call__(self, state: MultiAgentState) -> MultiAgentState:
        user_query = f"User Query:\n{state['user_query']}"
        td_agent_response = state.get("td_agent_response", None)
        plot_agent_response = state.get("plot_agent_response", None)

        if td_agent_response is not None:
            user_query += f"\n\nTeradata Agent Response:\n{td_agent_response}"
        if plot_agent_response is not None:
            user_query += f"\n\nPlot Agent Response:\n{plot_agent_response}"
        
        result = await self.agent_executor.ainvoke(
            {"input": user_query},
        )

        decision, message, explanation = None, None, None
        try:
            response = json.loads(result["output"])
            decision = response["decision"].lower()
            message = response["message"]
            explanation = response["explanation"]
        except:
            decision = result["output"].lower()

        if "teradata" in decision:
            state["manager_decision"] = "teradata"
        elif "plot" in decision:
            state["manager_decision"] = "plot"
        elif "done" in decision:
            state["manager_decision"] = "done"
        else:
            state["manager_decision"] = "done"

        state["response"] = message if message is not None else result["output"]
        state["explanation"] = explanation if explanation is not None else None

        state["messages"].append({"role": "manager", "content": state["response"]})

        return state