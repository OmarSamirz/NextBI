from langchain.base_language import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

import json
from typing import Self
from typing_extensions import override

from graph_agents.base import GraphAgent
from graph_agents.state import MultiAgentState
from constants import MANAGER_AGENT_SYSTEM_PROMPT_PATH


class ManagerAgent(GraphAgent):

    def __init__(self, llm: BaseLanguageModel) -> None:
        super().__init__(llm)
        with open(str(MANAGER_AGENT_SYSTEM_PROMPT_PATH), "r") as f:
            content = f.read()

        self.system_prompt = content

        self.chat_histories = {}
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    @override
    @classmethod
    async def create(cls: type[Self], llm: BaseLanguageModel) -> Self:
        self = cls(llm)
        self.tools = []

        agent = create_tool_calling_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        base_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=self.return_intermediate_steps
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
        result = await self.agent_executor.ainvoke(
            {"input": state["user_query"]},
            config={"configurable": {"session_id": self.session_id}}
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
            state["done"] = True
        else:
            state["manager_decision"] = "done"
            state["done"] = True
        
        state["response"] = message if message is not None else result["output"]
        state["explanation"] = explanation if explanation is not None else None

        return state