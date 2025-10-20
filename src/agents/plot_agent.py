from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_experimental.tools.python.tool import PythonAstREPLTool

from typing import Self
from string import Template
from typing_extensions import override

from modules.logger import logger
from agents.base import BaseAgent
from states.multi_agent_state import MultiAgentState
from constants import CHARTS_PATH, PLOT_AGENT_SYSTEM_PROMPT_PATH


class PlotAgent(BaseAgent):

    def __init__(self, llm: BaseLanguageModel, memory: BaseChatMemory) -> None:
        super().__init__(llm, memory)
        with open(str(PLOT_AGENT_SYSTEM_PROMPT_PATH), "r", encoding="utf-8") as f:
            content = Template(f.read())

        self.system_prompt = content.safe_substitute(
            charts_path=CHARTS_PATH
        )

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
        self.tools = [PythonAstREPLTool()]

        agent = create_tool_calling_agent(llm=llm, tools=self.tools, prompt=self.prompt)
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
        logger.log("[Agent]", "plot")
        explanation = state.get("explanation", None)
        input_message = f"Manager Request: {explanation}"

        response = await self.agent_executor.ainvoke(
            {"input": input_message},
        )
        if "intermediate_steps" in response and len(response["intermediate_steps"]) > 0:
            action, _ = response["intermediate_steps"][0]
            logger.log("[Used Tool]", action.tool)
            state["is_plot"] = True

        state["plot_agent_response"] = response["output"]
        logger.log("[Plot Agent Output]", response["output"])

        return state