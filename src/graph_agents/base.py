from langchain.base_language import BaseLanguageModel
from langchain_core.chat_history import InMemoryChatMessageHistory

import os
from typing import Self
from abc import ABC, abstractmethod

from graph_agents.state import MultiAgentState


class GraphAgent(ABC):

    def __init__(self, llm: BaseLanguageModel) -> None:
        self.llm = llm
        self.chat_histories = {}
        self.session_id = "default"
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 30))
        self.verbose = eval(os.getenv("VERBOSE", False))
        self.return_intermediate_steps = eval(os.getenv("RETURN_INTERMEDIATE_STEPS", False))

        self.tools = None
        self.agent_executor = None

    @classmethod
    @abstractmethod
    async def create(cls: type[Self], *args, **kwargs) -> Self:
        ...

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = InMemoryChatMessageHistory()

        return self.chat_histories[session_id]

    @abstractmethod
    async def __call__(self, state: MultiAgentState):
        ...