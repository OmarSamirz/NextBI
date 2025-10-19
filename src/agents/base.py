from langchain.base_language import BaseLanguageModel
from langchain.memory.chat_memory import BaseChatMemory

import os
from typing import Self
from abc import ABC, abstractmethod
from typing import TypedDict, Literal, Optional

from states import MultiAgentState


class Message(TypedDict, total=False):
    role: Literal["user", "ai", "assistant", "system"]
    content: str
    ts: Optional[str]


class BaseAgent(ABC):

    def __init__(self, llm: BaseLanguageModel, memory: BaseChatMemory) -> None:
        super().__init__()
        self.llm = llm
        self.memory = memory
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 30))
        self.verbose = eval(os.getenv("VERBOSE", False))
        self.return_intermediate_steps = eval(os.getenv("RETURN_INTERMEDIATE_STEPS", False))

        self.tools = None
        self.agent_executor = None

    @classmethod
    @abstractmethod
    async def create(cls: type[Self], llm: BaseLanguageModel, memory: BaseChatMemory) -> Self:
        ...

    @abstractmethod
    async def __call__(self, state: MultiAgentState) -> MultiAgentState:
        ...