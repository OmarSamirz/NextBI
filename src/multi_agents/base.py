
from abc import ABC, abstractmethod

from states.base import BaseState


class BaseMultiAgent(ABC):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def _build_graph(self) -> None:
        ...

    @abstractmethod
    async def run(self, user_query: str) -> BaseState:
        ...

    @abstractmethod
    def visualize(self) -> None:
        ...