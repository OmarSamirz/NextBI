from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory

from typing import Any
from typing_extensions import override

from constants import ENV_PATH
from agents.base import Agent, Message
from modules.config import get_openai_config

load_dotenv(ENV_PATH)


class GPTAgent(Agent):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        cfg = get_openai_config()
        self.api_key = cfg["api_key"]
        self.model = cfg["model"]
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
        )

    @override
    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        return super()._get_session_history(session_id)

    @override
    @classmethod
    async def create(cls, config: Any = None):
        return await super().create(config)

    @override
    def _process_intermediate_logs(self, result):
        return super()._process_intermediate_logs(result)

    @override
    async def generate_reply(self, messages: list[Message], session_id: str = "default") -> str:
        return await super().generate_reply(messages, session_id)