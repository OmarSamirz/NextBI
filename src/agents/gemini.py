from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory

from typing_extensions import override

from constants import ENV_PATH
from agents.base import Agent, Message
from modules.config import get_google_genai_config

load_dotenv(ENV_PATH)


class GeminiAgent(Agent):

    def __init__(self) -> None:
        super().__init__()
        cfg = get_google_genai_config()
        self.api_key = cfg["api_key"]
        self.model = cfg["model"]
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            api_key=self.api_key,
        )

    @override
    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        return super()._get_session_history(session_id)

    @override
    @classmethod
    async def create(cls):
        return await super().create()

    @override
    def _process_intermediate_logs(self, result):
        return super()._process_intermediate_logs(result)

    @override
    async def generate_reply(self, messages: list[Message], session_id: str = "default") -> str:
        return await super().generate_reply(messages, session_id)