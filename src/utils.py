
from agents.base import Agent
from agents import GPTAgent, GeminiAgent
from modules.config import get_ai_backend

async def get_ai() -> Agent:
    """Construct and return a concrete :class:`AI` backend based on ``AI_BACKEND``.

    Currently supported values:
    - "gpt": :class:`ai.gpt.AI_GPT`
    - "openai": :class:`ai.openai.AI_OpenAI`

    Returns
    -------
    AI
        A ready-to-use backend implementing :class:`AI`.
    """
    backend = get_ai_backend()
    
    if backend == "gpt":
        return await GPTAgent.create()
    if backend == "gemini":
        return await GeminiAgent.create()

    raise ValueError(f"Unknown AI backend: {backend}")