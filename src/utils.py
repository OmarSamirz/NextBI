
from ai_modules.base import AI
from ai_modules.gpt import AIGPT
from modules.config import get_ai_backend

def get_ai() -> AI:
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
    try:
        # Late import to avoid import-time .env side effects in tests
        from modules.logger import ChatLogger  # type: ignore
        ChatLogger().event("ai.backend.select", backend=backend)
    except Exception:
        pass
    if backend == "gpt":
        return AIGPT()
    if backend == "qwen":
        from ai_modules.qwen import AIQwen
        return AIQwen()
    raise ValueError(f"Unknown AI backend: {backend}")