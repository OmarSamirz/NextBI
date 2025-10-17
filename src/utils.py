from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory

from agents.base import Agent
from agents import GPTAgent, GeminiAgent
from graph_agents.plot_agent import PlotAgent
from graph_agents.multi_agent import MultiAgent
from graph_agents.manager_agent import ManagerAgent
from graph_agents.teradata_agent import TeradataAgent
from modules.config import get_ai_backend, get_openai_config, get_google_genai_config

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

async def get_multi_agent() -> MultiAgent:
    backend = get_ai_backend()
    
    llm = None
    if backend == "gpt":
        cfg = get_openai_config()
        llm = ChatOpenAI(
            model=cfg["model"],
            api_key=cfg["api_key"]
        )
    elif backend == "gemini":
        cfg = get_google_genai_config()
        llm = ChatGoogleGenerativeAI(
            model=cfg["model"],
            api_key=cfg["api_key"]
        )
    else:
        raise ValueError(f"Unknown AI backend: {backend}")

    memory = ConversationBufferWindowMemory(
        k=15,
        output_key="output",
        return_messages=True,
        memory_key="chat_history",
    )
    teradata_agent = await TeradataAgent.create(llm, memory)
    plot_agent = await PlotAgent.create(llm, memory)
    manager_agent = await ManagerAgent.create(llm, memory)

    multi_agent = MultiAgent(memory, manager_agent, plot_agent, teradata_agent)

    return multi_agent