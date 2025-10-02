import torch
from dotenv import load_dotenv
from mcp_use import MCPClient, MCPAgent
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models import BaseChatModel

import os
from typing import Any

from ai_modules.base import AI, Message
from constants import MCP_CONFIG, DTYPE_MAP

load_dotenv()


class QwenModelWrapper(BaseChatModel):

    def __init__(self):
        super().__init__()
        self.model_id = os.getenv("Q_MODEL_NAME")
        self.device = torch.device(os.getenv("Q_DEVICE"))
        self.enable_thinking = eval(os.getenv("Q_ENABLE_THINKING"))
        self.max_new_tokens = int(os.getenv("Q_MAX_NEW_TOKENS"))
        dtype = os.getenv("Q_DTYPE")
        if dtype in DTYPE_MAP:
            self.dtype = DTYPE_MAP[dtype]
        else:
            raise ValueError(f"This dtype is not supported {dtype}.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map=self.device
        )

    def _prepare_message(self, messages):
        return [
            {
                "role": messages[-1],
                "content": messages[-1]
            }
        ]

    def _generate(self, messages, stop = None, run_manager = None, **kwargs):
        msg = self._prepare_message(messages)
        text = self.tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content

    def _llm_type(self):
        return "Qwen3"
        
        

class AIQwen(AI):
    """Concrete AI implementation using OpenAI Chat Completions models.

    Configuration is sourced from :func:`config.get_openai_config` which returns
    the API key, model name (e.g., ``gpt-4o``), and a configured OpenAI client.
    """

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self.model_id = os.getenv("Q_MODEL_NAME")
        self.device = torch.device(os.getenv("Q_DEVICE"))
        self.enable_thinking = eval(os.getenv("Q_ENABLE_THINKING"))
        self.max_new_tokens = int(os.getenv("Q_MAX_NEW_TOKENS"))
        self.mcp_config = MCP_CONFIG.copy()
        self.mcp_config["mcpServers"]["teradata"]["env"]["DATABASE_URI"] = os.getenv("DATABASE_URI")
        self.llm = QwenModelWrapper()
        self.client = MCPClient.from_dict(config=self.mcp_config)
        self.agent = MCPAgent(llm=self.llm, client=self.client, max_steps=30)

    async def generate_reply(self, messages: list[Message], context: dict | None = None) -> str:
        """Generate an assistant reply using OpenAI Chat Completions.

        Notes:
        - Unknown roles are coerced to "user".
        - App's internal role "ai" is translated to OpenAI's "assistant".
        - Empty/whitespace-only contents are skipped.
        - Retries up to 3 times on exceptions with backoff (0.5s, 1s).
        """
        msg = messages[-1].get("content", "")

        return await self.agent.run(msg)