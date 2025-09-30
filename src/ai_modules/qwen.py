import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
from typing import Any, Dict, List, Optional

from constants import DTYPE_MAP
from modules.logger import ChatLogger
from ai_modules.base import AI, Message

load_dotenv()


class AIQwen(AI):

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
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

    def _prepare_message(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return [
            {
                "role": messages[-1].get("role", "user"),
                "content": messages[-1].get("content", "")
            }
        ]

    def generate_reply(self, messages: List[Message], context: Optional[Dict[str, Any]] = None):
        messages = self._prepare_message(messages)
        print(f"All messages: {messages}")
        text = self.tokenizer.apply_chat_template(
            messages,
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

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content