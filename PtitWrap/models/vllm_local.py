"""In-process vLLM backend — the "bring your own HuggingFace model" path.

Loads the model directly with ``vllm.LLM(...)`` in the current process (like
MediQ's own helper did, and like lm-eval's ``vllm`` model), so no separate
server needs to be started or managed. Chat messages are rendered to a prompt
with the tokenizer's chat template before generation.

``vllm`` / ``transformers`` are imported lazily inside ``__init__`` so the
harness package imports fine on a machine without them.
"""

from __future__ import annotations

from typing import Any

from ..schema import Message
from .base import LM, register_model


@register_model("vllm")
class VLLMLocalLM(LM):
    """Chat model backed by an in-process vLLM engine over a local HF checkpoint."""

    def __init__(
        self,
        pretrained: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        temperature: float = 0.6,
        max_tokens: int = 256,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams

        llm_kwargs = dict(
            model=pretrained,
            dtype=dtype,
            gpu_memory_utilization=float(gpu_memory_utilization),
            tensor_parallel_size=int(tensor_parallel_size),
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)
        self.llm = LLM(**llm_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)

    def chat(self, messages: list[Message], **gen_kwargs: Any) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        params = self._SamplingParams(
            temperature=gen_kwargs.get("temperature", self.temperature),
            max_tokens=gen_kwargs.get("max_tokens", self.max_tokens),
            top_p=gen_kwargs.get("top_p", self.top_p),
        )
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        return outputs[0].outputs[0].text
