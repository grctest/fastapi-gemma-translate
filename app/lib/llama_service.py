from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

from llama_cpp import Llama

from .model_registry import resolve_model_path


@dataclass
class LlamaConfig:
    n_ctx: int = 2048
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0


_model_cache: Dict[str, Llama] = {}
_cache_lock = Lock()


def get_llama(model_name: str, config: Optional[LlamaConfig] = None) -> Llama:
    if config is None:
        config = LlamaConfig()

    with _cache_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

        model_path = resolve_model_path(model_name)
        llama = Llama(
            model_path=str(model_path),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=config.n_gpu_layers,
        )
        _model_cache[model_name] = llama
        return llama


def translate_text(
    model_name: str,
    source_lang_code: str,
    target_lang_code: str,
    text: str,
    content_type: str = "text",
    max_new_tokens: int = 200,
) -> str:
    if content_type != "text":
        raise ValueError("Only 'text' content_type is supported by GGUF models in this API.")

    llama = get_llama(model_name)

    response = llama.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang_code,
                        "target_lang_code": target_lang_code,
                        "text": text,
                    }
                ],
            }
        ],
        max_tokens=max_new_tokens,
        temperature=0.2,
    )

    message = response["choices"][0]["message"]["content"]
    return message.strip()