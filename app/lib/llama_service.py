from __future__ import annotations

from dataclasses import dataclass
import os
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


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def _apply_env_overrides(config: LlamaConfig) -> LlamaConfig:
    max_threads = os.cpu_count() or 1
    max_gpu_layers = _clamp_int(int(os.getenv("LLAMA_MAX_GPU_LAYERS", "256")), 0, 4096)

    threads_env = os.getenv("LLAMA_N_THREADS")
    if threads_env is not None:
        try:
            threads = int(threads_env)
        except ValueError as exc:
            raise ValueError("LLAMA_N_THREADS must be an integer") from exc
        config.n_threads = _clamp_int(threads, 1, max_threads)

    gpu_layers_env = os.getenv("LLAMA_N_GPU_LAYERS")
    if gpu_layers_env is not None:
        try:
            gpu_layers = int(gpu_layers_env)
        except ValueError as exc:
            raise ValueError("LLAMA_N_GPU_LAYERS must be an integer") from exc
        if gpu_layers == -1:
            config.n_gpu_layers = -1
        else:
            config.n_gpu_layers = _clamp_int(gpu_layers, 0, max_gpu_layers)

    return config

def get_llama(model_name: str, config: Optional[LlamaConfig] = None) -> Llama:
    if config is None:
        config = LlamaConfig()
    config = _apply_env_overrides(config)

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
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
) -> str:
    if content_type != "text":
        raise ValueError("Only 'text' content_type is supported by GGUF models in this API.")

    llama = get_llama(model_name)

    sampling_kwargs: Dict[str, object] = {}
    if temperature is not None:
        sampling_kwargs["temperature"] = temperature
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    if min_p is not None:
        sampling_kwargs["min_p"] = min_p
    if repeat_penalty is not None:
        sampling_kwargs["repeat_penalty"] = repeat_penalty
    if presence_penalty is not None:
        sampling_kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        sampling_kwargs["frequency_penalty"] = frequency_penalty

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
        **sampling_kwargs,
    )

    message = response["choices"][0]["message"]["content"]
    return message.strip()