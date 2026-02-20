from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
import os
from threading import Lock, Semaphore
from typing import Dict, Optional
from contextlib import contextmanager

from llama_cpp import Llama

from .model_registry import resolve_model_path

logger = logging.getLogger(__name__)


@dataclass
class LlamaConfig:
    n_ctx: int = 2048
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0

_model_cache: Dict[str, Llama] = {}
_cache_lock = Lock()
_last_model_name: Optional[str] = None


class InferenceOverloadedError(RuntimeError):
    """Raised when the inference queue is saturated for too long."""


def _get_env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning("%s=%r is invalid; using default=%d", name, raw_value, default)
        return default

    clamped = _clamp_int(parsed, min_value, max_value)
    if clamped != parsed:
        logger.warning(
            "%s=%d is out of range [%d, %d]; clamped to %d",
            name,
            parsed,
            min_value,
            max_value,
            clamped,
        )
    return clamped


def _get_env_float(name: str, default: float, min_value: float, max_value: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = float(raw_value)
    except ValueError:
        logger.warning("%s=%r is invalid; using default=%.2f", name, raw_value, default)
        return default

    clamped = max(min_value, min(max_value, parsed))
    if clamped != parsed:
        logger.warning(
            "%s=%.3f is out of range [%.3f, %.3f]; clamped to %.3f",
            name,
            parsed,
            min_value,
            max_value,
            clamped,
        )
    return clamped


_max_concurrent_inferences = _get_env_int(
    "LLAMA_MAX_CONCURRENT_INFERENCES",
    default=1,
    min_value=1,
    max_value=64,
)
_inference_acquire_timeout_seconds = _get_env_float(
    "LLAMA_INFERENCE_ACQUIRE_TIMEOUT_SECONDS",
    default=45.0,
    min_value=0.1,
    max_value=600.0,
)
_inference_semaphore = Semaphore(_max_concurrent_inferences)


@contextmanager
def _acquire_inference_slot():
    acquired = _inference_semaphore.acquire(timeout=_inference_acquire_timeout_seconds)
    if not acquired:
        raise InferenceOverloadedError(
            "Inference queue timeout reached. Try again later or reduce request concurrency."
        )

    try:
        yield
    finally:
        _inference_semaphore.release()


def _build_sampling_kwargs(
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    min_p: Optional[float],
    repeat_penalty: Optional[float],
    presence_penalty: Optional[float],
    frequency_penalty: Optional[float],
) -> Dict[str, object]:
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
    return sampling_kwargs


def _normalize_lang_code(code: str) -> str:
    return code.replace("_", "-")


def _manual_prompt_text(
    source_lang: str,
    source_lang_code: str,
    target_lang: str,
    target_lang_code: str,
    text: str,
) -> str:
    return (
        f"You are a professional {source_lang} ({source_lang_code}) to {target_lang} "
        f"({target_lang_code}) translator. Your goal is to accurately convey the meaning and "
        f"nuances of the original {source_lang} text while adhering to {target_lang} grammar, "
        f"vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_lang} translation, without any additional explanations or "
        f"commentary. Please translate the following {source_lang} text into {target_lang}:\n\n\n"
        f"{text}"
    )


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
    global _last_model_name
    if config is None:
        config = LlamaConfig()
    config = _apply_env_overrides(config)

    with _cache_lock:
        if model_name in _model_cache:
            _last_model_name = model_name
            return _model_cache[model_name]

        model_path = resolve_model_path(model_name)
        llama = Llama(
            model_path=str(model_path),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=config.n_gpu_layers,
        )
        _model_cache[model_name] = llama
        _last_model_name = model_name
        return llama


def unload_model(model_name: str) -> bool:
    """Unloads a model from memory and frees resources."""
    global _last_model_name
    with _cache_lock:
        llama = _model_cache.pop(model_name, None)
        if _last_model_name == model_name:
            _last_model_name = None

    if llama is None:
        return False

    # Attempt to call close if available in llama-cpp-python
    if hasattr(llama, "close"):
        try:
            llama.close()
        except Exception:
            pass

    del llama
    gc.collect()
    return True


def _maybe_unload_previous_model(requested_model: str) -> None:
    """If a different model is currently loaded, unload it first."""
    with _cache_lock:
        prev_model = _last_model_name

    if prev_model is not None and prev_model != requested_model:
        unload_model(prev_model)

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

    with _acquire_inference_slot():
        _maybe_unload_previous_model(model_name)
        llama = get_llama(model_name)

        sampling_kwargs = _build_sampling_kwargs(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

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


def experimental_translate_text(
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
        raise ValueError("Only 'text' content_type is supported by the experimental endpoint.")

    with _acquire_inference_slot():
        _maybe_unload_previous_model(model_name)
        llama = get_llama(model_name)

        source_lang_code = _normalize_lang_code(source_lang_code)
        target_lang_code = _normalize_lang_code(target_lang_code)

        source_lang = source_lang_code
        target_lang = target_lang_code

        prompt_text = _manual_prompt_text(
            source_lang=source_lang,
            source_lang_code=source_lang_code,
            target_lang=target_lang,
            target_lang_code=target_lang_code,
            text=text,
        )

        full_prompt = (
            f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        sampling_kwargs = _build_sampling_kwargs(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        response = llama(
            full_prompt,
            max_tokens=max_new_tokens,
            stop=["<end_of_turn>"],
            **sampling_kwargs,
        )

        message = response["choices"][0]["text"]
        return message.strip()
