from __future__ import annotations

from dataclasses import dataclass
import gc
import os
import base64
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from llama_cpp import Llama

try:
    from llama_cpp.llama_chat_format import Llama3VisionAlphaChatHandler
except ImportError:  # pragma: no cover - depends on llama-cpp-python build
    Llama3VisionAlphaChatHandler = None

from .model_registry import resolve_model_path


@dataclass
class LlamaConfig:
    n_ctx: int = 2048
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0

_model_cache: Dict[str, Llama] = {}
_cache_lock = Lock()
_loaded_model_name: Optional[str] = None
_is_loading: bool = False
_loaded_model_is_vision_enabled: bool = False
_loaded_mmproj_path: Optional[str] = None


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


def _manual_image_prompt_text(
    source_lang: str,
    source_lang_code: str,
    target_lang: str,
    target_lang_code: str,
) -> str:
    return (
        f"You are a professional {source_lang} ({source_lang_code}) to {target_lang} ({target_lang_code}) translator.\n"
        f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_lang} translation, without any additional explanations or commentary.\n"
        f"Please translate the following {source_lang} text in the provided image into {target_lang}:\n\n\n"
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


def _resolve_mmproj_path_for_model(model_name: str, mmproj: Optional[str]) -> Optional[str]:
    if mmproj is None:
        return None

    mmproj_value = mmproj.strip()
    if not mmproj_value:
        raise ValueError("mmproj must be a non-empty file path when provided")

    model_path = resolve_model_path(model_name)
    model_dir = model_path.parent

    mmproj_path = Path(mmproj_value)
    if not mmproj_path.is_absolute():
        mmproj_path = model_dir / mmproj_path

    mmproj_path = mmproj_path.resolve()
    if not mmproj_path.exists() or not mmproj_path.is_file():
        raise ValueError(f"mmproj file not found: {mmproj_path}")
    if mmproj_path.suffix.lower() != ".gguf":
        raise ValueError("mmproj file must be a .gguf file")

    return str(mmproj_path)


def _build_image_data_uri(image_bytes: bytes, image_mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{image_mime_type};base64,{encoded}"


def _sanitize_translation_output(text: str) -> str:
    # Remove occasional model control markers that can leak into plain-text output.
    sanitized = text.replace("<|eot_id|>", "").replace("<|end_header_id|>", "")
    return sanitized.strip()

def get_llama(model_name: str, config: Optional[LlamaConfig] = None) -> Llama:
    # Keep backwards compatibility for callers that still expect get_llama,
    # while enforcing the explicit load-state lifecycle.
    load_model(model_name, config=config)
    return get_loaded_llama_or_raise(model_name)


def unload_model(model_name: str) -> bool:
    """Unloads a model from memory and frees resources."""
    global _loaded_model_name
    global _loaded_model_is_vision_enabled
    global _loaded_mmproj_path
    with _cache_lock:
        llama = _model_cache.pop(model_name, None)
        if _loaded_model_name == model_name:
            _loaded_model_name = None
            _loaded_model_is_vision_enabled = False
            _loaded_mmproj_path = None

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


def get_loaded_model_name() -> Optional[str]:
    with _cache_lock:
        return _loaded_model_name


def is_model_loading() -> bool:
    with _cache_lock:
        return _is_loading


def is_model_loaded(model_name: Optional[str] = None) -> bool:
    with _cache_lock:
        if _loaded_model_name is None:
            return False
        if model_name is None:
            return True
        return _loaded_model_name == model_name


def is_loaded_model_vision_enabled() -> bool:
    with _cache_lock:
        return _loaded_model_is_vision_enabled


def load_model(
    model_name: str,
    config: Optional[LlamaConfig] = None,
    mmproj: Optional[str] = None,
) -> bool:
    """
    Loads model_name into memory.

    Returns True if a new load was performed, False when the requested model
    was already loaded.
    """
    global _loaded_model_name
    global _is_loading
    global _loaded_model_is_vision_enabled
    global _loaded_mmproj_path

    if config is None:
        config = LlamaConfig()
    config = _apply_env_overrides(config)
    mmproj_path = _resolve_mmproj_path_for_model(model_name, mmproj)

    old_model_name: Optional[str] = None
    old_llama: Optional[Llama] = None

    with _cache_lock:
        if _is_loading:
            raise ValueError(
                "A model load is already in progress. Please wait and try again."
            )

        if (
            _loaded_model_name == model_name
            and model_name in _model_cache
            and _loaded_mmproj_path == mmproj_path
        ):
            return False

        _is_loading = True

        if _loaded_model_name is not None and (
            _loaded_model_name != model_name or _loaded_mmproj_path != mmproj_path
        ):
            old_model_name = _loaded_model_name
            old_llama = _model_cache.pop(_loaded_model_name, None)
            _loaded_model_name = None
            _loaded_model_is_vision_enabled = False
            _loaded_mmproj_path = None

    if old_llama is not None:
        if hasattr(old_llama, "close"):
            try:
                old_llama.close()
            except Exception:
                pass
        del old_llama
        gc.collect()

    try:
        model_path = resolve_model_path(model_name)
        chat_handler = None
        if mmproj_path is not None:
            if Llama3VisionAlphaChatHandler is None:
                raise ValueError(
                    "Vision mmproj file found, but Llama3VisionAlphaChatHandler is unavailable. "
                    "Upgrade llama-cpp-python to a recent build with Gemma/Llama3 vision chat handlers."
                )
            chat_handler = Llama3VisionAlphaChatHandler(clip_model_path=mmproj_path)

        llama = Llama(
            model_path=str(model_path),
            chat_handler=chat_handler,
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=config.n_gpu_layers,
        )
    except Exception as exc:
        with _cache_lock:
            _is_loading = False
        if old_model_name is not None:
            raise ValueError(
                f"Failed to load model '{model_name}' after unloading '{old_model_name}': {exc}"
            ) from exc
        raise ValueError(f"Failed to load model '{model_name}': {exc}") from exc

    with _cache_lock:
        _model_cache[model_name] = llama
        _loaded_model_name = model_name
        _loaded_model_is_vision_enabled = mmproj_path is not None
        _loaded_mmproj_path = mmproj_path
        _is_loading = False

    return True


def get_loaded_llama_or_raise(model_name: str) -> Llama:
    with _cache_lock:
        if _is_loading:
            raise ValueError(
                "A model is currently loading. Wait for loading to complete and retry."
            )

        if _loaded_model_name is None:
            raise ValueError(
                "No model is loaded. Load one with the model load endpoint before translating."
            )

        if _loaded_model_name != model_name:
            raise ValueError(
                f"Requested model '{model_name}' does not match loaded model '{_loaded_model_name}'. "
                "Load the requested model first."
            )

        llama = _model_cache.get(model_name)

    if llama is None:
        raise ValueError(
            "Model state is inconsistent (loaded model missing from cache). "
            "Reload the model and retry."
        )

    return llama


def _ensure_loaded_model_supports_vision_or_raise(model_name: str) -> Llama:
    llama = get_loaded_llama_or_raise(model_name)
    with _cache_lock:
        if not _loaded_model_is_vision_enabled:
            raise ValueError(
                f"Loaded model '{model_name}' is text-only. To enable image translation, place a "
                "matching mmproj .gguf file in the model folder and reload the model."
            )
    return llama


def _ensure_loaded_model_is_text_only_or_raise(model_name: str) -> Llama:
    llama = get_loaded_llama_or_raise(model_name)
    with _cache_lock:
        if _loaded_model_is_vision_enabled:
            raise ValueError(
                f"Loaded model '{model_name}' is vision-enabled. Text translation endpoints are "
                "disabled for vision-loaded models because the vision chat handler changes prompt "
                "formatting. Unload and reload without mmproj, then retry."
            )
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

    llama = _ensure_loaded_model_is_text_only_or_raise(model_name)

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

    llama = _ensure_loaded_model_is_text_only_or_raise(model_name)

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


def translate_image(
    model_name: str,
    source_lang_code: str,
    target_lang_code: str,
    image_bytes: bytes,
    image_mime_type: str,
    max_new_tokens: int = 200,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
) -> str:
    llama = _ensure_loaded_model_supports_vision_or_raise(model_name)
    source_lang_code = _normalize_lang_code(source_lang_code)
    target_lang_code = _normalize_lang_code(target_lang_code)
    source_lang = source_lang_code
    target_lang = target_lang_code

    prompt_text = _manual_image_prompt_text(
        source_lang=source_lang,
        source_lang_code=source_lang_code,
        target_lang=target_lang,
        target_lang_code=target_lang_code,
    )

    image_data_uri = _build_image_data_uri(image_bytes, image_mime_type)

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
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri},
                        "source_lang_code": source_lang_code,
                        "target_lang_code": target_lang_code,
                    }
                ],
            }
        ],
        max_tokens=max_new_tokens,
        **sampling_kwargs,
    )

    message = response["choices"][0]["message"]["content"]
    return _sanitize_translation_output(message.strip())


def experimental_translate_image(
    model_name: str,
    source_lang_code: str,
    target_lang_code: str,
    image_bytes: bytes,
    image_mime_type: str,
    max_new_tokens: int = 200,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
) -> str:
    source_lang_code = _normalize_lang_code(source_lang_code)
    target_lang_code = _normalize_lang_code(target_lang_code)
    return translate_image(
        model_name=model_name,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )