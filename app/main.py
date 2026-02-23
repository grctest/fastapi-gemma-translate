from __future__ import annotations

import asyncio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator, model_validator

from lib.llama_service import (
    experimental_translate_text,
    get_loaded_model_name,
    is_loaded_model_vision_enabled,
    is_model_loaded,
    is_model_loading,
    load_model,
    translate_image,
    translate_text,
    unload_model as unload_loaded_model,
)
from lib.locales import EXPERIMENTAL_SUPPORTED_LOCALES, SUPPORTED_LOCALES
from lib.model_registry import list_model_names

app = FastAPI(title="FastAPI Gemma Translate")
image_inference_lock = asyncio.Lock()

class TranslationRequest(BaseModel):
    model: str = Field(..., description="Model name present in app/models")
    source_lang_code: str = Field(..., min_length=2)
    target_lang_code: str = Field(..., min_length=2)
    text: str = Field(..., min_length=1)
    content_type: str = Field("text", description="Only 'text' is supported")
    max_new_tokens: int = Field(200, ge=1, le=2000)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0, le=1000)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    repeat_penalty: float | None = Field(None, ge=0.0, le=2.0)
    presence_penalty: float | None = Field(None, ge=0.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=0.0, le=2.0)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        available = list_model_names()
        if value not in available:
            allowed = ", ".join(available) if available else "(no models found)"
            raise ValueError(f"Model must be one of: {allowed}")
        return value

    @field_validator("source_lang_code")
    @classmethod
    def validate_source_lang_code(cls, value: str) -> str:
        if value not in SUPPORTED_LOCALES:
            raise ValueError("Unsupported source language code")
        return value

    @field_validator("target_lang_code")
    @classmethod
    def validate_target_lang_code(cls, value: str) -> str:
        if value not in SUPPORTED_LOCALES:
            raise ValueError("Unsupported target language code")
        return value

    @model_validator(mode="after")
    def validate_languages_differ(self) -> "TranslationRequest":
        if self.source_lang_code == self.target_lang_code:
            raise ValueError("source_lang_code and target_lang_code must differ")
        return self


class ExperimentalTranslationRequest(BaseModel):
    model: str = Field(..., description="Model name present in app/models")
    source_lang_code: str = Field(..., min_length=2)
    target_lang_code: str = Field(..., min_length=2)
    text: str = Field(..., min_length=1)
    content_type: str = Field("text", description="Only 'text' is supported")
    max_new_tokens: int = Field(200, ge=1, le=2000)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0, le=1000)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    repeat_penalty: float | None = Field(None, ge=0.0, le=2.0)
    presence_penalty: float | None = Field(None, ge=0.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=0.0, le=2.0)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        available = list_model_names()
        if value not in available:
            allowed = ", ".join(available) if available else "(no models found)"
            raise ValueError(f"Model must be one of: {allowed}")
        return value

    @field_validator("source_lang_code")
    @classmethod
    def validate_source_lang_code(cls, value: str) -> str:
        if value not in SUPPORTED_LOCALES and value not in EXPERIMENTAL_SUPPORTED_LOCALES:
            raise ValueError("Unsupported source language code")
        return value

    @field_validator("target_lang_code")
    @classmethod
    def validate_target_lang_code(cls, value: str) -> str:
        if value not in SUPPORTED_LOCALES and value not in EXPERIMENTAL_SUPPORTED_LOCALES:
            raise ValueError("Unsupported target language code")
        return value

    @model_validator(mode="after")
    def validate_languages_differ(self) -> "ExperimentalTranslationRequest":
        if self.source_lang_code == self.target_lang_code:
            raise ValueError("source_lang_code and target_lang_code must differ")
        return self


class TranslationResponse(BaseModel):
    model: str
    translated_text: str


class LoadModelRequest(BaseModel):
    model: str = Field(..., description="Model name present in app/models")
    mmproj: str | None = Field(
        None,
        description=(
            "Optional mmproj .gguf path for vision support. Relative paths are resolved "
            "from the selected model folder."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        available = list_model_names()
        if value not in available:
            allowed = ", ".join(available) if available else "(no models found)"
            raise ValueError(f"Model must be one of: {allowed}")
        return value


class LoadModelResponse(BaseModel):
    model: str
    loaded: bool
    message: str


class ModelStatusResponse(BaseModel):
    loaded: bool
    loading: bool
    loaded_model: str | None
    vision_enabled: bool
    requested_model: str | None = None
    requested_model_loaded: bool | None = None


class UnloadModelRequest(BaseModel):
    model: str | None = Field(
        None,
        description="Optional model name to unload. If omitted, unloads the currently loaded model.",
    )


class UnloadModelResponse(BaseModel):
    unloaded: bool
    model: str
    message: str


@app.get("/models")
def list_models():
    return {"models": list_model_names()}


@app.post("/model/load", response_model=LoadModelResponse)
def load_requested_model(request: LoadModelRequest):
    try:
        did_load = load_model(request.model, mmproj=request.mmproj)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if did_load:
        return LoadModelResponse(
            model=request.model,
            loaded=True,
            message=f"Model '{request.model}' loaded successfully.",
        )

    return LoadModelResponse(
        model=request.model,
        loaded=True,
        message=f"Model '{request.model}' is already loaded.",
    )


@app.get("/model/status", response_model=ModelStatusResponse)
def model_status(model: str | None = None):
    loaded_model = get_loaded_model_name()
    loading = is_model_loading()
    vision_enabled = is_loaded_model_vision_enabled()

    if model is None:
        return ModelStatusResponse(
            loaded=loaded_model is not None,
            loading=loading,
            loaded_model=loaded_model,
            vision_enabled=vision_enabled,
        )

    return ModelStatusResponse(
        loaded=loaded_model is not None,
        loading=loading,
        loaded_model=loaded_model,
        vision_enabled=vision_enabled,
        requested_model=model,
        requested_model_loaded=is_model_loaded(model),
    )


@app.post("/model/unload", response_model=UnloadModelResponse)
def unload_requested_model(request: UnloadModelRequest):
    if is_model_loading():
        raise HTTPException(
            status_code=400,
            detail="A model is currently loading. Wait for loading to complete before unloading.",
        )

    loaded_model = get_loaded_model_name()
    if loaded_model is None:
        raise HTTPException(
            status_code=400,
            detail="No model is currently loaded.",
        )

    requested_model = request.model or loaded_model
    if requested_model != loaded_model:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Requested unload model '{requested_model}' does not match loaded model "
                f"'{loaded_model}'."
            ),
        )

    did_unload = unload_loaded_model(requested_model)
    if not did_unload:
        raise HTTPException(
            status_code=409,
            detail=(
                "Model unload failed due to inconsistent state. "
                "Retry the request or reload the model first."
            ),
        )

    return UnloadModelResponse(
        unloaded=True,
        model=requested_model,
        message=f"Model '{requested_model}' unloaded successfully.",
    )


def _validate_image_route_inputs(model: str, source_lang_code: str, target_lang_code: str) -> None:
    available = list_model_names()
    if model not in available:
        allowed = ", ".join(available) if available else "(no models found)"
        raise ValueError(f"Model must be one of: {allowed}")

    if source_lang_code not in SUPPORTED_LOCALES and source_lang_code not in EXPERIMENTAL_SUPPORTED_LOCALES:
        raise ValueError("Unsupported source language code")
    if target_lang_code not in SUPPORTED_LOCALES and target_lang_code not in EXPERIMENTAL_SUPPORTED_LOCALES:
        raise ValueError("Unsupported target language code")
    if source_lang_code == target_lang_code:
        raise ValueError("source_lang_code and target_lang_code must differ")


def _ensure_image_translation_ready(model: str) -> None:
    if is_model_loading():
        raise ValueError("A model is currently loading. Wait for loading to complete and retry.")
    if not is_model_loaded(model):
        raise ValueError(
            f"Requested model '{model}' is not loaded. Load it first using /model/load."
        )
    if not is_loaded_model_vision_enabled():
        raise ValueError(
            "Image translation requires loading the model with an mmproj .gguf file. "
            "Reload via /model/load and provide the 'mmproj' field."
        )


def _ensure_text_translation_ready(model: str) -> None:
    if is_model_loading():
        raise ValueError("A model is currently loading. Wait for loading to complete and retry.")
    if not is_model_loaded(model):
        raise ValueError(
            f"Requested model '{model}' is not loaded. Load it first using /model/load."
        )
    if is_loaded_model_vision_enabled():
        raise ValueError(
            "Text translation endpoints are disabled when the loaded model uses mmproj/vision "
            "chat handling. Unload and reload the model without 'mmproj' to use /translate or "
            "/experimental_translation."
        )

def _validate_upload_is_image(file: UploadFile) -> str:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise ValueError("Uploaded file must be an image (content-type image/*).")
    return content_type


@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    try:
        _ensure_text_translation_ready(request.model)
        translated = translate_text(
            model_name=request.model,
            source_lang_code=request.source_lang_code,
            target_lang_code=request.target_lang_code,
            text=request.text,
            content_type=request.content_type,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repeat_penalty=request.repeat_penalty,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TranslationResponse(model=request.model, translated_text=translated)


@app.post("/experimental_translation", response_model=TranslationResponse)
def experimental_translate(request: ExperimentalTranslationRequest):
    try:
        _ensure_text_translation_ready(request.model)
        translated = experimental_translate_text(
            model_name=request.model,
            source_lang_code=request.source_lang_code,
            target_lang_code=request.target_lang_code,
            text=request.text,
            content_type=request.content_type,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repeat_penalty=request.repeat_penalty,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TranslationResponse(model=request.model, translated_text=translated)


@app.post("/translate_image", response_model=TranslationResponse)
async def translate_image_route(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    model: str = Form(...),
    source_lang_code: str = Form(...),
    target_lang_code: str = Form(...),
    max_new_tokens: int = Form(200, ge=1, le=2000),
    temperature: float | None = Form(None, ge=0.0, le=2.0),
    top_p: float | None = Form(None, ge=0.0, le=1.0),
    top_k: int | None = Form(None, ge=0, le=1000),
    min_p: float | None = Form(None, ge=0.0, le=1.0),
    repeat_penalty: float | None = Form(None, ge=0.0, le=2.0),
    presence_penalty: float | None = Form(None, ge=0.0, le=2.0),
    frequency_penalty: float | None = Form(None, ge=0.0, le=2.0),
):
    try:
        _validate_image_route_inputs(model=model, source_lang_code=source_lang_code, target_lang_code=target_lang_code)
        _ensure_image_translation_ready(model)

        image_bytes = None
        image_mime_type = None

        if file is not None and file.filename:
            image_mime_type = _validate_upload_is_image(file)
            image_bytes = await file.read()
            if not image_bytes:
                raise ValueError("Uploaded image is empty.")

        if image_bytes is None and text is None:
            raise ValueError("Either an image file or text must be provided.")

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with image_inference_lock:
        try:
            translated = translate_image(
                model_name=model,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                text=text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TranslationResponse(model=model, translated_text=translated)