from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from lib.llama_service import translate_text
from lib.locales import SUPPORTED_LOCALES
from lib.model_registry import list_model_names

app = FastAPI(title="FastAPI Gemma Translate")

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


class TranslationResponse(BaseModel):
    model: str
    translated_text: str


@app.get("/models")
def list_models():
    return {"models": list_model_names()}


@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    try:
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