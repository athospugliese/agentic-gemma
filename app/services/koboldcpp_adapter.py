"""KoboldCpp adapter - uses Ollama-compatible API emulated by KoboldCpp."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

import httpx

from app.models import Message
from app.services.llm_adapter import (
    LLMAdapter,
    LLMError,
    LLMErrorCategory,
    StreamEvent,
    ToolDefinition,
)
from app.services.ollama_adapter import OllamaAdapter, classify_ollama_error

logger = logging.getLogger(__name__)


class KoboldCppAdapter(OllamaAdapter):
    """KoboldCpp adapter using its Ollama-compatible /api/chat endpoint.

    KoboldCpp exposes Ollama, OpenAI, and native KoboldCpp APIs on the same
    port.  This adapter reuses the Ollama protocol but adds KoboldCpp-specific
    health checking via the native ``/api/v1/info`` endpoint, and adjusts
    error classification for KoboldCpp quirks.
    """

    def __init__(self, base_url: str = "http://localhost:5001"):
        super().__init__(base_url=base_url)

    # -- Health / introspection via native KoboldCpp API --

    async def check_health(self) -> dict[str, Any]:
        """Ping KoboldCpp and return server info (model loaded, version, etc.)."""
        try:
            resp = await self._http.get("/api/v1/info", timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(
                category=LLMErrorCategory.RETRYABLE,
                message=f"KoboldCpp health check failed: {exc}",
                original=exc,
            ) from exc

    async def get_loaded_model(self) -> str | None:
        """Return the model name currently loaded in KoboldCpp, or None."""
        try:
            info = await self.check_health()
            return info.get("model_name") or info.get("model")
        except Exception:
            return None

    async def get_available_models(self) -> list[str]:
        """List models via Ollama-compat /api/tags endpoint."""
        try:
            resp = await self._http.get("/api/tags", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            # Fall back to native API
            model = await self.get_loaded_model()
            return [model] if model else []
