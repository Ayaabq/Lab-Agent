"""Builds the ADK model adapter for the configured LLM provider.

``LLM_PROVIDER`` picks exactly one provider. Nothing here falls back to
another one at runtime: a silent switch would move patient conversations to a
different vendor and a different bill without anyone deciding it. A provider
that cannot run — missing key, missing model, unknown name, LiteLLM not
installed — is replaced by :class:`UnavailableModel`, which fails every turn
*before* any network call. The WhatsApp service turns that failure into its
ordinary fallback reply, and the startup log names the missing variable.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from app.config import (
    LLM_PROVIDER_GEMINI,
    LLM_PROVIDER_OPENROUTER,
    OpenRouterSettings,
    Settings,
)
from app.logging_utils import get_logger

logger = get_logger(__name__)


class ModelUnavailableError(RuntimeError):
    """The configured provider cannot serve requests. Carries no secrets."""


class UnavailableModel(BaseLlm):
    """Stands in for a provider that is not configured. Never touches the network."""

    reason: str = "the configured LLM provider is unavailable"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        raise ModelUnavailableError(self.reason)
        yield  # pragma: no cover - makes this an async generator


def build_model(settings: Settings) -> BaseLlm:
    """The model adapter for ``settings.llm_provider``."""
    if problems := settings.llm_problems():
        if settings.llm_provider == LLM_PROVIDER_GEMINI:
            # Unchanged from before this module existed: a missing Gemini key
            # still builds the Gemini adapter, which fails at call time.
            return _gemini(settings)
        return _unavailable(problems[0])
    if settings.llm_provider == LLM_PROVIDER_OPENROUTER:
        return _openrouter(settings.openrouter)
    return _gemini(settings)


def _unavailable(reason: str) -> UnavailableModel:
    # Names a variable or a package, never a value.
    logger.error(
        "LLM provider unavailable; every chat will get the fallback reply",
        extra={"context": {"reason": reason}},
    )
    return UnavailableModel(model="unavailable", reason=reason)


def describe(settings: Settings) -> dict[str, str]:
    """Provider and model for logs and health output. Never includes a key."""
    if settings.llm_provider == LLM_PROVIDER_OPENROUTER:
        return {"provider": "openrouter", "model": settings.openrouter.model}
    if settings.llm_provider == LLM_PROVIDER_GEMINI:
        return {"provider": "gemini", "model": settings.gemini.model}
    return {"provider": "invalid", "model": ""}


def _gemini(settings: Settings) -> BaseLlm:
    return Gemini(
        model=settings.gemini.model,
        # The AI Studio free tier returns transient 503 "high demand" often
        # enough that 3 attempts surfaces it to the customer as a failure.
        # Riding it out is cheaper than an apology message.
        retry_options=types.HttpRetryOptions(attempts=5),
    )


_NOT_INSTALLED = "LLM_PROVIDER=openrouter needs the 'openrouter' extra installed."


def _openrouter(cfg: OpenRouterSettings) -> BaseLlm:
    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError:
        return _unavailable(_NOT_INSTALLED)
    if not _prepare_litellm():
        return _unavailable(_NOT_INSTALLED)

    return LiteLlm(
        model=cfg.litellm_model,
        api_key=cfg.api_key.get(),
        api_base=cfg.api_base,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout_seconds,
        # One request per model call. OpenRouter already moves between its
        # upstream providers on failure; a client-side retry on top of that
        # can bill the same turn twice.
        num_retries=0,
        max_retries=0,
        extra_body=_openrouter_body(cfg),
    )


def _openrouter_body(cfg: OpenRouterSettings) -> dict[str, Any]:
    """OpenRouter-only request fields, sent as-is in the JSON body."""
    return {
        "provider": {
            # Without this, OpenRouter may route to an upstream that ignores
            # `tools` — two of this model's providers do — and the agent would
            # answer without ever calling search_lab_knowledge or verify_patient.
            "require_parameters": True,
            "data_collection": cfg.data_collection,
        },
        "reasoning": {"effort": cfg.reasoning_effort},
    }


def _prepare_litellm() -> bool:
    """Import LiteLLM with its network side effects and verbose output off."""
    # Without this, importing litellm downloads its model price map from
    # GitHub — an outbound call at startup that nothing here needs.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import litellm
    except ImportError:
        return False
    # Message bodies hold patient names and birth dates; keep them out of any
    # LiteLLM logging callback, and keep LiteLLM's own banners out of our logs.
    litellm.turn_off_message_logging = True
    litellm.suppress_debug_info = True
    return True
