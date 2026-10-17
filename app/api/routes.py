"""Webhook and health endpoints.

Mounted onto the ADK-generated FastAPI app in ``app/fast_api_app.py``.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.api.public_surface import hosted_publicly
from app.config import get_settings
from app.escalation import get_escalation_store
from app.logging_utils import get_logger
from app.patients.store import get_verification_store
from app.whatsapp.service import get_whatsapp_service

logger = get_logger(__name__)

router = APIRouter(tags=["khatib"])


def _authorize(request: Request) -> None:
    """Reject the request unless it carries the configured webhook token.

    GREEN API sends the value of its ``webhookUrlToken`` setting as
    ``Authorization: Bearer <token>``. When no token is configured the endpoint
    stays open — that is the only way to receive anything before the token is
    set in the console — but it is reported as a problem by
    ``/health/integrations`` whenever live sending is on.

    A hosted deployment has no such bootstrap period and is reachable by
    anyone, so there a missing token rejects every request instead.
    """
    expected = get_settings().green_api.webhook_token
    if not expected.is_set:
        if hosted_publicly():
            logger.error(
                "rejected webhook: GREEN_API_WEBHOOK_TOKEN is not set on a "
                "hosted deployment"
            )
            raise HTTPException(status_code=401, detail="unauthorized")
        return

    header = request.headers.get("authorization", "")
    presented = header[7:] if header[:7].lower() == "bearer " else header
    # Constant-time comparison: a plain == leaks the token through timing.
    if not presented or not hmac.compare_digest(presented, expected.get()):
        logger.warning(
            "rejected webhook with missing or invalid Authorization header",
            extra={"context": {"had_header": str(bool(header))}},
        )
        # 401 is deliberate here, unlike the 200s below: an unauthenticated
        # caller is not GREEN API, so there is no delivery to retry.
        raise HTTPException(status_code=401, detail="unauthorized")


@router.post("/webhooks/green-api")
async def green_api_webhook(request: Request) -> dict[str, Any]:
    """Receive a GREEN API notification.

    Answers 200 for anything authentic, including payloads we ignore — a non-2xx
    makes GREEN API redeliver, which would duplicate work forever for events we
    will never act on. The body says whether the message was accepted and why not.
    """
    _authorize(request)

    try:
        payload = await request.json()
    except Exception:
        logger.warning("webhook body was not valid JSON")
        return {"status": "ignored", "reason": "invalid_json"}

    if not isinstance(payload, dict):
        return {"status": "ignored", "reason": "invalid_payload"}

    ack = await get_whatsapp_service().handle_webhook(payload)
    return ack.as_dict()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness. No external calls, safe to poll."""
    return {"status": "ok"}


@router.get("/health/integrations")
async def integrations_health(probe: bool = False) -> dict[str, Any]:
    """Configuration readiness, and optionally a read-only Weaviate probe.

    ``?probe=true`` runs a schema read against Weaviate. It never writes, and
    it never touches GREEN API — checking the provider is a separate, explicit
    script so a health poll can never act on the live instance.
    """
    settings = get_settings()
    report: dict[str, Any] = {
        "status": "ok" if not settings.problems() else "degraded",
        "problems": settings.problems(),
        "gemini": {
            "model": settings.gemini.model,
            "api_key_configured": settings.gemini.api_key.is_set,
            "using_vertex_ai": settings.gemini.use_vertexai,
        },
        "weaviate": {
            "configured": settings.weaviate.is_configured,
            "collection": settings.weaviate.collection or "<auto-discover>",
            "query_mode": settings.weaviate.query_mode,
            "min_certainty": settings.weaviate.min_certainty,
        },
        "green_api": {
            "configured": settings.green_api.is_configured,
            # The safety switch is the first thing an operator should see.
            "live_sending_enabled": settings.green_api.live_enabled,
            "ignore_groups": settings.green_api.ignore_groups,
            "webhook_token_required": settings.green_api.webhook_token.is_set,
        },
        "khatib_backend": {
            # Configuration only. No login is attempted from a health poll:
            # that would spend a tenant credential on every probe and could
            # trip the backend's own rate limits.
            "configured": settings.khatib.is_configured,
            "base_url_set": bool(settings.khatib.base_url),
            "timeout_seconds": settings.khatib.timeout_seconds,
            "max_verification_attempts": settings.khatib.max_verification_attempts,
            "active_verifications": get_verification_store().count(),
            # Section 5: stays false until the backend publishes the contract.
            "result_image_support": bool(settings.khatib.result_image_field),
        },
        "escalation": {"awaiting_human": get_escalation_store().pending_count()},
        "whatsapp": get_whatsapp_service().stats(),
    }

    if probe:
        report["weaviate"]["probe"] = _probe_weaviate()
        if report["weaviate"]["probe"].get("status") != "ok":
            report["status"] = "degraded"
    return report


def _probe_weaviate() -> dict[str, Any]:
    from app.rag.weaviate_store import get_store

    try:
        return {"status": "ok", **get_store().describe()}
    except Exception as exc:
        logger.error(
            "weaviate probe failed",
            extra={"context": {"error_type": type(exc).__name__}},
        )
        # Type name only: the exception text can carry the cluster host.
        return {"status": "error", "error_type": type(exc).__name__}
