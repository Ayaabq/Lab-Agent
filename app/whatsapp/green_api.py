"""GREEN API client for WhatsApp.

Safety model
------------
``GREEN_API_LIVE_ENABLED`` is the single switch that allows an outbound
message to leave this process. While it is false, :meth:`GreenApiClient.send_text`
performs **no network call at all** — it logs a redacted dry-run line and
returns. The guard is checked before the request is built, not inside an error
handler, so it cannot be bypassed by a failure path.

Outbound endpoints are ``sendMessage`` and ``sendFileByUrl``. Neither is
retried: both are non-idempotent, and a retry after an ambiguous failure can
deliver a second copy — of a message, or of someone's medical report.

The read-only endpoints are:

* ``getStateInstance`` — non-mutating instance state.
* ``getSettings`` — non-mutating instance configuration.

``receiveNotification`` and ``deleteNotification`` are deliberately **not**
implemented: they consume and delete the provider-side notification queue,
which would interfere with the live instance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import GreenApiSettings, get_settings
from app.logging_utils import get_logger, mask_chat_id, summarize_text

logger = get_logger(__name__)

#: Only these are retried, and only for read-only requests.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_READ_RETRIES = 2


@dataclass(frozen=True)
class SendResult:
    """Outcome of an outbound message attempt.

    ``delivered`` means the provider *accepted* the request, which is not the
    same as the customer having received anything. Nothing in this project may
    tell a customer their report arrived on the strength of this flag or of
    ``provider_message_id``.
    """

    delivered: bool
    dry_run: bool
    chat_id: str
    reason: str = ""
    provider_message_id: str = ""
    #: True when we do not know whether the provider got the request — a
    #: timeout after the bytes went out. The send is NOT retried: the provider
    #: may already have queued it, and a retry would deliver a second copy of
    #: someone's medical report.
    ambiguous: bool = False


class GreenApiError(RuntimeError):
    """Raised for provider failures. Message never contains credentials."""


class GreenApiClient:
    def __init__(
        self,
        settings: GreenApiSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings().green_api
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    @property
    def settings(self) -> GreenApiSettings:
        return self._settings

    @property
    def live_enabled(self) -> bool:
        return self._settings.live_enabled

    def _url(self, method: str) -> str:
        instance = self._settings.id_instance.get()
        token = self._settings.token_instance.get()
        return f"{self._settings.api_url}/waInstance{instance}/{method}/{token}"

    async def _http(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self._settings.timeout_seconds),
                    transport=self._transport,
                )
            return self._client

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()

    # -- outbound ----------------------------------------------------------

    async def send_text(self, chat_id: str, message: str) -> SendResult:
        """Send a WhatsApp text message, or log a dry-run when live is disabled."""
        text = (message or "").strip()
        if not chat_id or not text:
            return SendResult(False, not self.live_enabled, chat_id, "empty_message")

        if not self.live_enabled:
            # No request is constructed and no socket is opened.
            logger.info(
                "DRY-RUN outbound WhatsApp message suppressed "
                "(GREEN_API_LIVE_ENABLED=false)",
                extra={
                    "context": {
                        "chat": mask_chat_id(chat_id),
                        "message": summarize_text(text, keep=40),
                    }
                },
            )
            return SendResult(False, True, chat_id, "live_disabled")

        if not self._settings.is_configured:
            raise GreenApiError("GREEN API is not fully configured")

        client = await self._http()
        try:
            # No retry on send: sendMessage is not idempotent and a retry after
            # an ambiguous failure can deliver the message twice.
            response = await client.post(
                self._url("sendMessage"),
                json={"chatId": chat_id, "message": text},
            )
        except httpx.HTTPError as exc:
            raise GreenApiError(f"send failed: {type(exc).__name__}") from None

        if response.status_code >= 400:
            raise GreenApiError(f"send rejected with status {response.status_code}")

        payload = _safe_json(response)
        logger.info(
            "sent WhatsApp message",
            extra={
                "context": {
                    "chat": mask_chat_id(chat_id),
                    "message": summarize_text(text, keep=40),
                }
            },
        )
        return SendResult(
            True, False, chat_id, provider_message_id=str(payload.get("idMessage", ""))
        )

    async def send_file_by_url(
        self,
        chat_id: str,
        url_file: str,
        file_name: str,
        caption: str = "",
    ) -> SendResult:
        """Send a file the provider fetches by URL, or log a dry run.

        ``url_file`` addresses a patient's medical report. It is never logged,
        never returned, and never put in an error message — not even on the
        failure paths, where a URL in a traceback is the easiest way for one
        to end up in a log aggregator.

        The caller is responsible for having validated the URL's scheme and
        host: by the time it reaches here, the only question left is whether
        the provider accepts it.
        """
        if not chat_id or not url_file or not file_name:
            return SendResult(False, not self.live_enabled, chat_id, "empty_request")

        if not self.live_enabled:
            # Same master switch as text. No request is constructed and no
            # socket is opened; the URL is not written to the log line.
            logger.info(
                "DRY-RUN outbound WhatsApp file suppressed "
                "(GREEN_API_LIVE_ENABLED=false)",
                extra={
                    "context": {
                        "chat": mask_chat_id(chat_id),
                        "fileName": file_name,
                        "caption": summarize_text(caption, keep=40),
                    }
                },
            )
            return SendResult(False, True, chat_id, "live_disabled")

        if not self._settings.is_configured:
            raise GreenApiError("GREEN API is not fully configured")

        payload: dict[str, Any] = {
            "chatId": chat_id,
            "urlFile": url_file,
            "fileName": file_name,
        }
        if caption:
            payload["caption"] = caption

        client = await self._http()
        try:
            # No retry, exactly as for sendMessage: sendFileByUrl is not
            # idempotent.
            response = await client.post(self._url("sendFileByUrl"), json=payload)
        except httpx.TimeoutException:
            # The request may have been queued provider-side. Report it as
            # ambiguous and let the caller tell the customer something that is
            # true either way.
            logger.error(
                "sendFileByUrl timed out; not retried",
                extra={"context": {"chat": mask_chat_id(chat_id)}},
            )
            return SendResult(
                False, False, chat_id, "send_timeout_ambiguous", ambiguous=True
            )
        except httpx.HTTPError as exc:
            raise GreenApiError(f"file send failed: {type(exc).__name__}") from None

        if response.status_code >= 400:
            # The body can quote the URL back at us, so only the status is kept.
            raise GreenApiError(
                f"file send rejected with status {response.status_code}"
            )

        data = _safe_json(response)
        logger.info(
            "sent WhatsApp file",
            extra={
                "context": {
                    "chat": mask_chat_id(chat_id),
                    "fileName": file_name,
                    "captioned": str(bool(caption)),
                }
            },
        )
        return SendResult(
            True, False, chat_id, provider_message_id=str(data.get("idMessage", ""))
        )

    # -- read-only ---------------------------------------------------------

    async def get_state_instance(self) -> dict[str, Any]:
        """Non-mutating instance state, e.g. ``{"stateInstance": "authorized"}``."""
        return await self._get("getStateInstance")

    async def get_settings_snapshot(self) -> dict[str, Any]:
        """Non-mutating instance settings (includes the configured webhook URL)."""
        return await self._get("getSettings")

    async def _get(self, method: str) -> dict[str, Any]:
        if not self._settings.is_configured:
            raise GreenApiError("GREEN API is not fully configured")
        client = await self._http()
        last_error = "unknown"
        for attempt in range(_MAX_READ_RETRIES + 1):
            try:
                response = await client.get(self._url(method))
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
            else:
                if response.status_code < 400:
                    return _safe_json(response)
                last_error = f"status {response.status_code}"
                if response.status_code not in _RETRY_STATUS:
                    break
            if attempt < _MAX_READ_RETRIES:
                await asyncio.sleep(0.5 * (2**attempt))
        raise GreenApiError(f"{method} failed: {last_error}")


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {"result": data}


_client: GreenApiClient | None = None


def get_green_api_client() -> GreenApiClient:
    global _client
    if _client is None:
        _client = GreenApiClient()
    return _client


def set_green_api_client(client: GreenApiClient | None) -> None:
    """Swap the process-wide client. Tests use this to inject a fake."""
    global _client
    _client = client
