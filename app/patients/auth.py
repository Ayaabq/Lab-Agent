"""Tenant authentication against the Khatib backend.

This is infrastructure, not a tool. The WhatsApp customer never supplies these
credentials, the model never sees them, and no agent-facing surface can trigger
a login directly — a login happens only as a side effect of a patient call that
has already passed its own checks.

Design notes
------------
* **One login at a time.** A burst of inbound messages must not turn into a
  burst of ``/auth/tenant/login`` calls. An ``asyncio.Lock`` plus a re-check
  inside it means the first caller logs in and the rest wait for that result.
* **Renew early.** The token is treated as expired ``auth_refresh_margin_seconds``
  before the backend would say so, so a request cannot leave here valid and
  arrive expired.
* **No refresh endpoint.** The backend returns a ``refreshToken``, but no
  refresh route has been specified. Inventing one would be a guess, so renewal
  is a fresh login. The refresh token is deliberately not stored.
* **No retry loop.** A failed login starts a cooldown; callers inside it fail
  fast instead of hammering a backend that is already unhappy.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import KhatibApiSettings, get_settings
from app.logging_utils import get_logger

logger = get_logger(__name__)

LOGIN_PATH = "/api/v1.0/auth/tenant/login"
GRANT_TYPE = "client_credentials"

#: Used when the backend omits ``expiresIn`` or sends something unparsable.
#: Short, so a bad value costs an extra login rather than a dead token.
_FALLBACK_EXPIRES_IN_SECONDS = 300


class KhatibAuthError(RuntimeError):
    """Tenant authentication failed. The message never carries credentials."""


@dataclass(frozen=True)
class TenantToken:
    """A tenant access token and the moment we stop trusting it.

    ``access_token`` is a credential: it goes into an ``Authorization`` header
    and nowhere else. It is not in ``__repr__`` by accident — see below.
    """

    access_token: str
    expires_at: float

    def is_fresh(self, *, now: float, margin: float) -> bool:
        return now < self.expires_at - margin

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TenantToken(access_token=***, expires_at={self.expires_at:.0f})"


class TenantAuthenticator:
    """Caches one tenant token for the configured tenant.

    Scoped per instance, and the process holds one instance per tenant
    (:func:`get_tenant_authenticator`). Two different tenants therefore never
    share a cached token, even in the same process.
    """

    def __init__(
        self,
        settings: KhatibApiSettings | None = None,
        *,
        client_factory: Any = None,
        time_source: Any = None,
    ) -> None:
        self._settings = settings or get_settings().khatib
        self._client_factory = client_factory
        self._now = time_source or time.monotonic
        self._token: TenantToken | None = None
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None
        #: Set after a failed login; no further attempt until this passes.
        self._retry_not_before: float = 0.0
        self._logins = 0

    @property
    def settings(self) -> KhatibApiSettings:
        return self._settings

    @property
    def login_count(self) -> int:
        """How many logins actually happened. Used by tests, not by callers."""
        return self._logins

    async def get_token(self) -> str:
        """Return a usable tenant access token, logging in only if needed."""
        if not self._settings.is_configured:
            raise KhatibAuthError("Khatib backend credentials are not configured")

        margin = self._settings.auth_refresh_margin_seconds
        cached = self._token
        if cached is not None and cached.is_fresh(now=self._now(), margin=margin):
            return cached.access_token

        async with self._lock:
            # Re-check: another caller may have refreshed while we waited, and
            # that is the whole point of holding the lock.
            cached = self._token
            if cached is not None and cached.is_fresh(now=self._now(), margin=margin):
                return cached.access_token

            now = self._now()
            if now < self._retry_not_before:
                raise KhatibAuthError("tenant login is cooling down after a failure")

            token = await self._login()
            self._token = token
            return token.access_token

    def invalidate(self, *, used_token: str | None = None) -> None:
        """Drop the cached token after the backend rejected it.

        ``used_token`` guards against a slow 401 from a previous token wiping a
        newer one that another caller just obtained.
        """
        cached = self._token
        if cached is None:
            return
        if used_token is not None and used_token != cached.access_token:
            return
        self._token = None

    def reset(self) -> None:
        """Forget the token and any cooldown. Tests and shutdown only."""
        self._token = None
        self._retry_not_before = 0.0

    # -- internals ---------------------------------------------------------

    async def _login(self) -> TenantToken:
        client = await self._http()
        url = f"{self._settings.base_url}{LOGIN_PATH}"
        payload = {
            "clientId": self._settings.client_id.get(),
            "clientSecret": self._settings.client_secret.get(),
            "grantType": GRANT_TYPE,
        }
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            self._start_cooldown()
            # Type name only: an httpx message can echo the request URL.
            raise KhatibAuthError(
                f"tenant login failed: {type(exc).__name__}"
            ) from None

        if response.status_code >= 400:
            self._start_cooldown()
            logger.error(
                "tenant login rejected by the Khatib backend",
                extra={"context": {"status": str(response.status_code)}},
            )
            raise KhatibAuthError(
                f"tenant login rejected with status {response.status_code}"
            )

        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self._start_cooldown()
            raise KhatibAuthError("tenant login returned an unexpected body")

        access_token = _first_str(body, ("accessToken", "access_token"))
        if not access_token:
            self._start_cooldown()
            raise KhatibAuthError("tenant login returned no access token")

        expires_in = _positive_int(
            _first_value(body, ("expiresIn", "expires_in")),
            default=_FALLBACK_EXPIRES_IN_SECONDS,
        )
        self._retry_not_before = 0.0
        self._logins += 1
        logger.info(
            "obtained a tenant access token",
            extra={
                "context": {
                    "tenant": self._settings.tenant_key,
                    "expires_in_seconds": str(expires_in),
                }
            },
        )
        # The response also carries a refreshToken. No refresh endpoint has
        # been specified, so it is not stored: renewal is a fresh login.
        return TenantToken(
            access_token=access_token, expires_at=self._now() + expires_in
        )

    def _start_cooldown(self) -> None:
        self._retry_not_before = (
            self._now() + self._settings.auth_retry_cooldown_seconds
        )

    async def _http(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=request_timeout(self._settings))
        return self._client

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()


def request_timeout(settings: KhatibApiSettings) -> httpx.Timeout:
    """Bounded on every phase, so no call can hang a chat turn indefinitely."""
    return httpx.Timeout(
        settings.timeout_seconds, connect=settings.connect_timeout_seconds
    )


def _first_value(body: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in body:
            return body[key]
    return None


def _first_str(body: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _first_value(body, keys)
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


_authenticator: TenantAuthenticator | None = None


def get_tenant_authenticator() -> TenantAuthenticator:
    global _authenticator
    if _authenticator is None:
        _authenticator = TenantAuthenticator()
    return _authenticator


def set_tenant_authenticator(authenticator: TenantAuthenticator | None) -> None:
    """Swap the process-wide authenticator. Tests use this to inject a fake."""
    global _authenticator
    _authenticator = authenticator
