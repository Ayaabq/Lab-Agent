"""Minimal Upstash Redis REST client on ``httpx``.

Upstash executes a Redis command sent as a JSON array to the database URL,
a list of them to ``/pipeline``, and a list run atomically (MULTI/EXEC) to
``/multi-exec``, authenticated with ``Authorization: Bearer <token>``.

Every failure — network, HTTP status, or a Redis error reply — surfaces as
:class:`StateUnavailable`, carrying a short reason code and never the URL,
the token, a key or a value.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.config import Secret
from app.logging_utils import get_logger

logger = get_logger(__name__)

Command = list[Any]


class StateUnavailable(RuntimeError):
    """Shared state could not be read or written. Never means "proceed"."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UpstashClient:
    def __init__(
        self,
        url: str,
        token: Secret,
        *,
        timeout_seconds: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._timeout = timeout_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _http(self) -> httpx.AsyncClient:
        # An AsyncClient belongs to the loop that created it. Serving uses one
        # loop; tests use one per test, so rebuild rather than share across.
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                transport=self._transport,
                headers={"Authorization": f"Bearer {self._token.get()}"},
            )
            self._loop = loop
        return self._client

    async def _post(self, path: str, body: Any) -> Any:
        try:
            response = await self._http().post(self._url + path, json=body)
        except httpx.TimeoutException:
            raise StateUnavailable("redis_timeout") from None
        except httpx.HTTPError:
            raise StateUnavailable("redis_unreachable") from None
        if response.status_code != 200:
            raise StateUnavailable(f"redis_http_{response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise StateUnavailable("redis_bad_response") from None

    @staticmethod
    def _unwrap(item: Any) -> Any:
        if not isinstance(item, dict) or "error" in item or "result" not in item:
            # The error text can quote a key; the reason code is enough.
            raise StateUnavailable("redis_command_error")
        return item["result"]

    @staticmethod
    def _encode(command: Command) -> list[str]:
        return [str(part) for part in command]

    async def execute(self, *command: Any) -> Any:
        return self._unwrap(await self._post("", self._encode(list(command))))

    async def pipeline(self, commands: list[Command]) -> list[Any]:
        """Several commands in one round trip. Not atomic."""
        return await self._batch("/pipeline", commands)

    async def transaction(self, commands: list[Command]) -> list[Any]:
        """Several commands in one round trip, applied atomically."""
        return await self._batch("/multi-exec", commands)

    async def _batch(self, path: str, commands: list[Command]) -> list[Any]:
        body = await self._post(path, [self._encode(c) for c in commands])
        if not isinstance(body, list) or len(body) != len(commands):
            raise StateUnavailable("redis_bad_response")
        return [self._unwrap(item) for item in body]

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover - closing is best effort
                logger.debug("closing the Upstash client failed")
            self._client = None


class UnavailableClient:
    """Stands in when Redis was requested but cannot be used.

    Every call fails, so every store backed by it fails closed. Falling back to
    process memory instead would make a misconfigured Vercel deployment look
    reliable while it is not.
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def execute(self, *command: Any) -> Any:
        raise StateUnavailable(self._reason)

    async def pipeline(self, commands: list[Command]) -> list[Any]:
        raise StateUnavailable(self._reason)

    async def transaction(self, commands: list[Command]) -> list[Any]:
        raise StateUnavailable(self._reason)

    async def aclose(self) -> None:
        return None
