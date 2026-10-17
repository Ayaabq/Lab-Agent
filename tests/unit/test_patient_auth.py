"""Tenant authentication: caching, early renewal, and one login at a time.

Every test drives a fake clock and a mock transport. No network, no real
credentials, and nothing that could reach the live backend.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import Secret
from app.patients.auth import (
    LOGIN_PATH,
    KhatibAuthError,
    TenantAuthenticator,
)
from tests.unit.conftest import khatib_settings


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class LoginRecorder:
    """Mock transport that answers the login endpoint and counts requests."""

    def __init__(
        self,
        *,
        expires_in: int | str | None = 3600,
        status_code: int = 200,
        body: dict | None = None,
        delay: float = 0.0,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self._expires_in = expires_in
        self._status = status_code
        self._body = body
        self._delay = delay
        self._issued = 0

    @property
    def login_count(self) -> int:
        return len(self.requests)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        import json

        self.bodies.append(json.loads(request.content or b"{}"))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._status >= 400:
            return httpx.Response(self._status, json={"error": "nope"})
        if self._body is not None:
            return httpx.Response(200, json=self._body)
        self._issued += 1
        payload = {
            "accessToken": f"tenant-token-{self._issued}",
            "refreshToken": f"refresh-{self._issued}",
        }
        if self._expires_in is not None:
            payload["expiresIn"] = self._expires_in
        return httpx.Response(200, json=payload)


def build(recorder: LoginRecorder, clock: FakeClock, **overrides):
    client = recorder.client()
    return TenantAuthenticator(
        khatib_settings(**overrides),
        client_factory=lambda: client,
        time_source=clock,
    )


@pytest.mark.asyncio
async def test_login_sends_the_configured_client_credentials():
    recorder = LoginRecorder()
    auth = build(recorder, FakeClock())

    token = await auth.get_token()

    assert token == "tenant-token-1"
    assert recorder.requests[0].url.path == LOGIN_PATH
    assert recorder.bodies[0] == {
        "clientId": "test-khatib-client-id",
        "clientSecret": "test-khatib-client-secret",
        "grantType": "client_credentials",
    }


@pytest.mark.asyncio
async def test_a_cached_token_is_reused_without_logging_in_again():
    recorder = LoginRecorder()
    auth = build(recorder, FakeClock())

    first = await auth.get_token()
    second = await auth.get_token()

    assert first == second
    assert recorder.login_count == 1


@pytest.mark.asyncio
async def test_the_token_is_renewed_before_it_actually_expires():
    clock = FakeClock()
    recorder = LoginRecorder(expires_in=100)
    # Renew 10s early: at t+95 the backend would still accept the old token,
    # but a request leaving now could arrive after it lapsed.
    auth = build(recorder, clock, auth_refresh_margin_seconds=10)

    assert await auth.get_token() == "tenant-token-1"
    clock.advance(85)
    assert await auth.get_token() == "tenant-token-1"
    clock.advance(10)
    assert await auth.get_token() == "tenant-token-2"
    assert recorder.login_count == 2


@pytest.mark.asyncio
async def test_concurrent_callers_trigger_exactly_one_login():
    recorder = LoginRecorder(delay=0.02)
    auth = build(recorder, FakeClock())

    tokens = await asyncio.gather(*[auth.get_token() for _ in range(12)])

    assert recorder.login_count == 1
    assert set(tokens) == {"tenant-token-1"}


@pytest.mark.asyncio
async def test_concurrent_renewal_after_expiry_also_logs_in_once():
    clock = FakeClock()
    recorder = LoginRecorder(expires_in=100, delay=0.02)
    auth = build(recorder, clock, auth_refresh_margin_seconds=10)

    await auth.get_token()
    clock.advance(95)
    tokens = await asyncio.gather(*[auth.get_token() for _ in range(8)])

    assert recorder.login_count == 2
    assert set(tokens) == {"tenant-token-2"}


@pytest.mark.asyncio
async def test_unconfigured_credentials_never_reach_the_network():
    recorder = LoginRecorder()
    auth = build(recorder, FakeClock(), client_secret=Secret(""))

    with pytest.raises(KhatibAuthError):
        await auth.get_token()
    assert recorder.login_count == 0


@pytest.mark.asyncio
async def test_a_rejected_login_does_not_become_a_retry_loop():
    clock = FakeClock()
    recorder = LoginRecorder(status_code=401)
    auth = build(recorder, clock, auth_retry_cooldown_seconds=30)

    for _ in range(5):
        with pytest.raises(KhatibAuthError):
            await auth.get_token()

    # One real attempt; the rest were refused locally by the cooldown.
    assert recorder.login_count == 1
    clock.advance(31)
    with pytest.raises(KhatibAuthError):
        await auth.get_token()
    assert recorder.login_count == 2


@pytest.mark.asyncio
async def test_a_transport_failure_also_starts_the_cooldown():
    def explode(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    client = httpx.AsyncClient(transport=httpx.MockTransport(explode))
    auth = TenantAuthenticator(
        khatib_settings(), client_factory=lambda: client, time_source=FakeClock()
    )

    with pytest.raises(KhatibAuthError) as first:
        await auth.get_token()
    with pytest.raises(KhatibAuthError) as second:
        await auth.get_token()

    assert "ConnectError" in str(first.value)
    assert "cooling down" in str(second.value)


@pytest.mark.asyncio
async def test_a_missing_access_token_is_not_treated_as_success():
    recorder = LoginRecorder(body={"refreshToken": "r", "expiresIn": 3600})
    auth = build(recorder, FakeClock())

    with pytest.raises(KhatibAuthError):
        await auth.get_token()


@pytest.mark.asyncio
async def test_an_unusable_expires_in_falls_back_to_a_short_ttl():
    clock = FakeClock()
    recorder = LoginRecorder(expires_in="not-a-number")
    auth = build(recorder, clock, auth_refresh_margin_seconds=0)

    assert await auth.get_token() == "tenant-token-1"
    clock.advance(299)
    assert recorder.login_count == 1
    clock.advance(2)
    await auth.get_token()
    assert recorder.login_count == 2


@pytest.mark.asyncio
async def test_invalidate_only_drops_the_token_that_was_actually_rejected():
    recorder = LoginRecorder()
    auth = build(recorder, FakeClock())

    current = await auth.get_token()
    # A straggler 401 from a token issued before the current one must not
    # evict the token other callers are successfully using.
    auth.invalidate(used_token="an-older-token")
    assert await auth.get_token() == current
    assert recorder.login_count == 1

    auth.invalidate(used_token=current)
    assert await auth.get_token() == "tenant-token-2"


@pytest.mark.asyncio
async def test_the_refresh_token_is_never_stored():
    recorder = LoginRecorder()
    auth = build(recorder, FakeClock())
    await auth.get_token()

    # No refresh endpoint was specified, so there is nothing to use it with.
    # Keeping it would be storing a credential for a flow that does not exist.
    issued_refresh_token = "refresh-1"
    assert issued_refresh_token not in repr(auth.__dict__)
    assert issued_refresh_token not in repr(vars(auth._token))


@pytest.mark.asyncio
async def test_the_token_never_renders_itself():
    recorder = LoginRecorder()
    auth = build(recorder, FakeClock())
    await auth.get_token()

    assert "tenant-token-1" not in repr(auth._token)
