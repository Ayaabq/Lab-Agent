"""Webhook authentication and reply-loop prevention.

These guard the live path: an open webhook lets anyone make the bot send
WhatsApp messages, and a missed echo turns one customer message into an endless
exchange between the bot and itself.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.config import Secret, Settings, get_settings, reset_settings_cache
from app.escalation import EscalationStore
from app.whatsapp.dedupe import TTLDedupeCache
from app.whatsapp.service import WhatsAppService, set_whatsapp_service
from tests.unit.conftest import FakeRunner
from tests.unit.test_whatsapp_parsing import incoming
from tests.unit.test_whatsapp_service import FakeGreenClient

TOKEN = "webhook-token-abcdef123456"


def build_client(
    settings: Settings, green: FakeGreenClient | None = None
) -> TestClient:
    service = WhatsAppService(
        settings=settings,
        green_client=green or FakeGreenClient(),
        escalation_store=EscalationStore(),
        dedupe=TTLDedupeCache(ttl_seconds=60, max_entries=10),
        runner_provider=lambda: FakeRunner(),
    )
    set_whatsapp_service(service)
    api = FastAPI()
    api.include_router(router)
    return TestClient(api)


@pytest.fixture
def secured(settings, monkeypatch):
    """Settings with a webhook token, installed as the process-wide settings."""
    secured_settings = replace(
        settings, green_api=replace(settings.green_api, webhook_token=Secret(TOKEN))
    )
    monkeypatch.setattr("app.api.routes.get_settings", lambda: secured_settings)
    yield secured_settings
    set_whatsapp_service(None)


@pytest.fixture
def open_endpoint(settings, monkeypatch):
    """Settings with no webhook token configured."""
    monkeypatch.setattr("app.api.routes.get_settings", lambda: settings)
    yield settings
    set_whatsapp_service(None)


def test_valid_bearer_token_is_accepted(secured):
    client = build_client(secured)

    response = client.post(
        "/webhooks/green-api",
        json=incoming("مرحبا"),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


def test_bare_token_without_bearer_prefix_is_accepted(secured):
    """Tolerated because the console lets you paste a raw value."""
    client = build_client(secured)

    response = client.post(
        "/webhooks/green-api",
        json=incoming("مرحبا"),
        headers={"Authorization": TOKEN},
    )

    assert response.status_code == 200


def test_missing_header_is_rejected(secured):
    green = FakeGreenClient()
    client = build_client(secured, green)

    response = client.post("/webhooks/green-api", json=incoming("مرحبا"))

    assert response.status_code == 401
    # Nothing was processed and nothing was sent.
    assert green.sent == []


def test_wrong_token_is_rejected(secured):
    green = FakeGreenClient()
    client = build_client(secured, green)

    response = client.post(
        "/webhooks/green-api",
        json=incoming("مرحبا"),
        headers={"Authorization": "Bearer not-the-right-token"},
    )

    assert response.status_code == 401
    assert green.sent == []


def test_token_prefix_is_not_enough(secured):
    """Guards against a comparison that accepts any prefix."""
    client = build_client(secured)

    response = client.post(
        "/webhooks/green-api",
        json=incoming("مرحبا"),
        headers={"Authorization": f"Bearer {TOKEN[:10]}"},
    )

    assert response.status_code == 401


def test_rejection_does_not_echo_the_token(secured, caplog):
    client = build_client(secured)

    with caplog.at_level("WARNING"):
        client.post(
            "/webhooks/green-api",
            json=incoming("مرحبا"),
            headers={"Authorization": "Bearer wrong"},
        )

    assert TOKEN not in caplog.text


def test_endpoint_is_open_when_no_token_is_configured(open_endpoint):
    client = build_client(open_endpoint)

    response = client.post("/webhooks/green-api", json=incoming("مرحبا"))

    assert response.status_code == 200


def test_health_reports_whether_a_token_is_required(secured):
    client = build_client(secured)

    body = client.get("/health/integrations").json()

    assert body["green_api"]["webhook_token_required"] is True


def test_live_without_a_token_is_reported_as_a_problem(settings, monkeypatch):
    monkeypatch.setenv("GREEN_API_API_URL", "https://0000.api.greenapi.com")
    monkeypatch.setenv("GREEN_API_ID_INSTANCE", "000011112222")
    monkeypatch.setenv("GREEN_API_TOKEN_INSTANCE", "token")
    monkeypatch.setenv("GREEN_API_LIVE_ENABLED", "true")
    monkeypatch.delenv("GREEN_API_WEBHOOK_TOKEN", raising=False)
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    reset_settings_cache()
    try:
        problems = get_settings().problems()
    finally:
        reset_settings_cache()

    assert any("GREEN_API_WEBHOOK_TOKEN" in p for p in problems)


# --- reply-loop prevention ----------------------------------------------


@pytest.mark.asyncio
async def test_echo_of_our_own_sent_message_is_ignored(settings):
    """If the provider replays a message we sent, we must not answer it."""

    class EchoingGreen(FakeGreenClient):
        async def send_text(self, chat_id: str, message: str):
            from app.whatsapp.green_api import SendResult

            self.sent.append((chat_id, message))
            return SendResult(True, False, chat_id, provider_message_id="ECHO-1")

    green = EchoingGreen()
    runner = FakeRunner()
    service = WhatsAppService(
        settings=settings,
        green_client=green,
        escalation_store=EscalationStore(),
        dedupe=TTLDedupeCache(ttl_seconds=60, max_entries=10),
        runner_provider=lambda: runner,
    )

    await service.handle_webhook(incoming("مرحبا", id_message="IN-1"), wait=True)
    ack = await service.handle_webhook(
        incoming("مرحبا", id_message="ECHO-1"), wait=True
    )

    assert ack.status == "ignored"
    assert ack.reason == "self_message"
    assert len(runner.calls) == 1
    assert len(green.sent) == 1


def test_message_whose_sender_is_the_bot_is_ignored():
    own = "963900000000@c.us"

    result = __import__(
        "app.whatsapp.models", fromlist=["parse_webhook"]
    ).parse_webhook(incoming(wid=own, sender=own))

    assert result.skip_reason.value == "self_message"


def test_bot_own_chat_is_ignored():
    """The 'message yourself' chat: sender and chatId are both the bot."""
    own = "963900000000@c.us"

    result = __import__(
        "app.whatsapp.models", fromlist=["parse_webhook"]
    ).parse_webhook(incoming(chat_id=own, sender=own, wid=own))

    assert result.skip_reason.value == "self_message"


def test_contains_does_not_record():
    cache = TTLDedupeCache(ttl_seconds=60, max_entries=10)

    assert cache.contains("X") is False
    assert cache.contains("X") is False
    assert len(cache) == 0


def test_remember_then_contains():
    cache = TTLDedupeCache(ttl_seconds=60, max_entries=10)

    cache.remember("X")

    assert cache.contains("X") is True


@pytest.mark.parametrize("env", ["RENDER", "PUBLIC_SURFACE_ONLY"])
def test_a_hosted_deployment_without_a_token_rejects_every_webhook(
    open_endpoint, monkeypatch, env
):
    # Without a token the endpoint is open, which a laptop behind a tunnel can
    # tolerate while the console is being set up. A public host cannot: anyone
    # could make the bot message any number.
    monkeypatch.setenv(env, "true")
    green = FakeGreenClient()
    client = build_client(open_endpoint, green)

    response = client.post("/webhooks/green-api", json=incoming("مرحبا"))

    assert response.status_code == 401
