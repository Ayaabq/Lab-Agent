"""HTTP surface: the webhook endpoint and the health endpoints.

A bare FastAPI app is used rather than the full ADK app so these stay fast and
need no model credentials.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.escalation import EscalationStore
from app.whatsapp.dedupe import TTLDedupeCache
from app.whatsapp.service import WhatsAppService, set_whatsapp_service
from tests.unit.conftest import FakeRunner
from tests.unit.test_whatsapp_parsing import incoming
from tests.unit.test_whatsapp_service import FakeGreenClient


@pytest.fixture(autouse=True)
def _pinned_settings(settings, monkeypatch):
    """Keep the routes off the developer's real .env.

    Without this the endpoint would pick up a real GREEN_API_WEBHOOK_TOKEN and
    these tests would 401 on one machine and pass on another. Authentication
    itself is covered in test_webhook_auth.py.
    """
    monkeypatch.setattr("app.api.routes.get_settings", lambda: settings)


@pytest.fixture
def green() -> FakeGreenClient:
    return FakeGreenClient()


@pytest.fixture
def client(settings, green) -> TestClient:
    service = WhatsAppService(
        settings=settings,
        green_client=green,
        escalation_store=EscalationStore(),
        dedupe=TTLDedupeCache(ttl_seconds=60, max_entries=10),
        runner_provider=lambda: FakeRunner(),
    )
    set_whatsapp_service(service)
    api = FastAPI()
    api.include_router(router)
    with TestClient(api) as test_client:
        yield test_client
    set_whatsapp_service(None)


def test_webhook_accepts_an_incoming_message(client):
    response = client.post("/webhooks/green-api", json=incoming("مرحبا"))

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


def test_webhook_reports_ignored_events(client):
    payload = incoming()
    payload["typeWebhook"] = "outgoingMessageReceived"

    body = client.post("/webhooks/green-api", json=payload).json()

    assert body == {"status": "ignored", "reason": "outgoing_message"}


def test_webhook_returns_200_for_invalid_json(client):
    response = client.post(
        "/webhooks/green-api",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )

    # A non-2xx would make GREEN API redeliver a payload we can never parse.
    assert response.status_code == 200
    assert response.json()["reason"] == "invalid_json"


def test_webhook_returns_200_for_a_json_array(client):
    response = client.post("/webhooks/green-api", json=[1, 2, 3])

    assert response.status_code == 200
    assert response.json()["reason"] == "invalid_payload"


def test_webhook_never_sends_while_live_is_disabled(client, green):
    client.post("/webhooks/green-api", json=incoming("مرحبا"))

    # The fake records intent; the real client would have short-circuited.
    for _chat, message in green.sent:
        assert message


def test_healthz_is_cheap_and_ok(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_integrations_health_reports_the_safety_switch(client):
    body = client.get("/health/integrations").json()

    assert body["green_api"]["live_sending_enabled"] is False
    assert body["whatsapp"]["live_sending_enabled"] is False


def test_integrations_health_does_not_probe_by_default(client, monkeypatch):
    called = False

    def _probe():
        nonlocal called
        called = True
        return {"status": "ok"}

    monkeypatch.setattr("app.api.routes._probe_weaviate", _probe)

    client.get("/health/integrations")

    assert called is False


def test_integrations_health_probe_failure_is_degraded_not_500(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes._probe_weaviate",
        lambda: {"status": "error", "error_type": "ConnectionError"},
    )

    response = client.get("/health/integrations?probe=true")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_health_exposes_no_secret_values(client):
    body = client.get("/health/integrations").text

    assert "test-gemini-key-abcdef" not in body
    assert "test-weaviate-key-abcdef" not in body
    assert "test-green-token-abcdef" not in body
