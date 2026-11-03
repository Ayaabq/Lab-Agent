"""What a Vercel deployment depends on, checked offline.

Entrypoint, public surface, inline webhook processing, dependency set and the
patient-backend URLs. No test here opens a network connection.
"""

from __future__ import annotations

import importlib
import json
import re
import tomllib
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import load_settings, reset_settings_cache

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
VERCEL_BACKEND = "https://khatib-db-ms.vercel.app"


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "VERCEL",
        "RENDER",
        "PUBLIC_SURFACE_ONLY",
        "GREEN_API_PROCESS_INLINE",
        "KHATIB_API_BASE_URL",
        "KHATIB_IMAGE_ALLOWED_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    reset_settings_cache()
    yield monkeypatch
    reset_settings_cache()


# --- entrypoint ------------------------------------------------------------


def test_the_vercel_entrypoint_is_the_fastapi_app():
    target = PYPROJECT["tool"]["vercel"]["entrypoint"]
    module_name, attr = target.split(":")

    app = getattr(importlib.import_module(module_name), attr)

    assert isinstance(app, FastAPI)


def test_the_entrypoint_is_not_shadowed_by_an_auto_detected_file():
    # Vercel auto-detects app.py, index.py, server.py, main.py, wsgi.py or
    # asgi.py at the root and in app/ or src/. A stray one would win over the
    # configured entrypoint and could serve an unprotected app.
    names = {"app.py", "index.py", "server.py", "main.py", "wsgi.py", "asgi.py"}
    found = [
        str(p.relative_to(ROOT))
        for d in (ROOT, ROOT / "app", ROOT / "src")
        if d.is_dir()
        for p in d.iterdir()
        if p.name in names
    ]

    assert found == []


# --- public surface on Vercel -------------------------------------------------


@pytest.fixture
def vercel_app(clean_env):
    clean_env.setenv("VERCEL", "1")
    from app.fast_api_app import app

    return TestClient(app)


@pytest.mark.parametrize(
    "path",
    [
        "/dev-ui/",
        "/docs",
        "/openapi.json",
        "/list-apps",
        "/health/integrations",
        "/apps/app/users/963991234567@c.us/sessions",
        "/a2a/app/.well-known/agent-card.json",
    ],
)
def test_vercel_hides_everything_but_the_two_public_routes(vercel_app, path):
    assert vercel_app.get(path).status_code == 404


def test_vercel_serves_the_health_check(vercel_app):
    response = vercel_app.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_vercel_rejects_an_unauthenticated_webhook(vercel_app):
    # With a token configured this is the ordinary 401; without one, hosted
    # mode fails closed. Either way an anonymous caller gets nothing.
    assert vercel_app.post("/webhooks/green-api", json={}).status_code == 401


# --- inline webhook processing -----------------------------------------------


def test_vercel_processes_webhooks_inline_by_default(clean_env):
    clean_env.setenv("VERCEL", "1")

    assert load_settings().green_api.process_inline is True


def test_inline_processing_stays_off_elsewhere(clean_env):
    assert load_settings().green_api.process_inline is False


def test_inline_processing_can_be_set_explicitly(clean_env):
    clean_env.setenv("VERCEL", "1")
    clean_env.setenv("GREEN_API_PROCESS_INLINE", "false")

    assert load_settings().green_api.process_inline is False


def test_inline_mode_replies_before_the_webhook_returns(settings, monkeypatch):
    """The function may be frozen once it answers, so the reply goes first."""
    from app.config import Secret
    from app.whatsapp.service import set_whatsapp_service
    from tests.unit.test_webhook_auth import TOKEN, build_client
    from tests.unit.test_whatsapp_parsing import incoming
    from tests.unit.test_whatsapp_service import FakeGreenClient

    inline = replace(
        settings,
        green_api=replace(
            settings.green_api, webhook_token=Secret(TOKEN), process_inline=True
        ),
    )
    monkeypatch.setattr("app.api.routes.get_settings", lambda: inline)
    green = FakeGreenClient()
    client = build_client(inline, green)
    try:
        response = client.post(
            "/webhooks/green-api",
            json=incoming("مرحبا"),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        # Already sent by the time the 200 arrives — no background task left.
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"
        assert len(green.sent) == 1
    finally:
        set_whatsapp_service(None)


# --- dependencies --------------------------------------------------------------


def _requirements() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if m := re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line):
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_litellm_is_a_runtime_dependency_at_a_patched_version():
    # Vercel installs from pyproject/uv.lock and may skip optional extras, so
    # LiteLLM must be a plain dependency. 1.96.2: see the pyproject comment.
    assert "litellm==1.96.2" in PYPROJECT["project"]["dependencies"]
    assert _requirements()["litellm"] == "1.96.2"


@pytest.mark.parametrize(
    "heavy", ["pyarrow", "gcsfs", "google-cloud-logging", "google-cloud-bigquery"]
)
def test_unused_google_cloud_packages_stay_out(heavy):
    # These took the install to 608 MB, over Vercel's 500 MB Python limit.
    assert heavy not in _requirements()


def test_google_adk_is_installed_without_cloud_extras():
    adk = [
        d for d in PYPROJECT["project"]["dependencies"] if d.startswith("google-adk")
    ]

    assert adk == ["google-adk>=2.8.0,<2.9.0"]


# --- patient backend on Vercel -------------------------------------------------


def test_the_backend_url_comes_from_the_environment(clean_env):
    clean_env.setenv("KHATIB_API_BASE_URL", VERCEL_BACKEND + "/")

    khatib = load_settings().khatib

    assert khatib.base_url == VERCEL_BACKEND
    # Report images are only released from the backend's own host by default.
    assert khatib.allowed_image_hosts == {"khatib-db-ms.vercel.app"}


@pytest.mark.asyncio
async def test_the_client_calls_the_expected_paths_on_the_new_host():
    from tests.unit.test_patient_client import VERIFY_OK, Backend, build

    backend = Backend(httpx.Response(200, json=VERIFY_OK))
    seen: list[str] = []
    original = backend.handle

    async def record(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url}")
        return await original(request)

    backend.handle = record
    client = build(backend, base_url=VERCEL_BACKEND)

    await client.verify_patient(full_name="مريض تجريبي", date_of_birth="1990-12-25")

    assert seen == [
        f"POST {VERCEL_BACKEND}/api/v1.0/auth/tenant/login",
        f"POST {VERCEL_BACKEND}/api/v1.0/patients/verify",
    ]


# --- function timeout ----------------------------------------------------------

VERCEL_JSON = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
#: GREEN API redelivers a webhook not answered with a 200 within this.
GREEN_API_WEBHOOK_WAIT_SECONDS = 180
#: Vercel's Fluid-compute ceiling on Hobby. Pro allows 800.
HOBBY_MAX_DURATION_SECONDS = 300
#: One reply, plus the rare report path: an overlong patientMessage, the file,
#: and two fallback messages.
MAX_SENDS_PER_TURN = 5


def _entrypoint_file() -> str:
    module = PYPROJECT["tool"]["vercel"]["entrypoint"].split(":")[0]
    return module.replace(".", "/") + ".py"


def _max_duration() -> int:
    return VERCEL_JSON["functions"][_entrypoint_file()]["maxDuration"]


def test_vercel_json_configures_the_entrypoint_function():
    # A functions key that matches no file is ignored, leaving the default.
    assert list(VERCEL_JSON["functions"]) == [_entrypoint_file()]
    assert (ROOT / _entrypoint_file()).is_file()
    # Without Fluid compute the Hobby ceiling is 60 s, too short for a turn.
    assert VERCEL_JSON["fluid"] is True


def test_the_worst_case_turn_fits_inside_the_function_timeout(clean_env):
    from app.whatsapp.service import AGENT_TURN_TIMEOUT_SECONDS

    send = load_settings().green_api.timeout_seconds
    worst = AGENT_TURN_TIMEOUT_SECONDS + MAX_SENDS_PER_TURN * send

    assert worst < _max_duration() <= HOBBY_MAX_DURATION_SECONDS


def test_an_ordinary_turn_answers_before_green_api_redelivers(clean_env):
    from app.whatsapp.service import AGENT_TURN_TIMEOUT_SECONDS

    send = load_settings().green_api.timeout_seconds

    assert AGENT_TURN_TIMEOUT_SECONDS + send < GREEN_API_WEBHOOK_WAIT_SECONDS


def test_a_hung_agent_still_answers_the_webhook_inline(settings, monkeypatch):
    """A stuck model costs the turn timeout, not the function's whole budget."""
    import asyncio

    from app.config import Secret
    from app.escalation import EscalationStore
    from app.whatsapp.dedupe import TTLDedupeCache
    from app.whatsapp.service import (
        FALLBACK_REPLY,
        WhatsAppService,
        set_whatsapp_service,
    )
    from tests.unit.test_webhook_auth import TOKEN
    from tests.unit.test_whatsapp_parsing import incoming
    from tests.unit.test_whatsapp_service import FakeGreenClient

    class HungRunner:
        async def run_async(self, *, user_id, session_id, new_message):
            await asyncio.sleep(3600)
            yield  # pragma: no cover

    monkeypatch.setattr("app.whatsapp.service.AGENT_TURN_TIMEOUT_SECONDS", 0.05)
    inline = replace(
        settings,
        green_api=replace(
            settings.green_api, webhook_token=Secret(TOKEN), process_inline=True
        ),
    )
    monkeypatch.setattr("app.api.routes.get_settings", lambda: inline)
    green = FakeGreenClient()
    set_whatsapp_service(
        WhatsAppService(
            settings=inline,
            green_client=green,
            escalation_store=EscalationStore(),
            dedupe=TTLDedupeCache(ttl_seconds=60, max_entries=10),
            runner_provider=HungRunner,
        )
    )
    from app.api.routes import router

    api = FastAPI()
    api.include_router(router)
    try:
        response = TestClient(api).post(
            "/webhooks/green-api",
            json=incoming("مرحبا"),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"
        assert [text for _, text in green.sent] == [FALLBACK_REPLY]
    finally:
        set_whatsapp_service(None)


def test_vercel_warns_about_per_instance_state_even_with_a_session_uri(
    clean_env, caplog
):
    # SESSION_SERVICE_URI moves ADK sessions only; verification and dedupe
    # would still be per instance, so it must not silence the warning.
    clean_env.setenv("VERCEL", "1")
    clean_env.setenv("SESSION_SERVICE_URI", "memory://")
    from app.fast_api_app import app

    with caplog.at_level("WARNING"), TestClient(app):
        pass

    assert any("per instance" in r.getMessage() for r in caplog.records)


# --- deployed files -------------------------------------------------------------


def test_vercelignore_ships_the_config_but_not_secrets_or_docker():
    rules = (ROOT / ".vercelignore").read_text(encoding="utf-8").splitlines()

    assert rules[rules.index("*")] == "*"  # allowlist: exclude all first
    assert "!vercel.json" in rules
    assert "!app" in rules
    assert not [r for r in rules if r.startswith("!") and ".env" in r]
    assert "!Dockerfile" not in rules
    assert "!WEBHOOK_TOKEN.txt" not in rules


def test_vercel_with_redis_does_not_warn_about_per_instance_state(clean_env, caplog):
    from app.state.backend import reset_state_backend
    from tests.unit.upstash_fake import KEY, TOKEN, URL

    clean_env.setenv("VERCEL", "1")
    clean_env.setenv("UPSTASH_REDIS_REST_URL", URL)
    clean_env.setenv("UPSTASH_REDIS_REST_TOKEN", TOKEN)
    clean_env.setenv("STATE_ENCRYPTION_KEY", KEY)
    reset_state_backend()
    try:
        from app.fast_api_app import app

        with caplog.at_level("WARNING"), TestClient(app):
            pass
    finally:
        reset_settings_cache()
        reset_state_backend()

    assert not any("per instance" in r.getMessage() for r in caplog.records)
