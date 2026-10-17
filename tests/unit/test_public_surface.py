"""The tunnel must not expose the ADK development surface."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.public_surface import PUBLIC_PATHS, restrict_public_surface

#: What Cloudflare puts on a request it forwards from the public internet.
EDGE = {"CF-Ray": "8f0a1b2c3d4e5f60-FRA", "CF-Connecting-IP": "203.0.113.9"}


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/webhooks/green-api")
    async def webhook():
        return {"status": "accepted"}

    # Stand-ins for the ADK dev surface the real app mounts.
    @app.get("/list-apps")
    async def list_apps():
        return ["app"]

    @app.get("/dev-ui")
    async def dev_ui():
        return {"ui": True}

    @app.get("/apps/{app_name}/users/{user_id}/sessions")
    async def sessions(app_name: str, user_id: str):
        return [{"id": "s1"}]

    @app.get("/health/integrations")
    async def integrations():
        return {"status": "ok"}

    app.middleware("http")(restrict_public_surface)
    return TestClient(app)


@pytest.mark.parametrize("path", sorted(PUBLIC_PATHS))
def test_the_public_paths_are_reachable_from_the_tunnel(client, path):
    response = (
        client.post(path, headers=EDGE)
        if path.startswith("/webhooks")
        else client.get(path, headers=EDGE)
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/list-apps",
        "/dev-ui",
        "/apps/app/users/963991234567@c.us/sessions",
        "/health/integrations",
    ],
)
def test_the_dev_surface_is_invisible_from_the_tunnel(client, path):
    response = client.get(path, headers=EDGE)

    # 404, not 403: a public caller should not learn that these exist.
    assert response.status_code == 404


@pytest.mark.parametrize(
    "path",
    ["/list-apps", "/dev-ui", "/apps/app/users/x/sessions", "/health/integrations"],
)
def test_the_operator_on_localhost_keeps_the_full_surface(client, path):
    response = client.get(path)

    assert response.status_code == 200


def test_a_single_edge_header_is_enough_to_be_treated_as_public(client):
    for header in ("CF-Ray", "CF-Connecting-IP", "CF-Visitor"):
        response = client.get("/list-apps", headers={header: "x"})

        assert response.status_code == 404, header


def test_a_trailing_slash_does_not_slip_past_the_allow_list(client):
    assert client.get("/healthz/", headers=EDGE).status_code in (200, 307)
    assert client.get("/list-apps/", headers=EDGE).status_code == 404


def test_session_reading_is_blocked_because_sessions_hold_verifications(client):
    # An ADK session for a verified chat is the record of someone's identity
    # check. It must not be readable by anyone holding the public URL.
    response = client.get("/apps/app/users/963991234567@c.us/sessions", headers=EDGE)

    assert response.status_code == 404


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.delenv("PUBLIC_SURFACE_ONLY", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "env",
    [{"RENDER": "true"}, {"PUBLIC_SURFACE_ONLY": "true"}, {"PUBLIC_SURFACE_ONLY": "1"}],
)
@pytest.mark.parametrize(
    "path",
    ["/list-apps", "/dev-ui", "/apps/app/users/x/sessions", "/health/integrations"],
)
def test_a_hosted_deployment_hides_the_dev_surface_without_edge_headers(
    client, hosted, env, path
):
    # A hosted process has no tunnel: a caller without Cloudflare headers is
    # still the public internet, not the operator.
    for name, value in env.items():
        hosted.setenv(name, value)

    assert client.get(path).status_code == 404


@pytest.mark.parametrize("env", [{"RENDER": "true"}, {"PUBLIC_SURFACE_ONLY": "true"}])
def test_a_hosted_deployment_keeps_the_public_paths(client, hosted, env):
    for name, value in env.items():
        hosted.setenv(name, value)

    assert client.get("/healthz").status_code == 200
    assert client.post("/webhooks/green-api").status_code == 200


def test_public_surface_only_false_leaves_localhost_unrestricted(client, hosted):
    hosted.setenv("PUBLIC_SURFACE_ONLY", "false")

    assert client.get("/list-apps").status_code == 200
