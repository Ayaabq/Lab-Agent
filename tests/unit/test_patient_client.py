"""The Khatib backend client: request shape, headers, and failure handling.

A mock transport stands in for the backend. Nothing here reaches the network,
and no real credential or patient identity appears.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.patients.auth import TenantAuthenticator
from app.patients.client import (
    RESULTS_PATH_TEMPLATE,
    VERIFICATION_HEADER,
    VERIFY_PATH,
    KhatibApiClient,
    ResultStatusCode,
    VerifyStatus,
)
from tests.unit.conftest import khatib_settings

VERIFY_OK = {
    "patientId": "pat_internal_7781",
    "verificationToken": "vtok-abcdef123456",
    "expiresInSeconds": 900,
    "verificationMethod": "DATE_OF_BIRTH",
}


class Backend:
    """Scripted backend. Records every request it was sent."""

    def __init__(self, routes) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []
        self.login_count = 0

    def body(self, index: int) -> dict:
        return json.loads(self.requests[index].content or b"{}")

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/auth/tenant/login"):
            self.login_count += 1
            return httpx.Response(
                200,
                json={
                    "accessToken": f"tenant-{self.login_count}",
                    "refreshToken": "r",
                    "expiresIn": 3600,
                },
            )
        self.requests.append(request)
        route = self.routes
        if callable(route):
            return route(request, len(self.requests))
        return route


def build(backend: Backend, **overrides) -> KhatibApiClient:
    transport = httpx.MockTransport(backend.handle)
    http = httpx.AsyncClient(transport=transport)
    settings = khatib_settings(**overrides)
    return KhatibApiClient(
        settings,
        authenticator=TenantAuthenticator(settings, client_factory=lambda: http),
        client_factory=lambda: http,
    )


# --- verify ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_by_date_of_birth_sends_only_the_supplied_fields():
    backend = Backend(httpx.Response(200, json=VERIFY_OK))
    client = build(backend)

    outcome = await client.verify_patient(
        full_name="أحمد الخطيب", date_of_birth="1990-03-04"
    )

    assert outcome.ok
    assert outcome.patient_id == "pat_internal_7781"
    assert outcome.verification_token == "vtok-abcdef123456"
    assert outcome.expires_in_seconds == 900
    assert outcome.method == "DATE_OF_BIRTH"
    assert backend.requests[0].url.path == VERIFY_PATH
    # phoneLast4 is omitted, not sent as null.
    assert backend.body(0) == {"fullName": "أحمد الخطيب", "dateOfBirth": "1990-03-04"}


@pytest.mark.asyncio
async def test_verify_by_phone_suffix_sends_a_string_with_its_leading_zero():
    backend = Backend(
        httpx.Response(200, json={**VERIFY_OK, "verificationMethod": "PHONE_LAST4"})
    )
    client = build(backend)

    outcome = await client.verify_patient(full_name="أحمد الخطيب", phone_last4="0123")

    assert outcome.ok
    assert outcome.method == "PHONE_LAST4"
    body = backend.body(0)
    assert body == {"fullName": "أحمد الخطيب", "phoneLast4": "0123"}
    # Not 123: JSON must carry the string, or the backend sees another patient.
    assert '"phoneLast4": "0123"' in json.dumps(body, ensure_ascii=False, indent=None)


@pytest.mark.asyncio
async def test_both_optional_fields_are_sent_when_both_were_given():
    backend = Backend(httpx.Response(200, json=VERIFY_OK))
    client = build(backend)

    await client.verify_patient(
        full_name="أحمد الخطيب", date_of_birth="1990-03-04", phone_last4="0123"
    )

    assert backend.body(0) == {
        "fullName": "أحمد الخطيب",
        "dateOfBirth": "1990-03-04",
        "phoneLast4": "0123",
    }


@pytest.mark.asyncio
async def test_the_tenant_token_is_sent_as_a_bearer_header():
    backend = Backend(httpx.Response(200, json=VERIFY_OK))
    client = build(backend)

    await client.verify_patient(full_name="أحمد الخطيب", phone_last4="0123")

    assert backend.requests[0].headers["authorization"] == "Bearer tenant-1"


@pytest.mark.parametrize("status_code", [401, 403, 404])
@pytest.mark.asyncio
async def test_wrong_details_and_no_such_patient_look_identical(status_code):
    backend = Backend(
        lambda request, n: httpx.Response(status_code, json={"message": "whatever"})
    )
    client = build(backend)

    outcome = await client.verify_patient(full_name="لا أحد", phone_last4="9999")

    # One outcome for all three: anything finer makes this a patient-existence
    # oracle for whoever can guess names.
    assert outcome.status is VerifyStatus.NOT_VERIFIED
    assert outcome.patient_id == ""
    assert outcome.verification_token == ""


@pytest.mark.asyncio
async def test_a_success_body_missing_half_the_credential_is_not_a_success():
    backend = Backend(httpx.Response(200, json={"patientId": "pat_1"}))
    client = build(backend)

    outcome = await client.verify_patient(full_name="أحمد الخطيب", phone_last4="0123")

    assert outcome.status is VerifyStatus.UNAVAILABLE
    assert not outcome.ok


@pytest.mark.asyncio
async def test_a_backend_timeout_is_reported_not_raised():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    backend = Backend(None)
    backend.handle = _login_then(explode, backend)
    client = build(backend)

    outcome = await client.verify_patient(full_name="أحمد الخطيب", phone_last4="0123")

    assert outcome.status is VerifyStatus.TIMEOUT


@pytest.mark.asyncio
async def test_a_server_error_is_reported_as_unavailable():
    backend = Backend(httpx.Response(503, text="upstream down"))
    client = build(backend)

    outcome = await client.verify_patient(full_name="أحمد الخطيب", phone_last4="0123")

    assert outcome.status is VerifyStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_an_unconfigured_backend_never_opens_a_socket():
    from app.config import Secret

    backend = Backend(httpx.Response(200, json=VERIFY_OK))
    client = build(backend, client_id=Secret(""))

    outcome = await client.verify_patient(full_name="أحمد الخطيب", phone_last4="0123")

    assert outcome.status is VerifyStatus.NOT_CONFIGURED
    assert backend.requests == []
    assert backend.login_count == 0


# --- latest result --------------------------------------------------------


@pytest.mark.asyncio
async def test_results_request_uses_the_verified_id_and_both_auth_headers():
    backend = Backend(httpx.Response(200, json={"status": "DRAFT"}))
    client = build(backend)

    outcome = await client.get_latest_result(
        patient_id="pat_internal_7781", verification_token="vtok-abcdef123456"
    )

    assert outcome.ok
    request = backend.requests[0]
    assert request.method == "GET"
    assert request.url.path == RESULTS_PATH_TEMPLATE.format(
        patient_id="pat_internal_7781"
    )
    assert request.headers["authorization"] == "Bearer tenant-1"
    assert request.headers[VERIFICATION_HEADER] == "vtok-abcdef123456"


@pytest.mark.asyncio
async def test_a_patient_id_with_url_characters_is_escaped_into_the_path():
    backend = Backend(httpx.Response(200, json={"status": "DRAFT"}))
    client = build(backend)

    await client.get_latest_result(patient_id="pat/../admin", verification_token="vtok")

    # Escaped on the wire, so a crafted id cannot walk out of the results
    # route. ``url.path`` reports the decoded form, so assert on the raw one.
    assert (
        backend.requests[0].url.raw_path == b"/api/v1.0/results/pat%2F..%2Fadmin/latest"
    )


@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.asyncio
async def test_a_rejected_verification_token_never_becomes_permission(status_code):
    backend = Backend(lambda request, n: httpx.Response(status_code, json={}))
    client = build(backend)

    outcome = await client.get_latest_result(
        patient_id="pat_1", verification_token="stale"
    )

    assert outcome.status is ResultStatusCode.VERIFICATION_REJECTED
    assert outcome.payload is None


@pytest.mark.asyncio
async def test_a_missing_result_is_distinct_from_a_rejected_one():
    backend = Backend(httpx.Response(404, json={}))
    client = build(backend)

    outcome = await client.get_latest_result(patient_id="pat_1", verification_token="v")

    assert outcome.status is ResultStatusCode.NO_RESULT


@pytest.mark.asyncio
async def test_an_empty_verification_token_is_refused_before_the_request():
    backend = Backend(httpx.Response(200, json={"status": "FINAL"}))
    client = build(backend)

    outcome = await client.get_latest_result(patient_id="pat_1", verification_token="")

    assert outcome.status is ResultStatusCode.VERIFICATION_REJECTED
    assert backend.requests == []


# --- tenant re-authentication --------------------------------------------


@pytest.mark.asyncio
async def test_a_401_triggers_exactly_one_re_authentication():
    def route(request: httpx.Request, call_number: int) -> httpx.Response:
        if call_number == 1:
            return httpx.Response(401, json={})
        return httpx.Response(200, json=VERIFY_OK)

    backend = Backend(route)
    client = build(backend)

    outcome = await client.verify_patient(full_name="أحمد الخطيب", phone_last4="0123")

    assert outcome.ok
    assert backend.login_count == 2
    assert backend.requests[0].headers["authorization"] == "Bearer tenant-1"
    assert backend.requests[1].headers["authorization"] == "Bearer tenant-2"


@pytest.mark.asyncio
async def test_a_second_401_is_not_retried_again():
    backend = Backend(lambda request, n: httpx.Response(401, json={}))
    client = build(backend)

    outcome = await client.verify_patient(full_name="أحمد الخطيب", phone_last4="0123")

    assert outcome.status is VerifyStatus.NOT_VERIFIED
    # Two attempts, not an unbounded loop.
    assert len(backend.requests) == 2
    assert backend.login_count == 2


@pytest.mark.asyncio
async def test_a_403_is_never_retried_with_a_new_tenant_token():
    backend = Backend(lambda request, n: httpx.Response(403, json={}))
    client = build(backend)

    outcome = await client.get_latest_result(patient_id="pat_1", verification_token="v")

    assert outcome.status is ResultStatusCode.VERIFICATION_REJECTED
    # A 403 on this route can mean the *patient* check failed. Re-authenticating
    # the tenant and trying again would be reinterpreting that refusal.
    assert len(backend.requests) == 1
    assert backend.login_count == 1


def _login_then(handler, backend: Backend):
    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/auth/tenant/login"):
            backend.login_count += 1
            return httpx.Response(
                200, json={"accessToken": "tenant-1", "expiresIn": 3600}
            )
        backend.requests.append(request)
        return handler(request)

    return handle
