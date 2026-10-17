"""Bounded HTTP client for the Khatib patient/results backend.

Two endpoints, two authentication layers, one rule: a failure never becomes a
permission. Every method returns a typed outcome — there is no path on which a
rejection, a timeout, or an unparsable body turns into "proceed anyway".

Authentication retry
--------------------
A ``401`` on a *tenant*-authenticated call can mean the tenant token aged out
early, so the token is dropped and the call is retried **once**, with a freshly
issued one. A ``403``, and any second ``401``, is not retried: on the results
endpoint those can equally mean the *patient* verification token was rejected,
and re-authenticating the tenant must never be mistaken for clearing a patient
check. The caller treats that outcome as "verify again", never as "go ahead".

Privacy
-------
Nothing here logs a name, a birth date, a phone suffix, a patient id, a token,
a request body, or a URL containing a patient id. Log lines carry an outcome,
an HTTP status, and a masked chat id — enough to debug, not enough to identify.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import quote

import httpx

from app.config import KhatibApiSettings, get_settings
from app.logging_utils import get_logger
from app.patients.auth import (
    KhatibAuthError,
    TenantAuthenticator,
    get_tenant_authenticator,
    request_timeout,
)

logger = get_logger(__name__)

VERIFY_PATH = "/api/v1.0/patients/verify"
RESULTS_PATH_TEMPLATE = "/api/v1.0/results/{patient_id}/latest"
VERIFICATION_HEADER = "x-patient-verification-token"

#: Used when the backend omits ``expiresInSeconds`` on a successful verify.
_FALLBACK_VERIFICATION_TTL = 300


class TransportFailure(StrEnum):
    """A call that never produced a usable HTTP status."""

    NOT_CONFIGURED = "not_configured"
    AUTH_ERROR = "auth_error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


class VerifyStatus(StrEnum):
    VERIFIED = "verified"
    #: The backend did not accept this identity. Deliberately indistinguishable
    #: from "no such patient" — see the privacy note above.
    NOT_VERIFIED = "not_verified"
    #: The backend rejected the request shape (e.g. neither optional field).
    INVALID_REQUEST = "invalid_request"
    RATE_LIMITED = "rate_limited"
    NOT_CONFIGURED = "not_configured"
    AUTH_ERROR = "auth_error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


class ResultStatusCode(StrEnum):
    OK = "ok"
    #: The patient verification token was not accepted. Verify again.
    VERIFICATION_REJECTED = "verification_rejected"
    NO_RESULT = "no_result"
    NOT_CONFIGURED = "not_configured"
    AUTH_ERROR = "auth_error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class VerifyOutcome:
    """Result of a verification attempt.

    ``patient_id`` and ``verification_token`` are credentials. They go
    straight into :class:`app.patients.store.VerifiedPatient` and are never
    returned to a tool caller, a prompt, or a log line.
    """

    status: VerifyStatus
    patient_id: str = ""
    verification_token: str = ""
    expires_in_seconds: int = 0
    method: str = ""

    @property
    def ok(self) -> bool:
        return self.status is VerifyStatus.VERIFIED

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"VerifyOutcome(status={self.status.value!r}, patient_id=***, token=***)"


@dataclass(frozen=True)
class ResultOutcome:
    status: ResultStatusCode
    payload: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status is ResultStatusCode.OK


class KhatibApiClient:
    def __init__(
        self,
        settings: KhatibApiSettings | None = None,
        *,
        authenticator: TenantAuthenticator | None = None,
        client_factory: Any = None,
    ) -> None:
        self._settings = settings or get_settings().khatib
        self._authenticator = authenticator
        self._client_factory = client_factory
        self._client: httpx.AsyncClient | None = None

    @property
    def settings(self) -> KhatibApiSettings:
        return self._settings

    @property
    def auth(self) -> TenantAuthenticator:
        if self._authenticator is None:
            self._authenticator = get_tenant_authenticator()
        return self._authenticator

    # -- endpoints ---------------------------------------------------------

    async def verify_patient(
        self,
        *,
        full_name: str,
        date_of_birth: str = "",
        phone_last4: str = "",
    ) -> VerifyOutcome:
        """POST ``/patients/verify``. Caller has already normalised the fields.

        The backend requires ``fullName`` plus at least one of the other two;
        whichever was not supplied is omitted from the body rather than sent
        as null, so the backend sees exactly what the customer gave us.
        """
        if not self._settings.is_configured:
            return VerifyOutcome(VerifyStatus.NOT_CONFIGURED)

        body: dict[str, Any] = {"fullName": full_name}
        if date_of_birth:
            body["dateOfBirth"] = date_of_birth
        if phone_last4:
            # Always a string: int() would eat a leading zero and silently
            # change which patient is being claimed.
            body["phoneLast4"] = str(phone_last4)

        response = await self._request(
            "POST", VERIFY_PATH, json=body, operation="verify_patient"
        )
        if isinstance(response, TransportFailure):
            return VerifyOutcome(_VERIFY_FAILURE[response])

        if response.status_code in (400, 422):
            return VerifyOutcome(VerifyStatus.INVALID_REQUEST)
        if response.status_code == 429:
            return VerifyOutcome(VerifyStatus.RATE_LIMITED)
        if response.status_code in (401, 403, 404):
            # One outcome for "wrong details" and "no such patient": telling
            # them apart would make this endpoint a patient-existence oracle.
            return VerifyOutcome(VerifyStatus.NOT_VERIFIED)
        if response.status_code >= 400:
            self._log_status("verify_patient", response.status_code)
            return VerifyOutcome(VerifyStatus.UNAVAILABLE)

        body_out = _json_object(response)
        if body_out is None:
            return VerifyOutcome(VerifyStatus.UNAVAILABLE)

        patient_id = _str(body_out, ("patientId", "patient_id"))
        token = _str(body_out, ("verificationToken", "verification_token"))
        if not patient_id or not token:
            # A 200 without both halves is not an authorisation.
            logger.error("verify response lacked patientId or verificationToken")
            return VerifyOutcome(VerifyStatus.UNAVAILABLE)

        return VerifyOutcome(
            status=VerifyStatus.VERIFIED,
            patient_id=patient_id,
            verification_token=token,
            expires_in_seconds=_positive_int(
                _value(body_out, ("expiresInSeconds", "expires_in_seconds")),
                default=self._settings.default_verification_ttl_seconds
                or _FALLBACK_VERIFICATION_TTL,
            ),
            method=_str(body_out, ("verificationMethod", "verification_method")),
        )

    async def get_latest_result(
        self, *, patient_id: str, verification_token: str
    ) -> ResultOutcome:
        """GET ``/results/{patientId}/latest`` with both auth headers.

        ``patient_id`` comes from the stored verification record, never from
        the model or the customer, and is URL-escaped on the way into the path.
        """
        if not self._settings.is_configured:
            return ResultOutcome(ResultStatusCode.NOT_CONFIGURED)
        if not patient_id or not verification_token:
            return ResultOutcome(ResultStatusCode.VERIFICATION_REJECTED)

        path = RESULTS_PATH_TEMPLATE.format(patient_id=quote(patient_id, safe=""))
        response = await self._request(
            "GET",
            path,
            headers={VERIFICATION_HEADER: verification_token},
            operation="get_latest_result",
        )
        if isinstance(response, TransportFailure):
            return ResultOutcome(_RESULT_FAILURE[response])

        if response.status_code in (401, 403):
            # Could be an expired patient verification token; could be the
            # tenant token rejected twice. Either way the safe reading is
            # "this conversation is not authorised", never "skip a check".
            return ResultOutcome(ResultStatusCode.VERIFICATION_REJECTED)
        if response.status_code == 404:
            return ResultOutcome(ResultStatusCode.NO_RESULT)
        if response.status_code >= 400:
            self._log_status("get_latest_result", response.status_code)
            return ResultOutcome(ResultStatusCode.UNAVAILABLE)

        payload = _json_object(response)
        if payload is None:
            return ResultOutcome(ResultStatusCode.UNAVAILABLE)
        return ResultOutcome(ResultStatusCode.OK, payload)

    # -- transport ---------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response | TransportFailure:
        """Perform one authenticated call, re-authenticating at most once.

        Returns the response, or a :class:`TransportFailure` describing a call
        that never reached a usable HTTP status.
        """
        url = f"{self._settings.base_url}{path}"
        client = await self._http()

        for attempt in (0, 1):
            try:
                token = await self.auth.get_token()
            except KhatibAuthError:
                logger.error(
                    "tenant authentication unavailable",
                    extra={"context": {"operation": operation}},
                )
                return TransportFailure.AUTH_ERROR

            request_headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                **(headers or {}),
            }
            try:
                response = await client.request(
                    method, url, json=json, headers=request_headers
                )
            except httpx.TimeoutException:
                logger.error(
                    "Khatib backend timed out",
                    extra={"context": {"operation": operation}},
                )
                return TransportFailure.TIMEOUT
            except httpx.HTTPError as exc:
                logger.error(
                    "Khatib backend request failed",
                    extra={
                        "context": {
                            "operation": operation,
                            "error_type": type(exc).__name__,
                        }
                    },
                )
                return TransportFailure.UNAVAILABLE

            # Exactly one re-auth, and only for 401. A 403 is an authorisation
            # decision, not a stale credential, so retrying it would just be a
            # second attempt at something we were already told no about.
            if response.status_code == 401 and attempt == 0:
                self.auth.invalidate(used_token=token)
                continue
            return response

        return TransportFailure.AUTH_ERROR  # pragma: no cover - loop always returns

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
        if self._authenticator is not None:
            await self._authenticator.aclose()

    def _log_status(self, operation: str, status_code: int) -> None:
        logger.error(
            "Khatib backend returned an error status",
            extra={"context": {"operation": operation, "status": str(status_code)}},
        )


#: Transport-level failures map onto an equivalent condition per endpoint.
_VERIFY_FAILURE = {
    TransportFailure.AUTH_ERROR: VerifyStatus.AUTH_ERROR,
    TransportFailure.TIMEOUT: VerifyStatus.TIMEOUT,
    TransportFailure.UNAVAILABLE: VerifyStatus.UNAVAILABLE,
    TransportFailure.NOT_CONFIGURED: VerifyStatus.NOT_CONFIGURED,
}
_RESULT_FAILURE = {
    TransportFailure.AUTH_ERROR: ResultStatusCode.AUTH_ERROR,
    TransportFailure.TIMEOUT: ResultStatusCode.TIMEOUT,
    TransportFailure.UNAVAILABLE: ResultStatusCode.UNAVAILABLE,
    TransportFailure.NOT_CONFIGURED: ResultStatusCode.NOT_CONFIGURED,
}


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    try:
        data = response.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _value(body: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in body:
            return body[key]
    return None


def _str(body: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = _value(body, keys)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


_client: KhatibApiClient | None = None


def get_khatib_client() -> KhatibApiClient:
    global _client
    if _client is None:
        _client = KhatibApiClient()
    return _client


def set_khatib_client(client: KhatibApiClient | None) -> None:
    """Swap the process-wide client. Tests use this to inject a fake."""
    global _client
    _client = client
