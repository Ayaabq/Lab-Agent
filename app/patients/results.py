"""Reading a latest-result payload into something safe to say out loud.

The backend is the source of truth for anything patient-specific, and its
``patientMessage`` is the text the customer should hear. This module decides
*which* safe outcome a payload represents; it never composes medical content,
never interprets a value, and never upgrades an unknown shape into a
reassuring one.

The dispositions specified so far:

``FINAL`` + ``BENIGN``
    Releasable. The backend's ``patientMessage`` goes to the customer unchanged
    and the report image at ``imageURL`` is sent to the chat. Both halves are
    required: ``FINAL`` with any other disposition releases nothing, and
    ``BENIGN`` on a record that is not ``FINAL`` is not a finished result.

``DRAFT`` + ``PENDING``
    Still being processed. Say the backend's processing message. Not "ready",
    and no report image. If ``expectedReadyAt`` is already in the past, the
    promise has lapsed — that becomes a staff handoff rather than a date we
    restate or replace with one we made up.

``FINAL`` + ``CLINICIAN_REVIEW_REQUIRED``
    Say the backend's message and hand off to a human. ``FINAL`` describes the
    *record*, not the customer's right to hear what is in it: it authorises no
    disclosure of medical detail on its own.

Anything else — a disposition we have never seen, a missing status, an empty
body — is ``UNKNOWN`` and routes to a human. Guessing is the one option that
is never available here.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from app.config import KhatibApiSettings
from app.rag.sanitize import sanitize_snippet

#: The backend's message is shown to a customer, so it is injection-defused
#: like any other text this process did not write. The cap is WhatsApp's own
#: text limit rather than an editorial one: ``patientMessage`` is relayed
#: unchanged, so the only length we may impose is the one the transport does.
MAX_PATIENT_MESSAGE_CHARS = 4096


class ResultStatus(StrEnum):
    DRAFT = "DRAFT"
    FINAL = "FINAL"
    UNKNOWN = "UNKNOWN"


class Disposition(StrEnum):
    PENDING = "PENDING"
    CLINICIAN_REVIEW_REQUIRED = "CLINICIAN_REVIEW_REQUIRED"
    #: Released to the patient, with the report image.
    BENIGN = "BENIGN"
    UNKNOWN = "UNKNOWN"


class ResultCase(StrEnum):
    """What the agent should actually do about this result."""

    #: DRAFT/PENDING, still within the expected window.
    PROCESSING = "processing"
    #: DRAFT/PENDING, but the promised time has already passed.
    OVERDUE = "overdue"
    #: FINAL/CLINICIAN_REVIEW_REQUIRED.
    CLINICIAN_REVIEW_REQUIRED = "clinician_review_required"
    #: FINAL/BENIGN with a usable report image: message plus image.
    BENIGN_RELEASE = "benign_release"
    #: FINAL/BENIGN but the image is absent or unusable. The message still
    #: goes out; nothing is invented to stand in for the missing report.
    BENIGN_WITHOUT_IMAGE = "benign_without_image"
    #: The patient has no result on file.
    NO_RESULT = "no_result"
    #: A shape we do not recognise. Always a human handoff.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReportImage:
    """A reference to the backend's report image.

    ``url`` is patient data: it addresses someone's medical report. It stays
    server-side — out of tool output, out of the prompt, and out of logs.
    """

    url: str
    field: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ReportImage(field={self.field!r}, url=<redacted>)"


@dataclass(frozen=True)
class LatestResult:
    case: ResultCase
    status: ResultStatus
    disposition: Disposition
    #: Backend-authored, sanitised. Empty when the backend sent none.
    patient_message: str
    #: The disposition exactly as the backend spelled it. Kept because the
    #: enum collapses every unknown value to UNKNOWN, which would make a
    #: future disposition impossible to allow-list without editing this file.
    raw_disposition: str = ""
    #: Backend record identifiers. Server-side only: ``result_id`` keys the
    #: "already sent this report" check, and none of the three is ever shown
    #: to the model, written to a log, or put in a filename.
    result_id: str = ""
    biopsy_id: str = ""
    sample_number: str = ""
    #: Raw backend value, kept only to decide OVERDUE. Never shown as a promise.
    expected_ready_at: str = ""
    #: Present only once the backend contract exists and is configured.
    image: ReportImage | None = None
    #: True when the backend omitted a message we would otherwise relay.
    message_missing: bool = False


def parse_latest_result(
    payload: Any,
    *,
    settings: KhatibApiSettings,
    now: _dt.datetime | None = None,
) -> LatestResult:
    """Classify a ``/results/{id}/latest`` body. Never raises."""
    body = _unwrap(payload)
    if body is None:
        return LatestResult(
            case=ResultCase.NO_RESULT,
            status=ResultStatus.UNKNOWN,
            disposition=Disposition.UNKNOWN,
            patient_message="",
            message_missing=True,
        )

    status = _enum(ResultStatus, body.get("status"))
    result_id = _first_str(body, ("resultId", "result_id"))
    raw_disposition = body.get("disposition")
    raw_disposition = (
        raw_disposition.strip().upper() if isinstance(raw_disposition, str) else ""
    )
    disposition = _enum(Disposition, raw_disposition)
    raw_message = _first_str(body, ("patientMessage", "patient_message", "message"))
    message = (
        sanitize_snippet(raw_message, max_chars=MAX_PATIENT_MESSAGE_CHARS).text
        if raw_message
        else ""
    )
    expected = _first_str(body, ("expectedReadyAt", "expected_ready_at"))
    image = extract_report_image(body, settings=settings)
    releasable = image is not None and is_release_permitted(
        status=status, raw_disposition=raw_disposition, image=image, settings=settings
    )

    case = _classify(
        status, disposition, expected, now, image=image, releasable=releasable
    )
    return LatestResult(
        case=case,
        status=status,
        disposition=disposition,
        raw_disposition=raw_disposition,
        result_id=result_id,
        biopsy_id=_first_str(body, ("biopsyId", "biopsy_id")),
        sample_number=_first_str(body, ("sampleNumber", "sample_number")),
        patient_message=message,
        expected_ready_at=expected,
        image=image,
        message_missing=not message,
    )


def _classify(
    status: ResultStatus,
    disposition: Disposition,
    expected_ready_at: str,
    now: _dt.datetime | None,
    *,
    image: ReportImage | None,
    releasable: bool,
) -> ResultCase:
    if disposition is Disposition.BENIGN:
        if status is not ResultStatus.FINAL:
            # BENIGN on a record that is not final is not a finished result.
            # Treating it as one would release a draft finding.
            return ResultCase.UNKNOWN
        # Explicit, so BENIGN can never drop through to UNKNOWN. Whether the
        # image actually goes out is decided by `releasable`, which also
        # weighs the host, the scheme and the file extension.
        return (
            ResultCase.BENIGN_RELEASE
            if image is not None and releasable
            else ResultCase.BENIGN_WITHOUT_IMAGE
        )
    if disposition is Disposition.PENDING:
        # Status is informative here, not decisive: a PENDING result is not
        # releasable whatever the record's draft state says.
        if _has_passed(expected_ready_at, now):
            return ResultCase.OVERDUE
        return ResultCase.PROCESSING
    if disposition is Disposition.CLINICIAN_REVIEW_REQUIRED:
        return ResultCase.CLINICIAN_REVIEW_REQUIRED
    return ResultCase.UNKNOWN


def _has_passed(value: str, now: _dt.datetime | None) -> bool:
    """True when ``expectedReadyAt`` is a timestamp already in the past.

    An unparsable or absent value is *not* treated as passed: we would rather
    relay the backend's own processing message than manufacture a delay.
    """
    parsed = parse_timestamp(value)
    if parsed is None:
        return False
    reference = now or _dt.datetime.now(_dt.UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=_dt.UTC)
    return parsed < reference


def parse_timestamp(value: str) -> _dt.datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    # Python's fromisoformat rejects the trailing "Z" before 3.11's relaxation
    # in some shapes; normalising it is cheaper than a dependency.
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.UTC)
    return parsed


# -- report image ---------------------------------------------------------
#
# Releasing a report is the one thing in this module that reaches outside the
# conversation, so it is gated on four independent facts, all of which must
# hold: the record is FINAL, the disposition is allow-listed, the URL is HTTPS
# on an approved host, and the file extension is one we can name honestly to
# the provider. A fifth — a live patient verification — is checked by the
# caller immediately before the send, because it can lapse between the two.


#: Extensions we will hand to GREEN API. The provider infers the media type
#: from ``fileName``, so an extension we cannot recognise is one we would be
#: guessing at, and a guess here mislabels a medical document.
ALLOWED_IMAGE_EXTENSIONS: dict[str, str] = {
    ".jpg": ".jpg",
    ".jpeg": ".jpeg",
    ".png": ".png",
    ".webp": ".webp",
    ".pdf": ".pdf",
}


class ImageRejection(StrEnum):
    """Why a report image will not be sent. For logs, never for the customer."""

    NONE = ""
    MISSING = "missing"
    NOT_HTTPS = "not_https"
    HOST_NOT_ALLOWED = "host_not_allowed"
    UNKNOWN_EXTENSION = "unknown_extension"
    DISPOSITION_NOT_RELEASABLE = "disposition_not_releasable"
    STATUS_NOT_FINAL = "status_not_final"


def extract_report_image(
    body: dict[str, Any], *, settings: KhatibApiSettings
) -> ReportImage | None:
    """Read the report image URL from the backend's ``imageURL`` field.

    The field name is configuration rather than a literal because it belongs
    to the backend's contract, not ours. A missing field, an empty value or a
    non-string is simply "no image" — never an error, because the rest of the
    result still has to reach the customer.
    """
    field = (settings.result_image_field or "").strip()
    if not field:
        return None
    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    return ReportImage(url=value.strip(), field=field)


def image_rejection(
    *,
    status: ResultStatus,
    raw_disposition: str,
    image: ReportImage | None,
    settings: KhatibApiSettings,
) -> ImageRejection:
    """The first reason this image may not be released, or ``NONE``."""
    if image is None or not image.url:
        return ImageRejection.MISSING
    if status is not ResultStatus.FINAL:
        # FINAL alone does not authorise release, but its absence forbids it.
        return ImageRejection.STATUS_NOT_FINAL
    allowed = {d.upper() for d in settings.image_release_dispositions}
    if not allowed or raw_disposition.upper() not in allowed:
        return ImageRejection.DISPOSITION_NOT_RELEASABLE

    parsed = urlparse(image.url)
    if parsed.scheme.lower() != "https":
        # Plain HTTP would put a patient's report on the wire in clear.
        return ImageRejection.NOT_HTTPS
    host = (parsed.hostname or "").lower()
    if not host or host not in settings.allowed_image_hosts:
        # Otherwise anyone who can write a result row could point the provider
        # at an address of their choosing.
        return ImageRejection.HOST_NOT_ALLOWED
    if image_extension(image.url) is None:
        return ImageRejection.UNKNOWN_EXTENSION
    return ImageRejection.NONE


def is_release_permitted(
    *,
    status: ResultStatus,
    raw_disposition: str,
    image: ReportImage | None,
    settings: KhatibApiSettings,
) -> bool:
    """Whether the laboratory's rules allow this report image to be sent.

    This answers the *backend's* half of the question only. It knows nothing
    about patient verification and cannot stand in for it: the caller checks
    that separately, and later, so a verification cannot lapse in between.
    """
    return (
        image_rejection(
            status=status,
            raw_disposition=raw_disposition,
            image=image,
            settings=settings,
        )
        is ImageRejection.NONE
    )


def image_extension(url: str) -> str | None:
    """The allow-listed file extension of a URL's path, or ``None``.

    Only the path is inspected: a query string can end in anything, and an
    extension taken from one would describe the request, not the file.
    """
    path = urlparse(url).path
    dot = path.rfind(".")
    if dot == -1:
        return None
    return ALLOWED_IMAGE_EXTENSIONS.get(path[dot:].lower())


def report_file_name(url: str) -> str | None:
    """A neutral filename carrying the right extension, or ``None``.

    Deliberately not derived from the URL or from ``resultId``/``biopsyId``:
    a filename is visible in the chat, and a report that arrives called
    ``biopsy-99134.pdf`` has leaked a record identifier onto the customer's
    phone and into every backup of it.
    """
    extension = image_extension(url)
    return f"lab-report{extension}" if extension else None


# -- helpers ---------------------------------------------------------------


def _unwrap(payload: Any) -> dict[str, Any] | None:
    """Accept the result object bare or inside a one-key envelope."""
    if not isinstance(payload, dict) or not payload:
        return None
    for key in ("result", "data", "latestResult", "latest_result"):
        inner = payload.get(key)
        if isinstance(inner, dict) and inner:
            return inner
    return payload


def _enum(enum_cls: Any, value: Any) -> Any:
    if isinstance(value, str):
        try:
            return enum_cls(value.strip().upper())
        except ValueError:
            return enum_cls.UNKNOWN
    return enum_cls.UNKNOWN


def _first_str(body: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
