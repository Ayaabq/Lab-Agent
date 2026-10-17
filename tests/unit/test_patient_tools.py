"""The two patient tools as the agent sees them.

These are the tests that matter most: they assert the boundary the model runs
against. Nothing a tool returns may contain a token, an internal patient id,
or a raw backend response, and no failure path may end in "proceed".
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.escalation import EscalationReason, EscalationStore
from app.patients.client import (
    ResultOutcome,
    ResultStatusCode,
    VerifyOutcome,
    VerifyStatus,
)
from app.patients.context import (
    CHANNEL_WHATSAPP,
    ChannelIdentity,
    bind_channel_identity,
    reset_channel_identity,
)
from app.patients.store import VerificationStore, set_verification_store
from app.tools import get_my_latest_results, verify_patient
from tests.unit.conftest import khatib_settings

PATIENT_ID = "pat_internal_7781"
TOKEN = "vtok-abcdef123456"

PENDING_PAYLOAD = {
    "status": "DRAFT",
    "disposition": "PENDING",
    "patientMessage": "نتيجتك لسا قيد المعالجة، رح تكون جاهزة قريبًا.",
    "expectedReadyAt": "2099-01-01T10:00:00Z",
}
REVIEW_PAYLOAD = {
    "status": "FINAL",
    "disposition": "CLINICIAN_REVIEW_REQUIRED",
    "patientMessage": "نتيجتك بحاجة لمراجعة من طبيب المخبر.",
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeBackend:
    """Stands in for KhatibApiClient. Records what it was asked for."""

    def __init__(
        self,
        *,
        verify: VerifyOutcome | list[VerifyOutcome] | None = None,
        result: ResultOutcome | None = None,
    ) -> None:
        self._verify = verify if verify is not None else _verified()
        self._result = result or ResultOutcome(ResultStatusCode.OK, PENDING_PAYLOAD)
        self.verify_calls: list[dict[str, Any]] = []
        self.result_calls: list[dict[str, Any]] = []

    async def verify_patient(self, **kwargs: Any) -> VerifyOutcome:
        self.verify_calls.append(kwargs)
        if isinstance(self._verify, list):
            index = min(len(self.verify_calls) - 1, len(self._verify) - 1)
            return self._verify[index]
        return self._verify

    async def get_latest_result(self, **kwargs: Any) -> ResultOutcome:
        self.result_calls.append(kwargs)
        return self._result


def _verified(**overrides) -> VerifyOutcome:
    return VerifyOutcome(
        status=VerifyStatus.VERIFIED,
        patient_id=overrides.pop("patient_id", PATIENT_ID),
        verification_token=overrides.pop("verification_token", TOKEN),
        expires_in_seconds=overrides.pop("expires_in_seconds", 900),
        method=overrides.pop("method", "DATE_OF_BIRTH"),
    )


class FakeToolContext:
    """ADK's ToolContext as our tools use it: state, user_id and session.id."""

    def __init__(self, user_id: str = "", session_id: str = "") -> None:
        self.state: dict[str, Any] = {}
        self.user_id = user_id
        self.session = type("S", (), {"id": session_id or user_id})()


class Chat:
    """One WhatsApp conversation, with the identity binding the service sets."""

    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id
        self.context = FakeToolContext(user_id=chat_id, session_id=chat_id)
        self._token = None

    def __enter__(self) -> FakeToolContext:
        self._token = bind_channel_identity(
            ChannelIdentity(channel=CHANNEL_WHATSAPP, chat_id=self.chat_id)
        )
        return self.context

    def __exit__(self, *exc: Any) -> None:
        reset_channel_identity(self._token)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock) -> VerificationStore:
    store = VerificationStore(time_source=clock)
    set_verification_store(store)
    yield store
    set_verification_store(None)


@pytest.fixture
def escalations(monkeypatch) -> EscalationStore:
    store = EscalationStore()
    monkeypatch.setattr("app.tools.get_escalation_store", lambda: store)
    return store


@pytest.fixture
def backend(monkeypatch) -> FakeBackend:
    fake = FakeBackend()
    monkeypatch.setattr("app.tools.get_khatib_client", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _configured_backend(monkeypatch, settings):
    monkeypatch.setattr("app.tools.get_settings", lambda: settings)


def dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


# --- verification: the two supported methods -----------------------------


@pytest.mark.asyncio
async def test_verification_by_date_of_birth(store, backend, escalations):
    with Chat("963991234567@c.us") as ctx:
        result = await verify_patient("أحمد الخطيب", "1990-03-04", None, ctx)

    assert result["verified"] is True
    assert result["status"] == "verified"
    assert backend.verify_calls == [
        {
            "full_name": "أحمد الخطيب",
            "date_of_birth": "1990-03-04",
            "phone_last4": "",
        }
    ]


@pytest.mark.asyncio
async def test_verification_by_phone_suffix_keeps_the_leading_zero(store, backend):
    with Chat("963991234567@c.us") as ctx:
        result = await verify_patient("أحمد الخطيب", None, "0123", ctx)

    assert result["verified"] is True
    assert backend.verify_calls[0]["phone_last4"] == "0123"
    assert backend.verify_calls[0]["date_of_birth"] == ""


@pytest.mark.asyncio
async def test_arabic_digits_are_accepted_in_both_fields(store, backend):
    with Chat("963991234567@c.us") as ctx:
        await verify_patient("أحمد الخطيب", "١٩٩٠-٠٣-٠٤", "٠١٢٣", ctx)

    assert backend.verify_calls[0]["date_of_birth"] == "1990-03-04"
    assert backend.verify_calls[0]["phone_last4"] == "0123"


@pytest.mark.asyncio
async def test_a_tool_result_never_carries_the_token_or_the_patient_id(store, backend):
    with Chat("963991234567@c.us") as ctx:
        result = await verify_patient("أحمد الخطيب", None, "0123", ctx)

    rendered = dump(result)
    assert PATIENT_ID not in rendered
    assert TOKEN not in rendered
    assert "patientId" not in rendered and "patient_id" not in rendered


# --- verification: what it refuses to guess ------------------------------


@pytest.mark.asyncio
async def test_a_name_alone_asks_for_one_more_detail_without_calling_out(
    store, backend
):
    with Chat("963991234567@c.us") as ctx:
        result = await verify_patient("أحمد الخطيب", None, None, ctx)

    assert result["verified"] is False
    assert result["status"] == "needs_date_of_birth_or_phone_last4"
    # Nothing was sent: half an identity is not a request worth making.
    assert backend.verify_calls == []


@pytest.mark.asyncio
async def test_an_ambiguous_date_asks_instead_of_choosing(store, backend):
    with Chat("963991234567@c.us") as ctx:
        result = await verify_patient("أحمد الخطيب", "03/04/1990", None, ctx)

    assert result["status"] == "ambiguous_date_of_birth"
    assert backend.verify_calls == []


@pytest.mark.asyncio
async def test_a_missing_name_is_asked_for(store, backend):
    with Chat("963991234567@c.us") as ctx:
        result = await verify_patient("", "1990-03-04", None, ctx)

    assert result["status"] == "needs_full_name"
    assert backend.verify_calls == []


@pytest.mark.asyncio
async def test_a_failed_check_reveals_nothing_about_who_exists(
    store, backend, monkeypatch
):
    monkeypatch.setattr(
        "app.tools.get_khatib_client",
        lambda: FakeBackend(verify=VerifyOutcome(VerifyStatus.NOT_VERIFIED)),
    )

    with Chat("963991234567@c.us") as ctx:
        result = await verify_patient("شخص ما", None, "9999", ctx)

    assert result["verified"] is False
    assert result["status"] == "not_verified"
    assert (
        "Do NOT say whether a patient with those details exists" in (result["guidance"])
    )


# --- verification state: expiry, isolation, identity switching ------------


@pytest.mark.asyncio
async def test_verification_expires_and_then_blocks_result_retrieval(
    store, backend, clock
):
    chat = Chat("963991234567@c.us")
    with chat as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)
        clock.advance(899)
        allowed = await get_my_latest_results(ctx)
        clock.advance(2)
        blocked = await get_my_latest_results(ctx)

    assert allowed["status"] == "processing"
    assert blocked["status"] == "verification_required"
    assert blocked["verified"] is False
    # One request before expiry, none after.
    assert len(backend.result_calls) == 1


@pytest.mark.asyncio
async def test_results_are_refused_without_any_verification(store, backend):
    with Chat("963991234567@c.us") as ctx:
        result = await get_my_latest_results(ctx)

    assert result["status"] == "verification_required"
    assert backend.result_calls == []


@pytest.mark.asyncio
async def test_one_chats_verification_is_invisible_to_another(store, backend):
    with Chat("963991234567@c.us") as alice:
        await verify_patient("أحمد الخطيب", None, "0123", alice)

    with Chat("963997654321@c.us") as bob:
        result = await get_my_latest_results(bob)

    assert result["status"] == "verification_required"
    assert backend.result_calls == []

    # Alice is unaffected by Bob's attempt.
    with Chat("963991234567@c.us") as alice:
        assert (await get_my_latest_results(alice))["status"] == "processing"


@pytest.mark.asyncio
async def test_a_new_adk_session_does_not_inherit_the_verification(store, backend):
    chat_id = "963991234567@c.us"
    with Chat(chat_id) as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)

    restarted = Chat(chat_id)
    restarted.context = FakeToolContext(user_id=chat_id, session_id="session-2")
    with restarted as ctx:
        assert (await get_my_latest_results(ctx))["status"] == "verification_required"


@pytest.mark.asyncio
async def test_verifying_a_different_person_revokes_the_first_authorisation(
    store, backend, monkeypatch
):
    chat = Chat("963991234567@c.us")
    with chat as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)

        # The second person's details fail. The first person's access must be
        # gone regardless — they are not the one holding the phone now.
        monkeypatch.setattr(
            "app.tools.get_khatib_client",
            lambda: FakeBackend(verify=VerifyOutcome(VerifyStatus.NOT_VERIFIED)),
        )
        second = await verify_patient("سارة الخطيب", None, "4567", ctx)
        after = await get_my_latest_results(ctx)

    assert second["verified"] is False
    assert after["status"] == "verification_required"


@pytest.mark.asyncio
async def test_switching_identity_revokes_even_before_the_backend_answers(
    store, backend, monkeypatch
):
    chat = Chat("963991234567@c.us")
    with chat as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)

        # A new identity whose date is ambiguous: the call never leaves, but
        # the previous authorisation must not survive the attempt.
        second = await verify_patient("سارة الخطيب", "03/04/1990", None, ctx)
        after = await get_my_latest_results(ctx)

    assert second["status"] == "ambiguous_date_of_birth"
    assert after["status"] == "verification_required"


@pytest.mark.asyncio
async def test_re_verifying_the_same_identity_is_not_a_switch(store, backend):
    with Chat("963991234567@c.us") as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)
        again = await verify_patient("أحمد الخطيب", None, "0123", ctx)
        after = await get_my_latest_results(ctx)

    assert again["verified"] is True
    assert after["status"] == "processing"


# --- the results request --------------------------------------------------


@pytest.mark.asyncio
async def test_the_results_request_uses_the_stored_id_and_token(store, backend):
    with Chat("963991234567@c.us") as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)
        await get_my_latest_results(ctx)

    assert backend.result_calls == [
        {"patient_id": PATIENT_ID, "verification_token": TOKEN}
    ]


@pytest.mark.asyncio
async def test_the_results_tool_takes_no_identity_arguments():
    import inspect

    signature = inspect.signature(get_my_latest_results)

    # The model cannot name a patient, a token, a tenant or a chat, because
    # there is nowhere to put one.
    assert list(signature.parameters) == ["tool_context"]


@pytest.mark.asyncio
async def test_a_rejected_verification_token_forces_verifying_again(
    store, monkeypatch, backend
):
    with Chat("963991234567@c.us") as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)

        rejecting = FakeBackend(
            result=ResultOutcome(ResultStatusCode.VERIFICATION_REJECTED)
        )
        monkeypatch.setattr("app.tools.get_khatib_client", lambda: rejecting)
        first = await get_my_latest_results(ctx)
        second = await get_my_latest_results(ctx)

    assert first["status"] == "verification_expired"
    # The stored record was dropped, so the second call never reaches the
    # backend at all: a rejection is never retried into an authorisation.
    assert second["status"] == "verification_required"
    assert len(rejecting.result_calls) == 1


# --- result cases ---------------------------------------------------------


async def _results_for(payload_or_outcome, store, monkeypatch) -> dict[str, Any]:
    outcome = (
        payload_or_outcome
        if isinstance(payload_or_outcome, ResultOutcome)
        else ResultOutcome(ResultStatusCode.OK, payload_or_outcome)
    )
    fake = FakeBackend(result=outcome)
    monkeypatch.setattr("app.tools.get_khatib_client", lambda: fake)
    with Chat("963991234567@c.us") as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)
        return await get_my_latest_results(ctx)


@pytest.mark.asyncio
async def test_pending_relays_the_backend_message_and_promises_nothing(
    store, escalations, monkeypatch
):
    result = await _results_for(PENDING_PAYLOAD, store, monkeypatch)

    assert result["status"] == "processing"
    assert result["patient_message"] == PENDING_PAYLOAD["patientMessage"]
    assert result["report_image_available"] is False
    assert "Do NOT" in result["guidance"] and "ready" in result["guidance"]
    assert "escalate" not in result


@pytest.mark.asyncio
async def test_an_overdue_pending_result_goes_to_a_human_without_a_new_date(
    store, escalations, monkeypatch
):
    overdue = {**PENDING_PAYLOAD, "expectedReadyAt": "2020-01-01T10:00:00Z"}

    result = await _results_for(overdue, store, monkeypatch)

    assert result["status"] == "processing_overdue"
    assert result["escalate"] is True
    assert result["human_notified"] is False
    # The lapsed promise is not repeated back to the customer.
    assert "patient_message" not in result
    assert escalations.get("963991234567@c.us").reason is (
        EscalationReason.RESULT_DELAYED
    )


@pytest.mark.asyncio
async def test_clinician_review_relays_the_message_and_escalates(
    store, escalations, monkeypatch
):
    result = await _results_for(REVIEW_PAYLOAD, store, monkeypatch)

    assert result["status"] == "clinician_review_required"
    assert result["patient_message"] == REVIEW_PAYLOAD["patientMessage"]
    assert result["escalate"] is True
    assert result["human_notified"] is False
    assert result["report_image_available"] is False
    assert escalations.get("963991234567@c.us").reason is (
        EscalationReason.RESULT_CLINICIAN_REVIEW
    )
    # FINAL describes the record, not a licence to discuss its contents.
    assert "does NOT authorise" in result["guidance"]
    assert "appointment" in result["guidance"]


@pytest.mark.asyncio
async def test_an_unknown_disposition_goes_to_a_human(store, escalations, monkeypatch):
    result = await _results_for(
        {"status": "FINAL", "disposition": "RELEASED_TO_SOMETHING"}, store, monkeypatch
    )

    assert result["status"] == "unknown_result_state"
    assert result["escalate"] is True
    assert result["human_notified"] is False


@pytest.mark.asyncio
async def test_a_pending_result_with_no_message_is_not_improvised(
    store, escalations, monkeypatch
):
    result = await _results_for(
        {"status": "DRAFT", "disposition": "PENDING"}, store, monkeypatch
    )

    assert result["status"] == "unknown_result_state"
    assert result["escalate"] is True


@pytest.mark.asyncio
async def test_no_result_on_file_is_stated_plainly(store, escalations, monkeypatch):
    result = await _results_for(
        ResultOutcome(ResultStatusCode.NO_RESULT), store, monkeypatch
    )

    assert result["status"] == "no_result_on_file"
    assert "escalate" not in result


@pytest.mark.asyncio
async def test_every_backend_failure_looks_the_same_to_the_model(
    store, escalations, monkeypatch
):
    payloads = []
    for code in (
        ResultStatusCode.TIMEOUT,
        ResultStatusCode.UNAVAILABLE,
        ResultStatusCode.AUTH_ERROR,
    ):
        payloads.append(await _results_for(ResultOutcome(code), store, monkeypatch))

    # A timeout, an outage and a broken tenant credential are one outcome
    # here: which internal thing failed is not the customer's business, and
    # telling the model would put it in a reply.
    assert all(p["status"] == "service_unavailable" for p in payloads)
    assert len({dump(p) for p in payloads}) == 1
    assert "http" not in dump(payloads[0]).lower()


@pytest.mark.asyncio
async def test_the_image_field_being_absent_changes_nothing(
    store, escalations, monkeypatch
):
    # Today's payloads have no image field at all. The integration must behave
    # exactly as it does in the specified cases.
    for payload, expected in (
        (PENDING_PAYLOAD, "processing"),
        (REVIEW_PAYLOAD, "clinician_review_required"),
    ):
        result = await _results_for(payload, store, monkeypatch)

        assert result["status"] == expected
        assert result["report_image_available"] is False
        assert "url" not in dump(result).lower()


# --- failure budget -------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_failed_checks_stop_and_hand_off(
    store, escalations, monkeypatch
):
    monkeypatch.setattr(
        "app.tools.get_khatib_client",
        lambda: FakeBackend(verify=VerifyOutcome(VerifyStatus.NOT_VERIFIED)),
    )

    with Chat("963991234567@c.us") as ctx:
        # khatib_settings() allows 3 attempts in the window.
        for _ in range(2):
            assert (await verify_patient("شخص ما", None, "9999", ctx))["status"] == (
                "not_verified"
            )
        final = await verify_patient("شخص ما", None, "9999", ctx)
        blocked = await verify_patient("شخص ما", None, "8888", ctx)

    assert final["status"] == "too_many_attempts"
    assert final["escalate"] is True
    assert final["human_notified"] is False
    assert blocked["status"] == "too_many_attempts"
    assert escalations.get("963991234567@c.us").reason is (
        EscalationReason.VERIFICATION_FAILED
    )


@pytest.mark.asyncio
async def test_one_chats_failures_do_not_lock_another_chat(
    store, escalations, monkeypatch
):
    monkeypatch.setattr(
        "app.tools.get_khatib_client",
        lambda: FakeBackend(verify=VerifyOutcome(VerifyStatus.NOT_VERIFIED)),
    )

    with Chat("963991234567@c.us") as ctx:
        for _ in range(3):
            await verify_patient("شخص ما", None, "9999", ctx)

    with Chat("963997654321@c.us") as other:
        result = await verify_patient("شخص آخر", None, "1111", other)

    assert result["status"] == "not_verified"


@pytest.mark.asyncio
async def test_a_backend_outage_does_not_spend_the_customers_attempts(
    store, escalations, monkeypatch
):
    monkeypatch.setattr(
        "app.tools.get_khatib_client",
        lambda: FakeBackend(verify=VerifyOutcome(VerifyStatus.TIMEOUT)),
    )

    with Chat("963991234567@c.us") as ctx:
        for _ in range(5):
            result = await verify_patient("أحمد الخطيب", None, "0123", ctx)

    # Still "try again", never "locked out": the outage was not their doing.
    assert result["status"] == "service_unavailable"
    assert "escalate" not in result


# --- configuration --------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unconfigured_backend_degrades_without_leaking_why(
    store, backend, monkeypatch, settings
):
    from dataclasses import replace

    from app.config import Secret

    unconfigured = replace(settings, khatib=khatib_settings(client_secret=Secret("")))
    monkeypatch.setattr("app.tools.get_settings", lambda: unconfigured)

    with Chat("963991234567@c.us") as ctx:
        verified = await verify_patient("أحمد الخطيب", None, "0123", ctx)
        results = await get_my_latest_results(ctx)

    assert verified["status"] == "service_unavailable"
    assert results["status"] == "service_unavailable"
    assert backend.verify_calls == []
    assert "KHATIB" not in dump(verified)


@pytest.mark.asyncio
async def test_without_a_trusted_conversation_nothing_is_attempted(store, backend):
    # No WhatsApp binding and no ADK identity: there is no conversation to
    # scope a verification to, so the tool must not invent one.
    bare = FakeToolContext(user_id="", session_id="")

    result = await verify_patient("أحمد الخطيب", None, "0123", bare)

    assert result["status"] == "service_unavailable"
    assert backend.verify_calls == []


# --- FINAL / BENIGN: the released case -----------------------------------

IMAGE_URL = "https://khatib.example.test/reports/abc123.jpg"
BENIGN_PAYLOAD = {
    "resultId": "res_9001",
    "biopsyId": "bio_441",
    "sampleNumber": "S-2026-0012",
    "status": "FINAL",
    "disposition": "BENIGN",
    "imageURL": IMAGE_URL,
    "patientMessage": "نتيجتك سليمة، ومرفق تقرير النتيجة.",
}


@pytest.fixture
def staged():
    from app.patients.release import ReleaseStore, set_release_store

    store = ReleaseStore()
    set_release_store(store)
    yield store
    set_release_store(None)


@pytest.mark.asyncio
async def test_benign_stages_a_release_without_telling_the_model_anything(
    store, escalations, staged, monkeypatch
):
    result = await _results_for(BENIGN_PAYLOAD, store, monkeypatch)

    assert result["status"] == "released_with_report"
    assert result["report_image_available"] is True

    rendered = dump(result)
    # The model is told a report is going out, never where it lives, which
    # record it is, or what the laboratory wrote.
    assert IMAGE_URL not in rendered
    assert "imageURL" not in rendered
    assert "res_9001" not in rendered
    assert "bio_441" not in rendered
    assert "S-2026-0012" not in rendered
    assert "patient_message" not in result


@pytest.mark.asyncio
async def test_the_staged_release_carries_what_the_transport_needs(
    store, escalations, staged, monkeypatch, settings
):
    await _results_for(BENIGN_PAYLOAD, store, monkeypatch)

    from app.patients.context import CHANNEL_WHATSAPP, ConversationScope

    scope = ConversationScope(
        tenant_key=settings.khatib.tenant_key,
        channel=CHANNEL_WHATSAPP,
        chat_id="963991234567@c.us",
        session_id="963991234567@c.us",
    )
    release = staged.take(scope)

    assert release is not None
    assert release.image_url == IMAGE_URL
    assert release.file_name == "lab-report.jpg"
    assert release.patient_message == BENIGN_PAYLOAD["patientMessage"]
    assert release.result_id == "res_9001"
    assert release.patient_id == PATIENT_ID


@pytest.mark.asyncio
async def test_the_model_is_told_not_to_describe_the_result(
    store, escalations, staged, monkeypatch
):
    result = await _results_for(BENIGN_PAYLOAD, store, monkeypatch)

    guidance = result["guidance"]
    assert "Do NOT describe" in guidance
    assert "interpret" in guidance


@pytest.mark.asyncio
async def test_benign_without_a_usable_image_relays_the_message_only(
    store, escalations, staged, monkeypatch
):
    payload = {**BENIGN_PAYLOAD, "imageURL": "https://somewhere-else.test/r.jpg"}

    result = await _results_for(payload, store, monkeypatch)

    assert result["status"] == "released_without_report"
    assert result["patient_message"] == BENIGN_PAYLOAD["patientMessage"]
    assert result["report_image_available"] is False
    assert "cannot be sent" in result["guidance"]
    assert staged.count() == 0


@pytest.mark.asyncio
async def test_a_missing_image_field_does_not_break_the_benign_case(
    store, escalations, staged, monkeypatch
):
    payload = {k: v for k, v in BENIGN_PAYLOAD.items() if k != "imageURL"}

    result = await _results_for(payload, store, monkeypatch)

    assert result["status"] == "released_without_report"
    assert staged.count() == 0


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "DRAFT",
            "disposition": "BENIGN",
            "imageURL": IMAGE_URL,
            "patientMessage": "x",
        },
        {
            "status": "FINAL",
            "disposition": "CLINICIAN_REVIEW_REQUIRED",
            "imageURL": IMAGE_URL,
            "patientMessage": "x",
        },
        {
            "status": "DRAFT",
            "disposition": "PENDING",
            "imageURL": IMAGE_URL,
            "patientMessage": "x",
            "expectedReadyAt": "2099-01-01T10:00:00Z",
        },
        {
            "status": "FINAL",
            "disposition": "SOMETHING_NEW",
            "imageURL": IMAGE_URL,
            "patientMessage": "x",
        },
    ],
)
@pytest.mark.asyncio
async def test_no_other_disposition_stages_a_release(
    store, escalations, staged, monkeypatch, payload
):
    result = await _results_for(payload, store, monkeypatch)

    assert staged.count() == 0
    assert result["status"] != "released_with_report"


@pytest.mark.asyncio
async def test_an_unverified_chat_never_reaches_the_benign_path(
    store, escalations, staged, monkeypatch
):
    fake = FakeBackend(result=ResultOutcome(ResultStatusCode.OK, BENIGN_PAYLOAD))
    monkeypatch.setattr("app.tools.get_khatib_client", lambda: fake)

    with Chat("963991234567@c.us") as ctx:
        result = await get_my_latest_results(ctx)

    assert result["status"] == "verification_required"
    assert staged.count() == 0
    assert fake.result_calls == []


@pytest.mark.asyncio
async def test_losing_the_verification_drops_a_staged_release(
    store, escalations, staged, monkeypatch
):
    fake = FakeBackend(result=ResultOutcome(ResultStatusCode.OK, BENIGN_PAYLOAD))
    monkeypatch.setattr("app.tools.get_khatib_client", lambda: fake)

    with Chat("963991234567@c.us") as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)
        await get_my_latest_results(ctx)
        assert staged.count() == 1

        # Starting a verification for someone else revokes the authorisation,
        # and an unsent report is still an authorisation.
        monkeypatch.setattr(
            "app.tools.get_khatib_client",
            lambda: FakeBackend(verify=VerifyOutcome(VerifyStatus.NOT_VERIFIED)),
        )
        await verify_patient("سارة الخطيب", None, "4567", ctx)

    assert staged.count() == 0


@pytest.mark.asyncio
async def test_a_rejected_verification_token_drops_a_staged_release(
    store, escalations, staged, monkeypatch
):
    fake = FakeBackend(result=ResultOutcome(ResultStatusCode.OK, BENIGN_PAYLOAD))
    monkeypatch.setattr("app.tools.get_khatib_client", lambda: fake)

    with Chat("963991234567@c.us") as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)
        await get_my_latest_results(ctx)
        assert staged.count() == 1

        rejecting = FakeBackend(
            result=ResultOutcome(ResultStatusCode.VERIFICATION_REJECTED)
        )
        monkeypatch.setattr("app.tools.get_khatib_client", lambda: rejecting)
        await get_my_latest_results(ctx)

    assert staged.count() == 0


@pytest.mark.asyncio
async def test_asking_again_restages_rather_than_accumulating(
    store, escalations, staged, monkeypatch
):
    fake = FakeBackend(result=ResultOutcome(ResultStatusCode.OK, BENIGN_PAYLOAD))
    monkeypatch.setattr("app.tools.get_khatib_client", lambda: fake)

    with Chat("963991234567@c.us") as ctx:
        await verify_patient("أحمد الخطيب", None, "0123", ctx)
        await get_my_latest_results(ctx)
        await get_my_latest_results(ctx)

    # One slot per conversation: no backlog of unsent medical reports.
    assert staged.count() == 1
