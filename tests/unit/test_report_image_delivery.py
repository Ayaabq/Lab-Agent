"""Sending a report image: who may, when, and what happens when it fails.

This is the only path in the project that pushes patient content out to a
customer unprompted by the model, so each test fixes one rule that an
accidental refactor could otherwise erase.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import Secret
from app.escalation import EscalationStore
from app.patients.context import CHANNEL_WHATSAPP, ConversationScope
from app.patients.release import PendingRelease, ReleaseStore, set_release_store
from app.patients.store import (
    VerificationStore,
    VerifiedPatient,
    identity_fingerprint,
    set_verification_store,
)
from app.whatsapp.dedupe import TTLDedupeCache
from app.whatsapp.green_api import GreenApiError, SendResult
from app.whatsapp.service import (
    MAX_CAPTION_CHARS,
    REPORT_NEEDS_REVERIFY_REPLY,
    REPORT_SEND_FAILED_REPLY,
    REPORT_SEND_UNCERTAIN_REPLY,
    WhatsAppService,
)
from tests.unit.conftest import FakeRunner
from tests.unit.test_whatsapp_parsing import incoming

CHAT = "963991234567@c.us"
OTHER_CHAT = "963997654321@c.us"
IMAGE = "https://khatib.example.test/reports/abc123.jpg"
MESSAGE = "نتيجتك سليمة، ومرفق تقرير النتيجة."
PATIENT_ID = "pat_internal_7781"
IDENTITY = identity_fingerprint(
    full_name="أحمد الخطيب", date_of_birth="1990-03-04", phone_last4=""
)


class RecordingGreen:
    """Records text and file sends. Never opens a socket."""

    def __init__(
        self,
        *,
        live: bool = True,
        file_result: SendResult | None = None,
        file_error: Exception | None = None,
    ) -> None:
        self.live_enabled = live
        self.sent: list[tuple[str, str]] = []
        self.files: list[dict[str, Any]] = []
        self._file_result = file_result
        self._file_error = file_error

    async def send_text(self, chat_id: str, message: str) -> SendResult:
        self.sent.append((chat_id, message))
        return SendResult(True, False, chat_id, provider_message_id="TXT-1")

    async def send_file_by_url(
        self, chat_id: str, url_file: str, file_name: str, caption: str = ""
    ) -> SendResult:
        self.files.append(
            {
                "chat_id": chat_id,
                "url_file": url_file,
                "file_name": file_name,
                "caption": caption,
            }
        )
        if self._file_error is not None:
            raise self._file_error
        if self._file_result is not None:
            return self._file_result
        return SendResult(True, False, chat_id, provider_message_id="IMG-1")


@pytest.fixture
def verifications() -> VerificationStore:
    store = VerificationStore()
    set_verification_store(store)
    yield store
    set_verification_store(None)


@pytest.fixture
def releases() -> ReleaseStore:
    store = ReleaseStore()
    set_release_store(store)
    yield store
    set_release_store(None)


def scope_for(settings, chat_id: str = CHAT) -> ConversationScope:
    return ConversationScope(
        tenant_key=settings.khatib.tenant_key,
        channel=CHANNEL_WHATSAPP,
        chat_id=chat_id,
        session_id=chat_id,
    )


def verify(verifications, settings, chat_id: str = CHAT, *, ttl: int = 900) -> None:
    verifications.put(
        scope_for(settings, chat_id),
        VerifiedPatient(
            patient_id=PATIENT_ID,
            verification_token="vtok-abcdef",
            expires_at=verifications.now() + ttl,
            method="DATE_OF_BIRTH",
            identity=IDENTITY,
        ),
    )


def stage(
    releases,
    settings,
    chat_id: str = CHAT,
    *,
    message: str = MESSAGE,
    patient_id: str = PATIENT_ID,
    identity: str = IDENTITY,
    result_id: str = "res_9001",
) -> None:
    releases.stage(
        scope_for(settings, chat_id),
        PendingRelease(
            chat_id=chat_id,
            patient_message=message,
            image_url=IMAGE,
            file_name="lab-report.jpg",
            result_id=result_id,
            patient_id=patient_id,
            identity=identity,
            created_at=releases.now(),
        ),
    )


def build(settings, green) -> WhatsAppService:
    return WhatsAppService(
        settings=settings,
        green_client=green,
        escalation_store=EscalationStore(),
        dedupe=TTLDedupeCache(ttl_seconds=60, max_entries=20),
        runner_provider=lambda: FakeRunner("تفضل، التقرير مرفق."),
    )


async def deliver(service, chat_id: str = CHAT, message_id: str = "MSG-1") -> None:
    await service.handle_webhook(
        incoming("شو صار بنتيجتي؟", chat_id=chat_id, id_message=message_id), wait=True
    )


# --- the happy path -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_staged_report_is_sent_with_the_message_as_its_caption(
    settings, verifications, releases
):
    green = RecordingGreen()
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(build(settings, green))

    assert len(green.files) == 1
    sent = green.files[0]
    assert sent["chat_id"] == CHAT
    assert sent["url_file"] == IMAGE
    assert sent["file_name"] == "lab-report.jpg"
    # Relayed unchanged — not paraphrased, not shortened.
    assert sent["caption"] == MESSAGE


@pytest.mark.asyncio
async def test_the_agent_reply_goes_out_before_the_image(
    settings, verifications, releases
):
    green = RecordingGreen()
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(build(settings, green))

    assert green.sent[0][1] == "تفضل، التقرير مرفق."
    assert len(green.files) == 1


@pytest.mark.asyncio
async def test_a_long_message_is_sent_whole_as_its_own_text(
    settings, verifications, releases
):
    green = RecordingGreen()
    long_message = "ا" * (MAX_CAPTION_CHARS + 50)
    verify(verifications, settings)
    stage(releases, settings, message=long_message)

    await deliver(build(settings, green))

    # Sent separately and in full rather than cut down to caption length.
    assert any(text == long_message for _chat, text in green.sent)
    assert green.files[0]["caption"] == ""


@pytest.mark.asyncio
async def test_nothing_claims_the_report_was_delivered(
    settings, verifications, releases
):
    green = RecordingGreen()
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(build(settings, green))

    # A provider message id is an acknowledgement, not a delivery receipt.
    for _chat, text in green.sent:
        for claim in ("تم إرسال", "وصلك التقرير", "تم التسليم"):
            assert claim not in text


# --- who may not receive one ---------------------------------------------


@pytest.mark.asyncio
async def test_an_expired_verification_blocks_the_send_at_the_last_moment(
    settings, verifications, releases
):
    green = RecordingGreen()
    verify(verifications, settings, ttl=900)
    stage(releases, settings)
    # The backend answered while verified; the verification lapsed before the
    # image could go out. The check that counts is this one.
    verifications.clear(scope_for(settings))

    await deliver(build(settings, green))

    assert green.files == []
    assert any(text == REPORT_NEEDS_REVERIFY_REPLY for _chat, text in green.sent)


@pytest.mark.asyncio
async def test_a_different_verified_patient_in_the_chat_blocks_the_send(
    settings, verifications, releases
):
    green = RecordingGreen()
    verify(verifications, settings)
    # Staged for someone else: being verified now is not enough, it must be
    # the same verification that authorised the release.
    stage(releases, settings, patient_id="pat_someone_else")

    await deliver(build(settings, green))

    assert green.files == []


@pytest.mark.asyncio
async def test_a_re_verified_different_identity_blocks_the_send(
    settings, verifications, releases
):
    green = RecordingGreen()
    verify(verifications, settings)
    stage(releases, settings, identity="a-different-persons-fingerprint")

    await deliver(build(settings, green))

    assert green.files == []


@pytest.mark.asyncio
async def test_a_report_staged_for_one_chat_never_goes_to_another(
    settings, verifications, releases
):
    green = RecordingGreen()
    verify(verifications, settings, CHAT)
    verify(verifications, settings, OTHER_CHAT)
    stage(releases, settings, CHAT)

    # A message arrives in the *other* chat.
    await deliver(build(settings, green), chat_id=OTHER_CHAT, message_id="OTHER-1")

    assert green.files == []


@pytest.mark.asyncio
async def test_no_release_staged_means_no_image(settings, verifications, releases):
    green = RecordingGreen()
    verify(verifications, settings)

    await deliver(build(settings, green))

    assert green.files == []


# --- duplicates -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_redelivered_webhook_does_not_send_the_report_twice(
    settings, verifications, releases
):
    green = RecordingGreen()
    service = build(settings, green)
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(service, message_id="SAME-1")
    await deliver(service, message_id="SAME-1")

    assert len(green.files) == 1


@pytest.mark.asyncio
async def test_a_release_can_only_be_collected_once(settings, verifications, releases):
    green = RecordingGreen()
    service = build(settings, green)
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(service, message_id="FIRST")
    # A second, genuinely new message with nothing newly staged.
    await deliver(service, message_id="SECOND")

    assert len(green.files) == 1


@pytest.mark.asyncio
async def test_the_same_report_is_not_resent_even_if_staged_again(
    settings, verifications, releases
):
    green = RecordingGreen()
    service = build(settings, green)
    verify(verifications, settings)

    stage(releases, settings, result_id="res_9001")
    await deliver(service, message_id="FIRST")
    stage(releases, settings, result_id="res_9001")
    await deliver(service, message_id="SECOND")

    assert len(green.files) == 1


@pytest.mark.asyncio
async def test_a_genuinely_new_result_is_sent(settings, verifications, releases):
    green = RecordingGreen()
    service = build(settings, green)
    verify(verifications, settings)

    stage(releases, settings, result_id="res_9001")
    await deliver(service, message_id="FIRST")
    stage(releases, settings, result_id="res_9002")
    await deliver(service, message_id="SECOND")

    assert len(green.files) == 2


# --- the live-sending switch ---------------------------------------------


@pytest.mark.asyncio
async def test_a_dry_run_sends_no_file_and_says_nothing_about_it(
    settings, verifications, releases
):
    green = RecordingGreen(file_result=SendResult(False, True, CHAT, "live_disabled"))
    green.live_enabled = False
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(build(settings, green))

    # The client itself is what suppresses the request; the service must not
    # then apologise to a customer who was never going to get anything.
    assert REPORT_SEND_FAILED_REPLY not in [t for _c, t in green.sent]
    assert REPORT_SEND_UNCERTAIN_REPLY not in [t for _c, t in green.sent]


# --- failures -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_provider_error_produces_an_arabic_fallback(
    settings, verifications, releases
):
    green = RecordingGreen(
        file_error=GreenApiError("file send rejected with status 400")
    )
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(build(settings, green))

    texts = [t for _c, t in green.sent]
    assert REPORT_SEND_FAILED_REPLY in texts
    # The laboratory's own message still reaches the customer.
    assert MESSAGE in texts


@pytest.mark.asyncio
async def test_an_ambiguous_timeout_is_neither_claimed_nor_denied(
    settings, verifications, releases
):
    green = RecordingGreen(
        file_result=SendResult(
            False, False, CHAT, "send_timeout_ambiguous", ambiguous=True
        )
    )
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(build(settings, green))

    texts = [t for _c, t in green.sent]
    assert REPORT_SEND_UNCERTAIN_REPLY in texts
    assert REPORT_SEND_FAILED_REPLY not in texts
    # Not retried: the provider may already have queued the file.
    assert len(green.files) == 1


@pytest.mark.asyncio
async def test_an_ambiguous_timeout_is_not_sent_again_later(
    settings, verifications, releases
):
    green = RecordingGreen(
        file_result=SendResult(
            False, False, CHAT, "send_timeout_ambiguous", ambiguous=True
        )
    )
    service = build(settings, green)
    verify(verifications, settings)
    stage(releases, settings)
    await deliver(service, message_id="FIRST")

    # The customer asks again. We still do not know whether the first attempt
    # was queued provider-side, so the same report is NOT pushed a second
    # time; the uncertain reply already told them to check with the lab.
    stage(releases, settings)
    await deliver(service, message_id="SECOND")

    assert len(green.files) == 1


@pytest.mark.asyncio
async def test_a_rejected_send_does_not_block_the_text_reply(
    settings, verifications, releases
):
    green = RecordingGreen(file_error=RuntimeError("boom"))
    verify(verifications, settings)
    stage(releases, settings)

    await deliver(build(settings, green))

    assert green.sent[0][1] == "تفضل، التقرير مرفق."


# --- logging --------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_log_line_carries_the_report_url(
    settings, verifications, releases, caplog
):
    from app.logging_utils import redact_text

    green = RecordingGreen(file_error=GreenApiError("rejected"))
    verify(verifications, settings)
    stage(releases, settings)

    with caplog.at_level("DEBUG"):
        await deliver(build(settings, green))

    for record in caplog.records:
        rendered = redact_text(
            record.getMessage() + str(getattr(record, "context", ""))
        )
        assert IMAGE not in rendered
        assert "abc123.jpg" not in rendered


def test_the_staged_release_hides_its_url_in_a_repr(settings, releases):
    stage(releases, settings)
    release = releases.take(scope_for(settings))

    assert IMAGE not in repr(release)
    assert PATIENT_ID not in repr(release)


def test_an_unconfigured_backend_still_scopes_releases(settings):
    from dataclasses import replace

    from tests.unit.conftest import khatib_settings

    other = replace(settings, khatib=khatib_settings(client_id=Secret("another")))

    # A different tenant is a different scope, so a release staged under one
    # cannot be collected under the other.
    assert scope_for(settings).key != scope_for(other).key


# --- the duplicate claim is taken only when the file might exist ---------


@pytest.mark.asyncio
async def test_a_definite_rejection_does_not_block_a_later_attempt(
    settings, verifications, releases
):
    """A send that definitely failed must not lock the report out.

    Claiming the key on a definite failure would mean a transient provider
    error costs the customer their report for the whole dedupe window, with
    no safety benefit: nothing was sent, so a later send is not a duplicate.
    """
    green = RecordingGreen(
        file_error=GreenApiError("file send rejected with status 500")
    )
    service = build(settings, green)
    verify(verifications, settings)

    stage(releases, settings)
    await deliver(service, message_id="FIRST")
    assert green.files and len(green.files) == 1

    # The customer asks again and the provider has recovered.
    green._file_error = None
    stage(releases, settings)
    await deliver(service, message_id="SECOND")

    assert len(green.files) == 2


@pytest.mark.asyncio
async def test_a_rejected_result_without_an_exception_also_does_not_claim(
    settings, verifications, releases
):
    green = RecordingGreen(file_result=SendResult(False, False, CHAT, "rejected"))
    service = build(settings, green)
    verify(verifications, settings)

    stage(releases, settings)
    await deliver(service, message_id="FIRST")
    green._file_result = None
    stage(releases, settings)
    await deliver(service, message_id="SECOND")

    assert len(green.files) == 2


@pytest.mark.asyncio
async def test_a_dry_run_does_not_claim_the_report(settings, verifications, releases):
    """Turning live sending on later must still deliver the report."""
    green = RecordingGreen(file_result=SendResult(False, True, CHAT, "live_disabled"))
    green.live_enabled = False
    service = build(settings, green)
    verify(verifications, settings)

    stage(releases, settings)
    await deliver(service, message_id="DRY")

    # Live sending comes on.
    green.live_enabled = True
    green._file_result = None
    stage(releases, settings)
    await deliver(service, message_id="LIVE")

    assert len(green.files) == 2
    assert green.files[-1]["caption"] == MESSAGE


@pytest.mark.asyncio
async def test_a_successful_send_still_claims_the_report(
    settings, verifications, releases
):
    green = RecordingGreen()
    service = build(settings, green)
    verify(verifications, settings)

    stage(releases, settings)
    await deliver(service, message_id="FIRST")
    stage(releases, settings)
    await deliver(service, message_id="SECOND")

    assert len(green.files) == 1
