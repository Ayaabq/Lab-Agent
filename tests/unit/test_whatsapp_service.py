"""End-to-end webhook handling with a fake runner and a fake GREEN API client."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.escalation import EscalationReason, EscalationStore
from app.patients.context import CHANNEL_WHATSAPP, current_channel_identity
from app.whatsapp.dedupe import TTLDedupeCache
from app.whatsapp.green_api import SendResult
from app.whatsapp.service import FALLBACK_REPLY, WhatsAppService
from tests.unit.conftest import FakeEvent, FakeEventContentPart, FakeRunner
from tests.unit.test_whatsapp_parsing import incoming


class FakeGreenClient:
    live_enabled = False

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, chat_id: str, message: str) -> SendResult:
        self.sent.append((chat_id, message))
        return SendResult(False, True, chat_id, "live_disabled")


def build_service(settings, runner=None, green=None, escalation=None):
    runner = runner or FakeRunner()
    return (
        WhatsAppService(
            settings=settings,
            green_client=green or FakeGreenClient(),
            escalation_store=escalation or EscalationStore(),
            dedupe=TTLDedupeCache(ttl_seconds=60, max_entries=10),
            runner_provider=lambda: runner,
        ),
        runner,
    )


@pytest.mark.asyncio
async def test_message_is_answered_and_reply_dispatched(settings):
    green = FakeGreenClient()
    service, runner = build_service(settings, green=green)

    ack = await service.handle_webhook(incoming("مرحبا"), wait=True)

    assert ack.status == "accepted"
    assert len(runner.calls) == 1
    assert green.sent == [("963991234567@c.us", runner.reply)]


@pytest.mark.asyncio
async def test_chat_id_is_used_as_user_and_session_id(settings):
    service, runner = build_service(settings)

    await service.handle_webhook(incoming("مرحبا"), wait=True)

    call = runner.calls[0]
    assert call["user_id"] == "963991234567@c.us"
    assert call["session_id"] == "963991234567@c.us"


@pytest.mark.asyncio
async def test_two_chats_get_separate_sessions(settings):
    service, runner = build_service(settings)

    await service.handle_webhook(
        incoming("مرحبا", chat_id="963991111111@c.us", id_message="A"), wait=True
    )
    await service.handle_webhook(
        incoming("مرحبا", chat_id="963992222222@c.us", id_message="B"), wait=True
    )

    session_ids = [c["session_id"] for c in runner.calls]
    assert session_ids == ["963991111111@c.us", "963992222222@c.us"]
    assert len(set(session_ids)) == 2


@pytest.mark.asyncio
async def test_same_chat_reuses_one_session(settings):
    service, runner = build_service(settings)

    await service.handle_webhook(incoming("مرحبا", id_message="A"), wait=True)
    await service.handle_webhook(incoming("وين المخبر؟", id_message="B"), wait=True)

    assert {c["session_id"] for c in runner.calls} == {"963991234567@c.us"}


@pytest.mark.asyncio
async def test_duplicate_message_id_is_processed_once(settings):
    green = FakeGreenClient()
    service, runner = build_service(settings, green=green)

    first = await service.handle_webhook(incoming(id_message="DUP-1"), wait=True)
    second = await service.handle_webhook(incoming(id_message="DUP-1"), wait=True)

    assert first.status == "accepted"
    assert second.status == "ignored"
    assert second.reason == "duplicate"
    assert len(runner.calls) == 1
    assert len(green.sent) == 1


@pytest.mark.asyncio
async def test_ignored_events_never_reach_the_agent(settings):
    green = FakeGreenClient()
    service, runner = build_service(settings, green=green)

    outgoing = incoming(id_message="O-1")
    outgoing["typeWebhook"] = "outgoingMessageReceived"
    group = incoming(chat_id="120363000000000000@g.us", id_message="G-1")
    empty = incoming("   ", id_message="E-1")

    for payload in (outgoing, group, empty):
        ack = await service.handle_webhook(payload, wait=True)
        assert ack.status == "ignored"

    assert runner.calls == []
    assert green.sent == []


@pytest.mark.asyncio
async def test_messages_from_one_chat_are_serialised(settings):
    order: list[str] = []

    class SlowRunner(FakeRunner):
        async def run_async(self, *, user_id, session_id, new_message):
            label = new_message.parts[0].text
            order.append(f"start:{label}")
            await asyncio.sleep(0.02)
            order.append(f"end:{label}")
            self.calls.append(
                {"user_id": user_id, "session_id": session_id, "text": label}
            )
            yield FakeEvent(self.reply)

    service, _ = build_service(settings, runner=SlowRunner())

    await asyncio.gather(
        service.handle_webhook(incoming("أول", id_message="S-1")),
        service.handle_webhook(incoming("تاني", id_message="S-2")),
    )
    await service.drain()

    # No interleaving: each turn finishes before the next one starts.
    assert order in (
        ["start:أول", "end:أول", "start:تاني", "end:تاني"],
        ["start:تاني", "end:تاني", "start:أول", "end:أول"],
    )


@pytest.mark.asyncio
async def test_different_chats_run_concurrently(settings):
    active = 0
    peak = 0

    class SlowRunner(FakeRunner):
        async def run_async(self, *, user_id, session_id, new_message):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            yield FakeEvent(self.reply)

    service, _ = build_service(settings, runner=SlowRunner())

    await asyncio.gather(
        service.handle_webhook(
            incoming("أ", chat_id="963991111111@c.us", id_message="C-1")
        ),
        service.handle_webhook(
            incoming("ب", chat_id="963992222222@c.us", id_message="C-2")
        ),
    )
    await service.drain()

    assert peak == 2


@pytest.mark.asyncio
async def test_agent_failure_falls_back_safely(settings):
    green = FakeGreenClient()

    class BrokenRunner(FakeRunner):
        async def run_async(self, *, user_id, session_id, new_message):
            raise RuntimeError("model exploded")
            yield  # pragma: no cover - makes this an async generator

    service, _ = build_service(settings, runner=BrokenRunner(), green=green)

    await service.handle_webhook(incoming("مرحبا"), wait=True)

    assert green.sent == [("963991234567@c.us", FALLBACK_REPLY)]


@pytest.mark.asyncio
async def test_empty_agent_response_falls_back_and_escalates(settings):
    green = FakeGreenClient()
    escalation = EscalationStore()

    class SilentRunner(FakeRunner):
        async def run_async(self, *, user_id, session_id, new_message):
            yield FakeEvent("")

    service, _ = build_service(
        settings, runner=SilentRunner(), green=green, escalation=escalation
    )

    await service.handle_webhook(incoming("مرحبا"), wait=True)

    assert green.sent[0][1] == FALLBACK_REPLY
    assert escalation.is_awaiting("963991234567@c.us")


@pytest.mark.asyncio
async def test_provider_send_failure_does_not_crash_the_task(settings):
    class FailingGreen(FakeGreenClient):
        async def send_text(self, chat_id: str, message: str):
            raise RuntimeError("provider down")

    service, _ = build_service(settings, green=FailingGreen())

    await service.handle_webhook(incoming("مرحبا"), wait=True)  # must not raise


@pytest.mark.asyncio
async def test_malformed_payload_is_ignored_not_raised(settings):
    service, runner = build_service(settings)

    ack = await service.handle_webhook({"typeWebhook": 12345}, wait=True)

    assert ack.status == "ignored"
    assert runner.calls == []


# --- deterministic medical safety triage ---------------------------------


@pytest.mark.asyncio
async def test_result_interpretation_never_reaches_the_model(settings):
    """The refusal must not depend on the LLM choosing to refuse."""
    green = FakeGreenClient()
    escalation = EscalationStore()
    service, runner = build_service(settings, green=green, escalation=escalation)

    await service.handle_webhook(
        incoming("نتيجة تحليلي طالعة مرتفعة، شو يعني هالشي؟"), wait=True
    )

    assert runner.calls == []
    record = escalation.get("963991234567@c.us")
    assert record.reason is EscalationReason.MEDICAL_ADVICE_REQUESTED
    reply = green.sent[0][1]
    assert "لست طبيبًا" in reply
    assert "الطبيب المعالج" in reply


@pytest.mark.asyncio
async def test_urgent_symptoms_get_the_emergency_reply(settings):
    green = FakeGreenClient()
    escalation = EscalationStore()
    service, runner = build_service(settings, green=green, escalation=escalation)

    await service.handle_webhook(incoming("عندي ألم بالصدر وما فيني تنفس"), wait=True)

    assert runner.calls == []
    assert escalation.get("963991234567@c.us").reason is EscalationReason.URGENT_MEDICAL
    assert "الطوارئ" in green.sent[0][1]


@pytest.mark.asyncio
async def test_medication_request_is_refused(settings):
    green = FakeGreenClient()
    service, runner = build_service(settings, green=green)

    await service.handle_webhook(incoming("شو العلاج المناسب لحالتي؟"), wait=True)

    assert runner.calls == []
    assert "لست طبيبًا" in green.sent[0][1]


@pytest.mark.asyncio
async def test_ordinary_question_still_reaches_the_agent(settings):
    service, runner = build_service(settings)

    await service.handle_webhook(incoming("شو أوقات الدوام عندكم؟"), wait=True)

    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_stats_report_live_sending_state(settings):
    service, _ = build_service(settings)

    await service.handle_webhook(incoming("مرحبا"), wait=True)

    stats = service.stats()
    assert stats["live_sending_enabled"] is False
    assert stats["deduped_messages"] == 1
    assert stats["tracked_chats"] == 1


# --- trusted conversation identity for the patient tools -----------------


class IdentityRecordingRunner(FakeRunner):
    """Captures the bound channel identity from inside the agent turn.

    This is the only place a patient tool can learn which chat it is serving,
    so what a tool would see is exactly what this records.
    """

    def __init__(self) -> None:
        super().__init__()
        self.identities: list[Any] = []

    async def run_async(self, *, user_id: str, session_id: str, new_message):
        self.identities.append(current_channel_identity())
        async for event in super().run_async(
            user_id=user_id, session_id=session_id, new_message=new_message
        ):
            yield event


@pytest.mark.asyncio
async def test_the_agent_turn_runs_bound_to_the_authenticated_chat(settings):
    runner = IdentityRecordingRunner()
    service, _ = build_service(settings, runner=runner)

    await service.handle_webhook(incoming("وين صارت نتيجتي؟"), wait=True)

    identity = runner.identities[0]
    assert identity is not None
    assert identity.channel == CHANNEL_WHATSAPP
    assert identity.chat_id == "963991234567@c.us"


@pytest.mark.asyncio
async def test_each_chat_binds_its_own_identity(settings):
    runner = IdentityRecordingRunner()
    service, _ = build_service(settings, runner=runner)

    await service.handle_webhook(incoming("مرحبا"), wait=True)
    await service.handle_webhook(
        incoming("مرحبا", chat_id="963997654321@c.us", id_message="OTHER"), wait=True
    )

    assert [i.chat_id for i in runner.identities] == [
        "963991234567@c.us",
        "963997654321@c.us",
    ]


@pytest.mark.asyncio
async def test_the_identity_is_unbound_once_the_turn_ends(settings):
    runner = IdentityRecordingRunner()
    service, _ = build_service(settings, runner=runner)

    await service.handle_webhook(incoming("مرحبا"), wait=True)

    # A leftover binding would let unrelated later work act as this customer.
    assert current_channel_identity() is None


@pytest.mark.asyncio
async def test_a_timed_out_turn_still_unbinds_the_identity(settings, monkeypatch):
    class HangingRunner(FakeRunner):
        async def run_async(self, *, user_id, session_id, new_message):
            await asyncio.sleep(10)
            yield FakeEvent("never")

    monkeypatch.setattr("app.whatsapp.service.AGENT_TURN_TIMEOUT_SECONDS", 0.01)
    service, _ = build_service(settings, runner=HangingRunner())

    await service.handle_webhook(incoming("مرحبا"), wait=True)

    assert current_channel_identity() is None


# --- reply assembly -------------------------------------------------------


class SplitPartsRunner(FakeRunner):
    """A final response delivered as several text parts, as the model does."""

    def __init__(self, pieces: list[str], events: int = 1) -> None:
        super().__init__()
        self.pieces = pieces
        self.events = events

    async def run_async(self, *, user_id, session_id, new_message):
        for _ in range(self.events):
            event = FakeEvent("")
            event.content.parts = [FakeEventContentPart(p) for p in self.pieces]
            yield event


@pytest.mark.asyncio
async def test_parts_of_one_response_are_joined_without_breaking_words(settings):
    green = FakeGreenClient()
    # "أعتذر" split mid-word, which is where the model's part boundary fell.
    service, _ = build_service(
        settings,
        runner=SplitPartsRunner(["أعت", "ذر منك، في مشكلة تقنية."]),
        green=green,
    )

    await service.handle_webhook(incoming("وين موقع المخبر؟"), wait=True)

    reply = green.sent[-1][1]
    assert reply == "أعتذر منك، في مشكلة تقنية."
    # A newline inside a word reaches the customer as mangled Arabic.
    assert "أعت\nذر" not in reply


@pytest.mark.asyncio
async def test_spacing_between_fragments_is_preserved(settings):
    green = FakeGreenClient()
    service, _ = build_service(
        settings, runner=SplitPartsRunner(["أهلاً ", "وسهلاً ", "فيك"]), green=green
    )

    await service.handle_webhook(incoming("مرحبا"), wait=True)

    # Stripping each fragment would glue these into "أهلاًوسهلاًفيك".
    assert green.sent[-1][1] == "أهلاً وسهلاً فيك"


@pytest.mark.asyncio
async def test_separate_final_events_stay_on_separate_lines(settings):
    green = FakeGreenClient()
    service, _ = build_service(
        settings, runner=SplitPartsRunner(["سطر"], events=2), green=green
    )

    await service.handle_webhook(incoming("مرحبا"), wait=True)

    assert green.sent[-1][1] == "سطر\nسطر"


@pytest.mark.asyncio
async def test_a_backlog_of_stale_messages_never_reaches_the_model(settings):
    """A replayed GREEN API queue must not turn into a burst of late replies."""
    import time as _time
    from dataclasses import replace

    green = FakeGreenClient()
    aged = replace(
        settings, green_api=replace(settings.green_api, max_message_age_seconds=600)
    )
    service, runner = build_service(aged, green=green)

    for n in range(5):
        payload = incoming("سؤال قديم", id_message=f"OLD-{n}")
        payload["timestamp"] = int(_time.time() - 7200)
        ack = await service.handle_webhook(payload, wait=True)
        assert ack.reason == "stale_message"

    # No model call, no WhatsApp message, no quota spent.
    assert runner.calls == []
    assert green.sent == []


@pytest.mark.asyncio
async def test_a_fresh_message_still_gets_through_with_the_guard_on(settings):
    import time as _time
    from dataclasses import replace

    green = FakeGreenClient()
    aged = replace(
        settings, green_api=replace(settings.green_api, max_message_age_seconds=600)
    )
    service, runner = build_service(aged, green=green)

    payload = incoming("مرحبا", id_message="FRESH-1")
    payload["timestamp"] = int(_time.time())

    ack = await service.handle_webhook(payload, wait=True)

    assert ack.status == "accepted"
    assert len(runner.calls) == 1
    assert len(green.sent) == 1


# --- GREEN_API_ALLOWED_CHATS, loaded from the environment -----------------


def _with_env_allowlist(settings, monkeypatch, value: str | None):
    """Settings whose allowlist comes from the real environment parser."""
    from dataclasses import replace

    from app.config import load_settings

    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    if value is None:
        monkeypatch.delenv("GREEN_API_ALLOWED_CHATS", raising=False)
    else:
        monkeypatch.setenv("GREEN_API_ALLOWED_CHATS", value)
    allowed = load_settings().green_api.allowed_chats
    return replace(
        settings, green_api=replace(settings.green_api, allowed_chats=allowed)
    )


@pytest.mark.asyncio
async def test_a_chat_outside_the_env_allowlist_never_reaches_the_agent(
    settings, monkeypatch
):
    green = FakeGreenClient()
    restricted = _with_env_allowlist(settings, monkeypatch, "+963 94 765 4321")
    service, runner = build_service(restricted, green=green)

    ack = await service.handle_webhook(incoming("مرحبا"), wait=True)

    # Acknowledged so GREEN API stops retrying, but no model call, no reply.
    assert ack.status == "ignored"
    assert ack.reason == "not_allowlisted"
    assert runner.calls == []
    assert green.sent == []


@pytest.mark.asyncio
async def test_a_chat_on_the_env_allowlist_is_answered(settings, monkeypatch):
    green = FakeGreenClient()
    restricted = _with_env_allowlist(settings, monkeypatch, "963947654321,963991234567")
    service, runner = build_service(restricted, green=green)

    ack = await service.handle_webhook(incoming("مرحبا"), wait=True)

    assert ack.status == "accepted"
    assert len(runner.calls) == 1
    assert green.sent == [("963991234567@c.us", runner.reply)]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, ""])
async def test_without_an_env_allowlist_every_private_chat_is_answered(
    settings, monkeypatch, value
):
    green = FakeGreenClient()
    open_settings = _with_env_allowlist(settings, monkeypatch, value)
    service, runner = build_service(open_settings, green=green)

    for n, chat in enumerate(("963991111111@c.us", "963992222222@c.us")):
        await service.handle_webhook(
            incoming("مرحبا", chat_id=chat, id_message=f"OPEN-{n}"), wait=True
        )

    assert [c["session_id"] for c in runner.calls] == [
        "963991111111@c.us",
        "963992222222@c.us",
    ]
    assert len(green.sent) == 2


@pytest.mark.asyncio
async def test_the_allowlist_does_not_bypass_duplicate_detection(settings, monkeypatch):
    green = FakeGreenClient()
    restricted = _with_env_allowlist(settings, monkeypatch, "963991234567")
    service, runner = build_service(restricted, green=green)

    first = await service.handle_webhook(incoming(id_message="DUP-A"), wait=True)
    second = await service.handle_webhook(incoming(id_message="DUP-A"), wait=True)

    assert first.status == "accepted"
    assert second.reason == "duplicate"
    assert len(runner.calls) == 1
    assert len(green.sent) == 1
