"""Escalation triage and the in-memory awaiting-human state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.escalation import (
    EscalationReason,
    EscalationStore,
    classify_message,
    message_for,
)


@pytest.mark.parametrize(
    "text",
    [
        "بدي احكي مع موظف",
        "ممكن حدا من خدمة العملاء؟",
        "بدي مسؤول",
        "I want to talk to a human",
        "can I speak to an agent",
    ],
)
def test_human_requests_are_detected(text):
    assert classify_message(text) is EscalationReason.USER_REQUESTED_HUMAN


@pytest.mark.parametrize(
    "text",
    [
        "نتيجتي طالعة مرتفعة شو يعني؟",
        "ممكن تفسير للنتيجة؟",
        "شو العلاج المناسب؟",
        "أي دواء لازم آخد؟",
        "what does this result mean",
        "can you diagnose me",
    ],
)
def test_medical_advice_requests_are_detected(text):
    assert classify_message(text) is EscalationReason.MEDICAL_ADVICE_REQUESTED


@pytest.mark.parametrize(
    "text",
    [
        "عندي ألم بالصدر",
        "ما فيني تنفس",
        "في نزيف شديد",
        "I have chest pain",
        "she is unconscious",
    ],
)
def test_urgent_symptoms_are_detected(text):
    assert classify_message(text) is EscalationReason.URGENT_MEDICAL


def test_urgency_wins_over_interpretation():
    """A message can be both; the emergency response is the safer one."""
    text = "نتيجتي مرتفعة وعندي ألم بالصدر، شو يعني؟"

    assert classify_message(text) is EscalationReason.URGENT_MEDICAL


@pytest.mark.parametrize(
    "text",
    [
        "مرحبا",
        "شو أوقات الدوام؟",
        "كم سعر تحليل فيتامين د؟",
        "وين موقع المخبر؟",
        "شكرا إلك",
        "",
        None,
    ],
)
def test_ordinary_messages_are_not_escalated(text):
    assert classify_message(text) is None


def test_every_reason_has_a_message():
    for reason in EscalationReason.__members__.values():
        assert message_for(reason).strip()


def test_medical_message_refuses_and_redirects():
    text = message_for(EscalationReason.MEDICAL_ADVICE_REQUESTED)

    assert "لست طبيبًا" in text
    assert "الطبيب المعالج" in text


def test_no_message_claims_staff_were_notified():
    """There is no notification channel yet — nothing may promise a callback."""
    for reason in EscalationReason.__members__.values():
        text = message_for(reason)
        assert "تم إبلاغ" not in text
        assert "رح يتصل" not in text
        assert "سيتصل بك" not in text


# --- store ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_marking_records_the_reason():
    store = EscalationStore()

    record = await store.mark(
        "963991234567@c.us", EscalationReason.USER_REQUESTED_HUMAN
    )

    assert record.reason is EscalationReason.USER_REQUESTED_HUMAN
    assert record.human_notified is False
    assert await store.is_awaiting("963991234567@c.us")


@pytest.mark.asyncio
async def test_unknown_chat_is_not_awaiting():
    assert await EscalationStore().is_awaiting("963990000000@c.us") is False


@pytest.mark.asyncio
async def test_remarking_replaces_the_previous_reason():
    store = EscalationStore()
    await store.mark("chat@c.us", EscalationReason.INFORMATION_UNAVAILABLE)

    await store.mark("chat@c.us", EscalationReason.URGENT_MEDICAL)

    assert (await store.get("chat@c.us")).reason is EscalationReason.URGENT_MEDICAL
    assert await store.pending_count() == 1


@pytest.mark.asyncio
async def test_clear_removes_the_record():
    store = EscalationStore()
    await store.mark("chat@c.us", EscalationReason.TOOL_FAILURE)

    await store.clear("chat@c.us")

    assert await store.is_awaiting("chat@c.us") is False


@pytest.mark.asyncio
async def test_store_is_bounded():
    store = EscalationStore(max_entries=3)

    for index in range(10):
        await store.mark(f"chat-{index}@c.us", EscalationReason.TOOL_FAILURE)

    assert await store.pending_count() == 3
    assert await store.is_awaiting("chat-0@c.us") is False
    assert await store.is_awaiting("chat-9@c.us") is True


@pytest.mark.asyncio
async def test_snapshot_masks_chat_ids():
    store = EscalationStore()
    await store.mark("963991234567@c.us", EscalationReason.USER_REQUESTED_HUMAN)

    snapshot = await store.snapshot()

    assert "963991234567@c.us" not in snapshot
    assert any("*" in key for key in snapshot)


# --- contact line ----------------------------------------------------------


def _phone(monkeypatch, number: str) -> None:
    monkeypatch.setattr(
        "app.escalation.get_settings",
        lambda: SimpleNamespace(lab_contact_phone=number),
    )


def test_replies_never_send_the_customer_to_an_employee(monkeypatch):
    """The assistant speaks for the laboratory; it is not a relay to staff."""
    _phone(monkeypatch, "")
    for reason in EscalationReason.__members__.values():
        assert "موظف" not in message_for(reason)


def test_replies_follow_up_here_and_quote_the_configured_number(monkeypatch):
    _phone(monkeypatch, "011 222 3344")

    text = message_for(EscalationReason.RESULT_DELAYED)

    assert "بنتابع معك هون" in text
    assert "على الرقم 011 222 3344" in text


def test_without_a_number_the_reply_invents_none(monkeypatch):
    _phone(monkeypatch, "")

    text = message_for(EscalationReason.INFORMATION_UNAVAILABLE)

    assert "بنتابع معك هون" in text
    assert "الرقم" not in text
    assert not any(ch.isdigit() for ch in text)


def test_an_emergency_reply_points_only_at_emergency_services(monkeypatch):
    _phone(monkeypatch, "011 222 3344")

    text = message_for(EscalationReason.URGENT_MEDICAL)

    assert "الطوارئ" in text
    assert "011 222 3344" not in text
