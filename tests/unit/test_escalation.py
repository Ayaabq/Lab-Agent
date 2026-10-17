"""Escalation triage and the in-memory awaiting-human state."""

from __future__ import annotations

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


def test_marking_records_the_reason():
    store = EscalationStore()

    record = store.mark("963991234567@c.us", EscalationReason.USER_REQUESTED_HUMAN)

    assert record.reason is EscalationReason.USER_REQUESTED_HUMAN
    assert record.human_notified is False
    assert store.is_awaiting("963991234567@c.us")


def test_unknown_chat_is_not_awaiting():
    assert EscalationStore().is_awaiting("963990000000@c.us") is False


def test_remarking_replaces_the_previous_reason():
    store = EscalationStore()
    store.mark("chat@c.us", EscalationReason.INFORMATION_UNAVAILABLE)

    store.mark("chat@c.us", EscalationReason.URGENT_MEDICAL)

    assert store.get("chat@c.us").reason is EscalationReason.URGENT_MEDICAL
    assert store.pending_count() == 1


def test_clear_removes_the_record():
    store = EscalationStore()
    store.mark("chat@c.us", EscalationReason.TOOL_FAILURE)

    store.clear("chat@c.us")

    assert store.is_awaiting("chat@c.us") is False


def test_store_is_bounded():
    store = EscalationStore(max_entries=3)

    for index in range(10):
        store.mark(f"chat-{index}@c.us", EscalationReason.TOOL_FAILURE)

    assert store.pending_count() == 3
    assert store.is_awaiting("chat-0@c.us") is False
    assert store.is_awaiting("chat-9@c.us") is True


def test_snapshot_masks_chat_ids():
    store = EscalationStore()
    store.mark("963991234567@c.us", EscalationReason.USER_REQUESTED_HUMAN)

    snapshot = store.snapshot()

    assert "963991234567@c.us" not in snapshot
    assert any("*" in key for key in snapshot)
