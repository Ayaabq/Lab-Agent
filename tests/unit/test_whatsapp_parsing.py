"""Webhook payload parsing: what we act on and what we drop."""

from __future__ import annotations

from typing import Any

import pytest

from app.whatsapp.models import SkipReason, parse_webhook

#: The instance id the shared `settings` fixture is configured with.
UNDER_TEST_INSTANCE = "000011112222"


def incoming(
    text: str = "مرحبا",
    *,
    chat_id: str = "963991234567@c.us",
    id_message: str = "MSG-1",
    type_message: str = "textMessage",
    extended: bool = False,
    wid: str | None = None,
    sender: str | None = None,
) -> dict[str, Any]:
    if extended:
        message_data: dict[str, Any] = {
            "typeMessage": "extendedTextMessage",
            "extendedTextMessageData": {"text": text, "description": "", "title": ""},
        }
    else:
        message_data = {
            "typeMessage": type_message,
            "textMessageData": {"textMessage": text},
        }
    return {
        "typeWebhook": "incomingMessageReceived",
        # Must match the `settings` fixture in conftest: the service passes its
        # configured instance id to parse_webhook, so a mismatch here would
        # make every payload look like it came from a retired instance.
        "instanceData": {
            "idInstance": UNDER_TEST_INSTANCE,
            "wid": wid or "963900000000@c.us",
        },
        "timestamp": 1730000000,
        "idMessage": id_message,
        "senderData": {
            "chatId": chat_id,
            "sender": sender or chat_id,
            "senderName": "زبون",
        },
        "messageData": message_data,
    }


def test_plain_text_message_is_accepted():
    result = parse_webhook(incoming("وين موقع المخبر؟"))

    assert result.accepted
    message = result.message
    assert message.message_id == "MSG-1"
    assert message.chat_id == "963991234567@c.us"
    assert message.text == "وين موقع المخبر؟"
    assert message.sender_name == "زبون"
    assert message.timestamp == 1730000000


def test_extended_text_message_is_accepted():
    result = parse_webhook(incoming("شو أوقات الدوام؟", extended=True))

    assert result.accepted
    assert result.message.text == "شو أوقات الدوام؟"


def test_extended_message_falls_back_to_description():
    payload = incoming(extended=True)
    payload["messageData"]["extendedTextMessageData"] = {
        "text": "",
        "description": "نص بديل",
        "title": "",
    }

    result = parse_webhook(payload)

    assert result.accepted
    assert result.message.text == "نص بديل"


def test_outgoing_message_is_ignored():
    payload = incoming()
    payload["typeWebhook"] = "outgoingMessageReceived"

    result = parse_webhook(payload)

    assert not result.accepted
    assert result.skip_reason is SkipReason.OUTGOING_MESSAGE


def test_outgoing_api_message_is_ignored():
    payload = incoming()
    payload["typeWebhook"] = "outgoingAPIMessageReceived"

    assert parse_webhook(payload).skip_reason is SkipReason.OUTGOING_MESSAGE


def test_delivery_status_event_is_ignored():
    payload = {
        "typeWebhook": "outgoingMessageStatus",
        "idMessage": "MSG-1",
        "status": "delivered",
    }

    assert parse_webhook(payload).skip_reason is SkipReason.STATUS_EVENT


def test_unsupported_webhook_type_is_ignored():
    assert (
        parse_webhook({"typeWebhook": "stateInstanceChanged"}).skip_reason
        is SkipReason.NOT_INCOMING_MESSAGE
    )


def test_unknown_webhook_type_is_ignored():
    assert (
        parse_webhook(
            {"typeWebhook": "somethingBrandNew", "idMessage": "X"}
        ).skip_reason
        is SkipReason.NOT_INCOMING_MESSAGE
    )


def test_group_chat_is_ignored_by_default():
    result = parse_webhook(incoming(chat_id="120363000000000000@g.us"))

    assert result.skip_reason is SkipReason.GROUP_CHAT


def test_group_chat_can_be_allowed_explicitly():
    result = parse_webhook(
        incoming(chat_id="120363000000000000@g.us"), ignore_groups=False
    )

    assert result.accepted


def test_message_from_the_bot_itself_is_ignored():
    own = "963900000000@c.us"
    result = parse_webhook(incoming(wid=own, sender=own))

    assert result.skip_reason is SkipReason.SELF_MESSAGE


def test_empty_text_is_ignored():
    assert parse_webhook(incoming("   ")).skip_reason is SkipReason.EMPTY_TEXT


def test_non_text_message_type_is_ignored():
    payload = incoming(type_message="imageMessage")
    payload["messageData"] = {
        "typeMessage": "imageMessage",
        "fileMessageData": {"downloadUrl": "https://example.test/x.jpg"},
    }

    assert parse_webhook(payload).skip_reason is SkipReason.UNSUPPORTED_MESSAGE_TYPE


def test_missing_chat_id_is_ignored():
    payload = incoming()
    payload["senderData"]["chatId"] = ""

    assert parse_webhook(payload).skip_reason is SkipReason.MISSING_CHAT_ID


def test_missing_message_id_is_ignored():
    payload = incoming()
    payload.pop("idMessage")

    assert parse_webhook(payload).skip_reason is SkipReason.MISSING_MESSAGE_ID


def test_unknown_extra_fields_do_not_break_parsing():
    payload = incoming()
    payload["brandNewFieldFromProvider"] = {"nested": [1, 2, 3]}
    payload["senderData"]["anotherNewField"] = "x"

    assert parse_webhook(payload).accepted


# --- stale backlog --------------------------------------------------------
#
# GREEN API holds notifications it could not deliver and replays the whole
# backlog once the webhook URL works again. Answering an hour-old question as
# though it were new is the failure these guard against.

NOW = 1_760_000_000.0


def test_a_fresh_message_is_accepted_with_the_age_check_on():
    payload = incoming("مرحبا")
    payload["timestamp"] = int(NOW - 5)

    result = parse_webhook(payload, max_age_seconds=600, now=NOW)

    assert result.accepted


def test_a_message_older_than_the_window_is_dropped():
    payload = incoming("سؤال قديم")
    payload["timestamp"] = int(NOW - 3600)

    result = parse_webhook(payload, max_age_seconds=600, now=NOW)

    assert not result.accepted
    assert result.skip_reason is SkipReason.STALE_MESSAGE


def test_a_message_right_on_the_boundary_is_still_answered():
    payload = incoming("مرحبا")
    payload["timestamp"] = int(NOW - 600)

    assert parse_webhook(payload, max_age_seconds=600, now=NOW).accepted


def test_the_age_check_is_off_by_default():
    payload = incoming("مرحبا")
    payload["timestamp"] = int(NOW - 86400)

    # Callers opt in; parse_webhook on its own does not second-guess age.
    assert parse_webhook(payload).accepted


def test_zero_disables_the_age_check():
    payload = incoming("مرحبا")
    payload["timestamp"] = int(NOW - 86400)

    assert parse_webhook(payload, max_age_seconds=0, now=NOW).accepted


def test_a_missing_timestamp_counts_as_fresh():
    payload = incoming("مرحبا")
    payload.pop("timestamp", None)

    # Dropping a real customer because the provider omitted a field would be
    # the worse error.
    assert parse_webhook(payload, max_age_seconds=600, now=NOW).accepted


def test_a_future_timestamp_is_clock_skew_not_a_backlog():
    payload = incoming("مرحبا")
    payload["timestamp"] = int(NOW + 300)

    assert parse_webhook(payload, max_age_seconds=600, now=NOW).accepted


def test_an_unparseable_timestamp_counts_as_fresh():
    payload = incoming("مرحبا")
    payload["timestamp"] = None

    assert parse_webhook(payload, max_age_seconds=600, now=NOW).accepted


# --- diagnosability of rejections ----------------------------------------
#
# A bare reason could not distinguish a routine status ping from a customer
# being silently refused. These pin the context a rejection must carry.


def test_a_rejection_reports_the_provider_webhook_type():
    payload = incoming("مرحبا")
    payload["typeWebhook"] = "someNewNotification"

    result = parse_webhook(payload)

    assert result.skip_reason is SkipReason.NOT_INCOMING_MESSAGE
    assert result.webhook_type == "someNewNotification"
    assert result.chat.endswith("@c.us")
    # Masked, never the raw number.
    assert "963991234567" not in result.chat


def test_quota_exceeded_is_its_own_reason_not_a_generic_unknown():
    payload = incoming("مرحبا")
    payload["typeWebhook"] = "quotaExceeded"

    result = parse_webhook(payload)

    # It explains an absence: while it holds, no customer message arrives.
    assert result.skip_reason is SkipReason.QUOTA_EXCEEDED
    assert result.webhook_type == "quotaExceeded"


def test_an_accepted_message_also_carries_its_type():
    result = parse_webhook(incoming("مرحبا"))

    assert result.accepted
    assert result.webhook_type == "incomingMessageReceived"


# --- optional development allowlist --------------------------------------


def test_the_allowlist_is_inactive_when_empty():
    # Default behaviour is unchanged: every private chat is served.
    assert parse_webhook(incoming("مرحبا"), allowed_chats=()).accepted


def test_an_allowlisted_chat_is_accepted():
    result = parse_webhook(incoming("مرحبا"), allowed_chats=("963991234567",))

    assert result.accepted


def test_a_chat_outside_the_allowlist_is_dropped():
    result = parse_webhook(incoming("مرحبا"), allowed_chats=("963990000000",))

    assert not result.accepted
    assert result.skip_reason is SkipReason.NOT_ALLOWLISTED


@pytest.mark.parametrize(
    "spelling",
    ["963991234567", "+963 99 123 4567", "963991234567@c.us", "+963-99-123-4567"],
)
def test_any_spelling_of_the_same_number_matches(spelling):
    digits = "".join(c for c in spelling if c.isdigit())

    assert parse_webhook(incoming("مرحبا"), allowed_chats=(digits,)).accepted


def test_the_allowlist_runs_after_the_other_filters():
    # A group chat is still a group chat, allowlisted or not.
    payload = incoming("مرحبا", chat_id="12345@g.us")

    result = parse_webhook(payload, allowed_chats=("12345",))

    assert result.skip_reason is SkipReason.GROUP_CHAT


def test_quota_exceeded_reports_the_plan_usage_counts():
    result = parse_webhook(
        {"typeWebhook": "quotaExceeded", "quotaData": {"usedChats": 4, "totalChats": 3}}
    )

    assert result.skip_reason is SkipReason.QUOTA_EXCEEDED
    # Integers only — the notification carries no chat id and no message.
    assert result.quota == "4/3"


def test_quota_exceeded_without_counts_still_classifies():
    result = parse_webhook({"typeWebhook": "quotaExceeded"})

    assert result.skip_reason is SkipReason.QUOTA_EXCEEDED
    assert result.quota == ""


def test_quota_handling_does_not_loosen_the_other_filters():
    # The parser must not have become more permissive while gaining detail.
    own = "963900000000@c.us"
    assert parse_webhook(incoming("hi", wid=own, sender=own)).skip_reason is (
        SkipReason.SELF_MESSAGE
    )
    assert parse_webhook(incoming("hi", chat_id="123@g.us")).skip_reason is (
        SkipReason.GROUP_CHAT
    )
    out = incoming("hi")
    out["typeWebhook"] = "outgoingMessageReceived"
    assert parse_webhook(out).skip_reason is SkipReason.OUTGOING_MESSAGE
    assert parse_webhook(incoming("مرحبا")).accepted


# --- only our own instance may drive the agent ---------------------------
#
# One WhatsApp number can stay linked to several GREEN API instances at once.
# A retired instance still pointed at this URL would otherwise have its
# deliveries answered too, and every customer message handled twice.

OURS = UNDER_TEST_INSTANCE
RETIRED = "710722743498"  # a different, retired instance


def _from_instance(id_instance, text: str = "مرحبا"):
    payload = incoming(text)
    payload["instanceData"] = {"idInstance": id_instance, "wid": "963900000000@c.us"}
    return payload


def test_our_own_instance_is_accepted():
    assert parse_webhook(_from_instance(OURS), expected_instance=OURS).accepted


def test_a_retired_instance_is_rejected():
    result = parse_webhook(_from_instance(RETIRED), expected_instance=OURS)

    assert not result.accepted
    assert result.skip_reason is SkipReason.FOREIGN_INSTANCE


def test_the_id_matches_whether_sent_as_text_or_number():
    # GREEN API serialises idInstance as a JSON number; a real id has no
    # leading zeros, so int and str spellings are the same identity.
    real = "1101234567"
    assert parse_webhook(_from_instance(real), expected_instance=real).accepted
    assert parse_webhook(_from_instance(int(real)), expected_instance=real).accepted


def test_the_guard_is_off_when_no_instance_is_configured():
    # Unchanged behaviour for anyone who has not set it.
    assert parse_webhook(_from_instance(RETIRED)).accepted


def test_a_payload_without_instance_data_is_not_rejected():
    # Rejecting on a field the provider merely omitted would be worse than
    # the duplicate this guards against.
    payload = incoming("مرحبا")
    payload["instanceData"] = None

    assert parse_webhook(payload, expected_instance=OURS).accepted


def test_the_guard_does_not_bypass_the_other_filters():
    payload = _from_instance(OURS)
    payload["typeWebhook"] = "outgoingMessageReceived"

    assert parse_webhook(payload, expected_instance=OURS).skip_reason is (
        SkipReason.OUTGOING_MESSAGE
    )
