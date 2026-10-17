"""Typed GREEN API webhook payloads and the rules for what we act on.

GREEN API sends many webhook types over one endpoint. Everything that is not a
one-to-one incoming text message from a real customer is dropped here, with an
explicit reason, before any agent work is scheduled.

That includes messages that are simply too old. GREEN API holds notifications
it could not deliver and replays the whole backlog as soon as the webhook URL
works again, so a tunnel that was down for an hour ends with a queue arriving
in one burst. Each one would otherwise be answered as though it had just been
asked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.logging_utils import mask_chat_id

INCOMING_MESSAGE = "incomingMessageReceived"
#: The provider's plan-limit notification. Documented webhook type.
QUOTA_EXCEEDED_WEBHOOK = "quotaExceeded"

#: Message shapes that carry plain text we can answer.
_TEXT_TYPES = {"textMessage", "extendedTextMessage", "quotedMessage"}


class _Base(BaseModel):
    # GREEN API adds fields over time; unknown ones must never break parsing.
    model_config = ConfigDict(extra="ignore")


class InstanceData(_Base):
    idInstance: int | str | None = None
    wid: str | None = None
    typeInstance: str | None = None


class SenderData(_Base):
    chatId: str | None = None
    chatName: str | None = None
    sender: str | None = None
    senderName: str | None = None
    senderContactName: str | None = None


class TextMessageData(_Base):
    textMessage: str | None = None


class ExtendedTextMessageData(_Base):
    text: str | None = None
    description: str | None = None
    title: str | None = None


class QuotedMessage(_Base):
    stanzaId: str | None = None
    participant: str | None = None
    typeMessage: str | None = None
    textMessage: str | None = None


class MessageData(_Base):
    typeMessage: str | None = None
    textMessageData: TextMessageData | None = None
    extendedTextMessageData: ExtendedTextMessageData | None = None
    quotedMessage: QuotedMessage | None = None


class QuotaData(_Base):
    """Counts carried by a ``quotaExceeded`` notification.

    Integers describing the account's plan usage — no chat id, no message,
    nothing about a person. Safe to log, and the only way to see how close
    the account is to its limit without guessing.
    """

    usedChats: int | None = None
    totalChats: int | None = None
    monthlyChatsLimit: int | None = None
    usedMonthlyChats: int | None = None


class GreenApiWebhook(_Base):
    typeWebhook: str | None = None
    instanceData: InstanceData | None = None
    timestamp: int | None = None
    idMessage: str | None = None
    senderData: SenderData | None = None
    messageData: MessageData | None = None
    #: Present on outgoing-status webhooks.
    statusMessage: str | None = None
    #: Present on ``quotaExceeded`` notifications.
    quotaData: QuotaData | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SkipReason(StrEnum):
    NOT_INCOMING_MESSAGE = "not_incoming_message"
    OUTGOING_MESSAGE = "outgoing_message"
    STATUS_EVENT = "status_event"
    GROUP_CHAT = "group_chat"
    UNSUPPORTED_MESSAGE_TYPE = "unsupported_message_type"
    EMPTY_TEXT = "empty_text"
    MISSING_CHAT_ID = "missing_chat_id"
    MISSING_MESSAGE_ID = "missing_message_id"
    SELF_MESSAGE = "self_message"
    DUPLICATE = "duplicate"
    STALE_MESSAGE = "stale_message"
    #: The provider told us the account's plan limit was hit. Not a customer
    #: message at all, and the reason no customer message will follow.
    QUOTA_EXCEEDED = "quota_exceeded"
    #: A chat outside GREEN_API_ALLOWED_CHATS while that allowlist is in use.
    NOT_ALLOWLISTED = "not_allowlisted"
    #: Delivered by a GREEN API instance that is not the one we are configured
    #: for. One WhatsApp number can stay linked to several instances at once,
    #: so a retired instance pointed at this same URL would otherwise get its
    #: deliveries answered too — every message handled twice.
    FOREIGN_INSTANCE = "foreign_instance"


@dataclass(frozen=True)
class IncomingMessage:
    message_id: str
    chat_id: str
    sender: str
    sender_name: str
    text: str
    timestamp: int | None


@dataclass(frozen=True)
class ParseResult:
    message: IncomingMessage | None
    skip_reason: SkipReason | None
    #: The provider's own ``typeWebhook``, kept so a rejection can say what
    #: was actually rejected. Without it every unexpected notification looks
    #: identical in the log, which is precisely what made one customer's
    #: silence impossible to explain.
    webhook_type: str = ""
    #: Masked chat id when the payload carried one. Masked, never raw.
    chat: str = ""
    #: Plan-usage counts from a ``quotaExceeded`` notification, as "used/total".
    quota: str = ""

    @property
    def accepted(self) -> bool:
        return self.message is not None


def _extract_text(data: MessageData) -> str:
    if data.textMessageData and data.textMessageData.textMessage:
        return data.textMessageData.textMessage.strip()
    if data.extendedTextMessageData:
        ext = data.extendedTextMessageData
        for candidate in (ext.text, ext.description, ext.title):
            if candidate and candidate.strip():
                return candidate.strip()
    if data.quotedMessage and data.quotedMessage.textMessage:
        return data.quotedMessage.textMessage.strip()
    return ""


def _quota_summary(quota: QuotaData | None) -> str:
    """ "used/total" for the log line, or "" when the provider sent neither."""
    if quota is None:
        return ""
    used = quota.usedChats if quota.usedChats is not None else quota.usedMonthlyChats
    total = (
        quota.totalChats if quota.totalChats is not None else quota.monthlyChatsLimit
    )
    if used is None and total is None:
        return ""
    return f"{used if used is not None else '?'}/{total if total is not None else '?'}"


def _is_stale(timestamp: int | None, max_age_seconds: int, now: float | None) -> bool:
    """Whether this message is too old to answer.

    A missing or unparsable timestamp counts as fresh: dropping a real
    customer because the provider omitted a field would be the worse error. A
    timestamp in the future is fresh too — that is clock skew, not a backlog.
    """
    if max_age_seconds <= 0 or timestamp is None:
        return False
    try:
        sent_at = float(timestamp)
    except (TypeError, ValueError):
        return False
    reference = time.time() if now is None else now
    return (reference - sent_at) > max_age_seconds


def parse_webhook(
    payload: dict[str, Any] | GreenApiWebhook,
    *,
    ignore_groups: bool = True,
    max_age_seconds: int = 0,
    now: float | None = None,
    allowed_chats: tuple[str, ...] = (),
    expected_instance: str = "",
) -> ParseResult:
    """Validate a webhook body and decide whether the agent should see it."""
    hook = (
        payload
        if isinstance(payload, GreenApiWebhook)
        else GreenApiWebhook.model_validate(payload)
    )

    kind = (hook.typeWebhook or "").strip()
    sender_chat = (hook.senderData.chatId or "").strip() if hook.senderData else ""
    masked = mask_chat_id(sender_chat) if sender_chat else ""

    def skip(reason: SkipReason) -> ParseResult:
        return ParseResult(None, reason, webhook_type=kind, chat=masked)

    # Checked before anything else: a payload from another instance is not
    # ours to act on, whatever it contains. Absence of the field is tolerated
    # — rejecting on a field the provider merely omitted would be worse than
    # the duplicate this guards against.
    if expected_instance:
        delivered_by = (
            str(hook.instanceData.idInstance or "").strip() if hook.instanceData else ""
        )
        if delivered_by and delivered_by != str(expected_instance).strip():
            return ParseResult(None, SkipReason.FOREIGN_INSTANCE, webhook_type=kind)

    if kind != INCOMING_MESSAGE:
        if kind in {"outgoingMessageReceived", "outgoingAPIMessageReceived"}:
            return skip(SkipReason.OUTGOING_MESSAGE)
        if kind == "outgoingMessageStatus" or hook.statusMessage:
            return skip(SkipReason.STATUS_EVENT)
        if kind == QUOTA_EXCEEDED_WEBHOOK:
            # Called out separately because it explains an absence: while the
            # plan limit is hit, the provider sends THIS instead of the
            # customer's message, so the message never arrives at all.
            return ParseResult(
                None,
                SkipReason.QUOTA_EXCEEDED,
                webhook_type=kind,
                chat=masked,
                quota=_quota_summary(hook.quotaData),
            )
        return skip(SkipReason.NOT_INCOMING_MESSAGE)

    if not hook.idMessage:
        return skip(SkipReason.MISSING_MESSAGE_ID)

    sender_data = hook.senderData
    chat_id = sender_chat
    if not chat_id:
        return skip(SkipReason.MISSING_CHAT_ID)

    if ignore_groups and chat_id.endswith("@g.us"):
        return skip(SkipReason.GROUP_CHAT)

    # The bot's own number: GREEN API reports it as instanceData.wid. An
    # incoming webhook whose sender is our own wid is an echo of something we
    # sent, and answering it would start a loop. The chatId check catches the
    # bot's own "message yourself" chat, where sender and chat are both us.
    own_wid = (hook.instanceData.wid or "").strip() if hook.instanceData else ""
    sender = (sender_data.sender or "").strip() if sender_data else ""
    if own_wid and (sender == own_wid or chat_id == own_wid):
        return skip(SkipReason.SELF_MESSAGE)

    message_data = hook.messageData
    if message_data is None:
        return skip(SkipReason.UNSUPPORTED_MESSAGE_TYPE)

    msg_type = (message_data.typeMessage or "").strip()
    if msg_type and msg_type not in _TEXT_TYPES:
        return skip(SkipReason.UNSUPPORTED_MESSAGE_TYPE)

    text = _extract_text(message_data)
    if not text:
        return skip(SkipReason.EMPTY_TEXT)

    if _is_stale(hook.timestamp, max_age_seconds, now):
        return skip(SkipReason.STALE_MESSAGE)

    if allowed_chats:
        digits = "".join(ch for ch in chat_id if ch.isdigit())
        if digits not in allowed_chats:
            return skip(SkipReason.NOT_ALLOWLISTED)

    return ParseResult(
        IncomingMessage(
            message_id=hook.idMessage,
            chat_id=chat_id,
            sender=sender or chat_id,
            sender_name=(sender_data.senderName or "").strip() if sender_data else "",
            text=text,
            timestamp=hook.timestamp,
        ),
        None,
        webhook_type=kind,
        chat=masked,
    )
