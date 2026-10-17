"""The trusted conversation identity a patient tool runs under.

A patient tool must never take ``chatId``, ``patientId``, ``tenantId`` or a
verification token as an argument: the model would then be able to name a
conversation it is not in, and a customer could dictate one over WhatsApp.
Instead the serving surface *binds* the identity it authenticated around the
agent turn, and the tool reads it back from here.

Two sources, in order of trust:

1. :func:`bind_channel_identity`, set by the WhatsApp service for the chat the
   webhook was actually authenticated for. This is the production path.
2. The ADK invocation's own ``user_id``/``session.id``, used by the playground
   and the local smoke scripts. Still server-side — the model cannot set it.

A ``ContextVar`` is the right carrier because ADK runs each turn inside the
caller's asyncio context: a task created for a tool call copies the context at
creation time, so the binding follows the turn without being global state that
two concurrent chats could trample.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

#: Surface name recorded on the conversation scope, for logging only.
CHANNEL_WHATSAPP = "whatsapp"
CHANNEL_ADK = "adk"


@dataclass(frozen=True)
class ChannelIdentity:
    """A conversation the serving surface has already authenticated."""

    channel: str
    chat_id: str


_CURRENT: contextvars.ContextVar[ChannelIdentity | None] = contextvars.ContextVar(
    "khatib_channel_identity", default=None
)


def bind_channel_identity(identity: ChannelIdentity | None):
    """Bind the identity for the current context. Returns a reset token."""
    return _CURRENT.set(identity)


def reset_channel_identity(token: Any) -> None:
    _CURRENT.reset(token)


def current_channel_identity() -> ChannelIdentity | None:
    return _CURRENT.get()


@dataclass(frozen=True)
class ConversationScope:
    """Where one conversation's verification state lives.

    ``session_id`` is part of the key, not decoration: a new ADK session is a
    new conversation even on the same WhatsApp number, and a verification must
    not survive into it implicitly.
    """

    tenant_key: str
    channel: str
    chat_id: str
    session_id: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.tenant_key, self.channel, self.chat_id, self.session_id)

    @property
    def chat_key(self) -> tuple[str, str, str]:
        """Scope for per-chat rate limiting, which outlives a single session."""
        return (self.tenant_key, self.channel, self.chat_id)


def resolve_scope(tool_context: Any, tenant_key: str) -> ConversationScope | None:
    """Work out which conversation a tool call belongs to.

    Returns ``None`` when no trusted identity can be established, which the
    tools treat as "cannot verify here" rather than falling back to anything
    the model said.
    """
    identity = current_channel_identity()
    session_id = _session_id(tool_context)

    if identity is not None and identity.chat_id:
        return ConversationScope(
            tenant_key=tenant_key,
            channel=identity.channel,
            chat_id=identity.chat_id,
            session_id=session_id or identity.chat_id,
        )

    user_id = _attr(tool_context, "user_id")
    if user_id:
        return ConversationScope(
            tenant_key=tenant_key,
            channel=CHANNEL_ADK,
            chat_id=str(user_id),
            session_id=session_id or str(user_id),
        )
    return None


def _session_id(tool_context: Any) -> str:
    session = _attr(tool_context, "session")
    session_id = _attr(session, "id") if session is not None else None
    return str(session_id) if session_id else ""


def _attr(obj: Any, name: str) -> Any:
    """Read an attribute that may be absent or may raise on access.

    ADK's context properties dereference an invocation context that a test
    double or a partially built runner may not have.
    """
    if obj is None:
        return None
    try:
        return getattr(obj, name, None)
    except Exception:
        return None
