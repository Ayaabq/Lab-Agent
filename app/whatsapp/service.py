"""Turns a GREEN API webhook into an agent reply.

Flow: validate -> drop non-customer events -> de-duplicate -> acknowledge the
webhook immediately -> process in the background, one message at a time per
chat, using a per-``chatId`` ADK session.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from google.genai import types

from app.config import Settings, get_settings
from app.escalation import (
    EscalationReason,
    EscalationStore,
    classify_message,
    get_escalation_store,
    message_for,
)
from app.logging_utils import get_logger, mask_chat_id, summarize_text
from app.patients.context import (
    CHANNEL_WHATSAPP,
    ChannelIdentity,
    ConversationScope,
    bind_channel_identity,
    reset_channel_identity,
)
from app.patients.release import PendingRelease, get_release_store
from app.patients.store import get_verification_store
from app.whatsapp.dedupe import TTLDedupeCache
from app.whatsapp.green_api import GreenApiClient, get_green_api_client
from app.whatsapp.models import IncomingMessage, ParseResult, SkipReason, parse_webhook

logger = get_logger(__name__)

#: Sent when the agent produces nothing usable. Never invents lab facts.
FALLBACK_REPLY = (
    "عذرًا، ما قدرت أعالج طلبك هلق. يرجى إعادة المحاولة بعد شوي "
    "أو التواصل مع موظف مخبر الخطيب."
)

#: WhatsApp's caption limit. A ``patientMessage`` longer than this is sent as
#: its own message rather than being cut down: the laboratory's wording about
#: someone's result is relayed whole or not at all.
MAX_CAPTION_CHARS = 1024

#: Said when the report could not be sent. Neither promises delivery nor
#: claims a staff member was told — there is still no notification channel.
REPORT_SEND_FAILED_REPLY = (
    "ما قدرت أبعتلك نسخة التقرير هلق بسبب خلل تقني. "
    "يرجى التواصل مع موظف مخبر الخطيب للحصول على نسخة."
)
#: Used when the provider timed out and may or may not have queued the file.
#: Deliberately says neither "sent" nor "failed", because we do not know.
REPORT_SEND_UNCERTAIN_REPLY = (
    "ما قدرت أتأكد إذا وصلتك نسخة التقرير. إذا ما وصلتك خلال شوي، "
    "يرجى التواصل مع موظف مخبر الخطيب."
)
#: The verification lapsed between fetching the result and sending the report.
REPORT_NEEDS_REVERIFY_REPLY = (
    "انتهت صلاحية التحقق من هويتك قبل ما أبعت التقرير، وما بقدر أبعتو هيك. "
    "إذا بتحب، منعيد التحقق من جديد."
)

#: Upper bound on one agent turn, so a stuck model cannot pin a chat lock.
AGENT_TURN_TIMEOUT_SECONDS = 90

#: Chat locks are cheap but unbounded growth is not. Oldest idle locks are
#: dropped once we exceed this.
_MAX_TRACKED_CHATS = 2000


@dataclass(frozen=True)
class WebhookAck:
    """What the endpoint returns. Always a 200 so GREEN API stops retrying."""

    status: str
    reason: str = ""
    message_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.reason:
            payload["reason"] = self.reason
        if self.message_id:
            payload["idMessage"] = self.message_id
        return payload


class WhatsAppService:
    def __init__(
        self,
        settings: Settings | None = None,
        green_client: GreenApiClient | None = None,
        escalation_store: EscalationStore | None = None,
        dedupe: TTLDedupeCache | None = None,
        runner_provider: Any = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._green = green_client
        self._escalation = escalation_store or get_escalation_store()
        cfg = self._settings.green_api
        self._dedupe = dedupe or TTLDedupeCache(
            ttl_seconds=cfg.dedupe_ttl_seconds, max_entries=cfg.dedupe_max_entries
        )
        self._runner_provider = runner_provider
        # Ids of messages we sent. If the provider ever echoes one back as an
        # incoming notification, answering it would start a reply loop.
        self._outbound_ids = TTLDedupeCache(
            ttl_seconds=cfg.dedupe_ttl_seconds, max_entries=cfg.dedupe_max_entries
        )
        # Reports already sent, keyed by chat + result. The inbound dedupe
        # stops a redelivered webhook, but a customer asking twice is not a
        # duplicate webhook, and nobody needs the same report file twice.
        self._sent_reports = TTLDedupeCache(
            ttl_seconds=cfg.dedupe_ttl_seconds, max_entries=cfg.dedupe_max_entries
        )
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._locks_guard = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    # -- collaborators resolved lazily so tests can inject ----------------

    @property
    def green(self) -> GreenApiClient:
        if self._green is None:
            self._green = get_green_api_client()
        return self._green

    def _runner(self) -> Any:
        if self._runner_provider is not None:
            return self._runner_provider()
        from app.app_utils.runner import get_runner

        return get_runner()

    # -- entry point -------------------------------------------------------

    async def handle_webhook(
        self, payload: dict[str, Any], *, wait: bool = False
    ) -> WebhookAck:
        """Validate and queue a webhook. Returns fast; the agent runs after.

        ``wait=True`` processes inline instead of in the background — used by
        tests and the local webhook smoke check.
        """
        try:
            parsed: ParseResult = parse_webhook(
                payload,
                ignore_groups=self._settings.green_api.ignore_groups,
                max_age_seconds=self._settings.green_api.max_message_age_seconds,
                allowed_chats=self._settings.green_api.allowed_chats,
                expected_instance=self._settings.green_api.id_instance.get(),
            )
        except Exception:
            logger.warning("rejected malformed webhook payload")
            return WebhookAck("ignored", SkipReason.NOT_INCOMING_MESSAGE.value)

        if not parsed.accepted:
            reason = parsed.skip_reason.value if parsed.skip_reason else "unknown"
            # The provider's own type and the masked chat go in the line. A
            # bare reason cannot distinguish "a status ping" from "this
            # customer is being refused", which is the difference between
            # noise and someone getting no reply.
            logger.info(
                "webhook ignored",
                extra={
                    "context": {
                        "reason": reason,
                        "typeWebhook": parsed.webhook_type or "<absent>",
                        "chat": parsed.chat or "<none>",
                    }
                },
            )
            if parsed.skip_reason is SkipReason.QUOTA_EXCEEDED:
                logger.error(
                    "provider reported the account plan limit was exceeded; "
                    "the customer's message was replaced by this notification "
                    "and will not arrive",
                    extra={
                        "context": {
                            "chat": parsed.chat or "<none>",
                            "chats_used": parsed.quota or "<not reported>",
                        }
                    },
                )
            return WebhookAck("ignored", reason)

        message = parsed.message
        assert message is not None  # narrowed by parsed.accepted

        if self._outbound_ids.contains(message.message_id):
            logger.warning(
                "ignored echo of a message this bot sent",
                extra={
                    "context": {
                        "chat": mask_chat_id(message.chat_id),
                        "idMessage": message.message_id,
                    }
                },
            )
            return WebhookAck(
                "ignored", SkipReason.SELF_MESSAGE.value, message.message_id
            )

        if self._dedupe.seen(message.message_id):
            logger.info(
                "duplicate webhook ignored",
                extra={
                    "context": {
                        "chat": mask_chat_id(message.chat_id),
                        "idMessage": message.message_id,
                    }
                },
            )
            return WebhookAck("ignored", SkipReason.DUPLICATE.value, message.message_id)

        logger.info(
            "accepted incoming WhatsApp message",
            extra={
                "context": {
                    "chat": mask_chat_id(message.chat_id),
                    "idMessage": message.message_id,
                    "text": summarize_text(message.text),
                }
            },
        )

        if wait:
            await self._process(message)
        else:
            task = asyncio.create_task(self._process(message))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        return WebhookAck("accepted", message_id=message.message_id)

    # -- processing --------------------------------------------------------

    async def _chat_lock(self, chat_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(chat_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[chat_id] = lock
            self._locks.move_to_end(chat_id)
            while len(self._locks) > _MAX_TRACKED_CHATS:
                oldest_key, oldest_lock = next(iter(self._locks.items()))
                if oldest_lock.locked():
                    break
                self._locks.pop(oldest_key, None)
            return lock

    async def _process(self, message: IncomingMessage) -> None:
        """Handle one message end to end. Never raises."""
        lock = await self._chat_lock(message.chat_id)
        # Messages from one chat are serialised; different chats run concurrently.
        async with lock:
            try:
                reply = await self._build_reply(message)
            except Exception:
                logger.exception(
                    "agent turn failed",
                    extra={"context": {"chat": mask_chat_id(message.chat_id)}},
                )
                reply = FALLBACK_REPLY
            try:
                result = await self.green.send_text(message.chat_id, reply)
            except Exception:
                logger.exception(
                    "failed to deliver reply",
                    extra={"context": {"chat": mask_chat_id(message.chat_id)}},
                )
            else:
                if result.provider_message_id:
                    self._outbound_ids.remember(result.provider_message_id)

            # After the text, and still inside the per-chat lock so two
            # messages from one customer cannot race to send the same report.
            try:
                await self._deliver_report_image(message)
            except Exception:
                logger.exception(
                    "report image delivery failed",
                    extra={"context": {"chat": mask_chat_id(message.chat_id)}},
                )

    async def _deliver_report_image(self, message: IncomingMessage) -> None:
        """Send a report image the results tool staged for this conversation.

        Everything that decides *whether* to send lives here rather than in the
        tool, and every input comes from server-side state: the chat id is the
        one the webhook authenticated, the URL and patient id come from the
        staged release, and none of it can be named by the model or typed by
        the customer.
        """
        scope = ConversationScope(
            tenant_key=self._settings.khatib.tenant_key,
            channel=CHANNEL_WHATSAPP,
            chat_id=message.chat_id,
            session_id=message.chat_id,
        )
        release = get_release_store().take(scope)
        if release is None:
            return

        # The staged chat must be this chat. Equal by construction today; an
        # assertion in code is cheaper than discovering otherwise in someone's
        # chat history.
        if release.chat_id != message.chat_id:
            logger.error("staged report image did not match the delivering chat")
            return

        # Re-check the verification *now*, not when the backend answered. It
        # can expire in between, and the check that matters is the one at the
        # moment the report leaves.
        record = get_verification_store().get(scope)
        if (
            record is None
            or record.patient_id != release.patient_id
            or record.identity != release.identity
        ):
            logger.warning(
                "report image withheld: verification no longer valid at send time",
                extra={"context": {"chat": mask_chat_id(message.chat_id)}},
            )
            await self._send_safely(message.chat_id, REPORT_NEEDS_REVERIFY_REPLY)
            return

        # Read-only check here; the claim is recorded further down, and only
        # for outcomes where the provider might actually have the file. A
        # definite failure or a dry run must NOT claim it, or the customer
        # could never obtain a report that was never sent. Same-chat turns are
        # serialised by the per-chat lock and the key is chat-scoped, so a
        # check-then-record pair cannot race with itself.
        if self._sent_reports.contains(release.dedupe_key()):
            logger.info(
                "report image already sent for this result; not sending again",
                extra={"context": {"chat": mask_chat_id(message.chat_id)}},
            )
            return

        caption = (
            release.patient_message
            if len(release.patient_message) <= MAX_CAPTION_CHARS
            else ""
        )
        if not caption and release.patient_message:
            # Too long to ride along as a caption. It goes out whole, as its
            # own message — the laboratory's wording is not ours to shorten.
            await self._send_safely(message.chat_id, release.patient_message)

        try:
            result = await self.green.send_file_by_url(
                chat_id=message.chat_id,
                url_file=release.image_url,
                file_name=release.file_name,
                caption=caption,
            )
        except Exception:
            # Type only: a provider error body can quote the report URL.
            logger.exception(
                "report image send failed",
                extra={"context": {"chat": mask_chat_id(message.chat_id)}},
            )
            await self._report_fallback(message.chat_id, release, delivered=False)
            return

        if result.ambiguous:
            # The provider may or may not have queued it. Claimed, because a
            # second copy of someone's medical report is the worse outcome;
            # not retried; and described to the customer as neither sent nor
            # failed, because we genuinely do not know.
            self._sent_reports.remember(release.dedupe_key())
            await self._report_fallback(message.chat_id, release, delivered=None)
            return
        if result.dry_run:
            # Nothing left this process, so nothing is claimed: turning live
            # sending on later must still be able to deliver this report.
            logger.info(
                "report image suppressed by the live-sending switch",
                extra={"context": {"chat": mask_chat_id(message.chat_id)}},
            )
            return
        if not result.delivered:
            # A definite rejection. The file was not sent, so the claim is not
            # taken and asking again can still produce the report.
            await self._report_fallback(message.chat_id, release, delivered=False)
            return

        # Accepted by the provider. That is NOT proof the customer received
        # it, so nothing is said to them about delivery — the image arriving
        # is its own evidence.
        self._sent_reports.remember(release.dedupe_key())
        if result.provider_message_id:
            self._outbound_ids.remember(result.provider_message_id)

    async def _report_fallback(
        self,
        chat_id: str,
        release: PendingRelease,
        *,
        delivered: bool | None,
    ) -> None:
        """Tell the customer something true when the report did not go out."""
        if (
            release.patient_message
            and len(release.patient_message) <= MAX_CAPTION_CHARS
        ):
            # It would have been the caption, so it has not been sent yet.
            await self._send_safely(chat_id, release.patient_message)
        await self._send_safely(
            chat_id,
            REPORT_SEND_UNCERTAIN_REPLY
            if delivered is None
            else REPORT_SEND_FAILED_REPLY,
        )

    async def _send_safely(self, chat_id: str, text: str) -> None:
        try:
            result = await self.green.send_text(chat_id, text)
        except Exception:
            logger.exception(
                "failed to deliver a follow-up message",
                extra={"context": {"chat": mask_chat_id(chat_id)}},
            )
            return
        if result.provider_message_id:
            self._outbound_ids.remember(result.provider_message_id)

    async def _build_reply(self, message: IncomingMessage) -> str:
        """Deterministic safety triage first, then the agent."""
        reason = classify_message(message.text)
        if reason in {
            EscalationReason.URGENT_MEDICAL,
            EscalationReason.MEDICAL_ADVICE_REQUESTED,
        }:
            # These never reach the model: the safe answer is fixed, and it must
            # not depend on the model choosing to refuse.
            self._escalation.mark(message.chat_id, reason)
            return message_for(reason)

        text = await self._run_agent(message)
        if not text:
            self._escalation.mark(
                message.chat_id, EscalationReason.INFORMATION_UNAVAILABLE
            )
            return FALLBACK_REPLY
        return text

    async def _run_agent(self, message: IncomingMessage) -> str:
        runner = self._runner()
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=message.text)]
        )
        chunks: list[str] = []
        # The patient tools read the conversation they belong to from here,
        # never from a tool argument. This binding is the only thing that says
        # "this turn speaks for this chat", and it is set from the webhook we
        # authenticated — not from anything in the message body.
        identity_token = bind_channel_identity(
            ChannelIdentity(channel=CHANNEL_WHATSAPP, chat_id=message.chat_id)
        )

        async def _consume() -> None:
            async for event in runner.run_async(
                # chatId identifies the *conversation*, not the person. It
                # scopes the ADK session and the verification record; it is
                # never treated as evidence of who is holding the phone.
                user_id=message.chat_id,
                session_id=message.chat_id,
                new_message=content,
            ):
                if not event.is_final_response():
                    continue
                parts = getattr(getattr(event, "content", None), "parts", None) or []
                # The parts of ONE event are fragments of one sentence and are
                # concatenated as-is. Stripping each and joining them with a
                # newline splits words down the middle — Arabic "أعتذر" arrived
                # as "أعت\nذر" — because a fragment boundary is not a word
                # boundary and carries no whitespace of its own.
                merged = "".join(
                    piece for part in parts if (piece := getattr(part, "text", None))
                ).strip()
                if merged:
                    chunks.append(merged)

        try:
            await asyncio.wait_for(_consume(), timeout=AGENT_TURN_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.error(
                "agent turn timed out",
                extra={"context": {"chat": mask_chat_id(message.chat_id)}},
            )
            return ""
        finally:
            # Unbind even on timeout, so a later turn in this task can never
            # inherit another chat's identity.
            reset_channel_identity(identity_token)
        # Separate final events are separate messages, so those do get a
        # newline between them.
        return "\n".join(chunks).strip()

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "tracked_chats": len(self._locks),
            "deduped_messages": len(self._dedupe),
            "in_flight": len(self._tasks),
            "awaiting_human": self._escalation.pending_count(),
            "reports_sent_recently": len(self._sent_reports),
            "live_sending_enabled": self.green.live_enabled,
        }

    async def drain(self) -> None:
        """Await background work. Used on shutdown and in tests."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


_service: WhatsAppService | None = None


def get_whatsapp_service() -> WhatsAppService:
    global _service
    if _service is None:
        _service = WhatsAppService()
    return _service


def set_whatsapp_service(service: WhatsAppService | None) -> None:
    """Swap the process-wide service. Tests use this to inject a fake."""
    global _service
    _service = service
