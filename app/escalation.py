"""Human-escalation state.

Held in process memory, or in Upstash Redis when shared state is configured
(:mod:`app.state.backend`), so every instance sees the same record.

MVP scope: there is **no** real notification channel to lab staff. This module
only records that a conversation is waiting for a human and supplies the safe
Arabic text to send back. Nothing here should ever claim an employee was
actually notified.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.logging_utils import get_logger, mask_chat_id
from app.state.backend import StateBackend, get_state_backend

logger = get_logger(__name__)


class EscalationReason(StrEnum):
    USER_REQUESTED_HUMAN = "user_requested_human"
    INFORMATION_UNAVAILABLE = "information_unavailable"
    MEDICAL_ADVICE_REQUESTED = "medical_advice_requested"
    URGENT_MEDICAL = "urgent_medical"
    TOOL_FAILURE = "tool_failure"
    #: A FINAL result the backend marked CLINICIAN_REVIEW_REQUIRED. The
    #: record exists; nobody here is allowed to read it out.
    RESULT_CLINICIAN_REVIEW = "result_clinician_review"
    #: A pending result whose promised ready time has already passed.
    RESULT_DELAYED = "result_delayed"
    #: Identity could not be confirmed, or too many attempts were made.
    VERIFICATION_FAILED = "verification_failed"


# Wording for each reason. Deliberately short — these go out over WhatsApp.
_MESSAGES: dict[EscalationReason, str] = {
    EscalationReason.USER_REQUESTED_HUMAN: (
        "أكيد. يرجى التواصل مع موظف مخبر الخطيب مباشرةً ليساعدك بشكل أفضل. "
        "سجّلت طلبك ليتابعه أحد الموظفين عند توفره."
    ),
    EscalationReason.INFORMATION_UNAVAILABLE: (
        "للأسف ما عندي معلومة موثوقة عن هالموضوع، وما بحب أعطيك معلومة غير مؤكدة. "
        "يرجى التواصل مع موظف المخبر ليجاوبك بدقة."
    ),
    EscalationReason.MEDICAL_ADVICE_REQUESTED: (
        "أنا مساعد إداري ولست طبيبًا، وما بقدر أفسّر نتائج التحاليل أو أنصح بعلاج. "
        "يرجى مراجعة الطبيب المعالج، وبإمكانك التواصل مع موظف المخبر لأي استفسار إداري."
    ),
    EscalationReason.URGENT_MEDICAL: (
        "إذا كانت الأعراض شديدة أو طارئة، يرجى التواصل فورًا مع خدمات الطوارئ "
        "أو مراجعة أقرب مشفى أو طبيب مختص. أنا ما بقدر أقيّم الحالة الطبية."
    ),
    EscalationReason.TOOL_FAILURE: (
        "في خلل تقني مؤقت عندي ومو قادر أجيب المعلومة هلق. "
        "يرجى التواصل مع موظف المخبر أو إعادة المحاولة بعد شوي."
    ),
    # Says nothing about what is in the result: "needs review" is a workflow
    # state, not a finding, and must not be heard as one.
    EscalationReason.RESULT_CLINICIAN_REVIEW: (
        "نتيجتك بحاجة لمراجعة من طبيب المخبر قبل ما تتسلّمها، وأنا ما بقدر "
        "أشارك تفاصيلها. يرجى التواصل مع موظف المخبر لمتابعة الموضوع."
    ),
    EscalationReason.RESULT_DELAYED: (
        "نتيجتك لسا قيد المعالجة وأخدت وقت أطول من المتوقع. "
        "يرجى التواصل مع موظف المخبر ليتابعلك حالة النتيجة."
    ),
    EscalationReason.VERIFICATION_FAILED: (
        "ما قدرت أتحقق من هويتك من المعلومات يلي وصلتني، وما بقدر أشارك أي "
        "معلومة عن النتائج بدون تحقق. يرجى التواصل مع موظف المخبر مباشرةً."
    ),
}

# Deliberately conservative keyword matching: a false positive costs one
# unnecessary "contact an employee" line; a false negative can mean the agent
# answering a medical question it must not touch.
_HUMAN_REQUEST = re.compile(
    r"(موظف|موظفة|محاسب|بشر|إنسان|انسان|شخص حقيقي|حدا يحكي|بدي احكي مع حدا|"
    r"خدمة العملاء|خدمة الزبائن|مسؤول|مدير|"
    r"\bhuman\b|\bagent\b|\brepresentative\b|\boperator\b|\breal person\b)"
)
_MEDICAL_ADVICE = re.compile(
    r"(شو يعني|شو معنى|فسر|فسّر|تفسير|تحليلي طالع|نتيجتي طالعة|نتيجتي مرتفعة|"
    r"مرتفع كتير|منخفض كتير|طبيعي ولا لأ|طبيعية ولا لأ|عندي مرض|شو مرضي|"
    r"شو العلاج|أي دواء|اي دواء|وصفة|جرعة|بدي دوا|شخص حالتي|تشخيص|"
    r"\bdiagnos|\binterpret\b|\bwhat does.*mean\b|\bmedication\b|\btreatment\b|"
    r"\bdosage\b|\bprescribe)"
)
_URGENT = re.compile(
    r"(ألم بالصدر|الم بالصدر|وجع صدر|ما فيني تنفس|صعوبة تنفس|ضيق تنفس|نزيف|"
    r"إغماء|اغماء|فقدان وعي|تشنج|سكتة|جلطة|انتحار|تسمم|حرارة عالية كتير|"
    r"\bchest pain\b|\bcan'?t breathe\b|\bbleeding\b|\bunconscious\b|"
    r"\bseizure\b|\bstroke\b|\bpoison|\bsuicid)"
)


def classify_message(text: str | None) -> EscalationReason | None:
    """Deterministic pre-LLM triage of an inbound message.

    Urgency is checked first: a message can ask for an interpretation *and*
    describe an emergency, and the emergency response is the safer one.
    """
    if not text:
        return None
    normalized = text.strip().lower()
    if _URGENT.search(normalized):
        return EscalationReason.URGENT_MEDICAL
    if _MEDICAL_ADVICE.search(normalized):
        return EscalationReason.MEDICAL_ADVICE_REQUESTED
    if _HUMAN_REQUEST.search(normalized):
        return EscalationReason.USER_REQUESTED_HUMAN
    return None


def message_for(reason: EscalationReason) -> str:
    return _MESSAGES[reason]


@dataclass(frozen=True)
class EscalationRecord:
    chat_id: str
    reason: EscalationReason
    created_at: float
    #: Always False in the MVP — no notification channel exists yet.
    human_notified: bool = False


class EscalationStore:
    """Bounded, in-memory record of conversations awaiting a human.

    State is lost on restart. Async to share an interface with
    :class:`RedisEscalationStore`; nothing here awaits.
    """

    def __init__(self, max_entries: int = 5000) -> None:
        self._max = max_entries
        self._items: OrderedDict[str, EscalationRecord] = OrderedDict()
        self._lock = threading.Lock()

    async def mark(self, chat_id: str, reason: EscalationReason) -> EscalationRecord:
        record = EscalationRecord(
            chat_id=chat_id, reason=reason, created_at=time.time()
        )
        with self._lock:
            self._items.pop(chat_id, None)
            self._items[chat_id] = record
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        logger.info(
            "conversation marked as awaiting human assistance",
            extra={"context": {"chat": mask_chat_id(chat_id), "reason": reason.value}},
        )
        return record

    async def get(self, chat_id: str) -> EscalationRecord | None:
        with self._lock:
            return self._items.get(chat_id)

    async def is_awaiting(self, chat_id: str) -> bool:
        return await self.get(chat_id) is not None

    async def clear(self, chat_id: str) -> None:
        with self._lock:
            self._items.pop(chat_id, None)

    async def pending_count(self) -> int:
        with self._lock:
            return len(self._items)

    async def snapshot(self) -> dict[str, str]:
        """Masked view for health reporting — never exposes raw chat ids."""
        with self._lock:
            return {
                mask_chat_id(cid): rec.reason.value for cid, rec in self._items.items()
            }


class EscalationStoreLike(Protocol):
    async def mark(
        self, chat_id: str, reason: EscalationReason
    ) -> EscalationRecord: ...
    async def get(self, chat_id: str) -> EscalationRecord | None: ...
    async def is_awaiting(self, chat_id: str) -> bool: ...
    async def clear(self, chat_id: str) -> None: ...
    async def pending_count(self) -> int | None: ...


class RedisEscalationStore:
    """Escalation records in Redis, each expiring after ``ttl_seconds``.

    The record is a note for staff, not a control: no code path reads it to
    decide what a customer is told. Callers therefore log and carry on when it
    cannot be written.
    """

    def __init__(self, backend: StateBackend, *, ttl_seconds: int) -> None:
        self._b = backend
        self._ttl = max(60, int(ttl_seconds))

    def _key(self, chat_id: str) -> str:
        return self._b.key("escalation", chat_id)

    def _index(self) -> str:
        return f"{self._b.prefix}:escalation-index"

    async def mark(self, chat_id: str, reason: EscalationReason) -> EscalationRecord:
        record = EscalationRecord(
            chat_id=chat_id, reason=reason, created_at=time.time()
        )
        payload = json.dumps(
            {
                "chat_id": chat_id,
                "reason": reason.value,
                "created_at": record.created_at,
            }
        )
        key = self._key(chat_id)
        await self._b.client.transaction(
            [
                ["SET", key, self._b.cipher.encrypt(payload), "EX", self._ttl],
                ["ZADD", self._index(), record.created_at, key],
                ["EXPIRE", self._index(), self._ttl],
            ]
        )
        logger.info(
            "conversation marked as awaiting human assistance",
            extra={"context": {"chat": mask_chat_id(chat_id), "reason": reason.value}},
        )
        return record

    async def get(self, chat_id: str) -> EscalationRecord | None:
        token = await self._b.client.execute("GET", self._key(chat_id))
        plain = self._b.cipher.decrypt(token) if isinstance(token, str) else None
        if plain is None:
            return None
        try:
            data = json.loads(plain)
            return EscalationRecord(
                chat_id=str(data["chat_id"]),
                reason=EscalationReason(data["reason"]),
                created_at=float(data["created_at"]),
            )
        except (TypeError, KeyError, ValueError):
            return None

    async def is_awaiting(self, chat_id: str) -> bool:
        return await self.get(chat_id) is not None

    async def clear(self, chat_id: str) -> None:
        key = self._key(chat_id)
        await self._b.client.transaction([["DEL", key], ["ZREM", self._index(), key]])

    async def pending_count(self) -> int | None:
        results = await self._b.client.transaction(
            [
                ["ZREMRANGEBYSCORE", self._index(), "-inf", time.time() - self._ttl],
                ["ZCARD", self._index()],
            ]
        )
        return int(results[1])


_store: EscalationStoreLike | None = None


def get_escalation_store() -> EscalationStoreLike:
    global _store
    if _store is None:
        backend = get_state_backend()
        if backend.shared:
            from app.config import get_settings

            _store = RedisEscalationStore(
                backend, ttl_seconds=get_settings().redis.escalation_ttl_seconds
            )
        else:
            _store = EscalationStore()
    return _store


def set_escalation_store(store: EscalationStoreLike | None) -> None:
    """Swap the process-wide store. Tests use this to inject a fresh one."""
    global _store
    _store = store


async def mark_escalation(
    store: EscalationStoreLike, chat_id: str, reason: EscalationReason
) -> None:
    """Record an escalation; if shared state is down, log and carry on.

    What the customer is told never depends on this record, so failing to
    write it must not change the reply.
    """
    from app.state.upstash import StateUnavailable

    try:
        await store.mark(chat_id, reason)
    except StateUnavailable as exc:
        logger.error(
            "could not record an escalation; shared state is unavailable",
            extra={
                "context": {
                    "chat": mask_chat_id(chat_id),
                    "reason": reason.value,
                    "error": exc.reason,
                }
            },
        )
