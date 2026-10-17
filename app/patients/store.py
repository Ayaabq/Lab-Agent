"""Server-side patient verification state.

What is kept here never leaves the process: the internal patient id and the
backend's verification token are the two values that, together, unlock someone
else's laboratory results. They are held in memory, keyed by the conversation
that earned them, and read back only by :mod:`app.tools` when it builds a
request the customer already authorised.

Scoping rules this module enforces
----------------------------------
* **Tenant + channel + chat + ADK session.** A verification granted in one
  WhatsApp chat is invisible to every other chat, and does not survive into a
  new ADK session.
* **Expiry is absolute.** ``get`` drops an expired record instead of returning
  it; there is no grace period, because the backend will reject the token
  anyway and a "nearly valid" verification is just an unverified one.
* **Switching identity revokes the old one.** Starting a verification for a
  different person clears whatever was authorised before, and so does a failed
  attempt — see :meth:`VerificationStore.begin_attempt`.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from app.logging_utils import get_logger, mask_chat_id
from app.patients.context import ConversationScope

logger = get_logger(__name__)

#: Bounded so a long-running process cannot grow without limit.
MAX_VERIFICATIONS = 2000
MAX_ATTEMPT_COUNTERS = 5000


def identity_fingerprint(
    *, full_name: str, date_of_birth: str, phone_last4: str
) -> str:
    """A non-reversible stand-in for "which person was this verification for".

    Comparing fingerprints lets us notice that the customer switched identity
    without keeping their name or birth date anywhere.
    """
    seed = f"{full_name.strip().casefold()}|{date_of_birth}|{phone_last4}".encode()
    return hashlib.sha256(seed).hexdigest()[:32]


@dataclass(frozen=True)
class VerifiedPatient:
    """Proof that this conversation may read one patient's results.

    Never serialised into session state, tool output, a prompt, or a log line.
    """

    patient_id: str
    verification_token: str
    expires_at: float
    method: str
    identity: str

    def is_valid(self, now: float) -> bool:
        return (
            bool(self.patient_id and self.verification_token) and now < self.expires_at
        )

    def seconds_remaining(self, now: float) -> int:
        return max(0, int(self.expires_at - now))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"VerifiedPatient(patient_id=***, token=***, method={self.method!r})"


class VerificationStore:
    """In-memory verification records and failed-attempt counters.

    Thread-safe rather than asyncio-only: ADK may call a tool from a worker
    thread, and the WhatsApp service runs turns concurrently.
    """

    def __init__(
        self,
        *,
        max_entries: int = MAX_VERIFICATIONS,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._max = max_entries
        self._now = time_source or time.monotonic
        self._items: OrderedDict[tuple[str, str, str, str], VerifiedPatient] = (
            OrderedDict()
        )
        #: (chat scope) -> [attempt timestamps], newest last.
        self._attempts: OrderedDict[tuple[str, str, str], list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def now(self) -> float:
        """The store's own clock.

        Callers build ``expires_at`` from this rather than from
        ``time.monotonic()`` directly: two clocks would let a record be
        written against one timeline and expired against another.
        """
        return self._now()

    # -- verification records ---------------------------------------------

    def get(self, scope: ConversationScope) -> VerifiedPatient | None:
        """Return a still-valid record, dropping it if it has expired."""
        now = self._now()
        with self._lock:
            record = self._items.get(scope.key)
            if record is None:
                return None
            if not record.is_valid(now):
                self._items.pop(scope.key, None)
                logger.info(
                    "patient verification expired",
                    extra={"context": {"chat": mask_chat_id(scope.chat_id)}},
                )
                return None
            return record

    def put(self, scope: ConversationScope, record: VerifiedPatient) -> None:
        with self._lock:
            self._items.pop(scope.key, None)
            self._items[scope.key] = record
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def clear(self, scope: ConversationScope) -> None:
        with self._lock:
            self._items.pop(scope.key, None)

    def begin_attempt(self, scope: ConversationScope) -> None:
        """Revoke any existing authorisation before a verification is tried.

        Called *before* the backend request, not after. If the attempt fails,
        times out, or the process is interrupted halfway, the conversation is
        left unverified rather than still holding the previous person's
        authorisation.
        """
        self.clear(scope)

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    # -- failed-attempt limiting ------------------------------------------

    def record_failure(self, scope: ConversationScope, *, window_seconds: int) -> int:
        """Record a failed verification and return the count inside the window."""
        now = self._now()
        with self._lock:
            stamps = [
                t
                for t in self._attempts.get(scope.chat_key, [])
                if t > now - window_seconds
            ]
            stamps.append(now)
            self._attempts.pop(scope.chat_key, None)
            self._attempts[scope.chat_key] = stamps
            while len(self._attempts) > MAX_ATTEMPT_COUNTERS:
                self._attempts.popitem(last=False)
            return len(stamps)

    def failure_count(self, scope: ConversationScope, *, window_seconds: int) -> int:
        now = self._now()
        with self._lock:
            stamps = [
                t
                for t in self._attempts.get(scope.chat_key, [])
                if t > now - window_seconds
            ]
            if stamps:
                self._attempts[scope.chat_key] = stamps
            else:
                self._attempts.pop(scope.chat_key, None)
            return len(stamps)

    def reset_failures(self, scope: ConversationScope) -> None:
        with self._lock:
            self._attempts.pop(scope.chat_key, None)

    def clear_all(self) -> None:
        with self._lock:
            self._items.clear()
            self._attempts.clear()


_store = VerificationStore()


def get_verification_store() -> VerificationStore:
    return _store


def set_verification_store(store: VerificationStore | None) -> None:
    """Swap the process-wide store. Tests use this to inject a fresh one."""
    global _store
    _store = store if store is not None else VerificationStore()
