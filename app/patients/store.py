"""Server-side patient verification state.

What is kept here never reaches the model, a log line or the customer: the
internal patient id and the backend's verification token are the two values
that, together, unlock someone else's laboratory results. They are keyed by
the conversation that earned them and read back only by :mod:`app.tools` and
the report sender, when building a request the customer already authorised.

Two implementations, chosen by :mod:`app.state.backend`: in process memory,
or in Upstash Redis — encrypted, under a hashed key, expiring with the
verification itself — so every serverless instance sees the same record and
the same attempt counter.

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
import json
import math
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.logging_utils import get_logger, mask_chat_id
from app.patients.context import ConversationScope
from app.state.backend import StateBackend, get_state_backend

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

    Async to share an interface with :class:`RedisVerificationStore`; nothing
    here awaits.

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

    async def get(self, scope: ConversationScope) -> VerifiedPatient | None:
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

    async def put(self, scope: ConversationScope, record: VerifiedPatient) -> None:
        with self._lock:
            self._items.pop(scope.key, None)
            self._items[scope.key] = record
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    async def clear(self, scope: ConversationScope) -> None:
        with self._lock:
            self._items.pop(scope.key, None)

    async def begin_attempt(self, scope: ConversationScope) -> None:
        """Revoke any existing authorisation before a verification is tried.

        Called *before* the backend request, not after. If the attempt fails,
        times out, or the process is interrupted halfway, the conversation is
        left unverified rather than still holding the previous person's
        authorisation.
        """
        await self.clear(scope)

    async def count(self) -> int:
        with self._lock:
            return len(self._items)

    # -- failed-attempt limiting ------------------------------------------

    async def record_failure(
        self, scope: ConversationScope, *, window_seconds: int
    ) -> int:
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

    async def failure_count(
        self, scope: ConversationScope, *, window_seconds: int
    ) -> int:
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

    async def reset_failures(self, scope: ConversationScope) -> None:
        with self._lock:
            self._attempts.pop(scope.chat_key, None)

    async def clear_all(self) -> None:
        with self._lock:
            self._items.clear()
            self._attempts.clear()


class VerificationStoreLike(Protocol):
    def now(self) -> float: ...
    async def get(self, scope: ConversationScope) -> VerifiedPatient | None: ...
    async def put(self, scope: ConversationScope, record: VerifiedPatient) -> None: ...
    async def clear(self, scope: ConversationScope) -> None: ...
    async def begin_attempt(self, scope: ConversationScope) -> None: ...
    async def count(self) -> int | None: ...
    async def record_failure(
        self, scope: ConversationScope, *, window_seconds: int
    ) -> int: ...
    async def failure_count(
        self, scope: ConversationScope, *, window_seconds: int
    ) -> int: ...
    async def reset_failures(self, scope: ConversationScope) -> None: ...


class RedisVerificationStore:
    """Verification records and attempt counters shared by every instance.

    Every method raises :class:`~app.state.upstash.StateUnavailable` when
    Redis cannot answer. Callers treat that as "not verified" — never as a
    reason to proceed — and never as a failed attempt that was not counted.

    Expiry is enforced twice: Redis drops the key when the verification
    lapses, and :meth:`get` re-checks ``expires_at`` against the wall clock,
    the only clock two instances share.
    """

    def __init__(
        self,
        backend: StateBackend,
        *,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._b = backend
        self._now = time_source or time.time

    def now(self) -> float:
        return self._now()

    def _record_key(self, scope: ConversationScope) -> str:
        return self._b.key("verify", *scope.key)

    def _attempts_key(self, scope: ConversationScope) -> str:
        return self._b.key("verify-attempts", *scope.chat_key)

    async def get(self, scope: ConversationScope) -> VerifiedPatient | None:
        key = self._record_key(scope)
        token = await self._b.client.execute("GET", key)
        if token is None:
            return None
        plain = self._b.cipher.decrypt(token) if isinstance(token, str) else None
        try:
            data = json.loads(plain) if plain is not None else None
            record = VerifiedPatient(
                patient_id=str(data["patient_id"]),
                verification_token=str(data["verification_token"]),
                expires_at=float(data["expires_at"]),
                method=str(data.get("method") or ""),
                identity=str(data["identity"]),
            )
        except (TypeError, KeyError, ValueError):
            # Unreadable — a rotated key or a corrupted value. Not verified.
            await self._b.client.execute("DEL", key)
            return None
        if not record.is_valid(self._now()):
            await self._b.client.execute("DEL", key)
            logger.info(
                "patient verification expired",
                extra={"context": {"chat": mask_chat_id(scope.chat_id)}},
            )
            return None
        return record

    async def put(self, scope: ConversationScope, record: VerifiedPatient) -> None:
        ttl = math.ceil(record.expires_at - self._now())
        if ttl <= 0 or not record.is_valid(self._now()):
            # Already expired: storing it would only be deleted on read.
            await self.clear(scope)
            return
        payload = json.dumps(
            {
                "patient_id": record.patient_id,
                "verification_token": record.verification_token,
                "expires_at": record.expires_at,
                "method": record.method,
                "identity": record.identity,
            }
        )
        await self._b.client.execute(
            "SET", self._record_key(scope), self._b.cipher.encrypt(payload), "EX", ttl
        )

    async def clear(self, scope: ConversationScope) -> None:
        await self._b.client.execute("DEL", self._record_key(scope))

    async def begin_attempt(self, scope: ConversationScope) -> None:
        await self.clear(scope)

    async def count(self) -> int | None:
        return None

    async def record_failure(
        self, scope: ConversationScope, *, window_seconds: int
    ) -> int:
        now = self._now()
        key = self._attempts_key(scope)
        # Atomic: prune the window, add this attempt, count, extend expiry.
        results = await self._b.client.transaction(
            [
                ["ZREMRANGEBYSCORE", key, "-inf", now - window_seconds],
                ["ZADD", key, now, f"{now:.6f}:{secrets.token_hex(4)}"],
                ["ZCARD", key],
                ["EXPIRE", key, max(1, int(window_seconds))],
            ]
        )
        return int(results[2])

    async def failure_count(
        self, scope: ConversationScope, *, window_seconds: int
    ) -> int:
        now = self._now()
        key = self._attempts_key(scope)
        results = await self._b.client.transaction(
            [
                ["ZREMRANGEBYSCORE", key, "-inf", now - window_seconds],
                ["ZCARD", key],
            ]
        )
        return int(results[1])

    async def reset_failures(self, scope: ConversationScope) -> None:
        await self._b.client.execute("DEL", self._attempts_key(scope))


_store: VerificationStoreLike | None = None


def get_verification_store() -> VerificationStoreLike:
    global _store
    if _store is None:
        backend = get_state_backend()
        _store = (
            RedisVerificationStore(backend) if backend.shared else VerificationStore()
        )
    return _store


def set_verification_store(store: VerificationStoreLike | None) -> None:
    """Swap the process-wide store. Tests use this to inject a fresh one.

    ``None`` makes the next :func:`get_verification_store` build one again
    from the configured backend.
    """
    global _store
    _store = store
