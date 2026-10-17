"""Staging a report image for delivery, between the tool and the transport.

Why this exists
---------------
Deciding that a report *may* be released and actually sending it are different
jobs, done at different moments, and putting them in one place would force a
bad trade-off either way:

* The tool runs inside the model's loop. If it sent the image itself, the send
  would happen before the model had finished composing the turn, and the
  ordering of what the customer sees would depend on the model.
* The service runs outside the model, already holds the per-chat lock, the
  inbound de-duplication and the live-sending switch, and already knows which
  chat the webhook authenticated. That is where an outbound message belongs.

So the tool *stages* a release here, and the service takes it and sends it —
after the text reply, and only after re-checking that the verification that
authorised it is still valid. The two checks are deliberately separate: a
verification can lapse between the backend answering and the image going out,
and the one that matters is the one at the moment of sending.

Nothing staged here is ever shown to the model. The URL, the record ids and
the patient id are read back only by the service.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from app.patients.context import ConversationScope

#: A staged release that is never collected is dropped. It is short because a
#: release is collected at the end of the very turn that staged it; anything
#: older is debris from a failed turn, not a pending delivery.
STAGED_TTL_SECONDS = 120

MAX_STAGED = 500


@dataclass(frozen=True)
class PendingRelease:
    """One report image, authorised and waiting for the service to send it.

    ``identity`` and ``patient_id`` are copied from the verification that
    authorised this release so the service can confirm, at send time, that it
    is still the same verified person in the same chat — not merely that
    *someone* in this chat is verified now.
    """

    chat_id: str
    patient_message: str
    image_url: str
    file_name: str
    #: Keys the "already sent this report" check. Server-side only.
    result_id: str
    #: Copied from the verification record that authorised the release.
    patient_id: str
    identity: str
    created_at: float

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PendingRelease(chat_id=<masked>, file_name={self.file_name!r}, "
            "image_url=<redacted>, patient_id=***)"
        )

    def dedupe_key(self) -> str:
        """Identifies this report in this chat, for duplicate suppression.

        Falls back to the URL when the backend sent no ``resultId`` — two
        sends of the same URL to the same chat are the duplicate we care
        about, however the backend chose to label the record.
        """
        return f"{self.chat_id}|{self.result_id or self.image_url}"


class ReleaseStore:
    """Holds at most one staged release per conversation.

    One slot, not a queue: a later result supersedes an earlier one in the
    same conversation, and a backlog of unsent medical reports is not
    something this process should be capable of accumulating.
    """

    def __init__(
        self,
        *,
        max_entries: int = MAX_STAGED,
        ttl_seconds: int = STAGED_TTL_SECONDS,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._max = max_entries
        self._ttl = ttl_seconds
        self._now = time_source or time.monotonic
        self._items: OrderedDict[tuple[str, str, str, str], PendingRelease] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def now(self) -> float:
        return self._now()

    def stage(self, scope: ConversationScope, release: PendingRelease) -> None:
        with self._lock:
            self._items.pop(scope.key, None)
            self._items[scope.key] = release
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def take(self, scope: ConversationScope) -> PendingRelease | None:
        """Remove and return the staged release, if it is still fresh.

        Taking rather than reading is the point: a release can only ever be
        collected once, so a retry of the surrounding code cannot turn into a
        second copy of the same report.
        """
        now = self._now()
        with self._lock:
            release = self._items.pop(scope.key, None)
        if release is None:
            return None
        if now - release.created_at > self._ttl:
            return None
        return release

    def discard(self, scope: ConversationScope) -> None:
        with self._lock:
            self._items.pop(scope.key, None)

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def clear_all(self) -> None:
        with self._lock:
            self._items.clear()


_store = ReleaseStore()


def get_release_store() -> ReleaseStore:
    return _store


def set_release_store(store: ReleaseStore | None) -> None:
    """Swap the process-wide store. Tests use this to inject a fresh one."""
    global _store
    _store = store if store is not None else ReleaseStore()
