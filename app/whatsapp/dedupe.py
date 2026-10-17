"""Bounded TTL cache used to drop duplicate webhook deliveries.

GREEN API retries a webhook until it gets a 200, so the same ``idMessage`` can
arrive several times. Processing it twice would send the customer two replies.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable


class TTLDedupeCache:
    """Remembers recently seen keys for ``ttl_seconds``, capped at ``max_entries``."""

    def __init__(
        self,
        ttl_seconds: int = 900,
        max_entries: int = 5000,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._now = time_source or time.monotonic
        self._items: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def seen(self, key: str) -> bool:
        """Atomically check-and-record. True means "already handled"."""
        if not key:
            return False
        now = self._now()
        with self._lock:
            self._evict(now)
            expires_at = self._items.get(key)
            if expires_at is not None and expires_at > now:
                return True
            self._items.pop(key, None)
            self._items[key] = now + self._ttl
            while len(self._items) > self._max:
                self._items.popitem(last=False)
            return False

    def remember(self, key: str) -> None:
        """Record a key without reporting whether it was already present."""
        if key:
            self.seen(key)

    def contains(self, key: str) -> bool:
        """Read-only membership test. Unlike :meth:`seen`, records nothing."""
        if not key:
            return False
        now = self._now()
        with self._lock:
            expires_at = self._items.get(key)
            return expires_at is not None and expires_at > now

    def _evict(self, now: float) -> None:
        # Entries are inserted in expiry order, so stop at the first live one.
        while self._items:
            key, expires_at = next(iter(self._items.items()))
            if expires_at > now:
                break
            self._items.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            self._evict(self._now())
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
