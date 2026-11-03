"""Expiring sets of keys: webhook de-duplication, echo and sent-report guards.

Same three operations as :class:`~app.whatsapp.dedupe.TTLDedupeCache`, made
async so a Redis implementation can sit behind them:

* ``seen(key)`` — atomic check-and-record; ``True`` means "already handled".
* ``remember(key)`` / ``contains(key)`` — record, and read-only test.
* ``forget(key)`` — undo a claim, so a message we did not process can be
  redelivered and processed then.
"""

from __future__ import annotations

from typing import Protocol

from app.state.backend import StateBackend
from app.whatsapp.dedupe import TTLDedupeCache


class KeySet(Protocol):
    async def seen(self, key: str) -> bool: ...
    async def remember(self, key: str) -> None: ...
    async def contains(self, key: str) -> bool: ...
    async def forget(self, key: str) -> None: ...
    def count(self) -> int | None: ...


class MemoryKeySet:
    def __init__(self, cache: TTLDedupeCache) -> None:
        self._cache = cache

    async def seen(self, key: str) -> bool:
        return self._cache.seen(key)

    async def remember(self, key: str) -> None:
        self._cache.remember(key)

    async def contains(self, key: str) -> bool:
        return self._cache.contains(key)

    async def forget(self, key: str) -> None:
        self._cache.forget(key)

    def count(self) -> int | None:
        return len(self._cache)


class RedisKeySet:
    """One Redis key per member, expiring on its own. Values carry nothing."""

    def __init__(self, backend: StateBackend, namespace: str, ttl_seconds: int):
        self._backend = backend
        self._namespace = namespace
        self._ttl = max(1, int(ttl_seconds))

    def _key(self, key: str) -> str:
        return self._backend.key(self._namespace, key)

    async def seen(self, key: str) -> bool:
        if not key:
            return False
        # SET NX is the atomic check-and-record: exactly one instance gets OK.
        result = await self._backend.client.execute(
            "SET", self._key(key), "1", "NX", "EX", self._ttl
        )
        return result is None

    async def remember(self, key: str) -> None:
        if key:
            await self._backend.client.execute(
                "SET", self._key(key), "1", "EX", self._ttl
            )

    async def contains(self, key: str) -> bool:
        if not key:
            return False
        return bool(await self._backend.client.execute("EXISTS", self._key(key)))

    async def forget(self, key: str) -> None:
        if key:
            await self._backend.client.execute("DEL", self._key(key))

    def count(self) -> int | None:
        # Counting would mean scanning the keyspace; health reports "unknown".
        return None


def key_set(
    backend: StateBackend, namespace: str, *, ttl_seconds: int, max_entries: int
) -> KeySet:
    if backend.shared:
        return RedisKeySet(backend, namespace, ttl_seconds)
    return MemoryKeySet(
        TTLDedupeCache(ttl_seconds=ttl_seconds, max_entries=max_entries)
    )
