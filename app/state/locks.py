"""One message at a time per chat.

In memory this is an ``asyncio.Lock`` per chat, as before. Across instances
it is a Redis lease: ``SET NX PX`` with a random token, released only by the
holder (compare-and-delete in Lua), and expiring by itself if the holder's
function is killed.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable

from app.logging_utils import get_logger, mask_chat_id
from app.state.backend import StateBackend
from app.state.upstash import StateUnavailable

logger = get_logger(__name__)

#: At least the Vercel function's maxDuration, so a lease cannot lapse while
#: the function holding it is still running. A crashed holder blocks its chat
#: for at most this long; GREEN API keeps redelivering meanwhile.
LOCK_LEASE_SECONDS = 300

#: How long a second message from the same chat waits for the first. Past
#: this the webhook answers 503 and GREEN API redelivers a minute later.
LOCK_WAIT_SECONDS = 60

#: Chat locks are cheap but unbounded growth is not. Oldest idle locks are
#: dropped once we exceed this.
_MAX_TRACKED_CHATS = 2000

_RELEASE_SCRIPT = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)


class ChatBusy(Exception):
    """Another instance is still answering this chat."""


class MemoryChatLocks:
    def __init__(self) -> None:
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._guard = asyncio.Lock()

    async def _lock_for(self, chat_id: str) -> asyncio.Lock:
        async with self._guard:
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

    @contextlib.asynccontextmanager
    async def hold(self, chat_id: str) -> AsyncIterator[None]:
        lock = await self._lock_for(chat_id)
        async with lock:
            yield

    def count(self) -> int | None:
        return len(self._locks)


class RedisChatLocks:
    def __init__(
        self,
        backend: StateBackend,
        *,
        lease_seconds: int = LOCK_LEASE_SECONDS,
        wait_seconds: float = LOCK_WAIT_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self._lease_ms = int(lease_seconds * 1000)
        self._wait = wait_seconds
        self._sleep = sleep
        self._clock = clock

    @contextlib.asynccontextmanager
    async def hold(self, chat_id: str) -> AsyncIterator[None]:
        """Raises :class:`ChatBusy` after the wait, ``StateUnavailable`` if
        Redis cannot be reached. Neither is raised once the body has run."""
        key = self._backend.key("lock", chat_id)
        token = secrets.token_hex(16)
        deadline = self._clock() + self._wait
        delay = 0.2
        while True:
            acquired = await self._backend.client.execute(
                "SET", key, token, "NX", "PX", self._lease_ms
            )
            if acquired is not None:
                break
            if self._clock() >= deadline:
                raise ChatBusy()
            await self._sleep(delay)
            delay = min(delay * 2, 2.0)
        try:
            yield
        finally:
            try:
                await self._backend.client.execute(
                    "EVAL", _RELEASE_SCRIPT, 1, key, token
                )
            except StateUnavailable:
                # The lease expires by itself; the reply has already gone.
                logger.warning(
                    "could not release a chat lock; it will expire",
                    extra={"context": {"chat": mask_chat_id(chat_id)}},
                )

    def count(self) -> int | None:
        return None


def chat_locks(backend: StateBackend) -> MemoryChatLocks | RedisChatLocks:
    return RedisChatLocks(backend) if backend.shared else MemoryChatLocks()
