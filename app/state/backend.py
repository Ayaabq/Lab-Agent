"""Which state backend this process uses.

* ``memory`` — no Redis variable is set. The original behaviour: correct on a
  single long-lived process, per instance on a serverless host.
* ``redis`` — Upstash URL, token and a valid encryption key are all set.
* ``unavailable`` — some Redis variable is set but the set is incomplete or
  invalid. Every shared-state operation then fails, so patient verification is
  refused and messages get the fallback reply. A half-configured deployment
  must not quietly run on per-instance memory while looking persistent.
"""

from __future__ import annotations

import base64
import functools
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.state.crypto import StateCipher
from app.state.upstash import UnavailableClient, UpstashClient

BACKEND_MEMORY = "memory"
BACKEND_REDIS = "redis"
BACKEND_UNAVAILABLE = "unavailable"

#: Used only to derive key names for the ``unavailable`` backend, whose
#: client never sends anything. Not a secret; nothing is encrypted with it.
_PLACEHOLDER_KEY = base64.urlsafe_b64encode(
    b"state-backend-is-unavailable-000"
).decode()


@dataclass(frozen=True)
class StateBackend:
    kind: str
    #: ``UpstashClient`` or ``UnavailableClient``; ``None`` for memory.
    client: Any = None
    cipher: StateCipher | None = None
    prefix: str = "khatib"
    reason: str = ""

    @property
    def shared(self) -> bool:
        """State lives outside this process (or would, if it were reachable)."""
        return self.kind != BACKEND_MEMORY

    def key(self, kind: str, *parts: str) -> str:
        """Full Redis key for ``parts``, which are hashed, never stored raw."""
        if self.cipher is None:
            raise RuntimeError("the memory backend has no Redis keys")
        return f"{self.prefix}:{kind}:{self.cipher.name(*parts)}"


def build_backend(settings: Settings) -> StateBackend:
    redis = settings.redis
    if not redis.requested:
        return StateBackend(kind=BACKEND_MEMORY)
    if problem := redis.problem():
        return StateBackend(
            kind=BACKEND_UNAVAILABLE,
            client=UnavailableClient("redis_misconfigured"),
            cipher=StateCipher(_PLACEHOLDER_KEY),
            prefix=redis.key_prefix,
            reason=problem,
        )
    return StateBackend(
        kind=BACKEND_REDIS,
        client=UpstashClient(
            redis.rest_url, redis.rest_token, timeout_seconds=redis.timeout_seconds
        ),
        cipher=StateCipher(redis.encryption_key.get()),
        prefix=redis.key_prefix,
    )


@functools.lru_cache(maxsize=1)
def get_state_backend() -> StateBackend:
    return build_backend(get_settings())


def reset_state_backend() -> None:
    """Forget the cached backend. Tests use this after changing settings."""
    get_state_backend.cache_clear()
