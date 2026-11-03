"""An in-memory Upstash Redis REST endpoint, served through httpx.MockTransport.

Implements exactly the commands the app sends, with Redis semantics for
expiry, NX/XX and sorted-set ranges, on a fake clock. Nothing here opens a
socket. ``fail`` makes every request fail the way a real outage would.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import Secret
from app.state.backend import BACKEND_REDIS, StateBackend
from app.state.crypto import StateCipher
from app.state.upstash import UpstashClient

URL = "https://fake-upstash.example"
TOKEN = "fake-upstash-token-for-tests"
#: A valid Fernet key that is obviously a test value.
KEY = "dGVzdC1rZXktZm9yLXN0YXRlLWVuY3J5cHRpb24tMDA="


class FakeUpstash:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0
        self.data: dict[str, Any] = {}
        self.expiry: dict[str, float] = {}
        self.requests: list[Any] = []
        #: None, or "timeout" / "connect" / "http_500" / "error_reply".
        self.fail: str | None = None

    # -- clock -----------------------------------------------------------

    def clock(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def ttl(self, key: str) -> float | None:
        return None if key not in self.expiry else self.expiry[key] - self.now

    # -- transport -------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self.fail == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if self.fail == "connect":
            raise httpx.ConnectError("refused", request=request)
        if self.fail == "http_500":
            return httpx.Response(500, json={"error": "internal"})
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"error": "Unauthorized"})
        body = json.loads(request.content)
        self.requests.append(body)
        path = request.url.path.rstrip("/")
        if self.fail == "error_reply":
            if path in ("/pipeline", "/multi-exec"):
                return httpx.Response(200, json=[{"error": "ERR"} for _ in body])
            return httpx.Response(400, json={"error": "ERR wrong type"})
        if path in ("/pipeline", "/multi-exec"):
            return httpx.Response(200, json=[{"result": self.run(c)} for c in body])
        return httpx.Response(200, json={"result": self.run(body)})

    # -- redis -----------------------------------------------------------

    def _alive(self, key: str) -> bool:
        if key in self.expiry and self.expiry[key] <= self.now:
            self.data.pop(key, None)
            self.expiry.pop(key, None)
        return key in self.data

    def run(self, command: list[str]) -> Any:
        name, *args = command
        return getattr(self, "_" + name.lower())(*args)

    def _get(self, key):
        return self.data[key] if self._alive(key) else None

    def _set(self, key, value, *opts):
        opts = [o.upper() for o in opts]
        exists = self._alive(key)
        if "NX" in opts and exists:
            return None
        if "XX" in opts and not exists:
            return None
        self.data[key] = value
        self.expiry.pop(key, None)
        if "EX" in opts:
            self.expiry[key] = self.now + float(opts[opts.index("EX") + 1])
        if "PX" in opts:
            self.expiry[key] = self.now + float(opts[opts.index("PX") + 1]) / 1000
        return "OK"

    def _del(self, *keys):
        removed = 0
        for key in keys:
            if self._alive(key):
                removed += 1
            self.data.pop(key, None)
            self.expiry.pop(key, None)
        return removed

    def _exists(self, key):
        return int(self._alive(key))

    def _expire(self, key, seconds):
        if not self._alive(key):
            return 0
        self.expiry[key] = self.now + float(seconds)
        return 1

    def _hset(self, key, *pairs):
        h = self.data.setdefault(key, {}) if self._alive(key) else {}
        self.data[key] = h
        for field, value in zip(pairs[0::2], pairs[1::2], strict=True):
            h[field] = value
        return len(pairs) // 2

    def _hgetall(self, key):
        if not self._alive(key):
            return []
        return [x for pair in self.data[key].items() for x in pair]

    def _rpush(self, key, *values):
        lst = self.data[key] if self._alive(key) else []
        lst.extend(values)
        self.data[key] = lst
        return len(lst)

    def _lrange(self, key, start, stop):
        if not self._alive(key):
            return []
        lst = self.data[key]
        start, stop = int(start), int(stop)
        if start < 0:
            start = max(0, len(lst) + start)
        stop = len(lst) if stop == -1 else stop + 1
        return lst[start:stop]

    def _zset(self, key) -> dict[str, float]:
        z = self.data[key] if self._alive(key) else {}
        self.data[key] = z
        return z

    def _zadd(self, key, score, member):
        z = self._zset(key)
        new = member not in z
        z[member] = float(score)
        return int(new)

    def _zrem(self, key, *members):
        if not self._alive(key):
            return 0
        z = self.data[key]
        return sum(1 for m in members if z.pop(m, None) is not None)

    def _zrange(self, key, start, stop):
        if not self._alive(key):
            return []
        ordered = sorted(self.data[key].items(), key=lambda kv: (kv[1], kv[0]))
        assert (start, stop) == ("0", "-1")
        return [m for m, _ in ordered]

    def _zremrangebyscore(self, key, low, high):
        if not self._alive(key):
            return 0
        assert low == "-inf"
        z = self.data[key]
        doomed = [m for m, s in z.items() if s <= float(high)]
        for m in doomed:
            del z[m]
        return len(doomed)

    def _zcard(self, key):
        return len(self.data[key]) if self._alive(key) else 0

    def _eval(self, script, numkeys, key, token):
        # Only the lock-release script is ever sent.
        assert "redis.call('get', KEYS[1]) == ARGV[1]" in script
        assert numkeys == "1"
        if self._get(key) == token:
            return self._del(key)
        return 0

    # -- helpers for tests ------------------------------------------------

    def dump(self) -> str:
        """Everything stored and every request sent, as one searchable string."""
        return json.dumps([self.data, self.requests], ensure_ascii=False, default=str)


def redis_backend(
    fake: FakeUpstash, *, key: str = KEY, prefix: str = "test"
) -> StateBackend:
    return StateBackend(
        kind=BACKEND_REDIS,
        client=UpstashClient(URL, Secret(TOKEN), transport=fake.transport()),
        cipher=StateCipher(key),
        prefix=prefix,
    )
