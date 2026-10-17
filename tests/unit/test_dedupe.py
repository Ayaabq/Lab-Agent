"""The TTL cache that stops GREEN API retries from double-answering."""

from __future__ import annotations

from app.whatsapp.dedupe import TTLDedupeCache


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_first_sighting_is_not_a_duplicate():
    cache = TTLDedupeCache(ttl_seconds=60, max_entries=10)

    assert cache.seen("MSG-1") is False


def test_second_sighting_is_a_duplicate():
    cache = TTLDedupeCache(ttl_seconds=60, max_entries=10)
    cache.seen("MSG-1")

    assert cache.seen("MSG-1") is True


def test_entries_expire_after_the_ttl():
    clock = Clock()
    cache = TTLDedupeCache(ttl_seconds=60, max_entries=10, time_source=clock)
    cache.seen("MSG-1")

    clock.advance(61)

    assert cache.seen("MSG-1") is False


def test_entry_survives_until_the_ttl():
    clock = Clock()
    cache = TTLDedupeCache(ttl_seconds=60, max_entries=10, time_source=clock)
    cache.seen("MSG-1")

    clock.advance(59)

    assert cache.seen("MSG-1") is True


def test_cache_is_bounded():
    cache = TTLDedupeCache(ttl_seconds=600, max_entries=3)

    for index in range(10):
        cache.seen(f"MSG-{index}")

    assert len(cache) == 3
    # The oldest entries were evicted, so they are no longer "seen".
    assert cache.seen("MSG-0") is False
    # The newest one is still remembered.
    assert cache.seen("MSG-9") is True


def test_empty_key_is_never_a_duplicate():
    cache = TTLDedupeCache()

    assert cache.seen("") is False
    assert cache.seen("") is False


def test_distinct_keys_do_not_collide():
    cache = TTLDedupeCache()

    assert cache.seen("A") is False
    assert cache.seen("B") is False
    assert cache.seen("A") is True
