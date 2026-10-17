"""Shared fixtures. Every external service is faked — no network in unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.config import (
    GeminiSettings,
    GreenApiSettings,
    KhatibApiSettings,
    Secret,
    Settings,
    WeaviateSettings,
)


@pytest.fixture
def settings() -> Settings:
    """Fully configured settings with live WhatsApp sending disabled."""
    return Settings(
        gemini=GeminiSettings(
            model="gemini-3.8-flash",
            api_key=Secret("test-gemini-key-abcdef"),
            use_vertexai=False,
        ),
        weaviate=WeaviateSettings(
            url="https://example.weaviate.cloud",
            api_key=Secret("test-weaviate-key-abcdef"),
            collection="",
            search_limit=3,
            min_certainty=0.62,
            timeout_seconds=5,
            query_mode="near_text",
        ),
        green_api=GreenApiSettings(
            api_url="https://0000.api.greenapi.com",
            id_instance=Secret("000011112222"),
            token_instance=Secret("test-green-token-abcdef"),
            live_enabled=False,
            ignore_groups=True,
            timeout_seconds=5,
            dedupe_ttl_seconds=60,
            dedupe_max_entries=100,
            # Off by default: the shared webhook fixtures carry a fixed
            # timestamp from 2024, so a live age check would reject every one
            # of them. The tests that care about staleness switch it on.
            max_message_age_seconds=0,
        ),
        khatib=khatib_settings(),
        log_level="INFO",
    )


def khatib_settings(**overrides) -> KhatibApiSettings:
    """A configured backend with short, test-friendly budgets.

    The image fields stay empty, matching production: the backend has not
    published that contract yet, so "no image" is the case under test.
    """
    defaults = {
        "base_url": "https://khatib.example.test",
        "client_id": Secret("test-khatib-client-id"),
        "client_secret": Secret("test-khatib-client-secret"),
        "timeout_seconds": 5,
        "connect_timeout_seconds": 2,
        "auth_refresh_margin_seconds": 10,
        "auth_retry_cooldown_seconds": 30,
        "default_verification_ttl_seconds": 300,
        "max_verification_attempts": 3,
        "verification_attempt_window_seconds": 900,
    }
    defaults.update(overrides)
    return KhatibApiSettings(**defaults)


# --- Fake Weaviate -------------------------------------------------------


@dataclass
class FakeMetadata:
    certainty: float | None = None
    distance: float | None = None
    score: float | None = None


@dataclass
class FakeObject:
    properties: dict[str, Any]
    metadata: FakeMetadata
    uuid: str = "0000-1111"


@dataclass
class FakeResponse:
    objects: list[FakeObject]


@dataclass
class FakeProperty:
    name: str


@dataclass
class FakeCollectionConfig:
    properties: list[FakeProperty]
    vector_config: dict[str, Any] | None = None
    vectorizer: str | None = None


class FakeQuery:
    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection

    def _respond(self, kind: str, **kwargs: Any) -> FakeResponse:
        self._collection.calls.append((kind, kwargs))
        if self._collection.raises is not None:
            raise self._collection.raises
        return FakeResponse(list(self._collection.objects))

    def near_text(self, **kwargs: Any) -> FakeResponse:
        return self._respond("near_text", **kwargs)

    def hybrid(self, **kwargs: Any) -> FakeResponse:
        return self._respond("hybrid", **kwargs)

    def bm25(self, **kwargs: Any) -> FakeResponse:
        return self._respond("bm25", **kwargs)

    def fetch_objects(self, **kwargs: Any) -> FakeResponse:
        return self._respond("fetch_objects", **kwargs)


class _FakeConfigAccessor:
    def __init__(self, config: FakeCollectionConfig) -> None:
        self._config = config

    def get(self) -> FakeCollectionConfig:
        return self._config


class _FakeAggregate:
    def __init__(self, total: int) -> None:
        self._total = total

    def over_all(self, **_: Any) -> Any:
        return type("Agg", (), {"total_count": self._total})()


class FakeCollection:
    def __init__(
        self,
        objects: list[FakeObject] | None = None,
        config: FakeCollectionConfig | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.objects = objects or []
        self.raises = raises
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.config = _FakeConfigAccessor(
            config
            or FakeCollectionConfig(
                properties=[
                    FakeProperty("faqId"),
                    FakeProperty("category"),
                    FakeProperty("question"),
                    FakeProperty("answer"),
                ],
                vector_config={"text2vec_weaviate": object()},
            )
        )
        self.query = FakeQuery(self)
        self.aggregate = _FakeAggregate(len(self.objects))


class FakeCollections:
    def __init__(self, collections: dict[str, FakeCollection]) -> None:
        self._collections = collections

    def list_all(self, simple: bool = True) -> dict[str, Any]:
        return {name: {} for name in self._collections}

    def get(self, name: str) -> FakeCollection:
        if name not in self._collections:
            raise KeyError(f"collection {name} does not exist")
        return self._collections[name]


class FakeWeaviateClient:
    def __init__(self, collections: dict[str, FakeCollection]) -> None:
        self.collections = FakeCollections(collections)
        self.closed = False

    def is_ready(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def make_object(
    faq_id: str,
    question: str,
    answer: str,
    certainty: float,
    category: str = "general",
) -> FakeObject:
    return FakeObject(
        properties={
            "faqId": faq_id,
            "category": category,
            "question": question,
            "answer": answer,
        },
        metadata=FakeMetadata(certainty=certainty, distance=(1 - certainty) * 2),
    )


@dataclass
class FakeEventContentPart:
    text: str


@dataclass
class FakeEventContent:
    parts: list[FakeEventContentPart] = field(default_factory=list)


class FakeEvent:
    def __init__(self, text: str, final: bool = True) -> None:
        self.content = FakeEventContent([FakeEventContentPart(text)])
        self._final = final

    def is_final_response(self) -> bool:
        return self._final


class FakeRunner:
    """Records the (user_id, session_id) of every turn and replays canned text."""

    def __init__(self, reply: str = "أهلاً وسهلاً بك في مخبر الخطيب.") -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def run_async(self, *, user_id: str, session_id: str, new_message: Any):
        text = ""
        parts = getattr(new_message, "parts", None) or []
        if parts:
            text = getattr(parts[0], "text", "") or ""
        self.calls.append({"user_id": user_id, "session_id": session_id, "text": text})
        yield FakeEvent(self.reply)
