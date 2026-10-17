"""Read-only Weaviate Cloud retrieval for lab facts.

Design notes
------------
* **Relevance gate.** Queries use ``near_text`` because its ``certainty`` is an
  absolute similarity measure we can threshold. ``hybrid`` scores are
  rank-normalised per query — the best match of a completely off-topic question
  still scores ~1.0 — so they cannot tell "found" from "not found".
* **No writes, ever.** This module only reads schema and runs queries.
* **Never raises at the tool boundary.** Every failure becomes a
  :class:`SearchOutcome` with a status the agent can act on, and the reason is
  logged without connection details or credentials.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.config import Settings, get_settings
from app.logging_utils import get_logger
from app.rag.sanitize import sanitize_snippet

logger = get_logger(__name__)

#: Property names we prefer when building a snippet, in order.
_QUESTION_KEYS = ("question", "title", "name")
_ANSWER_KEYS = ("answer", "text", "content", "body", "description")
_ID_KEYS = ("faqId", "faq_id", "sourceId", "source_id", "id", "slug")
_CATEGORY_KEYS = ("category", "topic", "section")


class RetrievalStatus(StrEnum):
    OK = "ok"
    NO_MATCH = "no_match"
    NOT_CONFIGURED = "not_configured"
    AUTH_ERROR = "auth_error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    COLLECTION_MISSING = "collection_missing"
    COLLECTION_AMBIGUOUS = "collection_ambiguous"
    ERROR = "error"


@dataclass(frozen=True)
class LabSnippet:
    source_id: str
    category: str
    question: str
    answer: str
    certainty: float
    injection_filtered: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "certainty": round(self.certainty, 4),
            "injection_filtered": self.injection_filtered,
        }


@dataclass(frozen=True)
class SearchOutcome:
    status: RetrievalStatus
    snippets: tuple[LabSnippet, ...] = ()
    collection: str = ""
    detail: str = ""
    #: Highest certainty seen, including matches rejected by the threshold.
    best_certainty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.status is RetrievalStatus.OK and bool(self.snippets)


class CollectionAmbiguousError(RuntimeError):
    """More than one collection exists and WEAVIATE_COLLECTION is empty."""

    def __init__(self, candidates: list[str]) -> None:
        self.candidates = candidates
        super().__init__(
            "WEAVIATE_COLLECTION is empty and the cluster holds "
            f"{len(candidates)} collections; refusing to guess."
        )


def _first_str(props: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if value not in (None, "") and not isinstance(value, dict | list):
            return str(value)
    return ""


def default_client_factory(settings: WeaviateSettingsLike) -> Any:
    """Connect to Weaviate Cloud. Imported lazily so tests need no network."""
    import weaviate
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout

    timeout = settings.timeout_seconds
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.url,
        auth_credentials=Auth.api_key(settings.api_key.get()),
        # The gRPC readiness ping is unreliable from WSL2//corporate networks and
        # can time out even when queries work fine. Queries still fail loudly if
        # gRPC really is blocked.
        skip_init_checks=True,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=timeout, query=timeout, insert=timeout)
        ),
    )


# Structural alias so the factory signature reads well without a circular import.
WeaviateSettingsLike = Any


class WeaviateStore:
    """Lazily-connected, thread-safe, read-only view of the lab knowledge base."""

    def __init__(
        self,
        settings: Settings | None = None,
        client_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._factory = client_factory or default_client_factory
        self._client: Any | None = None
        self._collection_name: str | None = None
        self._query_plan: dict[str, Any] | None = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    @property
    def config(self) -> Any:
        return self._settings.weaviate

    def _client_or_connect(self) -> Any:
        if self._client is None:
            self._client = self._factory(self.config)
        return self._client

    def close(self) -> None:
        client = self._client
        self._client = None
        self._collection_name = None
        self._query_plan = None
        if client is None:
            return
        try:
            client.close()
        except Exception:
            logger.warning("weaviate client close failed")

    # -- schema discovery --------------------------------------------------

    def resolve_collection_name(self) -> str:
        """Configured name, or the single collection in the cluster."""
        if self._collection_name:
            return self._collection_name
        configured = self.config.collection.strip()
        if configured:
            self._collection_name = configured
            return configured

        client = self._client_or_connect()
        names = sorted(client.collections.list_all(simple=True).keys())
        if not names:
            raise CollectionAmbiguousError([])
        if len(names) > 1:
            raise CollectionAmbiguousError(names)
        self._collection_name = names[0]
        logger.info(
            "discovered weaviate collection",
            extra={"context": {"collection": names[0]}},
        )
        return names[0]

    def _plan_query(self, collection: Any) -> dict[str, Any]:
        """Inspect vector config once to decide how this collection can be queried."""
        if self._query_plan is not None:
            return self._query_plan

        cfg = collection.config.get()
        named = getattr(cfg, "vector_config", None) or {}
        target_vector: str | None = None
        vectorized = False

        if named:
            keys = list(named.keys())
            vectorized = True
            # A single named vector must be targeted explicitly; with several we
            # cannot pick for the user, so fall back to keyword search.
            if len(keys) == 1:
                target_vector = keys[0]
            else:
                vectorized = False
        else:
            vectorizer = getattr(cfg, "vectorizer", None)
            vectorized = bool(vectorizer) and str(vectorizer).lower() != "none"

        properties = [p.name for p in getattr(cfg, "properties", [])]
        self._query_plan = {
            "mode": (self.config.query_mode if vectorized else "bm25"),
            "target_vector": target_vector,
            "vectorized": vectorized,
            "properties": properties,
        }
        logger.info(
            "weaviate query plan",
            extra={"context": {k: str(v) for k, v in self._query_plan.items()}},
        )
        return self._query_plan

    def describe(self) -> dict[str, Any]:
        """Read-only health summary. Contains no credentials or host details."""
        collection_name = self.resolve_collection_name()
        client = self._client_or_connect()
        collection = client.collections.get(collection_name)
        plan = self._plan_query(collection)
        summary: dict[str, Any] = {
            "collection": collection_name,
            "query_mode": plan["mode"],
            "vectorized": plan["vectorized"],
            "properties": plan["properties"],
        }
        try:
            summary["object_count"] = collection.aggregate.over_all(
                total_count=True
            ).total_count
        except Exception:
            summary["object_count"] = None
        return summary

    # -- query -------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        min_certainty: float | None = None,
    ) -> SearchOutcome:
        """Retrieve lab facts for ``query``. Never raises."""
        text = (query or "").strip()
        if not text:
            return SearchOutcome(RetrievalStatus.NO_MATCH, detail="empty query")
        if not self.config.is_configured:
            return SearchOutcome(
                RetrievalStatus.NOT_CONFIGURED,
                detail="WEAVIATE_URL / WEAVIATE_API_KEY missing",
            )

        limit = limit or self.config.search_limit
        threshold = (
            self.config.min_certainty if min_certainty is None else min_certainty
        )

        with self._lock:
            try:
                return self._search_locked(text, limit, threshold)
            except CollectionAmbiguousError as exc:
                logger.error(
                    "weaviate collection ambiguous",
                    extra={"context": {"candidates": ", ".join(exc.candidates)}},
                )
                return SearchOutcome(
                    RetrievalStatus.COLLECTION_AMBIGUOUS,
                    detail=str(exc),
                    metadata={"candidates": exc.candidates},
                )
            except Exception as exc:
                status = _classify_exception(exc)
                # Drop the connection so the next call reconnects cleanly.
                if status in {RetrievalStatus.UNAVAILABLE, RetrievalStatus.TIMEOUT}:
                    self.close()
                logger.error(
                    "weaviate search failed",
                    extra={
                        "context": {
                            "status": status.value,
                            "error_type": type(exc).__name__,
                        }
                    },
                )
                return SearchOutcome(status, detail=type(exc).__name__)

    def _search_locked(self, text: str, limit: int, threshold: float) -> SearchOutcome:
        from weaviate.classes.query import MetadataQuery

        collection_name = self.resolve_collection_name()
        client = self._client_or_connect()
        collection = client.collections.get(collection_name)
        plan = self._plan_query(collection)

        metadata = MetadataQuery(distance=True, certainty=True, score=True)
        if plan["mode"] == "bm25":
            response = collection.query.bm25(
                query=text, limit=limit, return_metadata=metadata
            )
        elif plan["mode"] == "hybrid":
            kwargs: dict[str, Any] = {}
            if plan["target_vector"]:
                kwargs["target_vector"] = plan["target_vector"]
            response = collection.query.hybrid(
                query=text, limit=limit, return_metadata=metadata, **kwargs
            )
        else:
            kwargs = {}
            if plan["target_vector"]:
                kwargs["target_vector"] = plan["target_vector"]
            response = collection.query.near_text(
                query=text, limit=limit, return_metadata=metadata, **kwargs
            )

        snippets: list[LabSnippet] = []
        best = 0.0
        for obj in getattr(response, "objects", []) or []:
            certainty = _certainty_of(obj, plan["mode"])
            best = max(best, certainty)
            if certainty < threshold:
                continue
            snippet = _to_snippet(obj, certainty)
            if snippet is not None:
                snippets.append(snippet)

        if not snippets:
            return SearchOutcome(
                RetrievalStatus.NO_MATCH,
                collection=collection_name,
                best_certainty=best,
                metadata={"threshold": threshold, "mode": plan["mode"]},
            )
        return SearchOutcome(
            RetrievalStatus.OK,
            snippets=tuple(snippets),
            collection=collection_name,
            best_certainty=best,
            metadata={"threshold": threshold, "mode": plan["mode"]},
        )


def _certainty_of(obj: Any, mode: str) -> float:
    meta = getattr(obj, "metadata", None)
    if meta is None:
        return 0.0
    certainty = getattr(meta, "certainty", None)
    if isinstance(certainty, int | float):
        return float(certainty)
    distance = getattr(meta, "distance", None)
    if isinstance(distance, int | float):
        # Cosine distance in [0, 2] -> certainty in [0, 1].
        return max(0.0, 1.0 - float(distance) / 2.0)
    score = getattr(meta, "score", None)
    if isinstance(score, int | float):
        return float(score)
    return 0.0


def _to_snippet(obj: Any, certainty: float) -> LabSnippet | None:
    props = getattr(obj, "properties", None)
    if not isinstance(props, dict):
        return None
    question = sanitize_snippet(_first_str(props, _QUESTION_KEYS))
    answer = sanitize_snippet(_first_str(props, _ANSWER_KEYS))
    if not question.text and not answer.text:
        # Malformed document: nothing usable to show the customer.
        return None
    source_id = _first_str(props, _ID_KEYS) or str(getattr(obj, "uuid", ""))[:8]
    category = sanitize_snippet(_first_str(props, _CATEGORY_KEYS)).text
    return LabSnippet(
        source_id=sanitize_snippet(source_id, max_chars=64).text,
        category=category,
        question=question.text,
        answer=answer.text,
        certainty=certainty,
        injection_filtered=question.injection_filtered or answer.injection_filtered,
    )


def _classify_exception(exc: Exception) -> RetrievalStatus:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "unauthorized" in message or "401" in message or "403" in message:
        return RetrievalStatus.AUTH_ERROR
    if "authentication" in name or "authentication" in message:
        return RetrievalStatus.AUTH_ERROR
    if "timeout" in name or "timeout" in message or "deadline" in message:
        return RetrievalStatus.TIMEOUT
    if "notfound" in name or "does not exist" in message or "404" in message:
        return RetrievalStatus.COLLECTION_MISSING
    if any(k in name for k in ("connection", "unavailable", "grpc", "closed")):
        return RetrievalStatus.UNAVAILABLE
    return RetrievalStatus.ERROR


# -- process-wide store -----------------------------------------------------

_store: WeaviateStore | None = None
_store_lock = threading.Lock()


def get_store() -> WeaviateStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = WeaviateStore()
        return _store


def set_store(store: WeaviateStore | None) -> None:
    """Swap the process-wide store. Tests use this to inject a fake."""
    global _store
    with _store_lock:
        _store = store


def close_store() -> None:
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
            _store = None
