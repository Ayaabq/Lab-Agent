"""Read-only Weaviate connectivity check.

Reads schema and runs sample queries. Never writes, never prints credentials.

    uv run python scripts/check_weaviate.py
    uv run python scripts/check_weaviate.py "كم سعر تحليل فيتامين د"
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.rag.weaviate_store import (  # noqa: E402
    CollectionAmbiguousError,
    RetrievalStatus,
    get_store,
)

DEFAULT_QUERIES = [
    "وين موقع المخبر؟",
    "ما هي أوقات الدوام؟",
    "كم سعر تحليل فيتامين د",
    "هل عندكم سحب دم منزلي؟",
    "ما هي عاصمة فرنسا",  # off-topic: must be rejected by the threshold
]


def main() -> int:
    settings = get_settings()
    if not settings.weaviate.is_configured:
        print("WEAVIATE_URL / WEAVIATE_API_KEY are not set — nothing to check.")
        return 1

    store = get_store()
    try:
        try:
            info = store.describe()
        except CollectionAmbiguousError as exc:
            print(f"AMBIGUOUS: {exc}")
            print("Candidates:", ", ".join(exc.candidates))
            print("Set WEAVIATE_COLLECTION in .env to resolve.")
            return 2

        print("Connected.")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print(f"  min_certainty: {settings.weaviate.min_certainty}")

        queries = sys.argv[1:] or DEFAULT_QUERIES
        for query in queries:
            outcome = store.search(query)
            verdict = "FOUND" if outcome.found else outcome.status.value.upper()
            print(f"\n[{verdict}] {query}")
            print(f"  best_certainty={outcome.best_certainty:.4f}")
            for snippet in outcome.snippets:
                print(
                    f"  - {snippet.source_id} ({snippet.category}) "
                    f"certainty={snippet.certainty:.4f}"
                )
                print(f"      {snippet.answer[:110]}")
            if outcome.status not in {RetrievalStatus.OK, RetrievalStatus.NO_MATCH}:
                print(f"  detail: {outcome.detail}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
