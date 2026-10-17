"""The agent-facing tools must never raise and never leak internals."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from app.escalation import EscalationReason, EscalationStore
from app.rag.weaviate_store import WeaviateStore, set_store
from app.tools import (
    TOOL_FAILURE_ESCALATION_THRESHOLD,
    request_human_agent,
    search_lab_knowledge,
)
from tests.unit.conftest import FakeCollection, FakeWeaviateClient, make_object


class FakeToolContext:
    """Stand-in for ADK's ToolContext — only ``.state`` is used by our tools."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state: dict[str, Any] = state if state is not None else {}


@pytest.fixture
def tool_context() -> FakeToolContext:
    return FakeToolContext()


@pytest.fixture(autouse=True)
def _reset_store():
    yield
    set_store(None)


def install_store(settings, collection: FakeCollection) -> None:
    set_store(
        WeaviateStore(
            settings=settings,
            client_factory=lambda _c: FakeWeaviateClient({"KhatibLabFAQ": collection}),
        )
    )


def test_tool_returns_results_on_success(settings, tool_context):
    install_store(
        settings,
        FakeCollection(
            objects=[
                make_object(
                    "general_02", "أين يقع المخبر؟", "دمشق، شارع بغداد.", certainty=0.8
                )
            ]
        ),
    )

    result = search_lab_knowledge("موقع المخبر", tool_context)

    assert result["found"] is True
    assert result["count"] == 1
    assert result["results"][0]["source_id"] == "general_02"
    assert "دمشق" in result["results"][0]["answer"]
    assert tool_context.state["consecutive_tool_failures"] == 0


def test_tool_reports_not_found_with_guidance(settings, tool_context):
    install_store(
        settings,
        FakeCollection(objects=[make_object("x", "سؤال", "جواب", certainty=0.3)]),
    )

    result = search_lab_knowledge("ما هي عاصمة فرنسا", tool_context)

    assert result["found"] is False
    assert result["reason"] == "no_relevant_entry"
    assert "do not guess" in result["guidance"]
    assert tool_context.state["last_retrieval_found"] is False


def test_tool_survives_backend_failure(settings, tool_context):
    install_store(
        settings, FakeCollection(objects=[], raises=TimeoutError("deadline exceeded"))
    )

    result = search_lab_knowledge("أوقات الدوام", tool_context)

    assert result["found"] is False
    assert result["reason"] == "knowledge_base_unavailable"
    # No error type, status code, or tool internals reach the model.
    assert "Timeout" not in str(result)
    assert "weaviate" not in str(result).lower()


def test_repeated_failures_request_escalation(settings, tool_context):
    install_store(
        settings, FakeCollection(objects=[], raises=TimeoutError("deadline exceeded"))
    )

    for _ in range(TOOL_FAILURE_ESCALATION_THRESHOLD):
        result = search_lab_knowledge("أوقات الدوام", tool_context)

    assert result["escalate"] is True
    assert "request_human_agent" in result["guidance"]


def test_unconfigured_backend_is_handled(settings, tool_context):
    from app.config import Secret

    broken = replace(settings, weaviate=replace(settings.weaviate, url=""))
    set_store(WeaviateStore(settings=broken, client_factory=lambda _c: None))
    assert Secret("").is_set is False

    result = search_lab_knowledge("أوقات الدوام", tool_context)

    assert result["found"] is False
    assert result["reason"] == "knowledge_base_unavailable"


# --- escalation tool -----------------------------------------------------


def test_request_human_agent_marks_state(monkeypatch, tool_context):
    store = EscalationStore()
    monkeypatch.setattr("app.tools.get_escalation_store", lambda: store)
    tool_context.state["chat_id"] = "963991234567@c.us"

    result = request_human_agent("user_requested_human", tool_context)

    assert result["status"] == "awaiting_human"
    assert result["reason"] == "user_requested_human"
    assert store.is_awaiting("963991234567@c.us")
    assert tool_context.state["awaiting_human"] is True


def test_request_human_agent_never_claims_staff_were_notified(
    monkeypatch, tool_context
):
    monkeypatch.setattr("app.tools.get_escalation_store", lambda: EscalationStore())

    result = request_human_agent("information_unavailable", tool_context)

    assert result["human_notified"] is False
    assert "Never claim" in result["guidance"]


def test_unknown_reason_falls_back_safely(monkeypatch, tool_context):
    monkeypatch.setattr("app.tools.get_escalation_store", lambda: EscalationStore())

    result = request_human_agent("something-invented", tool_context)

    assert result["reason"] == EscalationReason.INFORMATION_UNAVAILABLE.value
    assert result["reply"]
