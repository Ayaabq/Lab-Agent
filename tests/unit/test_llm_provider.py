"""LLM provider selection, and ADK function calling through OpenRouter's adapter.

Offline by construction: an autouse guard refuses every socket connection that
is not loopback, so no test here can reach OpenRouter, Gemini or anything
else, whatever the local ``.env`` says. The OpenRouter round trips run against
a fake LiteLLM client, or a throwaway HTTP server on 127.0.0.1.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.config import (
    OpenRouterSettings,
    Secret,
    load_settings,
    reset_settings_cache,
)
from app.llm import ModelUnavailableError, UnavailableModel, build_model, describe
from app.rag.weaviate_store import LabSnippet, RetrievalStatus, SearchOutcome

KEY = "sk-or-v1-test-key-0123456789abcdef"
QWEN = "qwen/qwen3.5-35b-a3b"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly on any connection that would leave this machine."""
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK:
            raise AssertionError(f"test tried to open a network connection to {host}")
        return real_connect(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in LOOPBACK:
            raise AssertionError(f"test tried to resolve {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


def openrouter(settings, **overrides):
    cfg = OpenRouterSettings(
        model=overrides.pop("model", QWEN),
        api_key=Secret(overrides.pop("api_key", KEY)),
        **overrides,
    )
    return replace(settings, llm_provider="openrouter", openrouter=cfg)


# --- configuration ---------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_API_BASE",
        "OPENROUTER_MAX_TOKENS",
        "OPENROUTER_TIMEOUT_SECONDS",
        "OPENROUTER_REASONING_EFFORT",
        "OPENROUTER_DATA_COLLECTION",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    reset_settings_cache()
    yield monkeypatch
    reset_settings_cache()


def test_gemini_stays_the_default_provider(clean_env):
    settings = load_settings()

    assert settings.llm_provider == "gemini"
    assert describe(settings)["provider"] == "gemini"


def test_openrouter_settings_load_from_the_environment(clean_env):
    clean_env.setenv("LLM_PROVIDER", "OpenRouter")
    clean_env.setenv("OPENROUTER_API_KEY", KEY)
    clean_env.setenv("OPENROUTER_MODEL", QWEN)
    clean_env.setenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1/")

    settings = load_settings()

    assert settings.llm_provider == "openrouter"
    assert settings.openrouter.model == QWEN
    assert settings.openrouter.api_base == "https://openrouter.ai/api/v1"
    assert settings.openrouter.is_configured
    assert settings.llm_problems() == []


def test_openrouter_defaults_are_the_cheap_and_private_ones(clean_env):
    cfg = load_settings().openrouter

    assert cfg.model == ""  # never a model the operator did not name
    assert cfg.api_base == "https://openrouter.ai/api/v1"
    assert cfg.reasoning_effort == "none"
    assert cfg.data_collection == "deny"
    assert cfg.max_tokens == 1024


def test_openrouter_does_not_need_a_gemini_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openrouter")
    clean_env.setenv("OPENROUTER_API_KEY", KEY)
    clean_env.setenv("OPENROUTER_MODEL", QWEN)

    assert not any("GOOGLE_API_KEY" in p for p in load_settings().problems())


@pytest.mark.parametrize(
    ("unset", "named"),
    [
        ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
        ("OPENROUTER_MODEL", "OPENROUTER_MODEL"),
    ],
)
def test_a_missing_openrouter_setting_is_named(clean_env, unset, named):
    clean_env.setenv("LLM_PROVIDER", "openrouter")
    clean_env.setenv("OPENROUTER_API_KEY", KEY)
    clean_env.setenv("OPENROUTER_MODEL", QWEN)
    clean_env.delenv(unset)

    problems = load_settings().problems()

    assert any(named in p for p in problems)
    assert not any(KEY in p for p in problems)


def test_an_unknown_provider_is_reported_not_replaced(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")

    problems = load_settings().llm_problems()

    assert len(problems) == 1
    assert "LLM_PROVIDER" in problems[0]


@pytest.mark.parametrize(
    "base",
    ["http://openrouter.ai/api/v1", "ftp://openrouter.ai", "openrouter.ai/api/v1"],
)
def test_the_key_is_never_sent_over_plain_http(clean_env, base):
    clean_env.setenv("LLM_PROVIDER", "openrouter")
    clean_env.setenv("OPENROUTER_API_KEY", KEY)
    clean_env.setenv("OPENROUTER_MODEL", QWEN)
    clean_env.setenv("OPENROUTER_API_BASE", base)

    settings = load_settings()

    assert not settings.openrouter.is_configured
    assert any("OPENROUTER_API_BASE" in p for p in settings.problems())


def test_the_openrouter_key_is_redacted_from_logs(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", KEY)

    settings = load_settings()

    assert KEY in settings.secret_values()
    assert KEY not in repr(settings.openrouter)


# --- adapter construction ----------------------------------------------------


def test_gemini_builds_the_same_adapter_as_before(settings):
    model = build_model(settings)

    assert isinstance(model, Gemini)
    assert model.model == settings.gemini.model
    assert model.retry_options.attempts == 5


def test_gemini_without_a_key_still_builds_gemini(settings):
    # Unchanged behaviour: the Gemini adapter fails at call time, as before.
    unset = replace(settings, gemini=replace(settings.gemini, api_key=Secret("")))

    assert isinstance(build_model(unset), Gemini)


def test_openrouter_builds_adks_litellm_adapter(settings):
    from google.adk.models.lite_llm import LiteLlm

    model = build_model(openrouter(settings))

    assert isinstance(model, LiteLlm)
    assert model.model == f"openrouter/{QWEN}"
    args = model._additional_args
    assert args["api_base"] == "https://openrouter.ai/api/v1"
    assert args["max_tokens"] == 1024
    assert args["num_retries"] == 0
    assert args["max_retries"] == 0
    assert args["extra_body"] == {
        "provider": {"require_parameters": True, "data_collection": "deny"},
        "reasoning": {"effort": "none"},
    }


def test_the_adapter_never_renders_the_key(settings):
    model = build_model(openrouter(settings))

    assert KEY not in repr(model)
    assert KEY not in json.dumps(model.model_dump(mode="json"), default=str)


def test_litellm_is_told_not_to_fetch_its_price_map(settings, monkeypatch):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

    build_model(openrouter(settings))

    import os

    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


@pytest.mark.parametrize(
    "broken",
    [
        {"api_key": ""},
        {"model": ""},
        {"api_base": "http://openrouter.ai/api/v1"},
    ],
)
@pytest.mark.asyncio
async def test_an_unconfigured_openrouter_fails_before_any_request(settings, broken):
    model = build_model(openrouter(settings, **broken))

    assert isinstance(model, UnavailableModel)
    with pytest.raises(ModelUnavailableError) as excinfo:
        async for _ in model.generate_content_async(None):  # type: ignore[arg-type]
            pass
    assert KEY not in str(excinfo.value)


def test_an_unknown_provider_never_falls_back_to_gemini(settings):
    model = build_model(replace(settings, llm_provider="anthropic"))

    assert isinstance(model, UnavailableModel)
    assert not isinstance(model, Gemini)


def test_a_missing_litellm_install_degrades_instead_of_crashing(settings, monkeypatch):
    monkeypatch.setattr("app.llm._prepare_litellm", lambda: False)

    assert isinstance(build_model(openrouter(settings)), UnavailableModel)


@pytest.mark.asyncio
async def test_an_unavailable_model_becomes_the_ordinary_fallback_reply(settings):
    """The WhatsApp customer gets the usual apology, not an error or silence."""
    from app.escalation import EscalationStore
    from app.whatsapp.dedupe import TTLDedupeCache
    from app.whatsapp.service import FALLBACK_REPLY, WhatsAppService
    from tests.unit.test_whatsapp_parsing import incoming
    from tests.unit.test_whatsapp_service import FakeGreenClient

    runner = _runner(build_model(openrouter(settings, api_key="")))
    green = FakeGreenClient()
    service = WhatsAppService(
        settings=settings,
        green_client=green,
        escalation_store=EscalationStore(),
        dedupe=TTLDedupeCache(ttl_seconds=60, max_entries=10),
        runner_provider=lambda: runner,
    )

    await service.handle_webhook(incoming("وين المخبر؟"), wait=True)

    assert green.sent == [("963991234567@c.us", FALLBACK_REPLY)]


# --- function calling through LiteLLM, with a fake client --------------------


def _runner(model) -> Runner:
    from app.agent import root_agent
    from app.prompts import SYSTEM_INSTRUCTION

    # The production tool list and instruction, on the adapter under test.
    agent = Agent(
        name="khatib_lab_support",
        model=model,
        instruction=SYSTEM_INSTRUCTION,
        tools=list(root_agent.tools),
    )
    return Runner(
        app_name="app",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )


async def _final_text(runner: Runner, text: str) -> str:
    out = []
    async for event in runner.run_async(
        user_id="963991234567@c.us",
        session_id="963991234567@c.us",
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            out.extend(p.text for p in event.content.parts if p.text)
    return "".join(out)


def _tool_call(name: str, args: dict[str, Any], call_id: str = "call_1"):
    import litellm

    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    ],
                },
            }
        ]
    )


def _text(content: str):
    import litellm

    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ]
    )


class FakeLiteLLMClient:
    """Replays canned OpenAI-format responses and records every request."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def acompletion(self, model, messages, tools, **kwargs):
        self.calls.append(
            {"model": model, "messages": messages, "tools": tools, **kwargs}
        )
        return self.responses.pop(0)


class FakeStore:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, **_: Any) -> SearchOutcome:
        self.queries.append(query)
        return SearchOutcome(
            RetrievalStatus.OK,
            snippets=(
                LabSnippet(
                    source_id="faq-hours",
                    category="hours",
                    question="أوقات الدوام",
                    answer="FAKE-HOURS-ANSWER",
                    certainty=0.9,
                    injection_filtered=False,
                ),
            ),
            collection="KhatibLabFAQ",
            best_certainty=0.9,
        )


@pytest.fixture
def fake_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr("app.tools.get_store", lambda: store)
    return store


def _with_client(settings, client):
    model = build_model(openrouter(settings))
    model.llm_client = client
    return model


def test_all_four_tools_are_offered_in_openai_format(settings, fake_store):
    import asyncio

    client = FakeLiteLLMClient([_text("أهلين")])
    asyncio.run(_final_text(_runner(_with_client(settings, client)), "مرحبا"))

    tools = {t["function"]["name"]: t["function"] for t in client.calls[0]["tools"]}
    assert set(tools) == {
        "search_lab_knowledge",
        "request_human_agent",
        "verify_patient",
        "get_my_latest_results",
    }
    # The trusted context is injected by ADK, never exposed as a parameter.
    for fn in tools.values():
        assert "tool_context" not in json.dumps(fn)
    # Identity arguments: none on the results tool, and the phone suffix is a
    # string so "0123" cannot become 123.
    assert not tools["get_my_latest_results"].get("parameters", {}).get("properties")
    phone = tools["verify_patient"]["parameters"]["properties"]["phone_last4"]
    assert "string" in json.dumps(phone)


@pytest.mark.asyncio
async def test_a_tool_call_and_its_result_round_trip_through_litellm(
    settings, fake_store
):
    client = FakeLiteLLMClient(
        [
            _tool_call("search_lab_knowledge", {"query": "أوقات الدوام"}),
            _text("الدوام حسب المعلومات المسجلة عنا."),
        ]
    )

    reply = await _final_text(
        _runner(_with_client(settings, client)), "شو أوقات الدوام؟"
    )

    assert reply == "الدوام حسب المعلومات المسجلة عنا."
    # The Arabic argument reached the real tool intact.
    assert fake_store.queries == ["أوقات الدوام"]
    # The second request carries the tool result, tied to the same call id.
    assert len(client.calls) == 2
    second = client.calls[1]["messages"]
    tool_messages = [m for m in second if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "FAKE-HOURS-ANSWER" in str(tool_messages[0]["content"])
    assistant = [m for m in second if m.get("role") == "assistant"]
    assert assistant and assistant[0].get("tool_calls")


@pytest.mark.asyncio
async def test_patient_results_stay_gated_on_verification(
    settings, fake_store, monkeypatch
):
    """A model asking for results before verify_patient gets nothing to read."""
    from app.patients.context import (
        CHANNEL_WHATSAPP,
        ChannelIdentity,
        bind_channel_identity,
        reset_channel_identity,
    )

    # A configured (fake) backend, so the tool reaches its verification gate.
    # Any attempt to call it would trip the network guard.
    monkeypatch.setattr("app.tools.get_settings", lambda: settings)
    client = FakeLiteLLMClient(
        [_tool_call("get_my_latest_results", {}), _text("لازم نتأكد من هويتك أولًا.")]
    )
    token = bind_channel_identity(
        ChannelIdentity(channel=CHANNEL_WHATSAPP, chat_id="963991234567@c.us")
    )
    try:
        await _final_text(_runner(_with_client(settings, client)), "بدي نتيجتي")
    finally:
        reset_channel_identity(token)

    tool_messages = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    result = json.loads(tool_messages[0]["content"])
    assert result["verified"] is False
    assert "patient_message" not in result


@pytest.mark.asyncio
async def test_the_request_carries_the_cost_and_privacy_settings(settings, fake_store):
    client = FakeLiteLLMClient([_text("أهلين")])

    await _final_text(_runner(_with_client(settings, client)), "مرحبا")

    call = client.calls[0]
    assert call["model"] == f"openrouter/{QWEN}"
    assert call["max_tokens"] == 1024
    assert call["num_retries"] == 0
    assert call["extra_body"]["provider"]["require_parameters"] is True
    assert call["extra_body"]["reasoning"] == {"effort": "none"}
    # The system instruction is sent as the system message.
    assert call["messages"][0]["role"] == "system"
    assert "search_lab_knowledge" in call["messages"][0]["content"]


# --- the real LiteLLM + OpenAI SDK stack, against a loopback server ----------


class _FakeOpenRouter(ThreadingHTTPServer):
    def __init__(self, replies: list[tuple[int, dict[str, Any]]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        super().__init__(("127.0.0.1", 0), _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _FakeOpenRouter

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        self.server.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("authorization", ""),
                "json": json.loads(body),
            }
        )
        status, payload = self.server.replies.pop(0)
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


def _completion(message: dict[str, Any], finish: str) -> dict[str, Any]:
    return {
        "id": "gen-test",
        "object": "chat.completion",
        "created": 0,
        "model": QWEN,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def fake_openrouter(request):
    server = _FakeOpenRouter(request.param)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _loopback(settings, server):
    host, port = server.server_address[:2]
    return build_model(openrouter(settings, api_base=f"http://{host}:{port}/api/v1"))


@pytest.mark.parametrize(
    "fake_openrouter",
    [
        [
            (
                200,
                _completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "type": "function",
                                "function": {
                                    "name": "search_lab_knowledge",
                                    "arguments": '{"query": "أوقات الدوام"}',
                                },
                            }
                        ],
                    },
                    "tool_calls",
                ),
            ),
            (200, _completion({"role": "assistant", "content": "تمام"}, "stop")),
        ]
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_the_wire_request_is_what_openrouter_expects(
    settings, fake_store, fake_openrouter
):
    reply = await _final_text(_runner(_loopback(settings, fake_openrouter)), "الدوام؟")

    assert reply == "تمام"
    assert len(fake_openrouter.requests) == 2
    first, second = (r["json"] for r in fake_openrouter.requests)
    assert fake_openrouter.requests[0]["path"] == "/api/v1/chat/completions"
    assert fake_openrouter.requests[0]["authorization"] == f"Bearer {KEY}"
    # LiteLLM strips its routing prefix: OpenRouter sees its own model id.
    assert first["model"] == QWEN
    assert first["max_tokens"] == 1024
    assert first["provider"] == {"require_parameters": True, "data_collection": "deny"}
    assert first["reasoning"] == {"effort": "none"}
    assert {t["function"]["name"] for t in first["tools"]} >= {"search_lab_knowledge"}
    tool_msg = [m for m in second["messages"] if m["role"] == "tool"]
    assert tool_msg[0]["tool_call_id"] == "call_9"
    assert "FAKE-HOURS-ANSWER" in tool_msg[0]["content"]


@pytest.mark.parametrize(
    "fake_openrouter",
    [[(500, {"error": {"message": "upstream down", "code": 500}})] * 5],
    indirect=True,
)
@pytest.mark.asyncio
async def test_a_failed_call_is_not_retried_and_does_not_leak_the_key(
    settings, fake_store, fake_openrouter
):
    with pytest.raises(Exception) as excinfo:
        await _final_text(_runner(_loopback(settings, fake_openrouter)), "مرحبا")

    # Exactly one request: no client-side retry can bill the same turn twice.
    assert len(fake_openrouter.requests) == 1
    assert KEY not in str(excinfo.value)


def test_importing_litellm_opens_no_connection():
    """With the cost-map flag set, LiteLLM's import stays offline."""
    probe = (
        "import os, socket\n"
        "attempts = []\n"
        "def deny(*a, **k):\n"
        "    attempts.append(a)\n"
        "    raise OSError('blocked')\n"
        "socket.socket.connect = deny\n"
        "socket.getaddrinfo = deny\n"
        "os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'\n"
        "import litellm\n"
        "print(len(attempts))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.strip().splitlines()[-1] == "0"


def test_an_unavailable_provider_is_logged_without_secrets(settings, caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="app.llm"):
        build_model(openrouter(settings, model=""))

    messages = [r.getMessage() for r in caplog.records]
    assert any("LLM provider unavailable" in m for m in messages)
    assert KEY not in caplog.text
