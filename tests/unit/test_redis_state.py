"""Shared state in Upstash Redis, against a mocked REST endpoint.

Two things are checked throughout. First, *sharing*: two store instances over
the same fake Redis stand in for two Vercel instances, and must agree.
Second, *failing closed*: with Redis down, nothing verifies, no report image
leaves, and the customer gets the fallback reply. No test here opens a
socket, calls an LLM or reaches the patient backend.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Secret, load_settings, reset_settings_cache
from app.escalation import (
    EscalationReason,
    EscalationStore,
    RedisEscalationStore,
    mark_escalation,
    with_contact,
)
from app.patients.context import CHANNEL_WHATSAPP, ConversationScope
from app.patients.store import (
    RedisVerificationStore,
    VerifiedPatient,
    set_verification_store,
)
from app.state.backend import (
    BACKEND_MEMORY,
    BACKEND_REDIS,
    BACKEND_UNAVAILABLE,
    build_backend,
)
from app.state.crypto import StateCipher
from app.state.keysets import RedisKeySet
from app.state.locks import ChatBusy, RedisChatLocks
from app.state.sessions import RedisSessionService
from app.state.upstash import StateUnavailable, UpstashClient
from tests.unit.upstash_fake import KEY, TOKEN, URL, FakeUpstash, redis_backend

CHAT = "963991234567@c.us"
OTHER_CHAT = "963997654321@c.us"
PATIENT_ID = "pat_internal_7781"
VTOKEN = "vtok-secret-abcdef123456"


@pytest.fixture
def fake() -> FakeUpstash:
    return FakeUpstash()


def scope(chat: str = CHAT, session: str | None = None) -> ConversationScope:
    return ConversationScope(
        tenant_key="tenant0123456789",
        channel=CHANNEL_WHATSAPP,
        chat_id=chat,
        session_id=session or chat,
    )


def record(store, ttl: float = 900, identity: str = "id-a") -> VerifiedPatient:
    return VerifiedPatient(
        patient_id=PATIENT_ID,
        verification_token=VTOKEN,
        expires_at=store.now() + ttl,
        method="DATE_OF_BIRTH",
        identity=identity,
    )


def digits(chat: str) -> str:
    return re.sub(r"\D", "", chat)


# --- REST client -----------------------------------------------------------


@pytest.mark.asyncio
async def test_client_sends_bearer_token_and_unwraps_results(fake):
    client = UpstashClient(URL, Secret(TOKEN), transport=fake.transport())

    assert await client.execute("SET", "k", "v", "EX", 10) == "OK"
    assert await client.execute("GET", "k") == "v"
    assert await client.pipeline([["GET", "k"], ["EXISTS", "k"]]) == ["v", 1]
    assert fake.requests[0] == ["SET", "k", "v", "EX", "10"]


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("timeout", "redis_timeout"),
        ("connect", "redis_unreachable"),
        ("http_500", "redis_http_500"),
        ("error_reply", "redis_http_400"),
    ],
)
@pytest.mark.asyncio
async def test_every_failure_is_state_unavailable_without_secrets(fake, mode, reason):
    fake.fail = mode
    client = UpstashClient(URL, Secret(TOKEN), transport=fake.transport())

    with pytest.raises(StateUnavailable) as caught:
        await client.execute("GET", "k")

    assert caught.value.reason == reason
    assert TOKEN not in str(caught.value) and URL not in str(caught.value)


@pytest.mark.asyncio
async def test_an_error_inside_a_batch_fails_the_whole_batch(fake):
    fake.fail = "error_reply"
    client = UpstashClient(URL, Secret(TOKEN), transport=fake.transport())

    with pytest.raises(StateUnavailable):
        await client.transaction([["GET", "a"], ["GET", "b"]])


@pytest.mark.asyncio
async def test_a_wrong_token_is_rejected(fake):
    client = UpstashClient(URL, Secret("not-the-token"), transport=fake.transport())

    with pytest.raises(StateUnavailable, match="redis_http_401"):
        await client.execute("GET", "k")


# --- encryption --------------------------------------------------------------


def test_values_round_trip_and_are_unreadable_without_the_key():
    cipher = StateCipher(KEY)
    token = cipher.encrypt("مريض تجريبي")

    assert "مريض" not in token
    assert cipher.decrypt(token) == "مريض تجريبي"
    other = StateCipher(StateCipher_generate())
    assert other.decrypt(token) is None
    assert cipher.decrypt("garbage") is None


def StateCipher_generate() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def test_key_names_are_stable_keyed_and_hide_the_chat():
    a, b = StateCipher(KEY), StateCipher(StateCipher_generate())

    assert a.name(CHAT) == a.name(CHAT)
    assert a.name(CHAT) != a.name(OTHER_CHAT)
    assert a.name(CHAT) != b.name(CHAT)
    assert digits(CHAT) not in a.name(CHAT)


# --- configuration -------------------------------------------------------------

REDIS_VARS = (
    "UPSTASH_REDIS_REST_URL",
    "UPSTASH_REDIS_REST_TOKEN",
    "KV_REST_API_URL",
    "KV_REST_API_TOKEN",
    "STATE_ENCRYPTION_KEY",
    "REDIS_KEY_PREFIX",
)


@pytest.fixture
def env(monkeypatch):
    for name in REDIS_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    reset_settings_cache()
    yield monkeypatch
    reset_settings_cache()


def test_no_redis_variables_means_memory(env):
    settings = load_settings()

    assert build_backend(settings).kind == BACKEND_MEMORY
    assert not [p for p in settings.problems() if "Shared state" in p]


def test_complete_redis_variables_mean_redis(env):
    env.setenv("UPSTASH_REDIS_REST_URL", URL + "/")
    env.setenv("UPSTASH_REDIS_REST_TOKEN", TOKEN)
    env.setenv("STATE_ENCRYPTION_KEY", KEY)
    settings = load_settings()

    assert settings.redis.rest_url == URL
    assert build_backend(settings).kind == BACKEND_REDIS
    assert TOKEN in settings.secret_values() and KEY in settings.secret_values()
    assert TOKEN not in repr(settings) and KEY not in repr(settings)


def test_the_vercel_marketplace_names_are_accepted(env):
    env.setenv("KV_REST_API_URL", URL)
    env.setenv("KV_REST_API_TOKEN", TOKEN)
    env.setenv("STATE_ENCRYPTION_KEY", KEY)

    assert build_backend(load_settings()).kind == BACKEND_REDIS


@pytest.mark.parametrize(
    ("variables", "problem"),
    [
        ({"UPSTASH_REDIS_REST_URL": URL}, "UPSTASH_REDIS_REST_TOKEN"),
        (
            {"UPSTASH_REDIS_REST_URL": URL, "UPSTASH_REDIS_REST_TOKEN": TOKEN},
            "STATE_ENCRYPTION_KEY",
        ),
        (
            {
                "UPSTASH_REDIS_REST_URL": URL,
                "UPSTASH_REDIS_REST_TOKEN": TOKEN,
                "STATE_ENCRYPTION_KEY": "not-a-fernet-key",
            },
            "not a valid Fernet key",
        ),
        (
            {
                "UPSTASH_REDIS_REST_URL": "http://insecure.example",
                "UPSTASH_REDIS_REST_TOKEN": TOKEN,
                "STATE_ENCRYPTION_KEY": KEY,
            },
            "https://",
        ),
    ],
)
@pytest.mark.asyncio
async def test_incomplete_redis_configuration_fails_closed(env, variables, problem):
    for name, value in variables.items():
        env.setenv(name, value)
    settings = load_settings()
    backend = build_backend(settings)

    assert backend.kind == BACKEND_UNAVAILABLE
    assert any(problem in p for p in settings.problems())
    # Never a silent fall back to memory: every operation fails.
    with pytest.raises(StateUnavailable):
        await RedisVerificationStore(backend).get(scope())


# --- verification records --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verification_is_shared_by_two_instances(fake):
    first = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    second = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)

    await first.put(scope(), record(first))
    got = await second.get(scope())

    assert got is not None
    assert (got.patient_id, got.verification_token) == (PATIENT_ID, VTOKEN)


@pytest.mark.asyncio
async def test_nothing_identifying_is_stored_or_sent_in_clear(fake):
    store = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)

    await store.put(scope(), record(store))
    await store.record_failure(scope(), window_seconds=900)

    stored = fake.dump()
    for clear in (VTOKEN, PATIENT_ID, CHAT, digits(CHAT), "id-a"):
        assert clear not in stored


@pytest.mark.asyncio
async def test_the_redis_ttl_matches_the_verification(fake):
    store = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)

    await store.put(scope(), record(store, ttl=300))

    (key,) = [k for k in fake.data if ":verify:" in k]
    assert fake.ttl(key) == 300


@pytest.mark.asyncio
async def test_an_expired_verification_is_gone(fake):
    store = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    await store.put(scope(), record(store, ttl=300))

    fake.advance(301)

    assert await store.get(scope()) is None


@pytest.mark.asyncio
async def test_expiry_is_rechecked_even_if_the_key_outlives_it(fake):
    # A clock disagreement between instances must not extend a verification.
    store = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    await store.put(scope(), record(store, ttl=300))
    fake.expiry.clear()  # the key survives...
    fake.advance(301)  # ...but the record's own expiry has passed

    assert await store.get(scope()) is None
    assert not [k for k in fake.data if ":verify:" in k]


@pytest.mark.asyncio
async def test_a_verification_is_scoped_to_its_chat_and_session(fake):
    store = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    await store.put(scope(), record(store))

    assert await store.get(scope(OTHER_CHAT)) is None
    assert await store.get(scope(CHAT, session="a-new-session")) is None


@pytest.mark.asyncio
async def test_a_record_written_with_another_key_is_not_a_verification(fake):
    writer = RedisVerificationStore(
        redis_backend(fake, key=StateCipher_generate()), time_source=fake.clock
    )
    await writer.put(scope(), record(writer))
    # Same key name derivation is keyed too, so plant the value under ours.
    reader_backend = redis_backend(fake)
    (planted,) = [v for k, v in fake.data.items() if ":verify:" in k]
    fake.data[reader_backend.key("verify", *scope().key)] = planted

    reader = RedisVerificationStore(reader_backend, time_source=fake.clock)
    assert await reader.get(scope()) is None


@pytest.mark.asyncio
async def test_clear_and_begin_attempt_revoke_for_every_instance(fake):
    first = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    second = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    await first.put(scope(), record(first))

    await second.begin_attempt(scope())

    assert await first.get(scope()) is None


@pytest.mark.asyncio
async def test_failed_attempts_are_counted_across_instances(fake):
    first = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    second = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)

    await first.record_failure(scope(), window_seconds=900)
    await second.record_failure(scope(), window_seconds=900)
    count = await first.record_failure(scope(), window_seconds=900)

    assert count == 3
    assert await second.failure_count(scope(), window_seconds=900) == 3
    # Per chat, not per session: a new session does not reset the budget.
    assert await second.failure_count(scope(session="new"), window_seconds=900) == 3


@pytest.mark.asyncio
async def test_failed_attempts_leave_the_window(fake):
    store = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    await store.record_failure(scope(), window_seconds=900)
    fake.advance(600)
    await store.record_failure(scope(), window_seconds=900)
    fake.advance(301)

    assert await store.failure_count(scope(), window_seconds=900) == 1
    await store.reset_failures(scope())
    assert await store.failure_count(scope(), window_seconds=900) == 0


@pytest.mark.asyncio
async def test_redis_down_raises_rather_than_answering(fake):
    store = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    await store.put(scope(), record(store))
    fake.fail = "connect"

    for call in (
        store.get(scope()),
        store.put(scope(), record(store)),
        store.clear(scope()),
        store.failure_count(scope(), window_seconds=900),
        store.record_failure(scope(), window_seconds=900),
    ):
        with pytest.raises(StateUnavailable):
            await call


# --- patient tools fail closed -------------------------------------------------------


@pytest.fixture
def tools_env(monkeypatch, settings, fake):
    from app.escalation import EscalationStore
    from tests.unit.test_patient_tools import FakeBackend

    backend = FakeBackend()
    store = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    set_verification_store(store)
    monkeypatch.setattr("app.tools.get_settings", lambda: settings)
    monkeypatch.setattr("app.tools.get_khatib_client", lambda: backend)
    escalations = EscalationStore()
    monkeypatch.setattr("app.tools.get_escalation_store", lambda: escalations)
    yield backend, store
    set_verification_store(None)


@pytest.mark.asyncio
async def test_verification_through_redis_then_results(tools_env):
    from app.tools import get_my_latest_results, verify_patient
    from tests.unit.test_patient_tools import Chat

    backend, _ = tools_env
    with Chat(CHAT) as ctx:
        verified = await verify_patient("أحمد الخطيب", "1990-03-04", None, ctx)
        result = await get_my_latest_results(ctx)

    assert verified["verified"] is True
    assert result["status"] == "processing"
    assert backend.result_calls[0]["verification_token"] == "vtok-abcdef123456"


@pytest.mark.asyncio
async def test_redis_down_refuses_verification_before_the_backend(tools_env, fake):
    from app.tools import verify_patient
    from tests.unit.test_patient_tools import Chat

    backend, _ = tools_env
    fake.fail = "timeout"
    with Chat(CHAT) as ctx:
        result = await verify_patient("أحمد الخطيب", "1990-03-04", None, ctx)

    assert result == {
        "verified": False,
        "status": "service_unavailable",
        "guidance": result["guidance"],
    }
    # The attempt counter could not be read, so the backend was never asked.
    assert backend.verify_calls == []


@pytest.mark.asyncio
async def test_a_verification_that_cannot_be_stored_is_not_a_verification(
    tools_env, monkeypatch
):
    from app.tools import get_my_latest_results, verify_patient
    from tests.unit.test_patient_tools import Chat

    backend, store = tools_env

    async def broken_put(*_a: Any, **_k: Any) -> None:
        raise StateUnavailable("redis_timeout")

    monkeypatch.setattr(store, "put", broken_put)
    with Chat(CHAT) as ctx:
        result = await verify_patient("أحمد الخطيب", "1990-03-04", None, ctx)
        after = await get_my_latest_results(ctx)

    assert backend.verify_calls  # the backend said yes...
    assert result["verified"] is False  # ...but nothing was stored, so no
    assert result["status"] == "service_unavailable"
    assert after["status"] == "verification_required"
    assert backend.result_calls == []


@pytest.mark.asyncio
async def test_redis_down_refuses_results_without_asking_the_backend(tools_env, fake):
    from app.tools import get_my_latest_results, verify_patient
    from tests.unit.test_patient_tools import Chat

    backend, _ = tools_env
    with Chat(CHAT) as ctx:
        await verify_patient("أحمد الخطيب", "1990-03-04", None, ctx)
        fake.fail = "connect"
        result = await get_my_latest_results(ctx)

    assert result["status"] == "service_unavailable"
    assert "patient_message" not in result
    assert backend.result_calls == []


@pytest.mark.asyncio
async def test_the_attempt_limit_holds_across_instances(tools_env, fake, settings):
    from app.patients.client import VerifyOutcome, VerifyStatus
    from app.tools import verify_patient
    from tests.unit.test_patient_tools import Chat

    backend, _ = tools_env
    backend._verify = VerifyOutcome(status=VerifyStatus.NOT_VERIFIED)
    limit = settings.khatib.max_verification_attempts
    statuses = []
    for attempt in range(limit + 1):
        # A fresh store per attempt: each webhook on a different instance.
        set_verification_store(
            RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
        )
        with Chat(CHAT) as ctx:
            result = await verify_patient(f"اسم {attempt}", "1990-03-04", None, ctx)
        statuses.append(result["status"])

    assert statuses[-1] == "too_many_attempts"
    assert statuses.count("not_verified") == limit - 1
    assert len(backend.verify_calls) == limit


# --- key sets and locks ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedupe_claims_are_atomic_across_instances(fake):
    first = RedisKeySet(redis_backend(fake), "wa-inbound", 900)
    second = RedisKeySet(redis_backend(fake), "wa-inbound", 900)

    assert await first.seen("MSG1") is False
    assert await second.seen("MSG1") is True
    await second.forget("MSG1")
    assert await first.seen("MSG1") is False


@pytest.mark.asyncio
async def test_dedupe_entries_expire(fake):
    keys = RedisKeySet(redis_backend(fake), "wa-inbound", 900)
    await keys.remember("MSG1")

    assert await keys.contains("MSG1")
    fake.advance(901)
    assert not await keys.contains("MSG1")


@pytest.mark.asyncio
async def test_sent_report_keys_do_not_store_the_report_url(fake):
    keys = RedisKeySet(redis_backend(fake), "wa-report-sent", 900)

    await keys.remember(f"{CHAT}|https://khatib-db-ms.vercel.app/r/abc.jpg")

    assert "khatib-db-ms" not in fake.dump() and digits(CHAT) not in fake.dump()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.asyncio
async def test_a_chat_lock_excludes_other_instances_until_released(fake):
    clock = Clock()
    first = RedisChatLocks(
        redis_backend(fake), wait_seconds=5, sleep=clock.sleep, clock=clock
    )
    second = RedisChatLocks(
        redis_backend(fake), wait_seconds=5, sleep=clock.sleep, clock=clock
    )

    async with first.hold(CHAT):
        with pytest.raises(ChatBusy):
            async with second.hold(CHAT):
                pytest.fail("entered a held lock")
        # A different chat is not blocked.
        async with second.hold(OTHER_CHAT):
            pass

    async with second.hold(CHAT):
        pass


@pytest.mark.asyncio
async def test_a_lock_is_released_only_by_its_holder(fake):
    clock = Clock()
    locks = RedisChatLocks(redis_backend(fake), sleep=clock.sleep, clock=clock)
    key = redis_backend(fake).key("lock", CHAT)

    async with locks.hold(CHAT):
        # Simulate the lease lapsing and another instance taking it.
        fake.data[key] = "someone-else"
    assert fake.data.get(key) == "someone-else"


@pytest.mark.asyncio
async def test_a_crashed_holder_blocks_only_until_the_lease_ends(fake):
    clock = Clock()
    locks = RedisChatLocks(
        redis_backend(fake), lease_seconds=300, sleep=clock.sleep, clock=clock
    )
    fake.run(["SET", redis_backend(fake).key("lock", CHAT), "dead", "PX", "300000"])

    fake.advance(301)
    async with locks.hold(CHAT):
        pass


@pytest.mark.asyncio
async def test_a_failed_release_does_not_raise_after_the_reply(fake):
    clock = Clock()
    locks = RedisChatLocks(redis_backend(fake), sleep=clock.sleep, clock=clock)

    async with locks.hold(CHAT):
        fake.fail = "connect"  # Redis drops out mid-turn
    # Reaching here is the assertion: the lease will simply expire.


# --- escalation -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalations_are_shared_encrypted_and_expire(fake, monkeypatch):
    monkeypatch.setattr("app.escalation.time.time", fake.clock)
    first = RedisEscalationStore(redis_backend(fake), ttl_seconds=3600)
    second = RedisEscalationStore(redis_backend(fake), ttl_seconds=3600)

    await first.mark(CHAT, EscalationReason.URGENT_MEDICAL)

    got = await second.get(CHAT)
    assert got is not None and got.reason is EscalationReason.URGENT_MEDICAL
    assert got.human_notified is False
    assert await second.pending_count() == 1
    assert digits(CHAT) not in fake.dump()
    fake.advance(3601)
    assert await second.get(CHAT) is None
    assert await second.pending_count() == 0


@pytest.mark.asyncio
async def test_an_escalation_that_cannot_be_recorded_does_not_raise(fake):
    store = RedisEscalationStore(redis_backend(fake), ttl_seconds=3600)
    fake.fail = "connect"

    await mark_escalation(store, CHAT, EscalationReason.TOOL_FAILURE)


# --- ADK sessions ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessions_round_trip_between_instances(fake):
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions
    from google.genai import types

    first = RedisSessionService(redis_backend(fake), ttl_seconds=86400)
    second = RedisSessionService(redis_backend(fake), ttl_seconds=86400)

    session = await first.create_session(
        app_name="app", user_id=CHAT, session_id=CHAT, state={"k": 1}
    )
    call = types.Part.from_function_call(
        name="verify_patient", args={"full_name": "أحمد"}
    )
    await first.append_event(
        session,
        Event(
            author="user",
            invocation_id="i1",
            content=types.Content(
                role="user", parts=[types.Part.from_text(text="مرحبا")]
            ),
        ),
    )
    await first.append_event(
        session,
        Event(
            author="agent",
            invocation_id="i1",
            content=types.Content(role="model", parts=[call]),
            actions=EventActions(
                state_delta={
                    "awaiting_human": True,
                    "user:lang": "ar",
                    "app:version": 2,
                    "temp:scratch": "never stored",
                }
            ),
        ),
    )

    loaded = await second.get_session(app_name="app", user_id=CHAT, session_id=CHAT)

    assert loaded is not None
    assert [e.author for e in loaded.events] == ["user", "agent"]
    assert loaded.events[0].content.parts[0].text == "مرحبا"
    assert loaded.events[1].content.parts[0].function_call.name == "verify_patient"
    assert loaded.state["k"] == 1
    assert loaded.state["awaiting_human"] is True
    assert loaded.state["user:lang"] == "ar"
    assert loaded.state["app:version"] == 2
    assert "temp:scratch" not in loaded.state
    assert await second.get_user_state(app_name="app", user_id=CHAT) == {"lang": "ar"}
    # Conversation content and the chat id never reach Redis in clear.
    for clear in ("مرحبا", "أحمد", "never stored", digits(CHAT)):
        assert clear not in fake.dump()


@pytest.mark.asyncio
async def test_session_options_list_and_delete(fake):
    from google.adk.errors.already_exists_error import AlreadyExistsError
    from google.adk.events.event import Event
    from google.adk.sessions.base_session_service import GetSessionConfig

    service = RedisSessionService(redis_backend(fake), ttl_seconds=86400)
    session = await service.create_session(
        app_name="app", user_id=CHAT, session_id=CHAT
    )
    for n in range(3):
        await service.append_event(session, Event(author=f"a{n}", invocation_id="i"))

    with pytest.raises(AlreadyExistsError):
        await service.create_session(app_name="app", user_id=CHAT, session_id=CHAT)
    recent = await service.get_session(
        app_name="app",
        user_id=CHAT,
        session_id=CHAT,
        config=GetSessionConfig(num_recent_events=2),
    )
    assert [e.author for e in recent.events] == ["a1", "a2"]
    listed = await service.list_sessions(app_name="app", user_id=CHAT)
    assert [s.id for s in listed.sessions] == [CHAT]
    assert listed.sessions[0].events == []
    assert [s.id for s in (await service.list_sessions(app_name="app")).sessions] == [
        CHAT
    ]

    await service.delete_session(app_name="app", user_id=CHAT, session_id=CHAT)
    assert (
        await service.get_session(app_name="app", user_id=CHAT, session_id=CHAT) is None
    )
    assert (await service.list_sessions(app_name="app", user_id=CHAT)).sessions == []


@pytest.mark.asyncio
async def test_sessions_expire_after_the_ttl(fake):
    service = RedisSessionService(redis_backend(fake), ttl_seconds=3600)
    await service.create_session(app_name="app", user_id=CHAT, session_id=CHAT)

    fake.advance(3601)

    assert (
        await service.get_session(app_name="app", user_id=CHAT, session_id=CHAT) is None
    )
    assert all(fake.ttl(k) is not None for k in fake.data)


@pytest.mark.asyncio
async def test_a_real_adk_runner_keeps_history_across_instances(fake):
    """Two turns, two 'instances', one conversation — with a fake model."""
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import Runner
    from google.genai import types

    seen_turns: list[int] = []

    class EchoLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            seen_turns.append(len(llm_request.contents))
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="أهلا")]
                )
            )

    def runner() -> Runner:
        return Runner(
            app_name="app",
            agent=LlmAgent(name="khatib", model=EchoLlm(model="fake"), instruction="x"),
            session_service=RedisSessionService(redis_backend(fake), ttl_seconds=86400),
            auto_create_session=True,
        )

    for text in ("مرحبا", "شو الدوام؟"):
        message = types.Content(role="user", parts=[types.Part.from_text(text=text)])
        async for _ in runner().run_async(
            user_id=CHAT, session_id=CHAT, new_message=message
        ):
            pass

    # The second instance's model saw the first turn too: user, model, user.
    assert seen_turns == [1, 3]


@pytest.mark.asyncio
async def test_sessions_raise_when_redis_is_down(fake):
    service = RedisSessionService(redis_backend(fake), ttl_seconds=86400)
    fake.fail = "connect"

    with pytest.raises(StateUnavailable):
        await service.get_session(app_name="app", user_id=CHAT, session_id=CHAT)


# --- WhatsApp service over Redis ------------------------------------------------------------


def redis_service(settings, fake, *, green=None, runner=None, locks=None):
    from app.whatsapp.service import WhatsAppService
    from tests.unit.conftest import FakeRunner
    from tests.unit.test_whatsapp_service import FakeGreenClient

    backend = redis_backend(fake)
    green = green or FakeGreenClient()
    runner = runner or FakeRunner()
    service = WhatsAppService(
        settings=settings,
        green_client=green,
        escalation_store=EscalationStore(),
        dedupe=RedisKeySet(backend, "wa-inbound", 900),
        outbound_ids=RedisKeySet(backend, "wa-outbound", 900),
        sent_reports=RedisKeySet(backend, "wa-report-sent", 900),
        locks=locks or RedisChatLocks(backend),
        runner_provider=lambda: runner,
    )
    return service, green, runner


@pytest.mark.asyncio
async def test_a_redelivery_to_another_instance_is_not_answered_twice(settings, fake):
    from tests.unit.test_whatsapp_parsing import incoming

    first, green_a, runner_a = redis_service(settings, fake)
    second, green_b, runner_b = redis_service(settings, fake)

    a = await first.handle_webhook(incoming("مرحبا"), wait=True)
    b = await second.handle_webhook(incoming("مرحبا"), wait=True)

    assert a.status == "accepted" and b.reason == "duplicate"
    assert len(green_a.sent) == 1 and green_b.sent == []
    assert len(runner_a.calls) == 1 and runner_b.calls == []


@pytest.mark.asyncio
async def test_redis_down_gets_the_fallback_reply_without_the_agent(settings, fake):
    from app.whatsapp.service import FALLBACK_REPLY
    from tests.unit.test_whatsapp_parsing import incoming

    service, green, runner = redis_service(settings, fake)
    fake.fail = "timeout"

    ack = await service.handle_webhook(incoming("بدي نتيجتي"), wait=True)

    assert ack.status == "accepted" and ack.reason == "state_unavailable"
    assert [text for _, text in green.sent] == [with_contact(FALLBACK_REPLY)]
    assert runner.calls == []


@pytest.mark.asyncio
async def test_a_busy_chat_releases_its_claim_for_the_redelivery(settings, fake):
    from tests.unit.test_whatsapp_parsing import incoming

    clock = Clock()
    busy = RedisChatLocks(
        redis_backend(fake), wait_seconds=1, sleep=clock.sleep, clock=clock
    )
    service, green, runner = redis_service(settings, fake, locks=busy)

    async with RedisChatLocks(redis_backend(fake)).hold(CHAT):  # another instance
        ack = await service.handle_webhook(incoming("مرحبا"), wait=True)
    assert ack.status == "retry" and green.sent == []

    redelivered = await service.handle_webhook(incoming("مرحبا"), wait=True)
    assert redelivered.status == "accepted"
    assert len(green.sent) == 1 and len(runner.calls) == 1


def test_a_busy_chat_answers_503_so_green_api_redelivers(settings, monkeypatch):
    from app.api.routes import router
    from app.whatsapp.service import WebhookAck, set_whatsapp_service

    class Busy:
        async def handle_webhook(self, payload, *, wait=False):
            return WebhookAck("retry", "chat_busy", "MSG1")

    token = "route-token-abcdef"
    secured = replace(
        settings, green_api=replace(settings.green_api, webhook_token=Secret(token))
    )
    monkeypatch.setattr("app.api.routes.get_settings", lambda: secured)
    set_whatsapp_service(Busy())  # type: ignore[arg-type]
    api = FastAPI()
    api.include_router(router)
    try:
        response = TestClient(api).post(
            "/webhooks/green-api",
            json={"typeWebhook": "incomingMessageReceived"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        set_whatsapp_service(None)

    assert response.status_code == 503
    # Authorization still comes first: no token, no 503, just 401.
    set_whatsapp_service(Busy())  # type: ignore[arg-type]
    try:
        anonymous = TestClient(api).post("/webhooks/green-api", json={})
    finally:
        set_whatsapp_service(None)
    assert anonymous.status_code == 401


# --- report images over Redis ---------------------------------------------------------------


@pytest.fixture
def staged(settings, fake):
    """A verified chat with a report staged, all in shared state."""
    from app.patients.release import PendingRelease, ReleaseStore, set_release_store

    verifications = RedisVerificationStore(redis_backend(fake), time_source=fake.clock)
    releases = ReleaseStore()
    set_verification_store(verifications)
    set_release_store(releases)
    s = ConversationScope(
        tenant_key=settings.khatib.tenant_key,
        channel=CHANNEL_WHATSAPP,
        chat_id=CHAT,
        session_id=CHAT,
    )
    releases.stage(
        s,
        PendingRelease(
            chat_id=CHAT,
            patient_message="رسالة المخبر",
            image_url="https://khatib-db-ms.vercel.app/r/abc.jpg",
            file_name="lab-report.jpg",
            result_id="res-1",
            patient_id=PATIENT_ID,
            identity="id-a",
            created_at=releases.now(),
        ),
    )
    yield verifications, s
    set_verification_store(None)
    set_release_store(None)


class FileGreen:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.files: list[str] = []

    async def send_text(self, chat_id, text):
        from app.whatsapp.green_api import SendResult

        self.sent.append((chat_id, text))
        return SendResult(
            True, False, chat_id, "ok", provider_message_id=f"t{len(self.sent)}"
        )

    async def send_file_by_url(self, *, chat_id, url_file, file_name, caption):
        from app.whatsapp.green_api import SendResult

        self.files.append(file_name)
        return SendResult(True, False, chat_id, "ok", provider_message_id="f1")


@pytest.mark.asyncio
async def test_a_report_goes_out_when_shared_state_confirms_it(settings, fake, staged):
    from tests.unit.test_whatsapp_parsing import incoming

    verifications, s = staged
    await verifications.put(s, record(verifications))
    service, green, _ = redis_service(settings, fake, green=FileGreen())

    await service.handle_webhook(incoming("نتيجتي"), wait=True)

    assert green.files == ["lab-report.jpg"]


@pytest.mark.asyncio
async def test_no_report_leaves_when_verification_cannot_be_rechecked(
    settings, fake, staged, monkeypatch
):
    from app.whatsapp.service import REPORT_SEND_FAILED_REPLY
    from tests.unit.test_whatsapp_parsing import incoming

    verifications, s = staged
    await verifications.put(s, record(verifications))

    async def down(*_a: Any, **_k: Any) -> None:
        raise StateUnavailable("redis_timeout")

    monkeypatch.setattr(verifications, "get", down)
    service, green, _ = redis_service(settings, fake, green=FileGreen())

    await service.handle_webhook(incoming("نتيجتي"), wait=True)

    assert green.files == []
    texts = [t for _, t in green.sent]
    assert texts[-1] == with_contact(REPORT_SEND_FAILED_REPLY)
    # Not even the laboratory's message about the result goes out unverified.
    assert "رسالة المخبر" not in texts


@pytest.mark.asyncio
async def test_no_report_leaves_when_the_duplicate_guard_is_down(
    settings, fake, staged
):
    from app.whatsapp.service import REPORT_SEND_FAILED_REPLY
    from tests.unit.test_whatsapp_parsing import incoming

    verifications, s = staged
    await verifications.put(s, record(verifications))

    class DownSet:
        async def seen(self, key):
            return False

        async def remember(self, key):
            return None

        async def contains(self, key):
            raise StateUnavailable("redis_timeout")

        async def forget(self, key):
            return None

        def count(self):
            return None

    from app.whatsapp.service import WhatsAppService
    from tests.unit.conftest import FakeRunner

    backend = redis_backend(fake)
    green = FileGreen()
    service = WhatsAppService(
        settings=settings,
        green_client=green,
        escalation_store=EscalationStore(),
        dedupe=RedisKeySet(backend, "wa-inbound", 900),
        outbound_ids=RedisKeySet(backend, "wa-outbound", 900),
        sent_reports=DownSet(),
        locks=RedisChatLocks(backend),
        runner_provider=FakeRunner,
    )

    await service.handle_webhook(incoming("نتيجتي"), wait=True)

    assert green.files == []
    assert [t for _, t in green.sent][-1] == with_contact(REPORT_SEND_FAILED_REPLY)


# --- wiring ------------------------------------------------------------------------------------


def test_the_redis_backend_provides_the_adk_session_service(monkeypatch, env):
    from app.app_utils import services
    from app.state.backend import reset_state_backend

    env.setenv("UPSTASH_REDIS_REST_URL", URL)
    env.setenv("UPSTASH_REDIS_REST_TOKEN", TOKEN)
    env.setenv("STATE_ENCRYPTION_KEY", KEY)
    monkeypatch.delenv("SESSION_SERVICE_URI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", raising=False)
    reset_state_backend()
    services.get_session_service.cache_clear()
    try:
        assert isinstance(services.get_session_service(), RedisSessionService)
    finally:
        reset_settings_cache()
        reset_state_backend()
        services.get_session_service.cache_clear()


def test_the_lock_lease_covers_the_whole_function_lifetime():
    import json as _json
    from pathlib import Path

    from app.state.locks import LOCK_LEASE_SECONDS, LOCK_WAIT_SECONDS
    from app.whatsapp.service import AGENT_TURN_TIMEOUT_SECONDS

    root = Path(__file__).resolve().parents[2]
    vercel = _json.loads((root / "vercel.json").read_text())
    max_duration = vercel["functions"]["app/fast_api_app.py"]["maxDuration"]
    send = load_settings().green_api.timeout_seconds

    # A lease shorter than the function could lapse while its turn still runs.
    assert LOCK_LEASE_SECONDS >= max_duration
    # Waiting for the lock and then the worst-case turn still fit.
    assert LOCK_WAIT_SECONDS + AGENT_TURN_TIMEOUT_SECONDS + 5 * send < max_duration
