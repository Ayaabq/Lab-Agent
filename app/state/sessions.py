"""ADK sessions in Upstash Redis.

Behaves like ADK's ``InMemorySessionService`` — same state scoping (``app:``,
``user:``, session, and ``temp:`` never stored), same event serialisation as
ADK's database service — but survives across serverless instances.

Layout, every key name hashed (see :meth:`StateBackend.key`), every value
encrypted:

* ``sess-meta``   — the session's ids and ``last_update_time``.
* ``sess-state``  — hash: state key -> JSON value.
* ``sess-events`` — list of event JSON, in order.
* ``user-state`` / ``app-state`` — hashes, for ``user:`` and ``app:`` state.
* ``sess-index``  — sorted sets of session handles by last update, per user
  and per app, for ``list_sessions``.

Every key expires ``SESSION_TTL_SECONDS`` after the last write. Conversation
history is patient data; it is kept as long as a conversation plausibly
continues and no longer.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.platform import time as platform_time
from google.adk.platform import uuid as platform_uuid
from google.adk.sessions import _session_util
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.adk.sessions.state import State

from app.state.backend import StateBackend
from app.state.upstash import Command, StateUnavailable

#: Marks a value that is missing or was not written with our key.
_UNREADABLE = object()


class RedisSessionService(BaseSessionService):
    def __init__(self, backend: StateBackend, *, ttl_seconds: int) -> None:
        self._b = backend
        self._ttl = max(60, int(ttl_seconds))

    # -- keys ------------------------------------------------------------

    def _handle(self, app_name: str, user_id: str, session_id: str) -> str:
        return self._b.cipher.name(app_name, user_id, session_id)  # type: ignore[union-attr]

    def _keys(self, app_name: str, user_id: str, session_id: str) -> dict[str, str]:
        handle = self._handle(app_name, user_id, session_id)
        p = self._b.prefix
        return {
            "meta": f"{p}:sess-meta:{handle}",
            "state": f"{p}:sess-state:{handle}",
            "events": f"{p}:sess-events:{handle}",
        }

    def _user_state_key(self, app_name: str, user_id: str) -> str:
        return self._b.key("user-state", app_name, user_id)

    def _app_state_key(self, app_name: str) -> str:
        return self._b.key("app-state", app_name)

    def _user_index(self, app_name: str, user_id: str) -> str:
        return self._b.key("sess-index", app_name, user_id)

    def _app_index(self, app_name: str) -> str:
        return self._b.key("sess-index", app_name)

    # -- encoding --------------------------------------------------------

    def _enc(self, value: Any) -> str:
        return self._b.cipher.encrypt(json.dumps(value, ensure_ascii=False))  # type: ignore[union-attr]

    def _dec(self, token: Any) -> Any:
        """Decoded value, or :data:`_UNREADABLE` when absent or not ours."""
        if not isinstance(token, str):
            return _UNREADABLE
        plain = self._b.cipher.decrypt(token)  # type: ignore[union-attr]
        return _UNREADABLE if plain is None else json.loads(plain)

    def _hash_fields(self, mapping: dict[str, Any]) -> list[str]:
        flat: list[str] = []
        for key, value in mapping.items():
            flat.extend([key, self._enc(value)])
        return flat

    def _read_hash(self, raw: Any) -> dict[str, Any]:
        # HGETALL over REST is a flat [field, value, field, value, ...] list.
        if not isinstance(raw, list):
            return {}
        out: dict[str, Any] = {}
        for field, token in zip(raw[0::2], raw[1::2], strict=False):
            value = self._dec(token)
            if value is not _UNREADABLE:
                out[field] = value
        return out

    # -- writes shared by create and append ------------------------------

    def _state_writes(
        self, app_name: str, user_id: str, keys: dict[str, str], delta: dict[str, Any]
    ) -> list[Command]:
        deltas = _session_util.extract_json_safe_state_delta(delta)
        commands: list[Command] = []
        if deltas["app"]:
            k = self._app_state_key(app_name)
            commands += [["HSET", k, *self._hash_fields(deltas["app"])]]
            commands += [["EXPIRE", k, self._ttl]]
        if deltas["user"]:
            k = self._user_state_key(app_name, user_id)
            commands += [["HSET", k, *self._hash_fields(deltas["user"])]]
            commands += [["EXPIRE", k, self._ttl]]
        if deltas["session"]:
            commands += [["HSET", keys["state"], *self._hash_fields(deltas["session"])]]
        return commands

    def _touch(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        keys: dict[str, str],
        last_update_time: float,
    ) -> list[Command]:
        handle = self._handle(app_name, user_id, session_id)
        meta = {
            "app_name": app_name,
            "user_id": user_id,
            "id": session_id,
            "last_update_time": last_update_time,
        }
        user_index = self._user_index(app_name, user_id)
        app_index = self._app_index(app_name)
        return [
            ["SET", keys["meta"], self._enc(meta), "EX", self._ttl],
            ["EXPIRE", keys["state"], self._ttl],
            ["EXPIRE", keys["events"], self._ttl],
            ["ZADD", user_index, last_update_time, handle],
            ["EXPIRE", user_index, self._ttl],
            ["ZADD", app_index, last_update_time, handle],
            ["EXPIRE", app_index, self._ttl],
        ]

    # -- BaseSessionService ----------------------------------------------

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        session_id = (
            session_id.strip()
            if session_id and session_id.strip()
            else platform_uuid.new_uuid()
        )
        keys = self._keys(app_name, user_id, session_id)
        now = platform_time.get_time()
        meta = {
            "app_name": app_name,
            "user_id": user_id,
            "id": session_id,
            "last_update_time": now,
        }
        # NX is the existence check and the creation in one step, so two
        # instances racing to create the same chat's session cannot both win.
        created = await self._b.client.execute(
            "SET", keys["meta"], self._enc(meta), "NX", "EX", self._ttl
        )
        if created is None:
            raise AlreadyExistsError(f"Session with id {session_id} already exists.")
        commands = self._state_writes(app_name, user_id, keys, state or {})
        commands += self._touch(app_name, user_id, session_id, keys, now)
        await self._b.client.transaction(commands)
        session = await self.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        if session is None:
            # Written a moment ago and already unreadable: treat as an outage.
            raise StateUnavailable("redis_bad_response")
        return session

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: GetSessionConfig | None = None,
    ) -> Session | None:
        keys = self._keys(app_name, user_id, session_id)
        start = 0
        if config and config.num_recent_events is not None:
            if config.num_recent_events == 0:
                start = None  # type: ignore[assignment]
            else:
                start = -config.num_recent_events
        commands: list[Command] = [
            ["GET", keys["meta"]],
            ["HGETALL", keys["state"]],
            ["HGETALL", self._user_state_key(app_name, user_id)],
            ["HGETALL", self._app_state_key(app_name)],
        ]
        if start is not None:
            commands.append(["LRANGE", keys["events"], start, -1])
        results = await self._b.client.pipeline(commands)
        meta = self._dec(results[0])
        if not isinstance(meta, dict):
            return None
        events: list[Event] = []
        if start is not None:
            for token in results[4] or []:
                data = self._dec(token)
                if isinstance(data, dict):
                    events.append(Event.model_validate(data))
        if config and config.after_timestamp:
            events = [e for e in events if e.timestamp >= config.after_timestamp]
        state = self._read_hash(results[1])
        for key, value in self._read_hash(results[3]).items():
            state[State.APP_PREFIX + key] = value
        for key, value in self._read_hash(results[2]).items():
            state[State.USER_PREFIX + key] = value
        return Session(
            app_name=app_name,
            user_id=user_id,
            id=session_id,
            state=state,
            events=events,
            last_update_time=float(meta.get("last_update_time") or 0.0),
        )

    async def list_sessions(
        self, *, app_name: str, user_id: str | None = None
    ) -> ListSessionsResponse:
        index = (
            self._app_index(app_name)
            if user_id is None
            else self._user_index(app_name, user_id)
        )
        handles = await self._b.client.execute("ZRANGE", index, 0, -1)
        if not handles:
            return ListSessionsResponse()
        p = self._b.prefix
        metas = await self._b.client.pipeline(
            [["GET", f"{p}:sess-meta:{h}"] for h in handles]
        )
        sessions: list[Session] = []
        expired: list[str] = []
        for handle, token in zip(handles, metas, strict=True):
            meta = self._dec(token)
            if not isinstance(meta, dict):
                expired.append(handle)
                continue
            session = await self.get_session(
                app_name=app_name,
                user_id=meta["user_id"],
                session_id=meta["id"],
                config=GetSessionConfig(num_recent_events=0),
            )
            if session is not None:
                sessions.append(session)
        if expired:
            await self._b.client.execute("ZREM", index, *expired)
        sessions.sort(key=lambda s: (s.last_update_time, s.user_id, s.id))
        return ListSessionsResponse(sessions=sessions)

    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        keys = self._keys(app_name, user_id, session_id)
        handle = self._handle(app_name, user_id, session_id)
        await self._b.client.transaction(
            [
                ["DEL", keys["meta"], keys["state"], keys["events"]],
                ["ZREM", self._user_index(app_name, user_id), handle],
                ["ZREM", self._app_index(app_name), handle],
            ]
        )

    async def get_user_state(self, *, app_name: str, user_id: str) -> dict[str, Any]:
        raw = await self._b.client.execute(
            "HGETALL", self._user_state_key(app_name, user_id)
        )
        return self._read_hash(raw)

    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event
        # Updates the caller's session object and strips temp: state.
        event = await super().append_event(session=session, event=event)
        session.last_update_time = event.timestamp
        keys = self._keys(session.app_name, session.user_id, session.id)
        commands: list[Command] = [
            [
                "RPUSH",
                keys["events"],
                self._enc(event.model_dump(exclude_none=True, mode="json")),
            ]
        ]
        if event.actions and event.actions.state_delta:
            commands += self._state_writes(
                session.app_name, session.user_id, keys, event.actions.state_delta
            )
        commands += self._touch(
            session.app_name, session.user_id, session.id, keys, event.timestamp
        )
        # One atomic round trip per event: history and state never disagree.
        await self._b.client.transaction(commands)
        return event
