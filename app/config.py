"""Typed application configuration, read once from the environment.

Every secret is wrapped in :class:`Secret` so it cannot reach a log line, a
traceback, or a ``repr()`` by accident. Read the raw value only at the call
site that actually needs it, via ``.get()``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlparse

from dotenv import load_dotenv

# The scaffold selected this model and this key can serve it. Overridable via
# GEMINI_MODEL so the name lives in configuration, not scattered through code.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# Tuned against the live KhatibLabFAQ collection: on-topic questions scored
# >= 0.637 certainty, off-topic ones <= 0.589. See README "Retrieval tuning".
DEFAULT_MIN_CERTAINTY = 0.62

#: Exact field the backend puts the report image URL in. Spelled by the
#: backend, not by us — note the capitalised "URL".
DEFAULT_RESULT_IMAGE_FIELD = "imageURL"

#: The one disposition the laboratory has said releases a report to the
#: patient. Adding to this is a clinical decision, not a code change.
DEFAULT_IMAGE_RELEASE_DISPOSITIONS = ("BENIGN",)


class Secret:
    """A string that refuses to render itself."""

    __slots__ = ("_value",)

    def __init__(self, value: str | None) -> None:
        self._value = value or ""

    def get(self) -> str:
        """Return the raw value. Only call this at the point of use."""
        return self._value

    @property
    def is_set(self) -> bool:
        return bool(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "Secret(<set>)" if self._value else "Secret(<unset>)"

    def __str__(self) -> str:
        return "***" if self._value else ""


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _chat_digits(name: str) -> tuple[str, ...]:
    """Comma-separated chats, reduced to digits so any spelling matches.

    "+963 99 123 4567", "963991234567" and "963991234567@c.us" are the same
    chat written three ways; comparing digits means an operator cannot lock
    themselves out with a formatting difference.
    """
    out = []
    for part in _env(name).split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            out.append(digits)
    return tuple(out)


def _csv_upper(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    parts = tuple(p.strip().upper() for p in _env(name).split(",") if p.strip())
    return parts or default


def _csv_lower(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    parts = tuple(p.strip().lower() for p in _env(name).split(",") if p.strip())
    return parts or default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class GeminiSettings:
    model: str
    api_key: Secret
    use_vertexai: bool

    @property
    def is_configured(self) -> bool:
        return self.api_key.is_set


@dataclass(frozen=True)
class WeaviateSettings:
    url: str
    api_key: Secret
    #: Empty means "discover it, but only if the cluster has exactly one".
    collection: str
    search_limit: int
    min_certainty: float
    timeout_seconds: int
    #: ``near_text`` gives an absolute certainty we can threshold on. ``hybrid``
    #: scores are rank-normalised per query and cannot gate relevance.
    query_mode: str

    @property
    def is_configured(self) -> bool:
        return bool(self.url) and self.api_key.is_set


@dataclass(frozen=True)
class GreenApiSettings:
    api_url: str
    id_instance: Secret
    token_instance: Secret
    #: Master safety switch. While False no WhatsApp message is ever sent.
    live_enabled: bool
    ignore_groups: bool
    timeout_seconds: int
    dedupe_ttl_seconds: int
    dedupe_max_entries: int
    #: Development guard. While EMPTY (the default) every private chat is
    #: served, exactly as before. When it holds numbers, only those chats
    #: reach the agent — everything else is acknowledged and dropped. This is
    #: a testing convenience, NOT a security control: it narrows who the bot
    #: talks to, and must never be mistaken for patient identity verification.
    allowed_chats: tuple[str, ...] = ()
    #: Inbound messages older than this are acknowledged and dropped. GREEN API
    #: queues notifications it could not deliver and replays the backlog the
    #: moment the webhook URL works again — answering a question somebody asked
    #: hours ago is worse than silence, and a backlog flushed through the model
    #: can exhaust a day's quota in one burst. 0 disables the check.
    max_message_age_seconds: int = 600
    #: Shared secret GREEN API sends as ``Authorization: Bearer <token>``. When
    #: set, requests without a matching header are rejected with 401.
    webhook_token: Secret = field(default_factory=lambda: Secret(""))

    @property
    def is_configured(self) -> bool:
        return (
            bool(self.api_url)
            and self.id_instance.is_set
            and self.token_instance.is_set
        )


@dataclass(frozen=True)
class KhatibApiSettings:
    """Khatib patient/results backend (tenant-authenticated REST API).

    Two independent authentication layers reach this backend and neither may
    be skipped: a *tenant* bearer token proving this deployment is a registered
    client, and a per-patient verification token proving the person in the chat
    is who they claim to be. Both live server-side; neither is ever shown to
    the model, the customer, or a log line.
    """

    base_url: str
    client_id: Secret
    client_secret: Secret
    #: Whole-request budget. Every call is bounded; none may hang a chat turn.
    timeout_seconds: int
    connect_timeout_seconds: int
    #: Renew the tenant token this many seconds before ``expiresIn`` elapses,
    #: so an in-flight request cannot land just after expiry.
    auth_refresh_margin_seconds: int
    #: Minimum gap between tenant-login attempts after a failure. Without it a
    #: backend outage turns every inbound message into a login storm.
    auth_retry_cooldown_seconds: int
    #: Fallback TTL used when the verify response omits ``expiresInSeconds``
    #: or returns something unparsable. Short on purpose.
    default_verification_ttl_seconds: int
    #: Failed verification attempts allowed per chat inside the window below.
    max_verification_attempts: int
    verification_attempt_window_seconds: int
    #: The backend's result-image URL field. Configurable because the name is
    #: the backend's to change, not ours to hard-code in six places.
    result_image_field: str = DEFAULT_RESULT_IMAGE_FIELD
    #: Dispositions whose report image may be released to the patient. A URL
    #: alone authorises nothing: release also needs status FINAL, a valid
    #: patient verification, an approved host and a known file extension.
    image_release_dispositions: tuple[str, ...] = DEFAULT_IMAGE_RELEASE_DISPOSITIONS
    #: Hosts whose report images we will hand to GREEN API. Empty means "only
    #: the backend host", computed from base_url. An attacker who could put a
    #: URL in the result payload must not be able to make the provider fetch
    #: an arbitrary address on our behalf.
    image_allowed_hosts: tuple[str, ...] = ()

    @property
    def allowed_image_hosts(self) -> frozenset[str]:
        """Hosts a report image may come from, lower-cased."""
        configured = {h.strip().lower() for h in self.image_allowed_hosts if h.strip()}
        if configured:
            return frozenset(configured)
        host = urlparse(self.base_url).hostname
        return frozenset({host.lower()}) if host else frozenset()

    @property
    def is_configured(self) -> bool:
        return (
            bool(self.base_url) and self.client_id.is_set and self.client_secret.is_set
        )

    @property
    def tenant_key(self) -> str:
        """Stable, non-reversible id for the configured tenant.

        Every piece of cached patient state is scoped by this, so rotating the
        client credentials (or pointing at another backend) cannot resurrect a
        verification granted under the previous tenant. It is a digest, not the
        client id, so it is safe to put in a cache key or a log line.
        """
        seed = f"{self.base_url}|{self.client_id.get()}".encode()
        return hashlib.sha256(seed).hexdigest()[:16]


@dataclass(frozen=True)
class Settings:
    gemini: GeminiSettings
    weaviate: WeaviateSettings
    green_api: GreenApiSettings
    log_level: str
    escalation_max_entries: int = field(default=5000)
    #: Defaulted so a Settings(...) built without it stays valid; an
    #: unconfigured backend disables the patient tools rather than failing.
    khatib: KhatibApiSettings = field(
        default_factory=lambda: KhatibApiSettings(
            base_url="",
            client_id=Secret(""),
            client_secret=Secret(""),
            timeout_seconds=15,
            connect_timeout_seconds=5,
            auth_refresh_margin_seconds=60,
            auth_retry_cooldown_seconds=30,
            default_verification_ttl_seconds=300,
            max_verification_attempts=5,
            verification_attempt_window_seconds=900,
        )
    )

    def problems(self) -> list[str]:
        """Human-readable configuration problems. Never includes secret values."""
        issues: list[str] = []
        if not self.gemini.api_key.is_set:
            issues.append("GOOGLE_API_KEY is not set — the agent cannot call Gemini.")
        if self.gemini.use_vertexai:
            issues.append(
                "GOOGLE_GENAI_USE_VERTEXAI must be FALSE — this project uses an "
                "AI Studio key, not Vertex AI."
            )
        if not self.weaviate.url:
            issues.append("WEAVIATE_URL is not set — lab retrieval is unavailable.")
        if not self.weaviate.api_key.is_set:
            issues.append("WEAVIATE_API_KEY is not set — lab retrieval is unavailable.")
        if not self.green_api.is_configured:
            issues.append(
                "GREEN API is not fully configured — the WhatsApp webhook will "
                "accept messages but cannot reply."
            )
        if not self.khatib.base_url:
            issues.append(
                "KHATIB_API_BASE_URL is not set — patient verification and "
                "laboratory results are unavailable; the agent still answers "
                "general questions from the knowledge base."
            )
        else:
            missing = [
                name
                for name, secret in (
                    ("KHATIB_CLIENT_ID", self.khatib.client_id),
                    ("KHATIB_CLIENT_SECRET", self.khatib.client_secret),
                )
                if not secret.is_set
            ]
            if missing:
                verb = "is" if len(missing) == 1 else "are"
                issues.append(
                    f"{' and '.join(missing)} {verb} not set — patient "
                    "verification and laboratory results are unavailable; the "
                    "agent still answers general questions from the knowledge "
                    "base."
                )
        if self.green_api.live_enabled and not self.green_api.webhook_token.is_set:
            issues.append(
                "GREEN_API_WEBHOOK_TOKEN is not set while live sending is "
                "enabled — the webhook endpoint is unauthenticated and anyone "
                "who finds the URL can make the bot send WhatsApp messages."
            )
        return issues

    def secret_values(self) -> list[str]:
        """Raw secret strings, for the log redaction filter only."""
        return [
            s.get()
            for s in (
                self.gemini.api_key,
                self.weaviate.api_key,
                self.green_api.id_instance,
                self.green_api.token_instance,
                self.green_api.webhook_token,
                self.khatib.client_id,
                self.khatib.client_secret,
            )
            if s.is_set
        ]


def load_settings() -> Settings:
    """Build :class:`Settings` from the current environment (and ``.env``)."""
    load_dotenv()
    return Settings(
        gemini=GeminiSettings(
            model=_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            # ADK itself reads GOOGLE_API_KEY / GEMINI_API_KEY from the
            # environment; we mirror it here only for health reporting.
            api_key=Secret(_env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")),
            use_vertexai=_env_bool("GOOGLE_GENAI_USE_VERTEXAI", False),
        ),
        weaviate=WeaviateSettings(
            url=_env("WEAVIATE_URL"),
            api_key=Secret(_env("WEAVIATE_API_KEY")),
            collection=_env("WEAVIATE_COLLECTION"),
            search_limit=_env_int("WEAVIATE_SEARCH_LIMIT", 4),
            min_certainty=_env_float("WEAVIATE_MIN_CERTAINTY", DEFAULT_MIN_CERTAINTY),
            timeout_seconds=_env_int("WEAVIATE_TIMEOUT_SECONDS", 15),
            query_mode=_env("WEAVIATE_QUERY_MODE", "near_text").lower(),
        ),
        green_api=GreenApiSettings(
            api_url=_env("GREEN_API_API_URL").rstrip("/"),
            id_instance=Secret(_env("GREEN_API_ID_INSTANCE")),
            token_instance=Secret(_env("GREEN_API_TOKEN_INSTANCE")),
            live_enabled=_env_bool("GREEN_API_LIVE_ENABLED", False),
            ignore_groups=_env_bool("GREEN_API_IGNORE_GROUPS", True),
            timeout_seconds=_env_int("GREEN_API_TIMEOUT_SECONDS", 20),
            dedupe_ttl_seconds=_env_int("GREEN_API_DEDUPE_TTL_SECONDS", 900),
            dedupe_max_entries=_env_int("GREEN_API_DEDUPE_MAX_ENTRIES", 5000),
            max_message_age_seconds=_env_int("GREEN_API_MAX_MESSAGE_AGE_SECONDS", 600),
            allowed_chats=_chat_digits("GREEN_API_ALLOWED_CHATS"),
            webhook_token=Secret(_env("GREEN_API_WEBHOOK_TOKEN")),
        ),
        khatib=KhatibApiSettings(
            # No default: which backend this deployment talks to is an
            # operator decision, made in the environment. Unset, the patient
            # tools are disabled and everything else keeps working.
            base_url=_env("KHATIB_API_BASE_URL").rstrip("/"),
            client_id=Secret(_env("KHATIB_CLIENT_ID")),
            client_secret=Secret(_env("KHATIB_CLIENT_SECRET")),
            timeout_seconds=_env_int("KHATIB_API_TIMEOUT_SECONDS", 15),
            connect_timeout_seconds=_env_int("KHATIB_API_CONNECT_TIMEOUT_SECONDS", 5),
            auth_refresh_margin_seconds=_env_int(
                "KHATIB_AUTH_REFRESH_MARGIN_SECONDS", 60
            ),
            auth_retry_cooldown_seconds=_env_int(
                "KHATIB_AUTH_RETRY_COOLDOWN_SECONDS", 30
            ),
            default_verification_ttl_seconds=_env_int(
                "KHATIB_VERIFICATION_DEFAULT_TTL_SECONDS", 300
            ),
            max_verification_attempts=_env_int("KHATIB_MAX_VERIFICATION_ATTEMPTS", 5),
            verification_attempt_window_seconds=_env_int(
                "KHATIB_VERIFICATION_ATTEMPT_WINDOW_SECONDS", 900
            ),
            result_image_field=_env(
                "KHATIB_RESULT_IMAGE_FIELD", DEFAULT_RESULT_IMAGE_FIELD
            ),
            image_release_dispositions=_csv_upper(
                "KHATIB_IMAGE_RELEASE_DISPOSITIONS",
                DEFAULT_IMAGE_RELEASE_DISPOSITIONS,
            ),
            image_allowed_hosts=_csv_lower("KHATIB_IMAGE_ALLOWED_HOSTS"),
        ),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return load_settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Used by tests after patching the environment."""
    get_settings.cache_clear()
