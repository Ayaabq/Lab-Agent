"""Configuration loading, secret hiding, and the safety defaults."""

from __future__ import annotations

import pytest

from app.config import DEFAULT_GEMINI_MODEL, Secret, load_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "GEMINI_MODEL",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "WEAVIATE_COLLECTION",
        "WEAVIATE_MIN_CERTAINTY",
        "GREEN_API_API_URL",
        "GREEN_API_ID_INSTANCE",
        "GREEN_API_TOKEN_INSTANCE",
        "GREEN_API_LIVE_ENABLED",
        "GREEN_API_IGNORE_GROUPS",
        "KHATIB_API_BASE_URL",
        "KHATIB_CLIENT_ID",
        "KHATIB_CLIENT_SECRET",
        "KHATIB_MAX_VERIFICATION_ATTEMPTS",
        "KHATIB_RESULT_IMAGE_FIELD",
        "KHATIB_IMAGE_RELEASE_DISPOSITIONS",
        "KHATIB_IMAGE_ALLOWED_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)
    # load_settings() calls load_dotenv(), which must not override the test env.
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_secret_never_renders_its_value():
    secret = Secret("super-secret-value")

    assert "super-secret-value" not in repr(secret)
    assert "super-secret-value" not in str(secret)
    assert "super-secret-value" not in f"{secret}"
    assert secret.get() == "super-secret-value"


def test_secret_reports_whether_it_is_set():
    assert Secret("x").is_set is True
    assert Secret("").is_set is False
    assert Secret(None).is_set is False


def test_defaults_are_safe(monkeypatch):
    settings = load_settings()

    # Live WhatsApp sending is off unless explicitly enabled.
    assert settings.green_api.live_enabled is False
    # Group chats are ignored unless explicitly enabled.
    assert settings.green_api.ignore_groups is True
    # Vertex AI is off: this project uses an AI Studio key.
    assert settings.gemini.use_vertexai is False
    assert settings.gemini.model == DEFAULT_GEMINI_MODEL


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_live_flag_accepts_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("GREEN_API_LIVE_ENABLED", value)

    assert load_settings().green_api.live_enabled is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off", "", "maybe"])
def test_live_flag_defaults_to_off_for_anything_else(monkeypatch, value):
    monkeypatch.setenv("GREEN_API_LIVE_ENABLED", value)

    assert load_settings().green_api.live_enabled is False


def test_gemini_api_key_falls_back_to_gemini_api_key_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini-var")

    assert load_settings().gemini.api_key.get() == "from-gemini-var"


def test_google_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-var")
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini-var")

    assert load_settings().gemini.api_key.get() == "from-google-var"


def test_model_is_configurable(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")

    assert load_settings().gemini.model == "gemini-3.5-flash"


def test_invalid_numeric_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("WEAVIATE_MIN_CERTAINTY", "not-a-number")

    assert load_settings().weaviate.min_certainty == pytest.approx(0.62)


def test_problems_flag_missing_credentials():
    problems = load_settings().problems()

    assert any("GOOGLE_API_KEY" in p for p in problems)
    assert any("WEAVIATE_URL" in p for p in problems)


def test_problems_flag_vertex_ai(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

    assert any("GOOGLE_GENAI_USE_VERTEXAI" in p for p in load_settings().problems())


def test_configured_project_reports_no_problems(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setenv("WEAVIATE_URL", "https://example.weaviate.cloud")
    monkeypatch.setenv("WEAVIATE_API_KEY", "key")
    monkeypatch.setenv("GREEN_API_API_URL", "https://0000.api.greenapi.com")
    monkeypatch.setenv("GREEN_API_ID_INSTANCE", "000011112222")
    monkeypatch.setenv("GREEN_API_TOKEN_INSTANCE", "token")
    monkeypatch.setenv("KHATIB_API_BASE_URL", "https://backend.example.test")
    monkeypatch.setenv("KHATIB_CLIENT_ID", "client-id")
    monkeypatch.setenv("KHATIB_CLIENT_SECRET", "client-secret")

    assert load_settings().problems() == []


def test_secret_values_are_collected_for_redaction(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-key")
    monkeypatch.setenv("WEAVIATE_API_KEY", "weaviate-key")
    monkeypatch.setenv("GREEN_API_TOKEN_INSTANCE", "green-token")

    values = load_settings().secret_values()

    assert set(values) >= {"gemini-key", "weaviate-key", "green-token"}


def test_trailing_slash_is_stripped_from_the_api_url(monkeypatch):
    monkeypatch.setenv("GREEN_API_API_URL", "https://0000.api.greenapi.com/")

    assert load_settings().green_api.api_url == "https://0000.api.greenapi.com"


# --- Khatib patient/results backend --------------------------------------


def test_the_backend_base_url_has_no_default():
    # Which backend holds patient records is the operator's call, made in the
    # environment — never a host baked into the code.
    khatib = load_settings().khatib

    assert khatib.base_url == ""
    assert not khatib.is_configured


def test_a_missing_backend_url_is_reported_without_blocking_startup(monkeypatch):
    monkeypatch.setenv("KHATIB_CLIENT_ID", "an-id")
    monkeypatch.setenv("KHATIB_CLIENT_SECRET", "a-secret")

    settings = load_settings()

    assert not settings.khatib.is_configured
    assert any("KHATIB_API_BASE_URL is not set" in p for p in settings.problems())
    assert settings.khatib.allowed_image_hosts == frozenset()


def test_a_trailing_slash_is_stripped_from_the_backend_url(monkeypatch):
    monkeypatch.setenv("KHATIB_API_BASE_URL", "https://backend.example.test/")

    assert load_settings().khatib.base_url == "https://backend.example.test"


def test_missing_backend_credentials_are_named_without_their_values(monkeypatch):
    monkeypatch.setenv("KHATIB_API_BASE_URL", "https://backend.example.test")
    monkeypatch.setenv("KHATIB_CLIENT_ID", "an-id")

    problems = load_settings().problems()

    assert any("KHATIB_CLIENT_SECRET is not set" in p for p in problems)
    # The one that IS set is not named as a problem, and no value is echoed.
    assert not any("KHATIB_CLIENT_ID is" in p for p in problems)
    assert not any("an-id" in p for p in problems)


def test_both_missing_credentials_are_reported_together(monkeypatch):
    monkeypatch.setenv("KHATIB_API_BASE_URL", "https://backend.example.test")
    problems = load_settings().problems()

    assert any(
        "KHATIB_CLIENT_ID and KHATIB_CLIENT_SECRET are not set" in p for p in problems
    )


def test_the_backend_is_only_configured_with_both_halves(monkeypatch):
    monkeypatch.setenv("KHATIB_API_BASE_URL", "https://backend.example.test")
    monkeypatch.setenv("KHATIB_CLIENT_ID", "an-id")
    assert load_settings().khatib.is_configured is False

    monkeypatch.setenv("KHATIB_CLIENT_SECRET", "a-secret")
    assert load_settings().khatib.is_configured is True


def test_backend_credentials_are_collected_for_log_redaction(monkeypatch):
    monkeypatch.setenv("KHATIB_CLIENT_ID", "khatib-id-value")
    monkeypatch.setenv("KHATIB_CLIENT_SECRET", "khatib-secret-value")

    values = load_settings().secret_values()

    assert {"khatib-id-value", "khatib-secret-value"} <= set(values)


def test_the_tenant_key_is_a_digest_not_the_client_id(monkeypatch):
    monkeypatch.setenv("KHATIB_CLIENT_ID", "khatib-id-value")
    monkeypatch.setenv("KHATIB_CLIENT_SECRET", "s")

    key = load_settings().khatib.tenant_key

    # It ends up in cache keys and log lines, so it must not be the credential.
    assert "khatib-id-value" not in key
    assert len(key) == 16


def test_rotating_the_client_id_changes_the_tenant_scope(monkeypatch):
    monkeypatch.setenv("KHATIB_CLIENT_SECRET", "s")
    monkeypatch.setenv("KHATIB_CLIENT_ID", "first")
    first = load_settings().khatib.tenant_key
    monkeypatch.setenv("KHATIB_CLIENT_ID", "second")

    # Verifications cached under the old tenant cannot be found under the new.
    assert load_settings().khatib.tenant_key != first


def test_the_image_field_defaults_to_the_backends_spelling():
    khatib = load_settings().khatib

    # The backend spells it with a capitalised "URL". Guessing "imageUrl"
    # would silently mean no report ever reaches a patient.
    assert khatib.result_image_field == "imageURL"


def test_only_benign_releases_a_report_by_default():
    assert load_settings().khatib.image_release_dispositions == ("BENIGN",)


def test_the_allowed_image_host_defaults_to_the_backend_host(monkeypatch):
    monkeypatch.setenv("KHATIB_API_BASE_URL", "https://backend.example.test")

    assert load_settings().khatib.allowed_image_hosts == {"backend.example.test"}


def test_allowed_image_hosts_can_be_listed_explicitly(monkeypatch):
    monkeypatch.setenv(
        "KHATIB_IMAGE_ALLOWED_HOSTS", " CDN.Example.Test , files.example.test "
    )

    hosts = load_settings().khatib.allowed_image_hosts

    assert hosts == {"cdn.example.test", "files.example.test"}


def test_image_release_dispositions_parse_as_an_upper_case_list(monkeypatch):
    monkeypatch.setenv(
        "KHATIB_IMAGE_RELEASE_DISPOSITIONS", " released_to_patient , Other "
    )

    assert load_settings().khatib.image_release_dispositions == (
        "RELEASED_TO_PATIENT",
        "OTHER",
    )
