"""Nothing sensitive may reach a log line."""

from __future__ import annotations

import logging

from app.logging_utils import (
    MASK,
    RedactionFilter,
    mask_chat_id,
    mask_phone,
    redact_text,
    summarize_text,
)


def test_known_secrets_are_replaced():
    result = redact_text("token=super-secret-value", ("super-secret-value",))

    assert "super-secret-value" not in result
    assert "***" in result


def test_green_api_url_credentials_are_stripped():
    url = (
        "https://0000.api.greenapi.com/waInstance000011112222/sendMessage/deadbeefcafe"
    )

    result = redact_text(url)

    assert "000011112222" not in result
    assert "deadbeefcafe" not in result
    # The method name survives, which is the part worth logging.
    assert "sendMessage" in result


def test_bearer_tokens_are_stripped():
    result = redact_text("Authorization: Bearer abc.def.ghi")

    assert "abc.def.ghi" not in result


def test_api_key_query_parameters_are_stripped():
    result = redact_text("https://example.test/v1?api_key=abcdef123456&x=1")

    assert "abcdef123456" not in result


def test_chat_ids_in_free_text_are_masked():
    result = redact_text("reply sent to 963991234567@c.us")

    assert "963991234567@c.us" not in result
    assert "@c.us" in result


def test_bare_phone_numbers_are_masked():
    result = redact_text("call 963991234567 now")

    assert "963991234567" not in result


def test_mask_chat_id_keeps_shape_and_hides_the_middle():
    masked = mask_chat_id("963991234567@c.us")

    assert masked.endswith("@c.us")
    assert masked.startswith("9639")
    assert "991234" not in masked


def test_mask_chat_id_handles_group_ids():
    assert mask_chat_id("120363000000000000@g.us").endswith("@g.us")


def test_mask_chat_id_handles_empty_input():
    assert mask_chat_id(None) == ""
    assert mask_chat_id("") == ""


def test_mask_phone_hides_the_middle():
    assert mask_phone("963991234567") == "9639*****567"


def test_summarize_text_never_includes_the_full_body():
    body = "نص طويل جدًا فيه معلومات شخصية عن المريض ويجب ألا يظهر في السجلات"

    summary = summarize_text(body, keep=10)

    assert body not in summary
    assert "chars" in summary


def test_summarize_text_handles_empty():
    assert summarize_text("") == "<empty>"
    assert summarize_text(None) == "<empty>"


def test_filter_redacts_the_rendered_record():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="sending to %s with key %s",
        args=("963991234567@c.us", "my-secret-key-123"),
        exc_info=None,
    )

    RedactionFilter(("my-secret-key-123",)).filter(record)

    rendered = record.getMessage()
    assert "my-secret-key-123" not in rendered
    assert "963991234567" not in rendered


def test_filter_redacts_structured_context():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    record.context = {"chat": "963991234567@c.us", "key": "my-secret-key-123"}

    RedactionFilter(("my-secret-key-123",)).filter(record)

    assert "my-secret-key-123" not in str(record.context)
    assert "963991234567" not in str(record.context)


def test_short_strings_are_not_treated_as_secrets():
    """A 3-character 'secret' would otherwise blank out ordinary words."""
    result = redact_text("the lab is open", ("lab",))

    assert result == "the lab is open"


# --- patient backend credentials and identifiers -------------------------


def test_a_patient_verification_token_header_is_masked():
    line = redact_text(
        "sending x-patient-verification-token: vtok-abcdef123456 upstream"
    )

    assert "vtok-abcdef123456" not in line
    assert MASK in line


def test_the_internal_patient_id_is_masked_out_of_a_results_url():
    line = redact_text(
        "GET https://khatib.example.test/api/v1.0/results/pat_internal_7781/latest"
    )

    assert "pat_internal_7781" not in line
    assert "/results/***/latest" in line


def test_the_tenant_bearer_token_is_masked():
    assert "tenant-token-abc" not in redact_text(
        "Authorization: Bearer tenant-token-abc"
    )


def test_configured_backend_credentials_are_scrubbed_from_any_line(settings):
    secrets = tuple(settings.secret_values())

    line = redact_text(
        "login body clientId=test-khatib-client-id "
        "clientSecret=test-khatib-client-secret",
        secrets,
    )

    assert "test-khatib-client-id" not in line
    assert "test-khatib-client-secret" not in line
