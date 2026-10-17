"""GREEN API client: the live-sending guard is the important test here."""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from app.whatsapp.green_api import GreenApiClient, GreenApiError


class RecordingTransport(httpx.AsyncBaseTransport):
    """Fails the test if the client opens a request it should not have."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._response is None:
            return httpx.Response(200, json={"idMessage": "OUT-1"})
        return self._response


@pytest.mark.asyncio
async def test_live_disabled_sends_nothing(settings):
    transport = RecordingTransport()
    client = GreenApiClient(settings.green_api, transport=transport)

    result = await client.send_text("963991234567@c.us", "مرحبا")

    assert result.delivered is False
    assert result.dry_run is True
    assert result.reason == "live_disabled"
    # The decisive assertion: no HTTP request was made at all.
    assert transport.requests == []
    await client.aclose()


@pytest.mark.asyncio
async def test_dry_run_logs_a_bounded_redacted_preview(settings, caplog):
    """The dry-run line exists to show intent, so it keeps a bounded prefix.

    The recipient is always masked, and the body is capped — an operator can
    see what would have gone out without the log becoming a message archive.
    """
    long_message = "تفاصيل الرد الكامل " * 10
    client = GreenApiClient(settings.green_api, transport=RecordingTransport())

    with caplog.at_level("INFO"):
        await client.send_text("963991234567@c.us", long_message)

    record = next(r for r in caplog.records if "DRY-RUN" in r.getMessage())
    context = (
        record.context if isinstance(getattr(record, "context", None), dict) else {}
    )
    assert "963991234567" not in str(context)
    logged = str(context.get("message", ""))
    assert logged.startswith("<")
    assert "chars" in logged
    assert long_message.strip() not in logged
    assert len(logged) < len(long_message)
    await client.aclose()


@pytest.mark.asyncio
async def test_live_enabled_posts_to_send_message(settings):
    live = replace(settings.green_api, live_enabled=True)
    transport = RecordingTransport()
    client = GreenApiClient(live, transport=transport)

    result = await client.send_text("963991234567@c.us", "أهلاً")

    assert result.delivered is True
    assert result.dry_run is False
    assert result.provider_message_id == "OUT-1"
    request = transport.requests[0]
    assert request.url.path.endswith("/sendMessage/test-green-token-abcdef")
    assert b"963991234567" in request.content
    await client.aclose()


@pytest.mark.asyncio
async def test_send_is_not_retried(settings):
    """sendMessage is not idempotent — one failure must not become two messages."""
    live = replace(settings.green_api, live_enabled=True)
    transport = RecordingTransport(httpx.Response(500, json={"error": "boom"}))
    client = GreenApiClient(live, transport=transport)

    with pytest.raises(GreenApiError):
        await client.send_text("963991234567@c.us", "أهلاً")

    assert len(transport.requests) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_error_message_does_not_leak_the_token(settings):
    live = replace(settings.green_api, live_enabled=True)
    transport = RecordingTransport(httpx.Response(401, json={"error": "bad token"}))
    client = GreenApiClient(live, transport=transport)

    with pytest.raises(GreenApiError) as excinfo:
        await client.send_text("963991234567@c.us", "أهلاً")

    assert "test-green-token-abcdef" not in str(excinfo.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_empty_message_is_not_sent(settings):
    live = replace(settings.green_api, live_enabled=True)
    transport = RecordingTransport()
    client = GreenApiClient(live, transport=transport)

    result = await client.send_text("963991234567@c.us", "   ")

    assert result.delivered is False
    assert transport.requests == []
    await client.aclose()


@pytest.mark.asyncio
async def test_get_state_instance_is_read_only(settings):
    transport = RecordingTransport(
        httpx.Response(200, json={"stateInstance": "authorized"})
    )
    client = GreenApiClient(settings.green_api, transport=transport)

    state = await client.get_state_instance()

    assert state["stateInstance"] == "authorized"
    assert transport.requests[0].method == "GET"
    assert "getStateInstance" in str(transport.requests[0].url)
    await client.aclose()


@pytest.mark.asyncio
async def test_no_notification_queue_methods_exist():
    """receiveNotification / deleteNotification would mutate the live instance."""
    assert not hasattr(GreenApiClient, "receive_notification")
    assert not hasattr(GreenApiClient, "delete_notification")


# --- sendFileByUrl --------------------------------------------------------

REPORT_URL = "https://khatib.example.test/reports/abc123.jpg"


@pytest.mark.asyncio
async def test_file_send_posts_the_documented_fields(settings):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"idMessage": "IMG-1"})

    client = GreenApiClient(
        replace(settings.green_api, live_enabled=True),
        transport=httpx.MockTransport(handler),
    )

    result = await client.send_file_by_url(
        "963991234567@c.us", REPORT_URL, "lab-report.jpg", "نتيجتك سليمة."
    )

    assert result.delivered is True
    assert result.provider_message_id == "IMG-1"
    assert calls[0].url.path.endswith("/sendFileByUrl/test-green-token-abcdef")
    assert json.loads(calls[0].content) == {
        "chatId": "963991234567@c.us",
        "urlFile": REPORT_URL,
        "fileName": "lab-report.jpg",
        "caption": "نتيجتك سليمة.",
    }


@pytest.mark.asyncio
async def test_an_empty_caption_is_omitted_rather_than_sent_blank(settings):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"idMessage": "IMG-1"})

    client = GreenApiClient(
        replace(settings.green_api, live_enabled=True),
        transport=httpx.MockTransport(handler),
    )

    await client.send_file_by_url("963991234567@c.us", REPORT_URL, "lab-report.jpg")

    assert "caption" not in json.loads(calls[0].content)


@pytest.mark.asyncio
async def test_a_file_send_opens_no_socket_while_live_is_disabled(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a request was built while live sending was off")

    client = GreenApiClient(settings.green_api, transport=httpx.MockTransport(handler))

    result = await client.send_file_by_url(
        "963991234567@c.us", REPORT_URL, "lab-report.jpg", "caption"
    )

    assert result.dry_run is True
    assert result.delivered is False
    assert result.reason == "live_disabled"


@pytest.mark.asyncio
async def test_a_file_send_timeout_is_ambiguous_and_not_retried(settings):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("slow", request=request)

    client = GreenApiClient(
        replace(settings.green_api, live_enabled=True),
        transport=httpx.MockTransport(handler),
    )

    result = await client.send_file_by_url(
        "963991234567@c.us", REPORT_URL, "lab-report.jpg"
    )

    assert result.ambiguous is True
    assert result.delivered is False
    # The provider may already have queued the file; a retry would deliver a
    # second copy of a medical report.
    assert attempts == 1


@pytest.mark.asyncio
async def test_a_rejected_file_send_raises_without_echoing_the_url(settings):
    client = GreenApiClient(
        replace(settings.green_api, live_enabled=True),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, text=f"bad url {REPORT_URL}")
        ),
    )

    with pytest.raises(GreenApiError) as exc:
        await client.send_file_by_url("963991234567@c.us", REPORT_URL, "lab-report.jpg")

    assert REPORT_URL not in str(exc.value)
    assert "400" in str(exc.value)


@pytest.mark.asyncio
async def test_an_incomplete_file_request_is_refused(settings):
    client = GreenApiClient(
        replace(settings.green_api, live_enabled=True),
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("should not send"))
        ),
    )

    for chat, url, name in (
        ("", REPORT_URL, "lab-report.jpg"),
        ("963991234567@c.us", "", "lab-report.jpg"),
        ("963991234567@c.us", REPORT_URL, ""),
    ):
        result = await client.send_file_by_url(chat, url, name)
        assert result.delivered is False
        assert result.reason == "empty_request"
