"""Structured JSON logging with secret and personal-data redaction.

Nothing in this project should log a raw API key, instance token, phone number,
full message body, patient verification token, internal patient id, or
report image URL.
:func:`configure_logging` installs a filter that scrubs records on the way out,
so a careless ``logger.info(payload)`` elsewhere still cannot leak.

The filter is a backstop, not the control. Call sites are expected not to pass
patient identity data in the first place — a name or a birth date has no
regular shape, so no pattern here can catch one.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any

from app.config import Settings, get_settings

MASK = "***"

#: GREEN API puts the instance id and token straight into the request path,
#: e.g. /waInstance710.../sendMessage/bc62...
_GREEN_PATH_RE = re.compile(
    r"/waInstance\d+/(?P<method>[A-Za-z0-9_]+)/[A-Za-z0-9]+",
)
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)
#: The Khatib per-patient verification token, however it is spelled.
_VERIFICATION_TOKEN_RE = re.compile(
    r"(x-patient-verification-token[\"'\s:=]+)[A-Za-z0-9._\-]+", re.IGNORECASE
)
#: /api/v1.0/results/<internal patient id>/latest — the id identifies a person.
_PATIENT_RESULTS_PATH_RE = re.compile(r"/results/[^/\s\"']+/latest", re.IGNORECASE)
#: A report image URL. The address of someone's medical report is as sensitive
#: as the report, and a signed one carries an access token in its query too.
_REPORT_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+?\.(?:jpe?g|png|webp|pdf)(?:\?[^\s\"'<>]*)?",
    re.IGNORECASE,
)
#: The GREEN API sendFileByUrl field, however it is quoted.
_URL_FILE_RE = re.compile(r"(urlFile[\"'\s:=]+)[^\s,}\"']+", re.IGNORECASE)
_APIKEY_QS_RE = re.compile(r"([?&](?:api[_-]?key|token|key)=)[^&\s]+", re.IGNORECASE)
#: WhatsApp chat ids look like 963991234567@c.us / @g.us
_CHAT_ID_RE = re.compile(r"\b(\d{6,20})@([cg])\.us\b")
#: Bare international-looking phone numbers.
_PHONE_RE = re.compile(r"(?<![\w@.])\+?\d{9,15}(?![\w.])")


def mask_chat_id(chat_id: str | None) -> str:
    """``963991234567@c.us`` -> ``9639*****567@c.us``. Stable, non-reversible."""
    if not chat_id:
        return ""
    match = _CHAT_ID_RE.fullmatch(chat_id.strip())
    if not match:
        return mask_phone(chat_id)
    digits, kind = match.group(1), match.group(2)
    return f"{_mask_digits(digits)}@{kind}.us"


def mask_phone(value: str | None) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) < 6:
        return MASK
    return _mask_digits(digits)


def _mask_digits(digits: str) -> str:
    if len(digits) <= 7:
        return digits[:2] + "*" * (len(digits) - 2)
    return f"{digits[:4]}{'*' * (len(digits) - 7)}{digits[-3:]}"


def summarize_text(text: str | None, keep: int = 12) -> str:
    """A loggable stand-in for message content: length plus a short prefix.

    Full customer message bodies are personal data and are never logged.
    """
    if not text:
        return "<empty>"
    cleaned = " ".join(text.split())
    prefix = cleaned[:keep]
    suffix = "…" if len(cleaned) > keep else ""
    return f"<{len(cleaned)} chars: {prefix}{suffix}>"


def redact_text(value: str, secrets: tuple[str, ...] = ()) -> str:
    """Scrub known secrets, credential-bearing URLs and phone numbers."""
    if not value:
        return value
    out = value
    for secret in secrets:
        if secret and len(secret) >= 6:
            out = out.replace(secret, MASK)
    out = _GREEN_PATH_RE.sub(r"/waInstance***/\g<method>/***", out)
    out = _BEARER_RE.sub(r"\1" + MASK, out)
    out = _VERIFICATION_TOKEN_RE.sub(r"\1" + MASK, out)
    out = _PATIENT_RESULTS_PATH_RE.sub("/results/" + MASK + "/latest", out)
    out = _URL_FILE_RE.sub(r"\1" + MASK, out)
    out = _REPORT_URL_RE.sub(MASK, out)
    out = _APIKEY_QS_RE.sub(r"\1" + MASK, out)
    out = _CHAT_ID_RE.sub(lambda m: mask_chat_id(m.group(0)), out)
    out = _PHONE_RE.sub(lambda m: mask_phone(m.group(0)), out)
    return out


class RedactionFilter(logging.Filter):
    """Redacts the rendered message and every structured extra."""

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._secrets = tuple(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        record.msg = redact_text(rendered, self._secrets)
        record.args = ()
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            record.context = {
                k: redact_text(str(v), self._secrets) for k, v in extra.items()
            }
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = context
        if record.exc_info:
            # Type name only — a formatted traceback can carry request URLs
            # and payload fragments into the log.
            payload["error"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else "Error"
            )
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Install the JSON handler and redaction filter on the root logger."""
    global _configured
    if _configured and not force:
        return
    settings = settings or get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactionFilter(tuple(settings.secret_values())))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    _configured = True


class _ContextLogger(logging.LoggerAdapter):
    """Keeps the caller's ``extra``.

    The stdlib adapter's ``process()`` overwrites ``kwargs["extra"]`` with the
    adapter's own dict (before Python 3.13), which would silently drop every
    ``extra={"context": ...}`` passed at the call site.
    """

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """A logger that accepts ``logger.info("msg", extra={"context": {...}})``."""
    return _ContextLogger(logging.getLogger(name), {})
