"""Defuse untrusted text coming back from the knowledge base.

Retrieved documents are *reference content*, never instructions. Anyone who can
write to the Weaviate collection would otherwise be able to steer the agent by
planting a sentence like "ignore your previous instructions". This module
strips control characters, caps length, and replaces recognised
instruction-injection spans with a visible marker so the model sees that
something was removed rather than silently obeying it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Shown in place of a removed injection attempt.
INJECTION_MARKER = "[تم حذف تعليمات غير موثوقة]"

MAX_SNIPPET_CHARS = 1200

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # English
    re.compile(
        r"(?i)\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
        r"(previous|prior|above|earlier|all)\b[^.\n]{0,40}"
        r"\b(instruction|instructions|prompt|prompts|rules?|context)\b"
    ),
    re.compile(
        r"(?i)\b(you are now|act as|pretend to be|from now on you)\b[^.\n]{0,80}"
    ),
    re.compile(
        r"(?i)\b(system prompt|developer message|reveal|print|output)\b"
        r"[^.\n]{0,40}\b(prompt|instructions?|api[_ ]?key|token|secret|credential)s?\b"
    ),
    re.compile(r"(?i)^\s*(system|assistant|developer)\s*:", re.MULTILINE),
    # Arabic
    re.compile(
        r"(تجاهل|تجاهلي|انسَ|انسى|ألغِ|الغي)\s+[^.\n]{0,40}(التعليمات|الأوامر|التوجيهات|السابقة)"
    ),
    re.compile(
        r"(أنت الآن|انت الان|تصرف كأنك|تصرفي كأنك|من الآن فصاعدًا)\s*[^.\n]{0,80}"
    ),
    re.compile(
        r"(اكشف|أظهر|اطبع)\s+[^.\n]{0,40}(التعليمات|النظام|المفتاح|كلمة السر|البرومبت)"
    ),
)


@dataclass(frozen=True)
class Sanitized:
    text: str
    injection_filtered: bool
    truncated: bool


def sanitize_snippet(raw: object, *, max_chars: int = MAX_SNIPPET_CHARS) -> Sanitized:
    """Return display-safe text for a single retrieved field."""
    if raw is None:
        return Sanitized("", False, False)
    text = raw if isinstance(raw, str) else str(raw)

    # Strip control characters and bidi/zero-width overrides, which can hide
    # text from a human reviewer while the model still reads it.
    text = "".join(
        ch for ch in text if ch == "\n" or unicodedata.category(ch) not in {"Cc", "Cf"}
    )
    text = re.sub(r"[ \t]+", " ", text).strip()

    filtered = False
    for pattern in _INJECTION_PATTERNS:
        text, count = pattern.subn(INJECTION_MARKER, text)
        if count:
            filtered = True

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip() + "…"

    return Sanitized(text, filtered, truncated)
