"""Turning what a customer typed into what the backend accepts.

Customers write dates and numbers the way they speak them: Arabic-Indic
digits, slashes instead of dashes, day-first ordering. The backend wants
``YYYY-MM-DD`` and a four-character digit string. Everything here is strict in
one direction only — it will reshape an unambiguous value, and it will refuse
an ambiguous one so the agent can ask, but it never *picks* a reading. A
guessed birth date is a guessed identity.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from enum import StrEnum

#: Arabic-Indic (U+0660..) and Extended/Persian (U+06F0..) digit forms.
_ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

#: Arabic comma/decimal separators that can appear inside a typed number.
_SEPARATORS = str.maketrans({"٫": ".", "٬": ",", "،": ","})

MIN_NAME_CHARS = 3
MAX_NAME_CHARS = 120

#: Nobody in the records predates this, and a future birth date is a typo.
_EARLIEST_BIRTH_YEAR = 1900


def to_ascii_digits(value: str) -> str:
    """Map Arabic-Indic digits to ASCII, leaving everything else alone."""
    return value.translate(_ARABIC_DIGITS).translate(_SEPARATORS)


class FieldStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    INVALID = "invalid"
    #: Readable two ways (03/04 — March or April?). The agent must ask.
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class FieldResult:
    """A normalised field, or a reason it could not be normalised.

    ``value`` is customer identity data: it is passed to the backend and never
    logged, never echoed to the model beyond what the tool chooses to return.
    """

    status: FieldStatus
    value: str = ""

    @property
    def ok(self) -> bool:
        return self.status is FieldStatus.OK


def normalize_full_name(raw: str | None) -> FieldResult:
    """Collapse whitespace and bound the length. No transliteration."""
    if raw is None:
        return FieldResult(FieldStatus.MISSING)
    text = " ".join(str(raw).split())
    # Strip the digits a customer sometimes pastes in with the name; a name
    # that is mostly digits is a phone number in the wrong field.
    if not text:
        return FieldResult(FieldStatus.MISSING)
    if len(text) < MIN_NAME_CHARS or len(text) > MAX_NAME_CHARS:
        return FieldResult(FieldStatus.INVALID)
    digits = sum(ch.isdigit() for ch in to_ascii_digits(text))
    if digits > len(text) / 2:
        return FieldResult(FieldStatus.INVALID)
    return FieldResult(FieldStatus.OK, text)


def normalize_date_of_birth(
    raw: str | None, *, today: _dt.date | None = None
) -> FieldResult:
    """Parse a birth date into ``YYYY-MM-DD``, or report why it cannot be.

    Accepted unambiguously:

    * ``1990-03-04`` / ``1990/3/4`` — year first, so ISO order is certain.
    * ``25/12/1990`` — the first part exceeds 12, so it can only be a day.

    Reported as :attr:`FieldStatus.AMBIGUOUS`:

    * ``03/04/1990`` — day-month or month-day, both readings are real dates.

    A two-digit year is ambiguous too (``90`` could be 1990 or 2090), so it is
    rejected rather than widened to a century we invented.
    """
    if raw is None:
        return FieldResult(FieldStatus.MISSING)
    text = to_ascii_digits(str(raw)).strip()
    if not text:
        return FieldResult(FieldStatus.MISSING)

    today = today or _dt.date.today()
    parts = re.split(r"[^\d]+", text)
    parts = [p for p in parts if p]
    if len(parts) != 3:
        return FieldResult(FieldStatus.INVALID)

    first, second, third = parts

    if len(first) == 4:
        return _build(first, second, third, today)

    if len(third) != 4:
        # Two-digit years are genuinely ambiguous; ask instead of assuming.
        return FieldResult(FieldStatus.AMBIGUOUS)

    day_first = _build(third, second, first, today)
    month_first = _build(third, first, second, today)
    if day_first.ok and month_first.ok and first != second:
        # Both orderings yield a valid, plausible date. Only the customer
        # knows which they meant.
        return FieldResult(FieldStatus.AMBIGUOUS)
    if day_first.ok:
        return day_first
    if month_first.ok:
        return month_first
    # Neither ordering is a real date — but say "ambiguous" only when the
    # shape was ambiguous; here it is simply wrong.
    return FieldResult(FieldStatus.INVALID)


def _build(year: str, month: str, day: str, today: _dt.date) -> FieldResult:
    try:
        parsed = _dt.date(int(year), int(month), int(day))
    except ValueError:
        return FieldResult(FieldStatus.INVALID)
    if parsed.year < _EARLIEST_BIRTH_YEAR or parsed > today:
        return FieldResult(FieldStatus.INVALID)
    return FieldResult(FieldStatus.OK, parsed.isoformat())


#: A full phone number a customer pasted instead of only its last four digits.
_FULL_PHONE_DIGITS = range(9, 16)


def normalize_phone_last4(raw: str | None) -> FieldResult:
    """Return exactly four digits **as a string**, leading zeros intact.

    ``"0123"`` must stay ``"0123"``. Nothing here ever converts the value to
    an int, because ``int("0123")`` silently becomes a different identity
    claim.

    A customer who sends their whole number gets the last four taken from it —
    that is a reading of what they sent, not a guess. Anything in between (five
    to eight digits) is a typo we refuse rather than truncate.
    """
    if raw is None:
        return FieldResult(FieldStatus.MISSING)
    text = to_ascii_digits(str(raw)).strip()
    if not text:
        return FieldResult(FieldStatus.MISSING)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return FieldResult(FieldStatus.INVALID)
    if len(digits) == 4:
        return FieldResult(FieldStatus.OK, digits)
    if len(digits) in _FULL_PHONE_DIGITS:
        return FieldResult(FieldStatus.OK, digits[-4:])
    return FieldResult(FieldStatus.INVALID)
