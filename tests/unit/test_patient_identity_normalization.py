"""Turning what a customer typed into what the backend accepts.

The rule under test throughout: reshape an unambiguous value, refuse an
ambiguous one, never pick a reading. A guessed birth date is a guessed
identity.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app.patients.normalize import (
    FieldStatus,
    normalize_date_of_birth,
    normalize_full_name,
    normalize_phone_last4,
    to_ascii_digits,
)

TODAY = dt.date(2026, 10, 11)


# --- Arabic digits --------------------------------------------------------


def test_arabic_indic_and_persian_digits_both_map_to_ascii():
    assert to_ascii_digits("١٩٩٠-٠٣-٠٤") == "1990-03-04"
    assert to_ascii_digits("۱۹۹۰/۰۳/۰۴") == "1990/03/04"


# --- dates ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["1990-03-04", "1990/03/04", "1990.3.4", "١٩٩٠-٠٣-٠٤", " 1990 - 3 - 4 "],
)
def test_year_first_dates_are_unambiguous(raw):
    result = normalize_date_of_birth(raw, today=TODAY)

    assert result.ok
    assert result.value == "1990-03-04"


def test_a_day_above_twelve_settles_the_order():
    assert normalize_date_of_birth("25/12/1990", today=TODAY).value == "1990-12-25"
    assert normalize_date_of_birth("٢٥/١٢/١٩٩٠", today=TODAY).value == "1990-12-25"


def test_a_date_readable_two_ways_is_refused_not_guessed():
    result = normalize_date_of_birth("03/04/1990", today=TODAY)

    # March 4th or April 3rd — only the customer knows.
    assert result.status is FieldStatus.AMBIGUOUS
    assert result.value == ""


def test_identical_day_and_month_is_not_ambiguous():
    assert normalize_date_of_birth("03/03/1990", today=TODAY).value == "1990-03-03"


def test_a_two_digit_year_is_ambiguous():
    assert normalize_date_of_birth("04/03/90", today=TODAY).status is (
        FieldStatus.AMBIGUOUS
    )


@pytest.mark.parametrize(
    "raw",
    [
        "1990-02-30",  # not a real date
        "1990-13-01",  # month 13
        "2030-01-01",  # in the future
        "1850-01-01",  # implausibly early
        "hello",
        "1990-03",  # incomplete
    ],
)
def test_unusable_dates_are_invalid(raw):
    assert normalize_date_of_birth(raw, today=TODAY).status is FieldStatus.INVALID


def test_a_missing_date_is_missing_not_invalid():
    # The agent should ask for it, not tell the customer they got it wrong.
    assert normalize_date_of_birth(None).status is FieldStatus.MISSING
    assert normalize_date_of_birth("   ").status is FieldStatus.MISSING


# --- phone suffix ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0123", "0123"),
        ("٠١٢٣", "0123"),
        ("0000", "0000"),
        (" 0045 ", "0045"),
        ("00-45", "0045"),
    ],
)
def test_leading_zeros_survive_as_a_string(raw, expected):
    result = normalize_phone_last4(raw)

    assert result.ok
    # int("0123") would be 123 — a different identity claim entirely.
    assert result.value == expected
    assert isinstance(result.value, str)
    assert len(result.value) == 4


def test_a_whole_phone_number_yields_its_own_last_four():
    assert normalize_phone_last4("0991234067").value == "4067"
    assert normalize_phone_last4("+963 991 234 067").value == "4067"


def test_a_number_with_a_trailing_zero_block_keeps_it():
    assert normalize_phone_last4("0991230000").value == "0000"


@pytest.mark.parametrize("raw", ["12", "12345", "12345678", "abcd", "٠١٢"])
def test_anything_between_four_and_a_phone_number_is_refused(raw):
    # Truncating a six-digit typo would silently claim a different patient.
    assert normalize_phone_last4(raw).status is FieldStatus.INVALID


def test_a_missing_suffix_is_missing():
    assert normalize_phone_last4(None).status is FieldStatus.MISSING
    assert normalize_phone_last4("").status is FieldStatus.MISSING


# --- names ----------------------------------------------------------------


def test_whitespace_in_a_name_is_collapsed():
    assert normalize_full_name("  أحمد   الخطيب  ").value == "أحمد الخطيب"


@pytest.mark.parametrize("raw", ["", "  ", None])
def test_an_absent_name_is_missing(raw):
    assert normalize_full_name(raw).status is FieldStatus.MISSING


@pytest.mark.parametrize("raw", ["ab", "0991234567", "x" * 200])
def test_an_unusable_name_is_invalid(raw):
    assert normalize_full_name(raw).status is FieldStatus.INVALID
