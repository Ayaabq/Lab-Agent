"""No real credential may live in tracked files.

A live OpenRouter key was once committed to app/config.py as a "temporary"
override. These checks fail the build if anything key-shaped comes back, and
pin that the LLM settings are read from the environment only. Fake test values
are deliberately short or hyphenated so they never match the real shapes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from app import config

ROOT = Path(__file__).resolve().parents[2]

#: Real-credential shapes. Each is long enough that no test fixture matches.
SECRET_SHAPES = {
    "OpenRouter key": r"sk-or-v1-[A-Za-z0-9]{32,}",
    "Google API key": r"AIza[0-9A-Za-z_\-]{35}",
    "Google OAuth access token": r"ya29\.[0-9A-Za-z_\-]{20,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{36,}",
}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / line for line in out.splitlines() if line]


@pytest.mark.parametrize(("name", "shape"), sorted(SECRET_SHAPES.items()))
def test_no_tracked_file_contains_a_credential(name, shape):
    pattern = re.compile(shape)
    offenders = [
        str(path.relative_to(ROOT))
        for path in _tracked_files()
        if path.is_file()
        and pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]

    # File names only: the assertion message must not echo the secret.
    assert offenders == [], f"{name} found in: {offenders}"


def test_no_env_or_token_file_is_tracked():
    tracked = {str(p.relative_to(ROOT)) for p in _tracked_files()}
    forbidden = {
        p
        for p in tracked
        if re.search(r"(^|/)\.env($|\.(?!example$))", p)
        or p.endswith(("WEBHOOK_TOKEN.txt", ".pem", "-key.json"))
    }

    assert forbidden == set()


@pytest.mark.parametrize(
    "ignore_file", [".gitignore", ".dockerignore", ".vercelignore"]
)
def test_env_files_are_ignored_everywhere(ignore_file):
    text = (ROOT / ignore_file).read_text(encoding="utf-8")

    # Either an explicit .env rule, or an allowlist that starts by excluding all.
    assert ".env" in text or re.search(r"(?m)^\*$", text)


def test_the_temporary_override_is_gone():
    assert not [name for name in vars(config) if name.startswith("_TEMPORARY")]


def test_llm_settings_come_only_from_the_environment(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_API_BASE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("app.config.load_dotenv", lambda *a, **k: False)
    config.reset_settings_cache()
    try:
        settings = config.get_settings()
    finally:
        config.reset_settings_cache()

    # Nothing set means nothing configured: no key, no model, no provider swap.
    assert not settings.openrouter.api_key.is_set
    assert settings.openrouter.model == ""
    assert settings.llm_provider == config.LLM_PROVIDER_GEMINI
