"""Encryption of stored values and hashing of key names.

One key does both jobs: Fernet (AES-128-CBC with HMAC-SHA256) for values, and
an HMAC subkey derived from it for key names. Rotating the key therefore
orphans all existing state — acceptable, because everything stored is
temporary and expires on its own.
"""

from __future__ import annotations

import hashlib
import hmac

from cryptography.fernet import Fernet, InvalidToken

#: Separates the key-name subkey from the encryption key.
_KEY_NAME_CONTEXT = b"khatib-state/key-names/v1"


class StateCipher:
    def __init__(self, key: str) -> None:
        self._fernet = Fernet(key.encode())
        self._name_key = hmac.new(
            key.encode(), _KEY_NAME_CONTEXT, hashlib.sha256
        ).digest()

    @staticmethod
    def is_valid_key(key: str) -> bool:
        try:
            Fernet(key.encode())
        except (ValueError, TypeError):
            return False
        return True

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str | None:
        """Plaintext, or ``None`` for anything not written with this key.

        ``None`` rather than an exception: a value that cannot be read is
        treated exactly like a value that is not there, which is the safe
        reading for every store (no verification, no session, not seen).
        """
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except (InvalidToken, ValueError, TypeError):
            return None

    def name(self, *parts: str) -> str:
        """Stable, non-reversible key-name component for ``parts``."""
        joined = "\x1f".join(parts).encode()
        return hmac.new(self._name_key, joined, hashlib.sha256).hexdigest()[:32]
