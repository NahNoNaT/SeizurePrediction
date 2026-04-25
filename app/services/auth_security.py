from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

PBKDF2_ITERATIONS = 240_000
PBKDF2_ALGORITHM = "sha256"
SALT_BYTES = 16


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(PBKDF2_ALGORITHM, password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        scheme, iterations_text, salt_b64, digest_b64 = encoded_hash.split("$", 3)
        if not scheme.startswith("pbkdf2_"):
            return False
        algorithm = scheme.replace("pbkdf2_", "", 1)
        iterations = int(iterations_text)
        salt = _unb64(salt_b64)
        expected = _unb64(digest_b64)
    except Exception:
        return False

    calculated = hashlib.pbkdf2_hmac(algorithm, password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(calculated, expected)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))
