from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.services.auth_security import hash_password, verify_password

VALID_ROLES = {"viewer", "clinician", "admin"}


@dataclass(slots=True)
class SessionUser:
    user_id: str
    username: str
    full_name: str
    role: str

    def to_session(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "full_name": self.full_name,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> SessionUser | None:
        if not value:
            return None
        user_id = str(value.get("user_id", "")).strip()
        username = str(value.get("username", "")).strip()
        full_name = str(value.get("full_name", "")).strip()
        role = str(value.get("role", "")).strip().lower()
        if not user_id or not username or not role:
            return None
        if role not in VALID_ROLES:
            return None
        return cls(
            user_id=user_id,
            username=username,
            full_name=full_name or username,
            role=role,
        )


def normalize_username(value: str) -> str:
    return value.strip().lower()


def bootstrap_first_role(user_count: int) -> str:
    return "admin" if user_count == 0 else "viewer"


def register_user(
    store: Any,
    *,
    username: str,
    full_name: str,
    password: str,
    role: str,
) -> SessionUser:
    normalized_username = normalize_username(username)
    resolved_role = role.strip().lower()
    if resolved_role not in VALID_ROLES:
        raise ValueError("Unsupported role.")
    if len(normalized_username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if store.get_user_by_username(normalized_username) is not None:
        raise ValueError("Username already exists.")

    user_id = str(uuid4())
    created_at = datetime.now(timezone.utc)
    store.create_user(
        user_id=user_id,
        username=normalized_username,
        full_name=full_name.strip() or normalized_username,
        password_hash=hash_password(password),
        role=resolved_role,
        created_at=created_at,
    )
    return SessionUser(
        user_id=user_id,
        username=normalized_username,
        full_name=full_name.strip() or normalized_username,
        role=resolved_role,
    )


def authenticate_user(store: Any, *, username: str, password: str) -> SessionUser | None:
    normalized_username = normalize_username(username)
    user = store.get_user_by_username(normalized_username)
    if user is None:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return SessionUser(
        user_id=str(user["id"]),
        username=str(user["username"]),
        full_name=str(user["full_name"]),
        role=str(user["role"]),
    )
