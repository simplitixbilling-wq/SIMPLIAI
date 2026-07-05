"""Local-only secret persistence for integration keys."""

from __future__ import annotations

import os
import secrets
from typing import Callable

from app_core.security_utils import constant_time_equals


class LocalSecretStore:
    """Manage local integration secrets without exposing them in app metadata."""

    def __init__(
        self,
        app_settings: dict,
        persist_settings: Callable[[dict], None],
        app_data_dir: str,
    ):
        self.app_settings = app_settings if isinstance(app_settings, dict) else {}
        self.persist_settings = persist_settings
        self.app_data_dir = app_data_dir

    def secret_path(self, name: str) -> str:
        return os.path.join(self.app_data_dir, name)

    def ensure_local_api_key(self) -> tuple[str, str]:
        key = str((self.app_settings or {}).get("local_api_key", "")).strip()

        if not key:
            key = secrets.token_urlsafe(24)
            self.app_settings["local_api_key"] = key
            try:
                self.persist_settings(self.app_settings)
            except Exception:
                pass

        key_file = self.secret_path("local_api_key.txt")
        try:
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, "w", encoding="utf-8") as f:
                f.write(key)
            try:
                os.chmod(key_file, 0o600)
            except OSError:
                pass
        except Exception:
            pass

        return key, key_file

    @staticmethod
    def check_secret(provided: str, expected: str) -> bool:
        return constant_time_equals(provided, expected)
