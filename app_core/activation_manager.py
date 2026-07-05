"""Trial and activation state management for SIMPLE_AI."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable


class ActivationManager:
    """Manage trial status, machine-bound keys, and activation persistence."""

    def __init__(
        self,
        app_settings: dict,
        persist_settings: Callable[[dict], None],
        app_data_dir: str,
        trial_days: int = 30,
        passkey_env: str = "SIMPLIAI_PASSKEY",
        machine_hint: str | None = None,
        activation_store_dir: str | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.app_settings = app_settings if isinstance(app_settings, dict) else {}
        self.persist_settings = persist_settings
        self.app_data_dir = app_data_dir
        self.trial_days = int(trial_days)
        self.passkey_env = passkey_env
        self.machine_hint = machine_hint
        self.activation_store_dir = activation_store_dir
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def _utc_now_iso(self) -> str:
        return self.now_fn().replace(microsecond=0).isoformat()

    @staticmethod
    def _parse_iso_datetime(value):
        try:
            raw = str(value or "").strip()
            if not raw:
                return None
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None

    def secret_file_path(self) -> str:
        return os.path.join(self.app_data_dir, "activation_passkey.txt")

    def activation_secret(self) -> str:
        env_secret = str(os.environ.get(self.passkey_env, "")).strip()
        if env_secret:
            return env_secret

        secret_file = self.secret_file_path()
        try:
            if os.path.exists(secret_file):
                with open(secret_file, "r", encoding="utf-8") as f:
                    file_secret = str(f.readline() or "").strip()
                if file_secret:
                    return file_secret
        except Exception:
            pass

        local_secret = secrets.token_urlsafe(48)
        try:
            os.makedirs(os.path.dirname(secret_file), exist_ok=True)
            with open(secret_file, "w", encoding="utf-8") as f:
                f.write(local_secret + "\n")
        except Exception:
            pass
        return local_secret

    def activation_machine_hint(self) -> str:
        if self.machine_hint is not None:
            return self.machine_hint
        return f"{uuid.getnode()}:{os.environ.get('COMPUTERNAME', '')}"

    def activation_system_code(self) -> str:
        raw = self.activation_machine_hint().encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12].upper()

    def build_machine_bound_key(self, system_code: str) -> str:
        secret = self.activation_secret().encode("utf-8")
        msg = str(system_code or "").strip().upper().encode("utf-8")
        digest = hmac.new(secret, msg, hashlib.sha256).hexdigest()[:24].upper()
        return f"{str(system_code).strip().upper()}-{digest}"

    def is_valid_activation_key(self, entered_key: str) -> bool:
        expected = self.build_machine_bound_key(self.activation_system_code())
        provided = str(entered_key or "").strip().upper()
        return bool(provided and hmac.compare_digest(provided, expected))

    def activation_signature(self, first_opened_at: str, activated_at: str) -> str:
        payload = (
            f"{first_opened_at}|{activated_at}|"
            f"{self.activation_machine_hint()}|{self.activation_secret()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def activation_store_path(self) -> str:
        base = self.activation_store_dir
        if not base:
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
            base = os.path.join(base, "SIMPLIAI")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "activation_state.json")

    def load_activation_store(self) -> dict:
        path = self.activation_store_path()
        try:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_activation_store(self, payload: dict):
        path = self.activation_store_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload or {}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ACTIVATION] Could not write activation store: {e}")

    def initialize(self):
        changed = False
        if not isinstance(self.app_settings, dict):
            self.app_settings = {}
            changed = True

        persisted = self.load_activation_store()

        first_opened_at = str(persisted.get("trial_first_opened_at", "")).strip()
        if not first_opened_at:
            first_opened_at = str(self.app_settings.get("trial_first_opened_at", "")).strip()
        if not first_opened_at:
            first_opened_at = self._utc_now_iso()
            changed = True
        self.app_settings["trial_first_opened_at"] = first_opened_at

        activation = persisted.get("activation", {}) if isinstance(persisted.get("activation", {}), dict) else {}
        if not activation:
            activation = self.app_settings.get("activation", {})
        if not isinstance(activation, dict):
            activation = {"activated": False}
            changed = True
        elif activation.get("activated"):
            activated_at = str(activation.get("activated_at", "")).strip()
            sig = str(activation.get("sig", "")).strip()
            expected = (
                self.activation_signature(first_opened_at, activated_at)
                if first_opened_at and activated_at
                else ""
            )
            if not expected or not sig or not hmac.compare_digest(sig, expected):
                activation = {"activated": False}
                changed = True

        self.app_settings["activation"] = activation

        persisted_payload = {
            "trial_first_opened_at": first_opened_at,
            "activation": activation,
        }
        if persisted_payload != persisted:
            self.save_activation_store(persisted_payload)

        if changed:
            self.persist_settings(self.app_settings)

    def status(self) -> dict:
        first_opened_at = str(self.app_settings.get("trial_first_opened_at", "")).strip()
        first_dt = self._parse_iso_datetime(first_opened_at)
        if not first_dt:
            first_opened_at = self._utc_now_iso()
            self.app_settings["trial_first_opened_at"] = first_opened_at
            self.persist_settings(self.app_settings)
            first_dt = self._parse_iso_datetime(first_opened_at)

        if first_dt and first_dt.tzinfo is None:
            first_dt = first_dt.replace(tzinfo=timezone.utc)

        now_dt = self.now_fn()
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        days_used = max(0, (now_dt - first_dt).days) if first_dt else 0
        is_trial_active = days_used < self.trial_days
        days_left = max(0, self.trial_days - days_used)
        trial_expires_at = (
            (first_dt + timedelta(days=self.trial_days)).replace(microsecond=0).isoformat()
            if first_dt
            else ""
        )

        activation = self.app_settings.get("activation", {})
        is_activated = False
        activated_at = ""
        if isinstance(activation, dict) and activation.get("activated"):
            activated_at = str(activation.get("activated_at", "")).strip()
            sig = str(activation.get("sig", "")).strip()
            expected = (
                self.activation_signature(first_opened_at, activated_at)
                if first_opened_at and activated_at
                else ""
            )
            is_activated = bool(expected and sig and hmac.compare_digest(sig, expected))

        return {
            "trial_days_total": self.trial_days,
            "trial_days_left": days_left,
            "first_opened_at": first_opened_at,
            "trial_started_at": first_opened_at,
            "trial_expires_at": trial_expires_at,
            "days_used": days_used,
            "days_left": days_left,
            "is_trial_active": is_trial_active,
            "is_activated": is_activated,
            "activated_at": activated_at,
            "requires_passkey": (not is_activated and not is_trial_active),
            "system_code": self.activation_system_code(),
        }

    def has_full_access(self) -> bool:
        status = self.status()
        return bool(status.get("is_trial_active") or status.get("is_activated"))

    def activate_full_access(self, passkey: str) -> dict:
        provided = str(passkey or "").strip().upper()
        if not provided:
            return {"ok": False, "error": "Passkey is required"}
        if not self.is_valid_activation_key(provided):
            return {"ok": False, "error": "Invalid passkey"}

        first_opened_at = str(self.app_settings.get("trial_first_opened_at", "")).strip() or self._utc_now_iso()
        activated_at = self._utc_now_iso()
        self.app_settings["trial_first_opened_at"] = first_opened_at
        self.app_settings["activation"] = {
            "activated": True,
            "activated_at": activated_at,
            "sig": self.activation_signature(first_opened_at, activated_at),
        }
        self.persist_settings(self.app_settings)
        self.save_activation_store({
            "trial_first_opened_at": first_opened_at,
            "activation": self.app_settings["activation"],
        })
        return {"ok": True, "status": self.status()}
