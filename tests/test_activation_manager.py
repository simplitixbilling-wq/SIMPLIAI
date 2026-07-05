import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app_core.activation_manager import ActivationManager


FIXED_NOW = datetime(2026, 7, 4, 12, 0, tzinfo=timezone.utc)


class ActivationManagerTests(unittest.TestCase):
    def make_manager(self, temp_dir, settings=None, machine_hint="machine:demo"):
        persisted = []

        def persist(payload):
            persisted.append(dict(payload))

        manager = ActivationManager(
            app_settings=settings if settings is not None else {},
            persist_settings=persist,
            app_data_dir=str(Path(temp_dir) / "app_data"),
            activation_store_dir=str(Path(temp_dir) / "activation_store"),
            machine_hint=machine_hint,
            now_fn=lambda: FIXED_NOW,
        )
        return manager, persisted

    def test_env_secret_wins_over_file_secret(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, _persisted = self.make_manager(temp_dir)
            secret_path = Path(manager.secret_file_path())
            secret_path.parent.mkdir(parents=True)
            secret_path.write_text("file-secret\n", encoding="utf-8")

            with patch.dict(os.environ, {"SIMPLIAI_PASSKEY": "env-secret"}):
                self.assertEqual(manager.activation_secret(), "env-secret")

    def test_file_secret_loads_when_env_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, _persisted = self.make_manager(temp_dir)
            secret_path = Path(manager.secret_file_path())
            secret_path.parent.mkdir(parents=True)
            secret_path.write_text("file-secret\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(manager.activation_secret(), "file-secret")

    def test_missing_secret_creates_local_secret(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, _persisted = self.make_manager(temp_dir)

            with patch.dict(os.environ, {}, clear=True):
                secret = manager.activation_secret()

            self.assertTrue(secret)
            self.assertEqual(Path(manager.secret_file_path()).read_text(encoding="utf-8").strip(), secret)

    def test_generated_activation_key_validates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, _persisted = self.make_manager(temp_dir)
            Path(manager.secret_file_path()).parent.mkdir(parents=True)
            Path(manager.secret_file_path()).write_text("master-secret\n", encoding="utf-8")

            key = manager.build_machine_bound_key(manager.activation_system_code())

            self.assertTrue(manager.is_valid_activation_key(key))
            self.assertFalse(manager.is_valid_activation_key(key + "BAD"))

    def test_initialize_persists_trial_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = {}
            manager, persisted = self.make_manager(temp_dir, settings=settings)

            manager.initialize()

            self.assertEqual(settings["trial_first_opened_at"], "2026-07-04T12:00:00+00:00")
            self.assertIn("activation", settings)
            self.assertTrue(persisted)
            self.assertTrue(Path(manager.activation_store_path()).exists())

    def test_expired_trial_requires_passkey(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = {
                "trial_first_opened_at": "2026-05-01T12:00:00+00:00",
                "activation": {"activated": False},
            }
            manager, _persisted = self.make_manager(temp_dir, settings=settings)

            status = manager.status()

            self.assertFalse(status["is_trial_active"])
            self.assertTrue(status["requires_passkey"])
            self.assertEqual(status["trial_days_left"], 0)
            self.assertEqual(status["trial_expires_at"], "2026-05-31T12:00:00+00:00")

    def test_status_includes_trial_period_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = {
                "trial_first_opened_at": "2026-06-25T12:00:00+00:00",
                "activation": {"activated": False},
            }
            manager, _persisted = self.make_manager(temp_dir, settings=settings)

            status = manager.status()

            self.assertEqual(status["trial_days_total"], 30)
            self.assertEqual(status["trial_days_left"], 21)
            self.assertEqual(status["days_left"], 21)
            self.assertEqual(status["trial_started_at"], "2026-06-25T12:00:00+00:00")
            self.assertEqual(status["trial_expires_at"], "2026-07-25T12:00:00+00:00")

    def test_activate_full_access_sets_signed_activation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = {"trial_first_opened_at": "2026-05-01T12:00:00+00:00"}
            manager, persisted = self.make_manager(temp_dir, settings=settings)
            Path(manager.secret_file_path()).parent.mkdir(parents=True)
            Path(manager.secret_file_path()).write_text("master-secret\n", encoding="utf-8")
            key = manager.build_machine_bound_key(manager.activation_system_code())

            result = manager.activate_full_access(key)

            self.assertTrue(result["ok"])
            self.assertTrue(settings["activation"]["activated"])
            self.assertTrue(result["status"]["is_activated"])
            self.assertTrue(persisted)


if __name__ == "__main__":
    unittest.main()
