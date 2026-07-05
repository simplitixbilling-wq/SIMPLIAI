import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app_core.local_api_server as local_api_server
from app_core.local_api_server import LocalApiServer
from app_core.local_secret_store import LocalSecretStore


class DummyBridge:
    def __init__(self, settings=None):
        self.app_settings = settings if settings is not None else {}
        self.saved_settings = []

    def save_app_settings(self, settings):
        self.app_settings.update(settings)
        self.saved_settings.append(dict(settings))
        return {"ok": True}


class LocalSecretStoreTests(unittest.TestCase):
    def test_existing_local_api_key_is_reused_and_written_to_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = {"local_api_key": "existing-key"}
            persisted = []
            store = LocalSecretStore(settings, lambda s: persisted.append(dict(s)), temp_dir)

            key, key_file = store.ensure_local_api_key()

            self.assertEqual(key, "existing-key")
            self.assertEqual(Path(key_file).read_text(encoding="utf-8"), "existing-key")
            self.assertEqual(persisted, [])

    def test_missing_local_api_key_is_generated_and_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = {}
            persisted = []
            store = LocalSecretStore(settings, lambda s: persisted.append(dict(s)), temp_dir)

            key, key_file = store.ensure_local_api_key()

            self.assertTrue(key)
            self.assertEqual(settings["local_api_key"], key)
            self.assertEqual(persisted[-1]["local_api_key"], key)
            self.assertEqual(Path(key_file).read_text(encoding="utf-8"), key)

    def test_check_secret_rejects_missing_or_wrong_key(self):
        self.assertTrue(LocalSecretStore.check_secret("abc", "abc"))
        self.assertFalse(LocalSecretStore.check_secret("", "abc"))
        self.assertFalse(LocalSecretStore.check_secret("wrong", "abc"))


class LocalApiServerSecretTests(unittest.TestCase):
    def test_server_reuses_settings_key_and_checks_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bridge = DummyBridge({"local_api_key": "server-key"})
            with patch.object(local_api_server, "app_data_path", lambda rel="": str(Path(temp_dir) / rel) if rel else temp_dir):
                server = LocalApiServer(bridge, port=0)

            self.assertEqual(server.api_key, "server-key")
            self.assertTrue(server._check_api_key({"X-API-Key": "server-key"}))
            self.assertFalse(server._check_api_key({"X-API-Key": "wrong"}))
            self.assertFalse(server._check_api_key({}))

    def test_start_metadata_does_not_expose_raw_api_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bridge = DummyBridge({"local_api_key": "server-key"})
            with patch.object(local_api_server, "app_data_path", lambda rel="": str(Path(temp_dir) / rel) if rel else temp_dir):
                server = LocalApiServer(bridge, port=0)
                meta = server.start()
                try:
                    self.assertTrue(meta["ok"])
                    self.assertNotIn("api_key", meta)
                    self.assertIn("api_key_file", meta)
                finally:
                    server.stop()

    def test_server_rejects_non_loopback_host(self):
        with self.assertRaises(ValueError):
            LocalApiServer(DummyBridge({"local_api_key": "server-key"}), host="0.0.0.0")

    def test_safe_jsonl_path_rejects_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as other_dir:
            outside = Path(other_dir) / "job.jsonl"
            outside.write_text("{}", encoding="utf-8")
            bridge = DummyBridge({"local_api_key": "server-key"})
            with patch.object(local_api_server, "app_data_path", lambda rel="": str(Path(app_dir) / rel) if rel else app_dir):
                server = LocalApiServer(bridge, port=0)

            with patch.object(server, "_allowed_file_roots", return_value=[app_dir]):
                with self.assertRaises(ValueError):
                    server._safe_jsonl_path(str(outside), must_exist=True)

    def test_sanitize_file_payloads_normalizes_allowed_path(self):
        with tempfile.TemporaryDirectory() as app_dir:
            allowed = Path(app_dir) / "input.csv"
            allowed.write_text("a,b\n1,2\n", encoding="utf-8")
            bridge = DummyBridge({"local_api_key": "server-key"})
            with patch.object(local_api_server, "app_data_path", lambda rel="": str(Path(app_dir) / rel) if rel else app_dir):
                server = LocalApiServer(bridge, port=0)

            sanitized = server._sanitize_file_payloads([str(allowed)])

            self.assertEqual(sanitized[0]["path"], str(allowed.resolve()))
            self.assertEqual(sanitized[0]["name"], "input.csv")
            self.assertGreater(sanitized[0]["size"], 0)


if __name__ == "__main__":
    unittest.main()
