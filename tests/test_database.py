import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from database import ChatDatabase, migrate_json_to_sqlite


class ChatDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "chats.db")
        self.db = ChatDatabase(self.db_path)

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_save_and_load_chat_round_trips_messages(self):
        messages = [{"role": "user", "content": "hello"}]

        self.db.save_chat("Chat 1", messages)

        self.assertEqual(self.db.load_chat("Chat 1"), messages)

    def test_load_all_chats_returns_most_recent_first(self):
        with patch("database.time.time", side_effect=[100.0, 200.0]):
            self.db.save_chat("Chat 1", [{"role": "user", "content": "first"}])
            self.db.save_chat("Chat 2", [{"role": "assistant", "content": "second"}])

        self.assertEqual(list(self.db.load_all_chats().keys()), ["Chat 2", "Chat 1"])

    def test_sorted_chat_ids_returns_most_recent_first(self):
        with patch("database.time.time", side_effect=[100.0, 200.0, 300.0]):
            self.db.save_chat("Chat 1", [])
            self.db.save_chat("Chat 2", [])
            self.db.save_chat("Chat 3", [])

        self.assertEqual(self.db.sorted_chat_ids(), ["Chat 3", "Chat 2", "Chat 1"])

    def test_delete_chat_removes_chat_metadata(self):
        self.db.save_chat("Chat 1", [{"role": "user", "content": "x"}])
        self.db.set_meta("Chat 1", "rag_db", "docs")

        self.db.delete_chat("Chat 1")

        self.assertIsNone(self.db.get_meta("Chat 1", "rag_db"))

    def test_rename_chat_moves_messages(self):
        messages = [{"role": "assistant", "content": "renamed"}]
        self.db.save_chat("Old", messages)

        self.db.rename_chat("Old", "New")

        self.assertEqual(self.db.load_chat("New"), messages)
        self.assertEqual(self.db.load_chat("Old"), [])

    def test_rename_chat_moves_metadata(self):
        self.db.save_chat("Old", [])
        self.db.set_meta("Old", "system_prompt", "be brief")

        self.db.rename_chat("Old", "New")

        self.assertEqual(self.db.get_meta("New", "system_prompt"), "be brief")
        self.assertIsNone(self.db.get_meta("Old", "system_prompt"))

    def test_save_all_chats_persists_multiple_chats(self):
        chats = {
            "Chat 1": [{"role": "user", "content": "alpha"}],
            "Chat 2": [{"role": "assistant", "content": "beta"}],
        }

        self.db.save_all_chats(chats)

        self.assertEqual(self.db.load_chat("Chat 1"), chats["Chat 1"])
        self.assertEqual(self.db.load_chat("Chat 2"), chats["Chat 2"])

    def test_set_and_get_meta_round_trip(self):
        self.db.set_meta("Chat 1", "rag_db", "research")

        self.assertEqual(self.db.get_meta("Chat 1", "rag_db"), "research")

    def test_get_all_meta_filters_by_key(self):
        self.db.set_meta("Chat 1", "rag_db", "alpha")
        self.db.set_meta("Chat 2", "rag_db", "beta")
        self.db.set_meta("Chat 1", "system_prompt", "short")

        self.assertEqual(self.db.get_all_meta("rag_db"), {"Chat 1": "alpha", "Chat 2": "beta"})

    def test_set_and_get_kv_round_trip_complex_object(self):
        payload = {"temperature": 0.2, "models": ["a", "b"]}

        self.db.set_kv("app_settings", payload)

        self.assertEqual(self.db.get_kv("app_settings"), payload)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.saved_dir = self.base / "saved_chats"
        self.saved_dir.mkdir(parents=True, exist_ok=True)
        self.db = ChatDatabase(str(self.base / "migrate.db"))

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def _write_json(self, path: Path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _fake_app_data_path(self, relative_path=""):
        return str(self.base / relative_path) if relative_path else str(self.base)

    def test_migrate_json_to_sqlite_imports_chat_files(self):
        self._write_json(self.saved_dir / "Chat 1.json", [{"role": "user", "content": "hello"}])

        with patch("database.app_data_path", side_effect=self._fake_app_data_path):
            migrated = migrate_json_to_sqlite(self.db)

        self.assertTrue(migrated)
        self.assertEqual(self.db.load_chat("Chat 1"), [{"role": "user", "content": "hello"}])
        self.assertTrue((self.saved_dir / "Chat 1.json.bak").exists())

    def test_migrate_json_to_sqlite_imports_rag_settings(self):
        self._write_json(self.saved_dir / "_rag_settings.json", {"Chat 1": "legal_docs"})

        with patch("database.app_data_path", side_effect=self._fake_app_data_path):
            migrated = migrate_json_to_sqlite(self.db)

        self.assertTrue(migrated)
        self.assertEqual(self.db.get_meta("Chat 1", "rag_db"), "legal_docs")
        self.assertTrue((self.saved_dir / "_rag_settings.json.bak").exists())

    def test_migrate_json_to_sqlite_imports_system_prompts(self):
        self._write_json(self.saved_dir / "_system_prompts.json", {"Chat 1": "Act formal."})

        with patch("database.app_data_path", side_effect=self._fake_app_data_path):
            migrated = migrate_json_to_sqlite(self.db)

        self.assertTrue(migrated)
        self.assertEqual(self.db.get_meta("Chat 1", "system_prompt"), "Act formal.")
        self.assertTrue((self.saved_dir / "_system_prompts.json.bak").exists())

    def test_migrate_json_to_sqlite_ignores_invalid_json_without_failing(self):
        (self.saved_dir / "Chat 1.json").write_text("{invalid", encoding="utf-8")

        with patch("database.app_data_path", side_effect=self._fake_app_data_path):
            migrated = migrate_json_to_sqlite(self.db)

        self.assertFalse(migrated)
        self.assertEqual(self.db.load_chat("Chat 1"), [])
