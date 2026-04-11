"""SQLite chat storage and JSON→SQLite migration."""

import json
import os
import sqlite3
import threading
import time
from typing import Dict, List

from utils import app_data_path


# ══════════════════════════════════════════════════════════════════════════
# #40  SQLite Chat Storage
# ══════════════════════════════════════════════════════════════════════════
class ChatDatabase:
    """Single-file SQLite storage for chats, RAG settings, system prompts."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        with self._lock:
            c = self._conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS chats (
                chat_id   TEXT PRIMARY KEY,
                messages  TEXT NOT NULL,
                updated   REAL NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS chat_meta (
                chat_id   TEXT NOT NULL,
                key       TEXT NOT NULL,
                value     TEXT,
                PRIMARY KEY (chat_id, key)
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT
            )""")
            self._conn.commit()

    # ── Chat CRUD ──────────────────────────────────────────────────
    def save_chat(self, chat_id: str, messages: list):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO chats (chat_id, messages, updated) VALUES (?, ?, ?)",
                (chat_id, json.dumps(messages, ensure_ascii=False), time.time())
            )
            self._conn.commit()

    def load_chat(self, chat_id: str) -> list:
        with self._lock:
            row = self._conn.execute(
                "SELECT messages FROM chats WHERE chat_id=?", (chat_id,)
            ).fetchone()
        return json.loads(row[0]) if row else []

    def load_all_chats(self) -> Dict[str, list]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chat_id, messages FROM chats ORDER BY updated DESC"
            ).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    def sorted_chat_ids(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chat_id FROM chats ORDER BY updated DESC"
            ).fetchall()
        return [r[0] for r in rows]

    def delete_chat(self, chat_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))
            self._conn.execute("DELETE FROM chat_meta WHERE chat_id=?", (chat_id,))
            self._conn.commit()

    def rename_chat(self, old_id: str, new_id: str):
        with self._lock:
            self._conn.execute(
                "UPDATE chats SET chat_id=? WHERE chat_id=?", (new_id, old_id))
            self._conn.execute(
                "UPDATE chat_meta SET chat_id=? WHERE chat_id=?", (new_id, old_id))
            self._conn.commit()

    def save_all_chats(self, chats: Dict[str, list]):
        with self._lock:
            now = time.time()
            self._conn.executemany(
                "INSERT OR REPLACE INTO chats (chat_id, messages, updated) VALUES (?, ?, ?)",
                [(cid, json.dumps(msgs, ensure_ascii=False), now) for cid, msgs in chats.items()]
            )
            self._conn.commit()

    # ── Per-chat metadata (RAG settings, system prompts) ───────────
    def set_meta(self, chat_id: str, key: str, value: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO chat_meta (chat_id, key, value) VALUES (?, ?, ?)",
                (chat_id, key, value)
            )
            self._conn.commit()

    def get_meta(self, chat_id: str, key: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM chat_meta WHERE chat_id=? AND key=?", (chat_id, key)
            ).fetchone()
        return row[0] if row else None

    def get_all_meta(self, key: str) -> Dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chat_id, value FROM chat_meta WHERE key=?", (key,)
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def delete_meta(self, chat_id: str, key: str):
        with self._lock:
            self._conn.execute(
                "DELETE FROM chat_meta WHERE chat_id=? AND key=?", (chat_id, key))
            self._conn.commit()

    # ── Key-value store (app settings, model configs) ──────────────
    def set_kv(self, key: str, value):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False))
            )
            self._conn.commit()

    def get_kv(self, key: str, default=None):
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else default

    def close(self):
        self._conn.close()


# ══════════════════════════════════════════════════════════════════════════
# #41  Config Migration — JSON → SQLite
# ══════════════════════════════════════════════════════════════════════════
def migrate_json_to_sqlite(db: ChatDatabase):
    """One-time migration: import existing JSON files into SQLite, then
    rename the originals to *.json.bak so they aren't re-imported."""

    saved_dir = app_data_path("saved_chats")
    migrated_any = False

    # ── Migrate individual chat JSON files ─────────────────────────
    if os.path.isdir(saved_dir):
        for fname in os.listdir(saved_dir):
            if not fname.endswith(".json") or fname.startswith("_"):
                continue
            chat_id = fname[:-5]  # strip .json
            fp = os.path.join(saved_dir, fname)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    messages = json.load(f)
                db.save_chat(chat_id, messages)
                os.rename(fp, fp + ".bak")
                migrated_any = True
            except Exception:
                pass

        # ── Migrate _rag_settings.json ─────────────────────────────
        rag_path = os.path.join(saved_dir, "_rag_settings.json")
        if os.path.exists(rag_path):
            try:
                with open(rag_path, "r", encoding="utf-8") as f:
                    rag_map = json.load(f)
                for cid, val in rag_map.items():
                    db.set_meta(cid, "rag_db", val)
                os.rename(rag_path, rag_path + ".bak")
                migrated_any = True
            except Exception:
                pass

        # ── Migrate _system_prompts.json ───────────────────────────
        sp_path = os.path.join(saved_dir, "_system_prompts.json")
        if os.path.exists(sp_path):
            try:
                with open(sp_path, "r", encoding="utf-8") as f:
                    sp_map = json.load(f)
                for cid, val in sp_map.items():
                    db.set_meta(cid, "system_prompt", val)
                os.rename(sp_path, sp_path + ".bak")
                migrated_any = True
            except Exception:
                pass

    # ── Migrate model_configs.json ─────────────────────────────────
    mc_path = app_data_path("model_configs.json")
    if os.path.exists(mc_path):
        try:
            with open(mc_path, "r", encoding="utf-8") as f:
                db.set_kv("model_configs", json.load(f))
            os.rename(mc_path, mc_path + ".bak")
            migrated_any = True
        except Exception:
            pass

    # ── Migrate app_settings.json ──────────────────────────────────
    as_path = app_data_path("app_settings.json")
    if os.path.exists(as_path):
        try:
            with open(as_path, "r", encoding="utf-8") as f:
                db.set_kv("app_settings", json.load(f))
            os.rename(as_path, as_path + ".bak")
            migrated_any = True
        except Exception:
            pass

    return migrated_any
