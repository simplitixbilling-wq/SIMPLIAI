import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from generation import GenerationMixin
from rag_handler import RAGHandlerMixin


class DummyGenerationApp(GenerationMixin):
    def __init__(self):
        self.message_history = []
        self.actual_n_ctx = 2048
        self.uploaded_content = ""


class DummyRAGDatabase:
    def __init__(self, chunks, sources):
        self.chunks = chunks
        self._sources = sources

    def get_chunk_source(self, chunk_idx):
        return self._sources[chunk_idx]


class DummyRAGManager:
    def __init__(self, results_by_name=None, dbs=None):
        self.results_by_name = results_by_name or {}
        self.databases = dbs or {}

    def list_databases(self):
        return list(self.results_by_name.keys() | self.databases.keys())

    def retrieve(self, rag_name, query, k=5):
        return self.results_by_name.get(rag_name, [])[:k]

    def _chunk_text(self, text, chunk_size=400, chunk_overlap=80):
        if len(text) <= chunk_size:
            return [text]
        return [text[i:i + chunk_size] for i in range(0, len(text), max(1, chunk_size - chunk_overlap))]


class DummyRAGApp(RAGHandlerMixin):
    def __init__(self):
        self.uploaded_content = ""
        self.model = None
        self.rag_manager = DummyRAGManager()
        self.temp_rag_db_name = None
        self.last_rag_hits = None
        self.current_rag_database = None
        self.current_chat_id = None
        self.chat_rag_settings = {}
        self.chat_system_prompts = {}
        self.status_updates = []
        self.signals = SimpleNamespace(update_status=SimpleNamespace(emit=lambda text: None), run_on_main=SimpleNamespace(emit=lambda func: None))

    def update_status(self, text):
        self.status_updates.append(text)

    def add_message(self, role, text):
        return (role, text)


class ConversationContextTests(unittest.TestCase):
    def setUp(self):
        self.app = DummyGenerationApp()

    def test_get_conversation_context_formats_roles(self):
        self.app.message_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        context = self.app._get_conversation_context()

        self.assertIn("User: Hello", context)
        self.assertIn("AI: Hi there", context)

    def test_get_conversation_context_strips_think_blocks(self):
        self.app.message_history = [
            {"role": "assistant", "content": "Visible<think>hidden</think> text"},
        ]

        context = self.app._get_conversation_context()

        self.assertEqual(context, "AI: Visible text")

    def test_get_conversation_context_drops_messages_that_become_empty_after_stripping(self):
        self.app.message_history = [{"role": "assistant", "content": "<think>hidden</think>"}]

        self.assertEqual(self.app._get_conversation_context(), "")

    def test_get_conversation_context_drops_single_oversized_message_when_it_cannot_fit(self):
        self.app.actual_n_ctx = 200
        self.app.message_history = [{"role": "user", "content": "x" * 500}]

        context = self.app._get_conversation_context(max_messages=1)

        self.assertEqual(context, "")

    def test_get_conversation_context_uses_only_recent_messages(self):
        self.app.message_history = [
            {"role": "user", "content": f"msg {i}"} for i in range(6)
        ]

        context = self.app._get_conversation_context(max_messages=2)

        self.assertIn("msg 4", context)
        self.assertIn("msg 5", context)
        self.assertNotIn("msg 0", context)

    def test_get_smart_context_prioritizes_relevant_lines(self):
        self.app.uploaded_content = "misc line\nalpha contract says salary\nother line\nalpha date line"

        context = self.app._get_smart_context("alpha salary", 30)

        self.assertIn("alpha contract says salary", context)

    def test_get_smart_context_respects_max_token_character_budget(self):
        self.app.uploaded_content = "\n".join(f"line {i} with alpha" for i in range(20))

        context = self.app._get_smart_context("alpha", 10)

        self.assertLessEqual(len(context), 40)

    def test_chunk_context_trims_on_section_boundaries(self):
        context = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100

        trimmed = self.app._chunk_context(context, 40)

        self.assertIn("A" * 100, trimmed)
        self.assertNotIn("C" * 100, trimmed)


class RAGHandlerContextTests(unittest.TestCase):
    def setUp(self):
        self.app = DummyRAGApp()

    def test_extract_rag_references_returns_clean_text_and_unique_names(self):
        clean, rag_names = self.app._extract_rag_references("Compare @legal and @finance with @legal")

        self.assertEqual(clean, "Compare  and  with")
        self.assertEqual(set(rag_names), {"legal", "finance"})

    def test_retrieve_rag_context_reports_missing_database(self):
        context, sources = self.app._retrieve_rag_context(["missing"], "query")

        self.assertIn("RAG database 'missing' not found", context)
        self.assertEqual(sources, [])

    def test_retrieve_rag_context_collects_context_and_sources(self):
        chunks = ["Chunk from alpha", "Chunk from beta"]
        db = DummyRAGDatabase(chunks, ["alpha.txt", "beta.txt"])
        self.app.rag_manager = DummyRAGManager(
            results_by_name={"docs": [(chunks[0], 0.9), (chunks[1], 0.7)]},
            dbs={"docs": db},
        )

        context, sources = self.app._retrieve_rag_context(["docs"], "alpha")

        self.assertIn("--- Context from [docs] ---", context)
        self.assertIn("Chunk from alpha", context)
        self.assertEqual(sources, ["alpha.txt", "beta.txt"])

    def test_retrieve_rag_context_falls_back_to_rag_name_when_sources_missing(self):
        chunks = ["Chunk with no resolvable source"]
        db = DummyRAGDatabase(chunks, [""])
        self.app.rag_manager = DummyRAGManager(
            results_by_name={"docs": [(chunks[0], 0.9)]},
            dbs={"docs": db},
        )

        _context, sources = self.app._retrieve_rag_context(["docs"], "alpha")

        self.assertEqual(sources, ["docs"])

    def test_get_relevant_chunk_uses_rag_result_when_available(self):
        chunk = "alpha salary clause with exact amount and start date"
        db = DummyRAGDatabase([chunk], ["alpha.txt"])
        self.app.uploaded_content = "fallback content"
        self.app.temp_rag_db_name = "temp"
        self.app.rag_manager = DummyRAGManager(results_by_name={"temp": [(chunk, 0.9)]}, dbs={"temp": db})

        result = self.app.get_relevant_chunk("salary amount")

        self.assertIn("salary clause", result)

    def test_get_relevant_chunk_returns_full_leading_content_for_generic_queries(self):
        self.app.uploaded_content = "alpha line one\nalpha line two\nalpha line three"

        result = self.app.get_relevant_chunk("summary")

        self.assertTrue(result.startswith("alpha line one"))

    def test_get_relevant_chunk_returns_matched_lines_for_specific_query(self):
        self.app.uploaded_content = "salary clause\ntravel policy\njoining date clause"

        result = self.app.get_relevant_chunk("joining date")

        self.assertIn("joining date clause", result)
        self.assertNotIn("travel policy", result)

    def test_create_temporary_rag_for_uploaded_file_registers_database(self):
        self.app.uploaded_content = "Alpha contract data. " * 40

        self.app._create_temporary_rag_for_uploaded_file("alpha.txt")

        self.assertIsNotNone(self.app.temp_rag_db_name)
        self.assertIn(self.app.temp_rag_db_name, self.app.rag_manager.databases)

    def test_create_temporary_rag_for_uploaded_file_replaces_previous_temp_database(self):
        self.app.uploaded_content = "First version. " * 40
        self.app._create_temporary_rag_for_uploaded_file("alpha.txt")
        first_name = self.app.temp_rag_db_name

        self.app.uploaded_content = "Second version. " * 40
        self.app._create_temporary_rag_for_uploaded_file("alpha.txt")

        self.assertNotEqual(first_name, self.app.temp_rag_db_name)
        self.assertNotIn(first_name, self.app.rag_manager.databases)

    def test_create_temporary_rag_for_uploaded_file_requires_content(self):
        with self.assertRaises(ValueError):
            self.app._create_temporary_rag_for_uploaded_file("alpha.txt")
