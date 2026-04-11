import os
import tempfile
import unittest
from pathlib import Path

from rag_manager import RAGDatabase, RAGManager


class RAGDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = RAGDatabase("demo")

    def test_add_chunks_stores_chunks_and_metadata(self):
        chunks = ["Alpha report about revenue.", "Beta report about hiring."]
        metadata = [{"source": "alpha.txt"}, {"source": "beta.txt"}]

        self.db.add_chunks(chunks, metadata)

        self.assertEqual(self.db.chunks, chunks)
        self.assertEqual(self.db.metadata, metadata)

    def test_build_keyword_index_includes_dates_words_and_amounts(self):
        self.db.add_chunks(["Invoice dated 12/04/2025 totals Rs. 1,250.00 for Widget Corp."])

        self.assertIn("invoice", self.db.keyword_index)
        self.assertIn("12/04/2025", self.db.keyword_index)
        self.assertIn("1,250.00", self.db.keyword_index)

    def test_extract_entities_finds_dates_amounts_percentages_and_key_terms(self):
        entities = self.db._extract_entities("Acme paid Rs. 4,500 on 01-02-2024 with 12.5% growth.")

        self.assertIn("01-02-2024", entities["dates"])
        self.assertTrue(any("4,500" in amount for amount in entities["amounts"]))
        self.assertIn("12.5%", entities["percentages"])
        self.assertIn("Acme", entities["key_terms"])

    def test_analyze_query_detects_quantitative_intent(self):
        self.db.add_chunks(["Revenue was Rs. 100.00 in alpha.txt"])

        info = self.db._analyze_query("How much revenue did Alpha make?")

        self.assertEqual(info["intent"], "quantitative")

    def test_analyze_query_detects_document_references_from_metadata(self):
        chunks = ["Alpha Corp revenue summary.", "Beta staffing plan."]
        metadata = [{"source": "alpha_report.txt"}, {"source": "beta_plan.txt"}]
        self.db.add_chunks(chunks, metadata)

        info = self.db._analyze_query("Summarize the alpha report")

        self.assertIn("alpha_report.txt", info["doc_refs"])

    def test_retrieve_ranks_best_matching_chunk_first(self):
        chunks = [
            "Alpha contract states the salary is Rs. 90,000 and joining date is 01/02/2025.",
            "Beta engineering roadmap covers platform migration.",
            "Gamma memo describes office seating changes.",
        ]
        self.db.add_chunks(chunks, [{"source": "alpha.txt"}, {"source": "beta.txt"}, {"source": "gamma.txt"}])

        results = self.db.retrieve("What salary is in the Alpha contract?", k=2)

        self.assertIn("Alpha contract", results[0][0])

    def test_retrieve_includes_document_reference_matches(self):
        chunks = ["Alpha project summary.", "Beta project summary."]
        metadata = [{"source": "alpha_notes.txt"}, {"source": "beta_notes.txt"}]
        self.db.add_chunks(chunks, metadata)

        results = self.db.retrieve("What is in beta notes?", k=2)

        self.assertTrue(any("Beta project summary" in chunk for chunk, _score in results))

    def test_retrieve_deduplicates_identical_chunks(self):
        duplicate = "Alpha duplicate content with revenue details and enough extra text to be indexed properly."
        self.db.add_chunks([duplicate, duplicate])

        results = self.db.retrieve("revenue details", k=5)

        self.assertEqual(len(results), 1)

    def test_get_chunk_source_prefers_metadata_source(self):
        self.db.add_chunks(["Alpha chunk content long enough for retrieval."], [{"source": "alpha.txt"}])

        self.assertEqual(self.db.get_chunk_source(0), "alpha.txt")

    def test_save_and_load_round_trip_database_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.db.add_chunks(["Alpha details with dates 01/01/2025 and Rs. 100.00."], [{"source": "alpha.txt"}])
            self.db.source_folder = "C:/docs"
            self.db.save(temp_dir)

            loaded = RAGDatabase.load(temp_dir)

        self.assertEqual(loaded.name, "demo")
        self.assertEqual(loaded.chunks, self.db.chunks)
        self.assertEqual(loaded.metadata, self.db.metadata)
        self.assertEqual(loaded.source_folder, "C:/docs")
        self.assertTrue(loaded.keyword_index)

    def test_load_rebuilds_indexes_when_cache_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.db.add_chunks(["Alpha details with dates 01/01/2025 and Rs. 100.00."], [{"source": "alpha.txt"}])
            self.db.save(temp_dir)
            os.remove(Path(temp_dir) / "indexes.pkl")

            loaded = RAGDatabase.load(temp_dir)

        self.assertIn("alpha", loaded.keyword_index)
        self.assertTrue(loaded.chunk_entities)


class RAGManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.manager = RAGManager(base_directory=str(self.base / "rag_store"))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_from_folder_builds_database_and_persists_it(self):
        docs = self.base / "docs"
        docs.mkdir()
        (docs / "alpha.txt").write_text(
            "Alpha contract signed on 01/02/2025 with revenue of Rs. 10,000.00.",
            encoding="utf-8",
        )

        db = self.manager.create_from_folder(str(docs), "contracts")

        self.assertIn("contracts", self.manager.databases)
        self.assertGreater(len(db.chunks), 0)
        self.assertTrue((self.base / "rag_store" / "contracts" / "metadata.json").exists())

    def test_create_from_folder_rejects_missing_folder(self):
        with self.assertRaises(ValueError):
            self.manager.create_from_folder(str(self.base / "missing"), "contracts")

    def test_create_from_folder_rejects_duplicate_name(self):
        docs = self.base / "docs"
        docs.mkdir()
        (docs / "alpha.txt").write_text("Alpha content with enough text for a chunk.", encoding="utf-8")
        self.manager.create_from_folder(str(docs), "contracts")

        with self.assertRaises(ValueError):
            self.manager.create_from_folder(str(docs), "contracts")

    def test_extract_texts_from_folder_only_returns_supported_files(self):
        docs = self.base / "docs"
        docs.mkdir()
        (docs / "alpha.txt").write_text("Alpha content with enough text for a chunk.", encoding="utf-8")
        (docs / "ignore.md").write_text("Should not be imported.", encoding="utf-8")

        items = self.manager._extract_texts_from_folder(str(docs))

        self.assertEqual([name for name, _text in items], ["alpha.txt"])

    def test_reindex_database_rebuilds_from_original_source_folder(self):
        docs = self.base / "docs"
        docs.mkdir()
        file_path = docs / "alpha.txt"
        file_path.write_text("Original alpha contract text with enough length to chunk properly.", encoding="utf-8")
        self.manager.create_from_folder(str(docs), "contracts")
        old_chunks = list(self.manager.databases["contracts"].chunks)

        file_path.write_text("Updated alpha contract text with revised compensation and start date details.", encoding="utf-8")
        db = self.manager.reindex_database("contracts")

        self.assertNotEqual(db.chunks, old_chunks)

    def test_delete_database_removes_memory_and_disk_state(self):
        docs = self.base / "docs"
        docs.mkdir()
        (docs / "alpha.txt").write_text("Alpha content with enough text for a chunk.", encoding="utf-8")
        self.manager.create_from_folder(str(docs), "contracts")

        deleted = self.manager.delete_database("contracts")

        self.assertTrue(deleted)
        self.assertNotIn("contracts", self.manager.databases)
        self.assertFalse((self.base / "rag_store" / "contracts").exists())

    def test_delete_database_returns_false_for_unknown_name(self):
        self.assertFalse(self.manager.delete_database("missing"))
