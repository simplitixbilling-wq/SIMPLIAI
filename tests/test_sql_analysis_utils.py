import unittest

from app_core.sql_analysis_utils import (
    clean_sql_text,
    apply_fuzzy_table_mapping,
    extract_direct_sql,
    normalize_table_token,
    repair_missing_table_errors,
    repair_sql_identifiers,
    repair_sql_join_predicates,
    split_sql_statements,
    strip_non_sql_prefix,
    validate_sql_basic_structure,
    validate_sql_table_usage,
)


class SQLAnalysisUtilsTests(unittest.TestCase):
    def test_normalize_table_token_removes_noise(self):
        self.assertEqual(normalize_table_token("df_Books 2A.xlsx"), "dfbooks2axlsx")

    def test_extract_direct_sql_prefers_fenced_sql(self):
        text = "Run this:\n```sql\nSELECT * FROM df_sales\n```"

        self.assertEqual(extract_direct_sql(text), "SELECT * FROM df_sales")

    def test_extract_direct_sql_finds_inline_select(self):
        text = "please execute SELECT * FROM df_sales WHERE amount > 0"

        self.assertEqual(extract_direct_sql(text), "SELECT * FROM df_sales WHERE amount > 0")

    def test_extract_direct_sql_rejects_non_query_text(self):
        self.assertEqual(extract_direct_sql("summarize sales"), "")

    def test_clean_sql_text_strips_fences_and_logs(self):
        logs = []

        result = clean_sql_text("```sql\nSELECT * FROM t\n```", debug=logs.append)

        self.assertEqual(result, "SELECT * FROM t")
        self.assertTrue(any("clean_sql input" in line for line in logs))

    def test_strip_non_sql_prefix_removes_query_label(self):
        self.assertEqual(strip_non_sql_prefix("Query-2: SELECT * FROM t"), "SELECT * FROM t")

    def test_strip_non_sql_prefix_keeps_from_first_sql_keyword(self):
        self.assertEqual(strip_non_sql_prefix("Here is SQL:\nWITH x AS (SELECT 1) SELECT * FROM x"), "WITH x AS (SELECT 1) SELECT * FROM x")

    def test_split_sql_statements_ignores_semicolons_inside_quotes(self):
        sql = "SELECT ';' AS semi; SELECT 'x'';y' AS escaped;"
        self.assertEqual(split_sql_statements(sql), ["SELECT ';' AS semi", "SELECT 'x'';y' AS escaped"])

    def test_split_sql_statements_ignores_semicolons_inside_comments(self):
        sql = "SELECT 1; -- ignore ; here\nSELECT 2 /* ignore ; here */; SELECT 3"
        self.assertEqual(split_sql_statements(sql), ["SELECT 1", "SELECT 2", "SELECT 3"])

    def test_validate_sql_basic_structure_accepts_select_with_table(self):
        self.assertIsNone(validate_sql_basic_structure("SELECT * FROM df_sales", ["df_sales"]))

    def test_validate_sql_basic_structure_rejects_mutation(self):
        self.assertEqual(
            validate_sql_basic_structure("DROP TABLE df_sales", ["df_sales"]),
            "Non-read-only SQL detected",
        )

    def test_validate_sql_basic_structure_requires_select_or_with(self):
        self.assertEqual(
            validate_sql_basic_structure("EXPLAIN SELECT * FROM df_sales", ["df_sales"]),
            "SQL must start with SELECT or WITH",
        )

    def test_validate_sql_basic_structure_requires_known_table(self):
        self.assertEqual(
            validate_sql_basic_structure("SELECT * FROM other_table", ["df_sales"]),
            "SQL does not reference available tables",
        )

    def test_validate_sql_table_usage_requires_both_recon_tables(self):
        self.assertEqual(
            validate_sql_table_usage("SELECT * FROM df_a", ["df_a", "df_b"], is_reconciliation=True),
            "Reconciliation requires using both input tables",
        )

    def test_validate_sql_table_usage_rejects_same_table_join(self):
        self.assertEqual(
            validate_sql_table_usage(
                "SELECT * FROM df_a A JOIN df_a B ON A.id = B.id WHERE EXISTS (SELECT 1 FROM df_b)",
                ["df_a", "df_b"],
                is_reconciliation=True,
            ),
            "Join uses the same table on both sides; use both input files",
        )

    def test_validate_sql_table_usage_rejects_scalar_join_on(self):
        self.assertIn(
            "Invalid JOIN ON condition",
            validate_sql_table_usage("SELECT * FROM df_a JOIN df_b ON df_a.id", ["df_a", "df_b"]) or "",
        )

    def test_validate_sql_table_usage_accepts_boolean_join_on(self):
        self.assertIsNone(
            validate_sql_table_usage("SELECT * FROM df_a JOIN df_b ON df_a.id = df_b.id", ["df_a", "df_b"], is_reconciliation=True)
        )

    def test_apply_fuzzy_table_mapping_maps_common_aliases(self):
        sql, replacements = apply_fuzzy_table_mapping(
            "SELECT * FROM books JOIN gstr2a ON books.id = gstr2a.id",
            ["df_Books_Register", "df_GSTR_2A"],
        )
        self.assertIn("FROM df_Books_Register", sql)
        self.assertIn("JOIN df_GSTR_2A", sql)
        self.assertEqual(replacements["books"], "df_Books_Register")

    def test_apply_fuzzy_table_mapping_preserves_cte_names(self):
        sql, replacements = apply_fuzzy_table_mapping(
            "WITH books AS (SELECT * FROM df_Books_Register) SELECT * FROM books",
            ["df_Books_Register"],
        )
        self.assertIn("FROM books", sql)
        self.assertNotIn("books", replacements)

    def test_repair_missing_table_errors_uses_fuzzy_mapping(self):
        sql, replacements = repair_missing_table_errors(
            "SELECT * FROM books",
            "Table with name books does not exist",
            ["df_Books_Register"],
        )
        self.assertEqual(sql, "SELECT * FROM df_Books_Register")
        self.assertEqual(replacements["books"], "df_Books_Register")

    def test_repair_sql_identifiers_renames_numeric_alias_and_quotes_columns(self):
        result = repair_sql_identifiers(
            "SELECT 2A.Vendor name FROM df_books 2A",
            {"df_books": ["Vendor name", "deductor_tan"]},
        )

        self.assertEqual(result, 'SELECT T2A."Vendor name" FROM df_books T2A')

    def test_repair_sql_identifiers_corrects_near_miss_column(self):
        result = repair_sql_identifiers(
            "SELECT A.dedector_tan FROM df_books A",
            {"df_books": ["deductor_tan"]},
        )

        self.assertEqual(result, "SELECT A.deductor_tan FROM df_books A")

    def test_repair_sql_join_predicates_replaces_coalesce_on_clause(self):
        result = repair_sql_join_predicates(
            "SELECT * FROM df_a A JOIN df_b B ON COALESCE(A.id, B.id)"
        )

        self.assertEqual(
            result,
            "SELECT * FROM df_a A JOIN df_b B ON A.id IS NOT DISTINCT FROM B.id",
        )


if __name__ == "__main__":
    unittest.main()
