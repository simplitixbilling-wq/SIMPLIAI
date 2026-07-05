import tempfile
import unittest
from pathlib import Path

from app_core.security_utils import (
    constant_time_equals,
    is_allowed_local_origin,
    is_loopback_host,
    parse_content_length,
    safe_local_path,
    validate_restricted_python_snippet,
)


class SecurityUtilsTests(unittest.TestCase):
    def test_constant_time_equals_rejects_missing_or_wrong_values(self):
        self.assertTrue(constant_time_equals("secret", "secret"))
        self.assertFalse(constant_time_equals("", "secret"))
        self.assertFalse(constant_time_equals("wrong", "secret"))

    def test_loopback_host_policy(self):
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertTrue(is_loopback_host("::1"))
        self.assertFalse(is_loopback_host("0.0.0.0"))
        self.assertFalse(is_loopback_host("192.168.1.5"))

    def test_local_origin_policy(self):
        self.assertTrue(is_allowed_local_origin(""))
        self.assertTrue(is_allowed_local_origin("http://127.0.0.1:8765"))
        self.assertTrue(is_allowed_local_origin("https://localhost:3000"))
        self.assertFalse(is_allowed_local_origin("https://example.com"))

    def test_parse_content_length_limits_large_requests(self):
        self.assertEqual(parse_content_length("10", max_bytes=20), (10, None))
        self.assertEqual(parse_content_length("-1", max_bytes=20)[1], "Invalid Content-Length")
        self.assertIn("too large", parse_content_length("21", max_bytes=20)[1])

    def test_safe_local_path_rejects_outside_allowed_root(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            allowed_file = Path(root) / "input.jsonl"
            allowed_file.write_text("{}", encoding="utf-8")
            outside_file = Path(other) / "input.jsonl"
            outside_file.write_text("{}", encoding="utf-8")

            self.assertEqual(safe_local_path(str(allowed_file), allowed_roots=[root], must_exist=True), str(allowed_file.resolve()))
            with self.assertRaises(ValueError):
                safe_local_path(str(outside_file), allowed_roots=[root], must_exist=True)

    def test_restricted_python_snippet_import_policy(self):
        self.assertEqual(validate_restricted_python_snippet("import math\nprint(math.sqrt(4))", ["math"]), "")
        self.assertIn("Import blocked", validate_restricted_python_snippet("import os\nprint(1)", ["math"]))

    def test_restricted_python_snippet_blocks_dunder_and_unsafe_calls(self):
        self.assertIn("Unsafe attribute", validate_restricted_python_snippet("print((1).__class__)", ["math"]))
        self.assertIn("Unsafe call", validate_restricted_python_snippet("eval('1+1')", ["math"]))


if __name__ == "__main__":
    unittest.main()
