"""Unit tests for path_template.normalize_url_to_template and match."""

import unittest

from engine.path_template import (
    normalize_url_to_template,
    query_keys_of,
    path_matches_template,
)


class TestNormalizeUrlToTemplate(unittest.TestCase):
    def test_numeric_id_replaced(self):
        self.assertEqual(
            normalize_url_to_template("https://x.com/api/users/42/balance"),
            "/api/users/{id}/balance",
        )

    def test_uuid_v4_replaced(self):
        self.assertEqual(
            normalize_url_to_template(
                "https://x.com/api/orders/550e8400-e29b-41d4-a716-446655440000"
            ),
            "/api/orders/{uuid}",
        )

    def test_mongo_oid_replaced(self):
        self.assertEqual(
            normalize_url_to_template("https://x.com/api/docs/507f1f77bcf86cd799439011"),
            "/api/docs/{oid}",
        )

    def test_iso_date_replaced(self):
        self.assertEqual(
            normalize_url_to_template("https://x.com/reports/2026-04-17/summary"),
            "/reports/{date}/summary",
        )

    def test_non_matching_segments_kept(self):
        self.assertEqual(
            normalize_url_to_template("https://x.com/api/auth/login"),
            "/api/auth/login",
        )

    def test_multiple_dynamic_segments(self):
        self.assertEqual(
            normalize_url_to_template("https://x.com/api/users/42/posts/99"),
            "/api/users/{id}/posts/{id}",
        )

    def test_query_string_dropped(self):
        self.assertEqual(
            normalize_url_to_template("https://x.com/search?q=abc&page=2"),
            "/search",
        )


class TestQueryKeysOf(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(query_keys_of("https://x.com/a"), [])

    def test_sorted_unique(self):
        self.assertEqual(
            query_keys_of("https://x.com/a?z=1&a=2&a=3&m=4"),
            ["a", "m", "z"],
        )


class TestPathMatchesTemplate(unittest.TestCase):
    def test_exact_literal_matches(self):
        self.assertTrue(path_matches_template("/api/auth/login", "/api/auth/login"))

    def test_id_placeholder_matches_numeric(self):
        self.assertTrue(path_matches_template("/api/users/42/balance", "/api/users/{id}/balance"))

    def test_id_placeholder_matches_non_numeric_string(self):
        self.assertTrue(path_matches_template("/api/users/alice/balance", "/api/users/{id}/balance"))

    def test_length_mismatch_fails(self):
        self.assertFalse(path_matches_template("/api/users/42", "/api/users/{id}/balance"))

    def test_different_literal_segment_fails(self):
        self.assertFalse(path_matches_template("/api/admins/42", "/api/users/{id}"))

    def test_uuid_placeholder_matches(self):
        self.assertTrue(path_matches_template(
            "/api/orders/550e8400-e29b-41d4-a716-446655440000",
            "/api/orders/{uuid}",
        ))


if __name__ == "__main__":
    unittest.main()
