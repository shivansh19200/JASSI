import unittest
from unittest.mock import patch
from app.tools.resource_lookup import resolve_resource_url


class TestResourceLookup(unittest.TestCase):
    def test_keeps_direct_url(self):
        url, source = resolve_resource_url("https://example.com/course")
        self.assertEqual(url, "https://example.com/course")
        self.assertEqual(source, "direct")

    def test_name_falls_back_to_youtube_search_link(self):
        with patch("app.tools.resource_lookup.DDGS", side_effect=ImportError, create=True):
            url, source = resolve_resource_url("Striver A2Z DSA Course")
        self.assertIn("youtube.com/results?search_query=", url)
        self.assertEqual(source, "search_link")

    def test_coursera_fallback(self):
        url, source = resolve_resource_url("Machine Learning Specialization", "Coursera")
        self.assertIn("coursera.org/search?query=", url)
        self.assertEqual(source, "search_link")

    def test_empty_input(self):
        self.assertEqual(resolve_resource_url(""), ("", "missing"))


if __name__ == "__main__":
    unittest.main()
