import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import anls_one, exact_one, text_normalize


class MetricTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(text_normalize("  Total: $1,200.00  "), "total: $1,200.00")

    def test_internal_spaces_are_significant(self):
        self.assertNotEqual(text_normalize("New  York"), text_normalize("New York"))

    def test_punctuation_is_significant(self):
        self.assertNotEqual(text_normalize("1,200"), text_normalize("1200"))

    def test_exact_multiple_references(self):
        self.assertEqual(exact_one("New York", ["NY", "New York"]), 1.0)

    def test_anls_threshold(self):
        self.assertEqual(anls_one("abcde", ["abcdf"], 0.5), 0.8)
        self.assertEqual(anls_one("x", ["abcdef"], 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
