import unittest

from web.api.runner import _long_term_rating


class LongTermRatingTests(unittest.TestCase):
    def test_reads_rating_from_first_line_only(self):
        report = "**Long-term rating: Overweight** — moat intact.\n\n| Principle | Evidence | Verdict |\nSell-side notes say Sell."
        self.assertEqual(_long_term_rating(report), "OVERWEIGHT")

    def test_missing_report_has_no_rating(self):
        self.assertIsNone(_long_term_rating(""))
        self.assertEqual(_long_term_rating("No rating given"), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
