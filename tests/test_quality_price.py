import unittest

from web.api.quality import data_warnings

FLAG = "never mentions the actual price"


def market_flagged(text: str, price: float) -> bool:
    reports = {"market": text, "sentiment": "ok", "news": "ok", "fundamentals": "revenue up"}
    return any(FLAG in w for w in data_warnings(reports, price))


class PriceMentionTests(unittest.TestCase):
    def test_unmarked_labelled_price_counts(self):
        self.assertFalse(market_flagged("**Current Position:**\n- Price: 32.50\n- RSI: 33.9", 32.5))
        self.assertFalse(market_flagged("The stock trades at 32.40 near support.", 32.5))

    def test_currency_marked_price_still_counts(self):
        self.assertFalse(market_flagged("Closed the week at $32.50.", 32.5))

    def test_wrong_or_missing_price_still_flagged(self):
        self.assertTrue(market_flagged("- Price: 242.74", 288.37))
        # Indicator readings near the price are not a price quote.
        self.assertTrue(market_flagged("RSI: 32.4, close_50_sma: 32.6", 32.5))


if __name__ == "__main__":
    unittest.main()
