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


class StatedPriceTests(unittest.TestCase):
    def stated_wrong(self, text, price):
        reports = {"market": text, "sentiment": "ok", "news": "ok", "fundamentals": "revenue up"}
        return [w for w in data_warnings(reports, price) if "states the current price" in w]

    def test_distance_from_price_is_not_a_price(self):
        text = ("- **Current Price:** $38.87\n"
                "- Upper Band: $46.81 (current price $7.94 below)\n- Lower Band: $36.13 (current price $2.74 above)")
        self.assertEqual(self.stated_wrong(text, 38.87), [])

    def test_wrong_stated_price_still_flagged(self):
        self.assertTrue(self.stated_wrong("Current price: $242.74, down 3%.", 288.37))


if __name__ == "__main__":
    unittest.main()
