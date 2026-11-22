import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from tradingagents.dataflows import y_finance


class StockDataRangeTests(unittest.TestCase):
    def test_end_date_bar_is_requested(self):
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame(
            {"Close": [32.06, 32.50]}, index=pd.to_datetime(["2026-11-19", "2026-11-20"]))
        with patch.object(y_finance.yf, "Ticker", return_value=ticker):
            out = y_finance.get_YFin_data_online("OTEX.TO", "2026-11-15", "2026-11-20")
        ticker.history.assert_called_once_with(start="2026-11-15", end="2026-11-21")
        self.assertIn("2026-11-20", out)
        self.assertIn("to 2026-11-20", out)


if __name__ == "__main__":
    unittest.main()
