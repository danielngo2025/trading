import time
import logging

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config

logger = logging.getLogger(__name__)


def _cache_path(category: str, key: str) -> str:
    """Return the file path for a text-based cache entry.

    Cache files live under ``data_cache_dir/<category>/<key>.txt``.
    """
    config = get_config()
    cache_dir = os.path.join(config["data_cache_dir"], category)
    os.makedirs(cache_dir, exist_ok=True)
    # Sanitise key for filesystem (replace slashes, dots are fine)
    safe_key = key.replace("/", "_").replace("\\", "_")
    return os.path.join(cache_dir, f"{safe_key}.txt")


# Default TTLs in seconds per cache category
CACHE_TTL: dict[str, int] = {
    "fundamentals": 24 * 3600,   # 24 hours — changes quarterly
    "financials":   24 * 3600,   # 24 hours — balance sheet, cashflow, income
    "news":          1 * 3600,   # 1 hour   — changes frequently
}
DEFAULT_CACHE_TTL = 24 * 3600    # fallback: 24 hours


def cached_fetch(category: str, key: str, fetch_fn, ttl_seconds: int | None = None):
    """Return cached text result or call *fetch_fn* and cache the output.

    *ttl_seconds* overrides the per-category default. If the cached file
    is older than the TTL it is re-fetched.
    """
    from tradingagents import perf_logger

    ttl = ttl_seconds if ttl_seconds is not None else CACHE_TTL.get(category, DEFAULT_CACHE_TTL)
    path = _cache_path(category, key)

    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < ttl:
            with open(path, "r", encoding="utf-8") as f:
                perf_logger.log_time(f"data:{category}", "cache_hit", 0.0, key)
                return f.read()

    t0 = time.time()
    result = fetch_fn()
    elapsed = time.time() - t0
    perf_logger.log_time(f"data:{category}", "fetch", elapsed, key)

    with open(path, "w", encoding="utf-8") as f:
        f.write(result)
    return result


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


OHLCV_TTL = 24 * 3600  # re-download price data once per day

_ohlcv_memory_cache: dict[str, pd.DataFrame] = {}


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 5 years of data and caches per symbol. The disk cache expires
    after OHLCV_TTL seconds (default 24 h). An in-memory cache avoids
    re-reading the CSV within the same process.
    Rows after curr_date are filtered out so backtests never see future prices.
    """
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-ohlcv.csv",
    )

    from tradingagents import perf_logger

    # In-memory cache: avoid re-reading CSV for repeated indicator calls
    mem_key = symbol
    disk_fresh = os.path.exists(data_file) and (time.time() - os.path.getmtime(data_file)) < OHLCV_TTL

    if mem_key in _ohlcv_memory_cache and disk_fresh:
        perf_logger.log_time("data:ohlcv", "memory_hit", 0.0, symbol)
        data = _ohlcv_memory_cache[mem_key]
    elif disk_fresh:
        perf_logger.log_time("data:ohlcv", "disk_hit", 0.0, symbol)
        data = pd.read_csv(data_file, on_bad_lines="skip")
        data = _clean_dataframe(data)
        _ohlcv_memory_cache[mem_key] = data
    else:
        t0 = time.time()
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        elapsed = time.time() - t0
        perf_logger.log_time("data:ohlcv", "download", elapsed, symbol)
        data = data.reset_index()
        data.to_csv(data_file, index=False)
        data = _clean_dataframe(data)
        _ohlcv_memory_cache[mem_key] = data

    # Filter to curr_date to prevent look-ahead bias in backtesting
    data = data[data["Date"] <= curr_date_dt].copy()

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
