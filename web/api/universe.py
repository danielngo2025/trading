"""S&P/TSX 60 constituents enriched with market cap and last price."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf

from cli.stock_lists import fetch_stock_list
from web.api.store import ROOT

CACHE = ROOT / "universe_tsx60.json"
CACHE_TTL = 12 * 3600


def _quote(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).fast_info
        last, prev = info.last_price, info.previous_close
        change = (last / prev - 1) * 100 if last and prev else None
        return {"market_cap": info.market_cap, "price": last, "change_pct": change, "currency": info.currency}
    except Exception:
        return {"market_cap": None, "price": None, "change_pct": None, "currency": None}


def get_universe(refresh: bool = False) -> dict:
    if not refresh and CACHE.exists() and time.time() - CACHE.stat().st_mtime < CACHE_TTL:
        return json.loads(CACHE.read_text(encoding="utf-8"))

    stocks = fetch_stock_list("S&P/TSX 60")
    with ThreadPoolExecutor(max_workers=8) as pool:
        quotes = list(pool.map(lambda s: _quote(s["ticker"]), stocks))
    rows = [{**s, **q} for s, q in zip(stocks, quotes)]
    rows.sort(key=lambda r: r["market_cap"] or 0, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    data = {"name": "S&P/TSX 60", "fetched_at": time.time(), "stocks": rows}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
