"""Price history, technical indicators and scheduled events for the stock detail page."""

import math
import threading
import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

CACHE_TTL = 15 * 60
CHART_DAYS = 66  # ~3 trading months; the 1M view slices the tail in the UI
RECENT_EVENT_DAYS = 30

_cache: dict[tuple, tuple[float, dict]] = {}
_lock = threading.Lock()


def _cached(key: tuple, build):
    with _lock:
        hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    value = build()
    with _lock:
        _cache[key] = (time.time(), value)
    return value


def _num(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 4)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def _pct(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return _num((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100)


def _build_chart(ticker: str) -> dict:
    # A year of history so the 50-day SMA and MACD are warmed up at the start of the 3M window.
    hist = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if hist.empty:
        return {"points": [], "summary": {}}
    close = hist["Close"]
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    frame = pd.DataFrame({
        "close": close, "volume": hist["Volume"],
        "sma20": close.rolling(20).mean(), "sma50": close.rolling(50).mean(),
        "rsi": _rsi(close), "macd": macd, "signal": signal, "hist": macd - signal,
    }).tail(CHART_DAYS)
    points = [{"date": idx.strftime("%Y-%m-%d"), **{k: _num(v) for k, v in row.items()}}
              for idx, row in frame.iterrows()]
    last = frame.iloc[-1]
    summary = {
        "change_1w": _pct(close, 5), "change_1m": _pct(close, 21), "change_3m": _pct(close, 63),
        "change_1y": _pct(close, len(close) - 1),
        "high_52w": _num(hist["High"].max()), "low_52w": _num(hist["Low"].min()),
        "rsi": _num(last["rsi"]), "macd": _num(last["macd"]), "signal": _num(last["signal"]),
        "sma20": _num(last["sma20"]), "sma50": _num(last["sma50"]),
        "sma200": _num(close.rolling(200).mean().iloc[-1]),
    }
    return {"points": points, "summary": summary}


def chart(ticker: str) -> dict:
    return _cached(("chart", ticker), lambda: _build_chart(ticker))


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _money(value) -> str:
    value = _num(value)
    if value is None:
        return ""
    return f"${value / 1e9:.2f}B" if abs(value) >= 1e9 else f"${value / 1e6:.0f}M" if abs(value) >= 1e6 else f"${value:.2f}"


def _build_events(ticker: str) -> list[dict]:
    stock = yf.Ticker(ticker)
    today = date.today()
    events: list[dict] = []
    try:
        cal = stock.calendar or {}
    except Exception:
        cal = {}

    for when in cal.get("Earnings Date") or []:
        when = _as_date(when)
        if when and when >= today:
            eps = _num(cal.get("Earnings Average"))
            rev = _money(cal.get("Revenue Average"))
            detail = ", ".join(x for x in (f"EPS est ${eps:.2f}" if eps is not None else "", f"revenue est {rev}" if rev else "") if x)
            events.append({"date": when.isoformat(), "kind": "earnings", "label": "Earnings report", "detail": detail})
            break
    for key, label in (("Ex-Dividend Date", "Ex-dividend"), ("Dividend Date", "Dividend paid")):
        when = _as_date(cal.get(key))
        if when and when >= today - timedelta(days=RECENT_EVENT_DAYS):
            events.append({"date": when.isoformat(), "kind": "dividend", "label": label, "detail": ""})

    # The last report shows whether the next one starts from a beat or a miss.
    try:
        past = stock.get_earnings_dates(limit=8)
    except Exception:
        past = None
    if past is not None and not past.empty:
        for idx, row in past.iterrows():
            when = idx.date()
            if when < today and (today - when).days <= 120 and not pd.isna(row.get("Reported EPS")):
                surprise = _num(row.get("Surprise(%)"))
                detail = f"EPS ${row['Reported EPS']:.2f} vs est ${row['EPS Estimate']:.2f}"
                if surprise is not None:
                    detail += f" ({'beat' if surprise >= 0 else 'miss'} {abs(surprise):.1f}%)"
                events.append({"date": when.isoformat(), "kind": "earnings", "label": "Last earnings", "detail": detail})
                break

    for event in events:
        event["days"] = (date.fromisoformat(event["date"]) - today).days
    return sorted(events, key=lambda e: e["date"])


def events(ticker: str) -> list[dict]:
    return _cached(("events", ticker), lambda: _build_events(ticker))
