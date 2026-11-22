"""Flags analyses whose agents worked from missing data, so a confident rating isn't taken at face value."""

import re

ANALYST_REPORTS = {"market": "Technical", "sentiment": "Sentiment", "news": "News", "fundamentals": "Fundamentals"}

FAILURE_PHRASES = (
    "unable to complete", "currently down", "unavailable", "i apologize", "error fetching",
    "could not retrieve", "cannot access", "unable to retrieve", "unable to access", "no data available",
)

FUNDAMENTAL_MARKERS = ("p/e", "pe ratio", "eps", "earnings per share", "revenue", "net income", "book value")


# Only currency-marked numbers, so RSI or MACD readings near the price don't count as quoting it.
_PRICE = re.compile(r"(?:\$|\bCAD|\bC\$|\bUSD)\s*\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
# Prices the report labels as such, written without a currency marker ("Price: 32.50", "trades at 32.50").
_LABELLED_PRICE = re.compile(
    r"\b(?:price|close|closed at|trading at|trades at)\b[^\d\n]{0,20}?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Saved quotes are taken when the run ends, which can be a session after the analysed close.
PRICE_TOLERANCE = 0.05


_STATED_PRICE = re.compile(
    r"current (?:price|close)[^\d\n]{0,40}?(?:\$|\bCAD|\bC\$)\s*~?\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    # Skip distances such as "current price $7.94 below" the upper band.
    r"(?!\d|[.,]\d)(?!\s*(?:below|above|under|over|higher|lower|away|from|points|%))",
    re.IGNORECASE,
)


def _stated_price_mismatch(text: str, price: float) -> float | None:
    for raw in _STATED_PRICE.findall(text):
        value = float(raw.replace(",", ""))
        if abs(value / price - 1) > PRICE_TOLERANCE:
            return value
    return None


def _mentions_price(text: str, price: float) -> bool:
    for raw in _PRICE.findall(text) + _LABELLED_PRICE.findall(text):
        value = float(raw.replace(",", ""))
        if abs(value / price - 1) <= PRICE_TOLERANCE:
            return True
    return False


def data_warnings(reports: dict, price: float | None = None) -> list[str]:
    warnings = []
    for key, label in ANALYST_REPORTS.items():
        text = (reports.get(key) or "").strip()
        if not text:
            warnings.append(f"{label} report is empty")
            continue
        # Only the opening matters; a full report may legitimately mention "unavailable" in passing.
        head = text[:600].lower()
        if any(p in head for p in FAILURE_PHRASES):
            warnings.append(f"{label} analyst could not get its data")
    fundamentals = (reports.get("fundamentals") or "").lower()
    if fundamentals and not any(m in fundamentals for m in FUNDAMENTAL_MARKERS):
        warnings.append("Fundamentals report has no valuation or earnings figures")
    if price:
        for key, label in ANALYST_REPORTS.items():
            wrong = _stated_price_mismatch(reports.get(key) or "", price)
            if wrong is not None:
                warnings.append(f"{label} report states the current price as {wrong:,.2f}, not {price:,.2f}")
    market = reports.get("market") or ""
    # A technical report that never quotes the real price was written from memory, not from data.
    if price and market and not _mentions_price(market, price):
        warnings.append(f"Technical report never mentions the actual price ({price:,.2f}); its levels are unreliable")
    return warnings
