"""Flags analyses whose agents worked from missing data, so a confident rating isn't taken at face value."""

ANALYST_REPORTS = {"market": "Technical", "sentiment": "Sentiment", "news": "News", "fundamentals": "Fundamentals"}

FAILURE_PHRASES = (
    "unable to complete", "currently down", "unavailable", "i apologize", "error fetching",
    "could not retrieve", "cannot access", "unable to retrieve", "unable to access", "no data available",
)

FUNDAMENTAL_MARKERS = ("p/e", "pe ratio", "eps", "earnings per share", "revenue", "net income", "book value")


def data_warnings(reports: dict) -> list[str]:
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
    return warnings
