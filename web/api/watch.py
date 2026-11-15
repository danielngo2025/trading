"""Price watcher: re-rates a stock when its price moves away from where it was last analysed."""

import logging
import re
import threading
import time
import uuid
from datetime import date, datetime
from zoneinfo import ZoneInfo

from web.api import runner, store
from web.api.universe import _quote

log = logging.getLogger(__name__)

STATE = store.ROOT / "watch.json"
TSX_TZ = ZoneInfo("America/Toronto")
DEFAULT_SETTINGS = {
    "enabled": True,  # check prices
    "auto": True,  # re-run on a trigger; when off, the alert waits for a manual re-run
    "interval_min": 10,
    "move_pct": 5.0,
    "max_reruns_per_day": 3,
    "quick": True,  # skip the risk debate on automatic re-rates to halve the cost
}
PENDING = ("waiting", "running")
_lock = threading.Lock()
_wake = threading.Event()


def _load() -> dict:
    state = store._read(STATE) or {}
    state["settings"] = DEFAULT_SETTINGS | state.get("settings", {})
    state.setdefault("levels", {})
    state.setdefault("events", [])
    state.setdefault("prices", {})
    return state


def _save(state: dict) -> None:
    state["events"] = state["events"][-200:]
    store._write(STATE, state)


def get_state() -> dict:
    with _lock:
        state = _load()
    state["events"] = list(reversed(state["events"]))[:50]
    state["unseen"] = sum(1 for e in state["events"] if not e.get("seen"))
    state["market_open"] = market_open()
    state["suggestions"] = _suggestions()
    return state


def update_settings(changes: dict) -> dict:
    with _lock:
        state = _load()
        state["settings"] |= {k: v for k, v in changes.items() if k in DEFAULT_SETTINGS}
        _save(state)
    _wake.set()
    return state["settings"]


def set_levels(ticker: str, above: float | None, below: float | None) -> None:
    with _lock:
        state = _load()
        if above is None and below is None:
            state["levels"].pop(ticker, None)
        else:
            state["levels"][ticker] = {"above": above, "below": below}
        _save(state)


def mark_seen(event_id: str | None = None) -> None:
    with _lock:
        state = _load()
        for event in state["events"]:
            if event_id in (None, event["id"]):
                event["seen"] = True
        _save(state)


def rerun(event_id: str) -> dict | None:
    with _lock:
        state = _load()
        event = next((e for e in state["events"] if e["id"] == event_id), None)
        if event and event["status"] in ("triggered", "skipped", "failed"):
            event.update(status="waiting", error=None, seen=True)
            _save(state)
    _wake.set()
    return event


# Price levels the decision itself names, offered as one-click alert levels.
_LEVEL_WORDS = {"stop": "stop-loss", "support": "support", "entry": "entry", "pullback": "entry",
                "accumulat": "entry", "re-entry": "entry", "resistance": "resistance", "target": "target",
                "breakout": "breakout"}
_LEVEL = re.compile(r"(" + "|".join(_LEVEL_WORDS) + r")[^$\n]{0,60}?\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s?[–-]\s?\$?(\d[\d,]*(?:\.\d+)?))?",
                    re.IGNORECASE)


def suggested_levels(text: str, price: float) -> list[dict]:
    found: dict[float, dict] = {}
    for word, low, high in _LEVEL.findall(text or ""):
        values = [float(v.replace(",", "")) for v in (low, high) if v]
        side = "below" if values[0] < price else "above"
        # A range is crossed at its near end: the top when falling into it, the bottom when rising.
        value = max(values) if side == "below" else min(values)
        # Far-off figures are usually debt or revenue; ones at today's price would fire on noise.
        if not 0.01 <= abs(value / price - 1) <= 0.3 or value in found:
            continue
        found[value] = {"price": value, "side": side, "label": _LEVEL_WORDS[word.lower()]}
    return sorted(found.values(), key=lambda s: -s["price"])[:6]


def _suggestions() -> dict:
    out = {}
    for summary in store.latest_analyses():
        price = (summary.get("quote") or {}).get("price")
        full = store.get_analysis(summary["ticker"], summary["trade_date"]) or {}
        if price:
            out[summary["ticker"]] = suggested_levels((full.get("reports") or {}).get("final", ""), price)
    return out


def check_now() -> None:
    with _lock:
        state = _load()
        state["force_check"] = True
        _save(state)
    _wake.set()


def market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(TSX_TZ)
    minutes = now.hour * 60 + now.minute
    return now.weekday() < 5 and 9 * 60 + 30 <= minutes <= 16 * 60


def _notify(title: str, message: str) -> None:
    # Alerts surface in the web UI from the events list; this is just the server log.
    log.info("%s — %s", title, message)


def _trigger_reason(price: float, base: float, move_pct: float, level: dict | None) -> str | None:
    level = level or {}
    above, below = level.get("above"), level.get("below")
    # Levels fire on crossing, so a re-rate taken above the level doesn't fire again.
    if above is not None and base < above <= price:
        return f"rose above your level {above:g}"
    if below is not None and base > below >= price:
        return f"fell below your level {below:g}"
    move = (price / base - 1) * 100
    if abs(move) >= move_pct:
        return f"moved {move:+.1f}% since the last analysis"
    return None


def _check_prices(state: dict) -> None:
    settings = state["settings"]
    today = date.today().isoformat()
    for summary in store.latest_analyses():
        ticker = summary["ticker"]
        base = (summary.get("quote") or {}).get("price")
        if not base:
            continue
        price = _quote(ticker)["price"]
        if not price:
            continue
        state["prices"][ticker] = {"price": price, "at": time.time(), "base": base}

        events = [e for e in state["events"] if e["ticker"] == ticker]
        if any(e["status"] in PENDING for e in events) or any(e["day"] == today for e in events):
            continue  # one automatic re-rate per stock per day; a failing one mustn't loop
        reason = _trigger_reason(price, base, settings["move_pct"], state["levels"].get(ticker))
        if not reason:
            continue
        rerun = sum(1 for e in state["events"] if e["day"] == today and e["status"] not in ("skipped", "triggered"))
        capped = rerun >= settings["max_reruns_per_day"]
        status = "triggered" if not settings["auto"] else "skipped" if capped else "waiting"
        state["events"].append({
            "id": uuid.uuid4().hex[:8], "ticker": ticker, "company": summary.get("company"),
            "at": time.time(), "day": today, "price": price, "base_price": base,
            "base_date": summary["trade_date"], "reason": reason, "prev_rating": summary["rating"],
            "model": summary.get("model"), "status": status,
            "run_id": None, "new_rating": None, "seen": False,
        })
        currency = (summary.get("quote") or {}).get("currency") or ""
        tail = {"triggered": "re-run it when ready", "waiting": "re-rating now",
                "skipped": f"daily re-rate limit ({settings['max_reruns_per_day']}) reached"}[status]
        _notify(f"{ticker} {reason}", f"{price:.2f} {currency} (was {base:.2f}). Rated {summary['rating']} — {tail}.")


def _start_waiting(state: dict) -> None:
    waiting = [e for e in state["events"] if e["status"] == "waiting"]
    if not waiting or any(r["status"] in ("queued", "running") for r in store.list_runs()):
        return  # never compete with a run the user started; retry next tick
    model = waiting[0].get("model") or {}
    tickers = [{
        "ticker": e["ticker"], "company": e.get("company"),
        # Daily price data stops at the previous close, so tell the agents what triggered the re-rate.
        "note": (f"Price update: {e['ticker']} is now trading at {e['price']:.2f}. It was {e['base_price']:.2f} "
                 f"when last analysed on {e['base_date']} and rated {e['prev_rating']}; since then it {e['reason']}. "
                 "Reassess the rating in light of this move."),
    } for e in waiting]
    run = runner.create_run(tickers, None, state["settings"]["quick"], provider=model.get("provider"),
                            deep_model=model.get("deep"), quick_model=model.get("quick"), force=True)
    for e in waiting:
        e.update(status="running", run_id=run["id"])


def _collect_results(state: dict) -> None:
    for e in state["events"]:
        if e["status"] != "running":
            continue
        run = store.get_run(e["run_id"]) or {}
        item = next((i for i in run.get("items", []) if i["ticker"] == e["ticker"]), None)
        if not item or item["status"] in ("queued", "running"):
            if run.get("status") not in ("queued", "running"):
                e["status"] = "failed"  # run was cancelled or interrupted before reaching this stock
            continue
        if item["status"] != "done":
            e.update(status="failed", error=item.get("error"))
            _notify(f"{e['ticker']} re-rate failed", item.get("error") or item["status"])
            continue
        e.update(status="done", new_rating=item["rating"], seen=False)
        changed = e["new_rating"] != e["prev_rating"]
        verdict = f"{e['prev_rating']} → {e['new_rating']}" if changed else f"still {e['new_rating']}"
        _notify(f"{e['ticker']} re-rated: {verdict}", f"After it {e['reason']} ({e['price']:.2f}). Check the report.")


def _tick(last_check: float) -> float:
    with _lock:
        state = _load()
        _collect_results(state)
        settings = state["settings"]
        due = time.time() - last_check >= settings["interval_min"] * 60
        if state.pop("force_check", False) or (settings["enabled"] and market_open() and due):
            _check_prices(state)
            state["last_check"] = last_check = time.time()
        _start_waiting(state)
        _save(state)
    return last_check


def _loop() -> None:
    last_check = 0.0
    while True:
        try:
            last_check = _tick(last_check)
        except Exception:
            log.exception("Watcher tick failed")
        _wake.wait(60)
        _wake.clear()


def start_watcher() -> None:
    threading.Thread(target=_loop, daemon=True, name="ta-watcher").start()
