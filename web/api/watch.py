"""Price watcher: re-rates a stock when its price moves away from where it was last analysed."""

import logging
import subprocess
import sys
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
    "enabled": True,
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


def mark_seen() -> None:
    with _lock:
        state = _load()
        for event in state["events"]:
            event["seen"] = True
        _save(state)


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
    log.info("%s — %s", title, message)
    if sys.platform != "darwin":
        return
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{esc(message)}" with title "{esc(title)}" sound name "Glass"'
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)


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
        rerun = sum(1 for e in state["events"] if e["day"] == today and e["status"] != "skipped")
        capped = rerun >= settings["max_reruns_per_day"]
        state["events"].append({
            "id": uuid.uuid4().hex[:8], "ticker": ticker, "company": summary.get("company"),
            "at": time.time(), "day": today, "price": price, "base_price": base,
            "base_date": summary["trade_date"], "reason": reason, "prev_rating": summary["rating"],
            "model": summary.get("model"), "status": "skipped" if capped else "waiting",
            "run_id": None, "new_rating": None, "seen": False,
        })
        currency = (summary.get("quote") or {}).get("currency") or ""
        tail = f"daily re-rate limit ({settings['max_reruns_per_day']}) reached" if capped else "re-rating now"
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
        if settings["enabled"]:
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
