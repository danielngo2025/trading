"""Background worker that runs TradingAgents over a batch of tickers, one at a time."""

import logging
import queue
import re
import threading
import time
import traceback
import uuid
from datetime import date, datetime, timedelta

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.default_config import DEFAULT_CONFIG
from web.api import store
from web.api.quality import data_warnings
from web.api.universe import _quote

log = logging.getLogger(__name__)

RATINGS = ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL")

# (stage key, label, predicate on streamed state)
STAGES = [
    ("market", "Market analyst", lambda s: s.get("market_report")),
    ("sentiment", "Sentiment analyst", lambda s: s.get("sentiment_report")),
    ("news", "News analyst", lambda s: s.get("news_report")),
    ("fundamentals", "Fundamentals analyst", lambda s: s.get("fundamentals_report")),
    ("research", "Bull/bear debate", lambda s: (s.get("investment_debate_state") or {}).get("judge_decision")),
    ("trader", "Trader", lambda s: s.get("trader_investment_plan")),
    ("risk", "Risk & portfolio manager", lambda s: s.get("final_trade_decision")),
]

# Timing phases, in graph order. Parallel nodes land in one streamed update, so analysts are one phase.
PHASES = [
    ("analysts", "Analysts", lambda s: all(s.get(k) for k in
                                           ("market_report", "sentiment_report", "news_report", "fundamentals_report"))),
    ("debate", "Bull/bear debate", lambda s: (s.get("investment_debate_state") or {}).get("count", 0) > 0),
    ("research_manager", "Research manager", lambda s: (s.get("investment_debate_state") or {}).get("judge_decision")),
    ("trader", "Trader", lambda s: s.get("trader_investment_plan")),
    ("risk", "Risk debate", lambda s: (s.get("risk_debate_state") or {}).get("count", 0) > 0),
    ("portfolio_manager", "Portfolio manager", lambda s: s.get("final_trade_decision")),
]


def _phase_timings(started: float, marks: dict[str, float]) -> list[dict]:
    timings, prev = [], started
    for key, label in [(k, label) for k, label, _ in PHASES] + [("rating", "Rating extraction")]:
        if key in marks:
            timings.append({"key": key, "label": label, "seconds": round(marks[key] - prev, 1)})
            prev = marks[key]
    return timings


_queue: "queue.Queue[str]" = queue.Queue()
_run_lock = threading.Lock()
# Kept in memory because the worker's own run copy is what gets saved on every progress tick.
_cancelled: set[str] = set()
_RATING_RE = re.compile(r"\b(" + "|".join(RATINGS) + r")\b")


def _normalize_rating(text: str) -> str:
    match = _RATING_RE.search((text or "").upper())
    return match.group(1) if match else "UNKNOWN"


def _long_term_rating(report: str) -> str | None:
    # The prompt pins the rating to the first line, so later mentions of other ratings don't count.
    return _normalize_rating(report.strip().splitlines()[0]) if report.strip() else None


def _extract_reports(state: dict) -> dict:
    debate = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}
    return {
        "market": state.get("market_report", ""),
        "sentiment": state.get("sentiment_report", ""),
        "news": state.get("news_report", ""),
        "fundamentals": state.get("fundamentals_report", ""),
        "bull": debate.get("bull_history", ""),
        "bear": debate.get("bear_history", ""),
        "research_manager": debate.get("judge_decision", ""),
        "trader": state.get("trader_investment_plan", ""),
        "aggressive": risk.get("aggressive_history", ""),
        "conservative": risk.get("conservative_history", ""),
        "neutral": risk.get("neutral_history", ""),
        "final": state.get("final_trade_decision", ""),
        "value": state.get("value_investor_report", ""),
    }


# Fundamentals only move on earnings, so a recent report is as good as a fresh one.
FUNDAMENTALS_MAX_AGE_DAYS = 7
ALL_ANALYSTS = ["market", "social", "news", "fundamentals"]


class TokenCounter(BaseCallbackHandler):
    """Sums LLM token usage; analysts run in parallel threads, hence the lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.input = self.output = self.calls = 0
            self.cost_usd = 0.0

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for gen in gens:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                cost = (getattr(message, "response_metadata", None) or {}).get("cost_usd", 0)
                with self._lock:
                    self.input += usage.get("input_tokens", 0)
                    self.output += usage.get("output_tokens", 0)
                    self.calls += 1
                    self.cost_usd += cost

    def snapshot(self) -> dict:
        with self._lock:
            return {"input": self.input, "output": self.output, "calls": self.calls,
                    "cost_usd": round(self.cost_usd, 4)}


def _recent_fundamentals(ticker: str, trade_date: str) -> dict | None:
    target = datetime.strptime(trade_date, "%Y-%m-%d")
    for d in store.analysis_dates(ticker):
        age = (target - datetime.strptime(d, "%Y-%m-%d")).days
        if 0 <= age <= FUNDAMENTALS_MAX_AGE_DAYS:
            source = store.get_analysis(ticker, d) or {}
            reports = source.get("reports") or {}
            # Don't carry forward a report from a run that had any data gap; it may rest on a wrong price.
            if reports.get("fundamentals") and not data_warnings(reports, (source.get("quote") or {}).get("price")):
                return {"date": d, "text": reports["fundamentals"]}
    return None


def _resolve_trade_date(requested: str) -> str:
    # Same roll-back the graph applies, so a Saturday request finds Friday's saved analysis.
    from tradingagents.graph.propagation import _last_trading_day
    return _last_trading_day(requested)


def _update_item(run: dict, ticker: str, **fields) -> None:
    with _run_lock:
        if run["id"] in _cancelled:
            run["cancel_requested"] = True
        for item in run["items"]:
            if item["ticker"] == ticker:
                item.update(fields)
        store.save_run(run)


def _analyze(graph, counter: TokenCounter, run: dict, ticker: str, company: str | None,
             reused_fundamentals: dict | None, note: str | None = None) -> None:
    started = time.time()
    counter.reset()
    _update_item(run, ticker, status="running", stage=STAGES[0][0], started_at=started)

    state = graph.propagator.create_initial_state(ticker, run["trade_date"])
    if reused_fundamentals:
        state["fundamentals_report"] = reused_fundamentals["text"]
    if note:
        state["messages"] = [("human", f"{ticker}\n\n{note}")]
    final = None
    marks: dict[str, float] = {}
    for chunk in graph.graph.stream(state, **graph.propagator.get_graph_args()):
        final = chunk
        for key, _, done in PHASES:
            if key not in marks and done(chunk):
                marks[key] = time.time()
        done = sum(1 for _, _, has in STAGES if has(chunk))
        stage = STAGES[min(done, len(STAGES) - 1)][0]
        _update_item(run, ticker, stage=stage, stages_done=done, tokens=counter.snapshot())

    if final is None:
        raise RuntimeError("Graph produced no output")

    graph.ticker = ticker
    graph.curr_state = final
    graph._log_state(final["trade_date"], final)
    raw_rating = graph.process_signal(final["final_trade_decision"])
    marks["rating"] = time.time()

    analysis = {
        "ticker": ticker,
        "company": company,
        "trade_date": final["trade_date"],
        "rating": _normalize_rating(raw_rating),
        "value_rating": _long_term_rating(final.get("value_investor_report", "")),
        "run_id": run["id"],
        "created_at": time.time(),
        "duration_s": round(time.time() - started),
        "model": {"provider": run["config"]["llm_provider"], "deep": run["config"]["deep_think_llm"],
                  "quick": run["config"]["quick_think_llm"],
                  "analyst": getattr(graph.analyst_llm, "model", None) or run["config"]["quick_think_llm"]},
        "quote": _quote(ticker),
        "tokens": counter.snapshot(),
        "timings": _phase_timings(started, marks),
        "reused": {"fundamentals": reused_fundamentals["date"]} if reused_fundamentals else {},
        "note": note,
        "reports": _extract_reports(final),
    }
    store.save_analysis(analysis)
    _update_item(run, ticker, status="done", stage=None, stages_done=len(STAGES),
                 rating=analysis["rating"], tokens=analysis["tokens"], finished_at=time.time())


def _execute(run_id: str) -> None:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    run = store.get_run(run_id)
    if not run:
        return
    run["status"] = "running"
    run["started_at"] = time.time()
    store.save_run(run)

    counter = TokenCounter()
    graphs: dict[tuple, object] = {}

    def graph_for(analysts: list[str]):
        # Graphs are compiled per analyst set; tickers with reused fundamentals skip that analyst.
        key = tuple(analysts)
        if key not in graphs:
            graphs[key] = TradingAgentsGraph(selected_analysts=analysts, config=full_config(run), callbacks=[counter])
        return graphs[key]

    try:
        graph_for(ALL_ANALYSTS)
        trade_date = _resolve_trade_date(run["trade_date"])
    except Exception as exc:
        run.update(status="failed", error=f"Could not start agents: {exc}", finished_at=time.time())
        store.save_run(run)
        return

    for item in run["items"]:
        ticker = item["ticker"]
        if run_id in _cancelled:
            _update_item(run, ticker, status="cancelled")
            continue
        existing = None if run.get("force") else store.get_analysis(ticker, trade_date)
        # A result built on missing data isn't worth keeping; analyse it again.
        if existing and not data_warnings(existing.get("reports") or {}, (existing.get("quote") or {}).get("price")):
            now = time.time()
            _update_item(run, ticker, status="done", stages_done=len(STAGES), rating=existing["rating"],
                         cached=True, tokens={"input": 0, "output": 0, "calls": 0}, started_at=now, finished_at=now)
            continue
        try:
            reuse = run.get("reuse_fundamentals") and not run.get("force")
            fundamentals = _recent_fundamentals(ticker, trade_date) if reuse else None
            analysts = [a for a in ALL_ANALYSTS if not (fundamentals and a == "fundamentals")]
            _analyze(graph_for(analysts), counter, run, ticker, item.get("company"), fundamentals, item.get("note"))
        except Exception as exc:
            log.exception("Analysis failed for %s", ticker)
            _update_item(run, ticker, status="failed", error=str(exc),
                         trace=traceback.format_exc()[-2000:], finished_at=time.time())

    cancelled = run_id in _cancelled
    run["status"] = "cancelled" if cancelled else "done"
    run["cancel_requested"] = cancelled
    run["finished_at"] = time.time()
    store.save_run(run)


def _worker() -> None:
    while True:
        run_id = _queue.get()
        try:
            _execute(run_id)
        except Exception:
            log.exception("Run %s crashed", run_id)


def start_worker() -> None:
    # Runs left mid-flight by a previous server process can't resume; mark them so the UI doesn't spin forever.
    for run in store.list_runs():
        if run["status"] in ("queued", "running"):
            run["status"] = "interrupted"
            for item in run["items"]:
                if item["status"] in ("queued", "running"):
                    item["status"] = "interrupted"
            store.save_run(run)
    threading.Thread(target=_worker, daemon=True, name="ta-runner").start()


def create_run(tickers: list[dict], trade_date: str | None, quick: bool, provider: str | None = None,
               deep_model: str | None = None, quick_model: str | None = None, force: bool = False,
               reuse_fundamentals: bool = True) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["skip_risk_debate"] = quick
    if provider:
        config["llm_provider"] = provider
    if deep_model:
        config["deep_think_llm"] = deep_model
    if quick_model:
        config["quick_think_llm"] = quick_model
    run = {
        "id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        "created_at": time.time(),
        "trade_date": trade_date or date.today().isoformat(),
        "status": "queued",
        "quick": quick,
        "force": force,
        "reuse_fundamentals": reuse_fundamentals,
        "config": {k: config[k] for k in ("llm_provider", "deep_think_llm", "quick_think_llm", "analyst_llm",
                                          "max_debate_rounds", "max_risk_discuss_rounds",
                                          "skip_risk_debate")} | {"data_vendors": config["data_vendors"]},
        "items": [{"ticker": t["ticker"], "company": t.get("company"), "note": t.get("note"),
                   "status": "queued", "stage": None, "stages_done": 0} for t in tickers],
        "source": "watcher" if any(t.get("note") for t in tickers) else "user",
        "stages": [{"key": k, "label": label} for k, label, _ in STAGES],
    }
    store.save_run(run)
    _queue.put(run["id"])
    return run


def full_config(run: dict) -> dict:
    return DEFAULT_CONFIG | run["config"]


def cancel_run(run_id: str) -> dict | None:
    run = store.get_run(run_id)
    if run and run["status"] in ("queued", "running"):
        _cancelled.add(run_id)
        run["cancel_requested"] = True
        store.save_run(run)
    return run
