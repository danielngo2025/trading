"""JSON-file storage for web runs and per-ticker analyses."""

import json
import threading
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from web.api.quality import data_warnings

ROOT = Path(DEFAULT_CONFIG["results_dir"]).resolve() / "web"
ANALYSES = ROOT / "analyses"
RUNS = ROOT / "runs"

_lock = threading.Lock()


def _safe(ticker: str) -> str:
    # Tickers like BIP.UN.TO are fine as dir names; guard only against path tricks.
    return ticker.replace("/", "_").replace("..", "_")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with _lock:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_analysis(analysis: dict) -> None:
    _write(ANALYSES / _safe(analysis["ticker"]) / f"{analysis['trade_date']}.json", analysis)


def get_analysis(ticker: str, trade_date: str) -> dict | None:
    return _read(ANALYSES / _safe(ticker) / f"{trade_date}.json")


def analysis_dates(ticker: str) -> list[str]:
    folder = ANALYSES / _safe(ticker)
    if not folder.exists():
        return []
    return sorted((p.stem for p in folder.glob("*.json")), reverse=True)


def latest_analyses() -> list[dict]:
    """Newest analysis per ticker, without the long report bodies."""
    out = []
    if not ANALYSES.exists():
        return out
    for folder in ANALYSES.iterdir():
        dates = analysis_dates(folder.name)
        if not dates:
            continue
        full = get_analysis(folder.name, dates[0])
        if full:
            summary = {k: v for k, v in full.items() if k != "reports"}
            summary["warnings"] = data_warnings(full.get("reports") or {}, (full.get("quote") or {}).get("price"))
            summary["history"] = [
                {"trade_date": d, "rating": (get_analysis(folder.name, d) or {}).get("rating")}
                for d in dates[:6]
            ]
            out.append(summary)
    return out


def save_run(run: dict) -> None:
    _write(RUNS / f"{run['id']}.json", run)


def get_run(run_id: str) -> dict | None:
    return _read(RUNS / f"{run_id}.json")


def list_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    runs = [r for r in (_read(p) for p in RUNS.glob("*.json")) if r]
    return sorted(runs, key=lambda r: r["created_at"], reverse=True)
