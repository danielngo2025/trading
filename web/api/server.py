"""HTTP API for the TradingAgents web UI. Run: uv run python -m web.api.server"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.api import models, runner, store
from web.api.quality import data_warnings
from web.api.universe import get_universe

load_dotenv()

UI_DIST = Path(__file__).resolve().parent.parent / "ui" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    runner.start_worker()
    yield


app = FastAPI(title="TradingAgents", lifespan=lifespan)


class RunTicker(BaseModel):
    ticker: str
    company: str | None = None


class RunRequest(BaseModel):
    tickers: list[RunTicker] = Field(min_length=1, max_length=60)
    trade_date: str | None = None
    quick: bool = False
    provider: str | None = None
    deep_model: str | None = None
    quick_model: str | None = None
    force: bool = False
    reuse_fundamentals: bool = True


@app.get("/api/universe")
def universe(refresh: bool = False):
    return get_universe(refresh=refresh)


@app.get("/api/models")
def list_models():
    return models.list_models()


@app.get("/api/analyses")
def analyses():
    return {"analyses": store.latest_analyses()}


@app.get("/api/analyses/{ticker}")
def analysis(ticker: str, date: str | None = None):
    dates = store.analysis_dates(ticker)
    if not dates:
        raise HTTPException(404, f"No analysis for {ticker}")
    found = store.get_analysis(ticker, date or dates[0])
    if not found:
        raise HTTPException(404, f"No analysis for {ticker} on {date}")
    return {**found, "dates": dates, "warnings": data_warnings(found.get("reports") or {})}


@app.get("/api/runs")
def runs():
    return {"runs": store.list_runs()[:30]}


@app.get("/api/runs/{run_id}")
def run(run_id: str):
    found = store.get_run(run_id)
    if not found:
        raise HTTPException(404, "Run not found")
    return found


@app.post("/api/runs")
def create_run(body: RunRequest):
    active = [r for r in store.list_runs() if r["status"] in ("queued", "running")]
    if active:
        raise HTTPException(409, f"Run {active[0]['id']} is still in progress")
    if body.provider:
        default = models.list_models()["default"]
        error = models.validate(body.provider, body.deep_model or default["deep"], body.quick_model or default["quick"])
        if error:
            raise HTTPException(400, error)
    return runner.create_run(
        [t.model_dump() for t in body.tickers], body.trade_date, body.quick,
        provider=body.provider, deep_model=body.deep_model, quick_model=body.quick_model,
        force=body.force, reuse_fundamentals=body.reuse_fundamentals,
    )


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str):
    found = runner.cancel_run(run_id)
    if not found:
        raise HTTPException(404, "Run not found")
    return found


# Serve the built UI when present so one process is enough outside of dev.
if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        return FileResponse(UI_DIST / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("TA_WEB_PORT", "8765")))
