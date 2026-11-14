# TradingAgents web UI

Batch-run TradingAgents over S&P/TSX 60 stocks and browse the results.

```bash
# one-time
cd web/ui && npm install && npm run build && cd ../..

# serve API + built UI at http://127.0.0.1:8765
uv run python -m web.api.server

# UI dev with hot reload (http://localhost:5180, proxies /api to 8765)
cd web/ui && npm run dev
```

- Results land in `results/web/` (`analyses/<TICKER>/<date>.json`, `runs/<id>.json`).
- One run at a time; stocks are analysed sequentially. Restarting the server marks in-flight runs as interrupted.
- LLM provider/models come from `tradingagents/default_config.py`.
- Port: `TA_WEB_PORT` (default 8765).
