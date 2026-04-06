"""Generate a concise HTML summary report with Chart.js charts from trading analysis state."""

import datetime
import json
import re
from pathlib import Path


def _extract_price(text: str) -> float | None:
    m = re.search(r"\*\*Close:?\*\*\s*\$([0-9,.]+)", text)
    if not m:
        m = re.search(r"Close:?\s*\$([0-9,.]+)", text)
    return float(m.group(1).replace(",", "")) if m else None


def _extract_dollar(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1).replace(",", "")) if m else None


def _extract_table_series(header_pattern: str, text: str) -> list[dict]:
    """Extract a markdown table with Date | Value columns."""
    results = []
    m = re.search(header_pattern, text)
    if not m:
        return results
    pos = m.end()
    lines = text[pos:].split("\n")
    found_data = False
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            if found_data:
                break
            continue
        if line.startswith("|---") or line.startswith("| ---"):
            continue
        cells = [c.strip().strip("*") for c in line.split("|")[1:-1]]
        if len(cells) >= 2:
            try:
                val = float(re.sub(r"[^0-9.\-]", "", cells[1]))
                results.append({"date": cells[0], "value": val})
                found_data = True
            except (ValueError, IndexError):
                continue
    return results


def _extract_summary_table(text: str) -> list[dict]:
    """Extract the Summary Table with Indicator | Current Value | Signal | Key Insight."""
    results = []
    m = re.search(r"## Summary Table", text)
    if not m:
        return results
    pos = m.end()
    lines = text[pos:].split("\n")
    found_data = False
    for line in lines:
        line = line.strip()
        if line.startswith("---") and not line.startswith("|"):
            break
        if not line.startswith("|"):
            if found_data:
                break
            continue
        if line.startswith("|---") or line.startswith("| ---"):
            continue
        cells = [c.strip().strip("*") for c in line.split("|")[1:-1]]
        if len(cells) >= 3 and cells[0] not in ("Indicator", ""):
            results.append({
                "indicator": cells[0],
                "value": cells[1],
                "signal": cells[2],
                "insight": cells[3] if len(cells) > 3 else "",
            })
            found_data = True
    return results


def _extract_decision(text: str) -> str:
    """Extract final decision from portfolio manager text."""
    m = re.search(r"Rating:\s*\*\*(\w+)\*\*", text)
    if m:
        return m.group(1).capitalize()
    m = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*\*(\w+)\*\*", text)
    if m:
        return m.group(1).capitalize()
    return "N/A"


def _extract_executive_summary(text: str) -> str:
    """Extract executive summary paragraph from a report section."""
    m = re.search(r"##\s*(?:Executive Summary|1\.\s*Executive Summary)\s*\n+(.*?)(?:\n##|\n---)", text, re.DOTALL)
    if m:
        summary = m.group(1).strip()
        # Take first paragraph only
        para = summary.split("\n\n")[0].strip()
        # Truncate to ~300 chars
        if len(para) > 300:
            para = para[:297] + "..."
        return para
    # Fallback: first meaningful paragraph
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 80 and not line.startswith("#") and not line.startswith("|") and not line.startswith("---"):
            return line[:300] + ("..." if len(line) > 300 else "")
    return ""


def _extract_action_items(text: str) -> list[str]:
    """Extract action items from portfolio manager decision."""
    items = []
    # Look for Entry/Exit, Hedging, Rotation sections
    for pattern in [
        r"\*\*(?:Entry/Exit Strategy|Action):?\*\*\s*\n((?:[-•]\s+.*\n?)+)",
        r"\*\*Hedging:?\*\*\s*\n((?:[-•]\s+.*\n?)+)",
        r"\*\*Rotation:?\*\*\s*\n((?:[-•]\s+.*\n?)+)",
    ]:
        m = re.search(pattern, text)
        if m:
            for line in m.group(1).strip().split("\n"):
                line = line.strip().lstrip("-•").strip()
                if line:
                    items.append(line)
    # Also try inline format
    if not items:
        for pattern in [
            r"\*\*Action:?\*\*\s*(.+)",
            r"Hard stop at\s*\*\*(\$[0-9,.]+)\*\*",
        ]:
            m = re.search(pattern, text)
            if m:
                items.append(m.group(1).strip())
    # Strip markdown bold/italic markers
    items = [re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", i) for i in items]
    return items[:6]


def _extract_bullets(text: str, max_count: int = 3) -> list[str]:
    """Extract first N bullet points from debate text."""
    bullets = []
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"^[-•*]\s+", line) or re.match(r"^\d+\.\s+", line):
            clean = re.sub(r"^[-•*\d.]+\s*", "", line).strip()
            if len(clean) > 20:
                if len(clean) > 200:
                    clean = clean[:197] + "..."
                bullets.append(clean)
                if len(bullets) >= max_count:
                    break
    return bullets


def extract_key_data(final_state: dict, ticker: str) -> dict:
    """Parse LLM prose reports to extract structured data for the summary."""
    market = final_state.get("market_report", "")
    sentiment = final_state.get("sentiment_report", "")
    news = final_state.get("news_report", "")
    fundamentals = final_state.get("fundamentals_report", "")
    debate = final_state.get("investment_debate_state", {})
    risk = final_state.get("risk_debate_state", {})
    portfolio_decision = risk.get("judge_decision", "")
    trader_plan = final_state.get("trader_investment_plan", "")

    # Summary table from market report (extract first - used for price levels)
    summary_table = _extract_summary_table(market)

    # Build a lookup from summary table for reliable values
    st_lookup = {}
    for row in summary_table:
        val_str = row["value"].strip().lstrip("$").replace(",", "")
        try:
            st_lookup[row["indicator"].lower()] = float(re.sub(r"[^0-9.\-]", "", val_str))
        except (ValueError, IndexError):
            pass

    # Price data - prefer summary table, fallback to regex
    current_price = st_lookup.get("price (close)") or _extract_price(market)
    sma_50 = st_lookup.get("50 sma") or _extract_dollar(r"\*\*50 SMA:?\*\*\s*\$([0-9,.]+)", market)
    sma_200 = st_lookup.get("200 sma") or _extract_dollar(r"\*\*200 SMA:?\*\*\s*\$([0-9,.]+)", market)
    support = _extract_dollar(r"(?:Recent Low|Lower Band|support)[^$]*\$([0-9,.]+)", market)
    stop_loss = _extract_dollar(r"(?:stop|Stop)[^$]*\$([0-9,.]+)", portfolio_decision)

    # Final decision
    decision = _extract_decision(portfolio_decision)
    if decision == "N/A":
        decision = _extract_decision(trader_plan)

    # Time series
    rsi_series = _extract_table_series(r"\|\s*Date\s*\|\s*RSI\s*\|", market)
    macd_hist_series = _extract_table_series(r"\|\s*Date\s*\|\s*Histogram\s*\|", market)
    atr_series = _extract_table_series(r"\|\s*Date\s*\|\s*ATR\s*\|", market)
    macd_series = _extract_table_series(r"\|\s*Date\s*\|\s*MACD\s*\|", market)

    # Analyst summaries
    analyst_summaries = {}
    for key, name, text in [
        ("market", "Market", market),
        ("sentiment", "Sentiment", sentiment),
        ("news", "News", news),
        ("fundamentals", "Fundamentals", fundamentals),
    ]:
        tx_m = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*?\*?(\w+)", text)
        tx = tx_m.group(1).upper() if tx_m else ""
        summary = _extract_executive_summary(text)
        analyst_summaries[key] = {"signal": tx, "summary": summary, "name": name}

    # Bull/Bear bullets
    bull_bullets = _extract_bullets(debate.get("bull_history", ""))
    bear_bullets = _extract_bullets(debate.get("bear_history", ""))

    # Action items
    action_items = _extract_action_items(portfolio_decision)

    return {
        "ticker": ticker,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "decision": decision,
        "current_price": current_price,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "support": support,
        "stop_loss": stop_loss,
        "summary_table": summary_table,
        "rsi_series": rsi_series,
        "macd_hist_series": macd_hist_series,
        "macd_series": macd_series,
        "atr_series": atr_series,
        "analyst_summaries": analyst_summaries,
        "bull_bullets": bull_bullets,
        "bear_bullets": bear_bullets,
        "action_items": action_items,
    }


def _decision_color(decision: str) -> str:
    d = decision.lower()
    if d in ("buy", "overweight"):
        return "#16a34a"
    if d in ("hold",):
        return "#d97706"
    if d in ("sell", "underweight"):
        return "#dc2626"
    return "#6b7280"


def _signal_color(signal: str) -> str:
    s = signal.lower()
    if "bullish" in s or "constructive" in s:
        return "#16a34a"
    if "bearish" in s or "elevated" in s:
        return "#dc2626"
    return "#d97706"


def generate_summary_html(data: dict) -> str:
    """Generate a single-page HTML summary report with Chart.js charts."""
    ticker = data["ticker"]
    decision = data["decision"]
    decision_clr = _decision_color(decision)
    price = data["current_price"]
    price_str = f"${price:,.2f}" if price else "N/A"

    # Key levels row
    levels = []
    if data["sma_200"]:
        levels.append(f'<div class="level"><span class="label">200 SMA</span><span class="val">${data["sma_200"]:,.2f}</span></div>')
    if data["sma_50"]:
        levels.append(f'<div class="level"><span class="label">50 SMA</span><span class="val">${data["sma_50"]:,.2f}</span></div>')
    if data["support"]:
        levels.append(f'<div class="level"><span class="label">Support</span><span class="val">${data["support"]:,.2f}</span></div>')
    if data["stop_loss"]:
        levels.append(f'<div class="level"><span class="label">Stop Loss</span><span class="val">${data["stop_loss"]:,.2f}</span></div>')
    levels_html = "\n".join(levels)

    # Summary table rows
    table_rows = ""
    for row in data["summary_table"]:
        sig_clr = _signal_color(row["signal"])
        table_rows += f"""<tr>
            <td>{row['indicator']}</td>
            <td class="mono">{row['value']}</td>
            <td><span class="signal-badge" style="background:{sig_clr}">{row['signal']}</span></td>
            <td class="insight">{row['insight']}</td>
        </tr>\n"""

    # Chart data
    rsi_labels = json.dumps([p["date"] for p in data["rsi_series"]])
    rsi_values = json.dumps([p["value"] for p in data["rsi_series"]])
    macd_h_labels = json.dumps([p["date"] for p in data["macd_hist_series"]])
    macd_h_values = json.dumps([p["value"] for p in data["macd_hist_series"]])
    macd_h_colors = json.dumps(["#16a34a" if v["value"] >= 0 else "#dc2626" for v in data["macd_hist_series"]])
    atr_labels = json.dumps([p["date"] for p in data["atr_series"]])
    atr_values = json.dumps([p["value"] for p in data["atr_series"]])

    # Analyst summaries
    analyst_cards = ""
    for key in ("market", "sentiment", "news", "fundamentals"):
        a = data["analyst_summaries"].get(key, {})
        sig = a.get("signal", "")
        sig_badge = f'<span class="signal-badge" style="background:{_decision_color(sig)}">{sig}</span>' if sig else ""
        analyst_cards += f"""<div class="analyst-card">
            <div class="analyst-header"><strong>{a.get('name', key.title())}</strong> {sig_badge}</div>
            <p>{a.get('summary', '')}</p>
        </div>\n"""

    # Bull/Bear bullets
    bull_html = "".join(f"<li>{b}</li>" for b in data["bull_bullets"])
    bear_html = "".join(f"<li>{b}</li>" for b in data["bear_bullets"])

    # Action items
    action_html = "".join(f"<li>{a}</li>" for a in data["action_items"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Analysis: {ticker}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"></script>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0; --muted: #94a3b8; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: var(--text); padding: 24px; max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; }}
  .header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
  .header-left {{ display: flex; align-items: center; gap: 16px; }}
  .decision-badge {{ font-size: 1.1rem; font-weight: 700; padding: 6px 18px; border-radius: 6px; color: #fff; }}
  .price {{ font-size: 2rem; font-weight: 700; }}
  .date {{ color: var(--muted); font-size: 0.85rem; }}
  .levels {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
  .level {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; min-width: 120px; }}
  .level .label {{ display: block; color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .level .val {{ font-size: 1.1rem; font-weight: 600; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 16px; }}
  .card h2 {{ font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 14px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  @media (max-width: 768px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }}
  .mono {{ font-family: 'SF Mono', Consolas, monospace; }}
  .insight {{ color: var(--muted); font-size: 0.8rem; }}
  .signal-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; color: #fff; font-size: 0.75rem; font-weight: 600; }}
  .analyst-card {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
  .analyst-header {{ margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }}
  .analyst-card p {{ color: var(--muted); font-size: 0.83rem; line-height: 1.5; }}
  .debate {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .debate-col h3 {{ font-size: 0.85rem; margin-bottom: 8px; }}
  .debate-col.bull h3 {{ color: #16a34a; }}
  .debate-col.bear h3 {{ color: #dc2626; }}
  .debate-col ul {{ padding-left: 18px; color: var(--muted); font-size: 0.83rem; line-height: 1.6; }}
  .actions ul {{ padding-left: 18px; line-height: 1.8; }}
  canvas {{ max-height: 220px; }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>{ticker}</h1>
    <span class="decision-badge" style="background:{decision_clr}">{decision.upper()}</span>
  </div>
  <div style="text-align:right">
    <div class="price">{price_str}</div>
    <div class="date">{data['date']}</div>
  </div>
</div>

<div class="levels">{levels_html}</div>

<!-- Indicator Table -->
<div class="card">
  <h2>Technical Indicators</h2>
  <table>
    <thead><tr><th>Indicator</th><th>Value</th><th>Signal</th><th>Insight</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>

<!-- Charts -->
<div class="grid-3">
  <div class="card">
    <h2>RSI</h2>
    <canvas id="rsiChart"></canvas>
  </div>
  <div class="card">
    <h2>MACD Histogram</h2>
    <canvas id="macdChart"></canvas>
  </div>
  <div class="card">
    <h2>ATR (Volatility)</h2>
    <canvas id="atrChart"></canvas>
  </div>
</div>

<!-- Analyst Summaries -->
<div class="card">
  <h2>Analyst Summaries</h2>
  <div class="grid-2">{analyst_cards}</div>
</div>

<!-- Bull vs Bear -->
<div class="card">
  <h2>Research Debate</h2>
  <div class="debate">
    <div class="debate-col bull"><h3>Bull Case</h3><ul>{bull_html}</ul></div>
    <div class="debate-col bear"><h3>Bear Case</h3><ul>{bear_html}</ul></div>
  </div>
</div>

<!-- Action Plan -->
<div class="card actions">
  <h2>Action Plan</h2>
  <ul>{action_html}</ul>
</div>

<script>
const chartOpts = {{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color: '#94a3b8', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
    y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 10 }} }}, grid: {{ color: '#334155' }} }}
  }}
}};

// RSI Chart
new Chart(document.getElementById('rsiChart'), {{
  type: 'line',
  data: {{
    labels: {rsi_labels},
    datasets: [{{
      data: {rsi_values},
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 3
    }}]
  }},
  options: {{
    ...chartOpts,
    plugins: {{
      ...chartOpts.plugins,
      annotation: {{
        annotations: {{
          oversold: {{ type: 'line', yMin: 30, yMax: 30, borderColor: '#dc2626', borderDash: [4,4], borderWidth: 1,
                       label: {{ content: 'Oversold', display: true, color: '#dc2626', font: {{ size: 9 }}, position: 'start' }} }},
          overbought: {{ type: 'line', yMin: 70, yMax: 70, borderColor: '#16a34a', borderDash: [4,4], borderWidth: 1,
                        label: {{ content: 'Overbought', display: true, color: '#16a34a', font: {{ size: 9 }}, position: 'start' }} }}
        }}
      }}
    }}
  }}
}});

// MACD Histogram Chart
new Chart(document.getElementById('macdChart'), {{
  type: 'bar',
  data: {{
    labels: {macd_h_labels},
    datasets: [{{
      data: {macd_h_values},
      backgroundColor: {macd_h_colors},
      borderRadius: 2
    }}]
  }},
  options: chartOpts
}});

// ATR Chart
new Chart(document.getElementById('atrChart'), {{
  type: 'line',
  data: {{
    labels: {atr_labels},
    datasets: [{{
      data: {atr_values},
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245,158,11,0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 3
    }}]
  }},
  options: chartOpts
}});
</script>

</body>
</html>"""


def save_summary_report(final_state: dict, ticker: str, save_path: Path) -> Path:
    """Extract key data from final_state and generate an HTML summary report."""
    data = extract_key_data(final_state, ticker)
    html = generate_summary_html(data)
    out = save_path / "summary.html"
    out.write_text(html, encoding="utf-8")
    return out
