"""Stock list providers for Canadian and other market indices.

Fetches index constituents from Wikipedia and presents them for interactive selection.
"""

import requests
import pandas as pd
import questionary
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# Wikipedia URLs for stock index constituent lists
STOCK_LISTS = {
    "S&P/TSX 60": {
        "url": "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        "suffix": ".TO",
        "description": "Top 60 Canadian blue-chip stocks",
    },
    "S&P/TSX Composite (top companies)": {
        "url": "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index",
        "suffix": ".TO",
        "description": "~250 largest companies on TSX",
    },
    "S&P 500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "suffix": "",
        "description": "Top 500 US large-cap stocks",
    },
    "NASDAQ-100": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "suffix": "",
        "description": "Top 100 US tech-heavy stocks",
    },
    "Dow Jones 30": {
        "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        "suffix": "",
        "description": "30 US blue-chip stocks",
    },
    "Enter ticker manually": {
        "url": None,
        "suffix": "",
        "description": "Type a ticker symbol directly",
    },
}


def _find_ticker_table(tables: list[pd.DataFrame], suffix: str) -> pd.DataFrame | None:
    """Find the most likely stock constituent table from a list of HTML tables.

    Looks for tables containing a 'Ticker' or 'Symbol' column, or infers
    ticker/company columns by matching known patterns.
    """
    for df in tables:
        cols_lower = [str(c).lower().strip() for c in df.columns]
        # Direct match on common column names
        for keyword in ("ticker", "ticker symbol", "symbol", "stock symbol"):
            if keyword in cols_lower:
                return df

    # Fallback: look for a column whose values look like ticker symbols
    for df in tables:
        if len(df) < 5:
            continue
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(20)
            # Tickers are short uppercase strings, possibly with dots
            if sample.str.match(r"^[A-Z]{1,5}(\.[A-Z]{1,3})?$").mean() > 0.5:
                return df

    return None


def _extract_tickers(df: pd.DataFrame, suffix: str) -> list[dict]:
    """Extract ticker and company name from a dataframe.

    Returns a list of dicts with keys: ticker, company, sector (if available).
    """
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    # Find ticker column
    ticker_col = None
    for keyword in ("ticker", "ticker symbol", "symbol", "stock symbol"):
        if keyword in cols_lower:
            ticker_col = cols_lower[keyword]
            break

    # Fallback: find column with ticker-like values
    if ticker_col is None:
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(20)
            if sample.str.match(r"^[A-Z]{1,5}(\.[A-Z]{1,3})?$").mean() > 0.5:
                ticker_col = col
                break

    if ticker_col is None:
        return []

    # Find company name column
    company_col = None
    for keyword in ("company", "name", "company name", "security", "issue"):
        if keyword in cols_lower:
            company_col = cols_lower[keyword]
            break

    # Find sector column
    sector_col = None
    for keyword in ("sector", "gics sector", "industry", "sub-industry", "gics sub-industry"):
        if keyword in cols_lower:
            sector_col = cols_lower[keyword]
            break

    results = []
    for _, row in df.iterrows():
        raw_ticker = str(row[ticker_col]).strip()
        if not raw_ticker or raw_ticker == "nan":
            continue

        # Clean ticker: remove anything after newline, take first word
        raw_ticker = raw_ticker.split("\n")[0].split()[0].strip()

        # Apply exchange suffix if not already present
        if suffix and not raw_ticker.endswith(suffix):
            ticker = f"{raw_ticker}{suffix}"
        else:
            ticker = raw_ticker

        company = str(row[company_col]).strip() if company_col else ""
        sector = str(row[sector_col]).strip() if sector_col else ""

        if company == "nan":
            company = ""
        if sector == "nan":
            sector = ""

        results.append({"ticker": ticker, "company": company, "sector": sector})

    return results


def fetch_stock_list(list_name: str) -> list[dict]:
    """Fetch stock constituents for the given list name.

    Returns list of dicts: [{"ticker": "RY.TO", "company": "Royal Bank", "sector": "Financials"}, ...]
    """
    config = STOCK_LISTS[list_name]
    if config["url"] is None:
        return []

    try:
        console.print(f"[dim]Fetching {list_name} from Wikipedia...[/dim]")
        # Wikipedia requires a proper User-Agent header
        resp = requests.get(
            config["url"],
            headers={"User-Agent": "TradingAgents/1.0 (stock screener)"},
            timeout=15,
        )
        resp.raise_for_status()
        from io import StringIO
        tables = pd.read_html(StringIO(resp.text))
        df = _find_ticker_table(tables, config["suffix"])
        if df is None:
            console.print(f"[yellow]Could not find constituent table for {list_name}[/yellow]")
            return []
        return _extract_tickers(df, config["suffix"])
    except Exception as e:
        console.print(f"[yellow]Error fetching {list_name}: {e}[/yellow]")
        return []


def display_stock_table(stocks: list[dict], list_name: str) -> None:
    """Display stocks in a rich table grouped by sector."""
    table = Table(
        title=f"{list_name} Constituents ({len(stocks)} stocks)",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Ticker", style="bold cyan", width=10)
    table.add_column("Company", width=35)
    table.add_column("Sector", style="green", width=25)

    # Sort by sector then company
    sorted_stocks = sorted(stocks, key=lambda s: (s.get("sector", ""), s.get("company", "")))
    for i, stock in enumerate(sorted_stocks, 1):
        table.add_row(str(i), stock["ticker"], stock["company"], stock.get("sector", ""))

    console.print(table)


def select_stock_list() -> str:
    """Let the user choose which stock index list to browse."""
    choices = [
        questionary.Choice(
            f"{name} — {cfg['description']}",
            value=name,
        )
        for name, cfg in STOCK_LISTS.items()
    ]

    choice = questionary.select(
        "Select a stock list to browse:",
        choices=choices,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()

    if choice is None:
        console.print("\n[red]No list selected. Exiting...[/red]")
        exit(1)

    return choice


def select_tickers_from_list(stocks: list[dict]) -> list[str]:
    """Let the user pick one or more tickers from the fetched stock list.

    Supports filtering by sector first, then selecting tickers via checkboxes.
    Returns a list of selected ticker symbols.
    """
    sectors = sorted(set(s["sector"] for s in stocks if s.get("sector")))

    filter_choice = "All stocks"
    if sectors:
        sector_choices = [questionary.Choice("All stocks", value="All stocks")]
        sector_choices += [questionary.Choice(s, value=s) for s in sectors]

        filter_choice = questionary.select(
            "Filter by sector (or show all):",
            choices=sector_choices,
            instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
            style=questionary.Style([
                ("selected", "fg:green noinherit"),
                ("highlighted", "fg:green noinherit"),
                ("pointer", "fg:green noinherit"),
            ]),
        ).ask()

        if filter_choice is None:
            filter_choice = "All stocks"

    # Filter stocks
    if filter_choice == "All stocks":
        filtered = stocks
    else:
        filtered = [s for s in stocks if s.get("sector") == filter_choice]

    # Sort and create choices
    filtered = sorted(filtered, key=lambda s: s.get("company", ""))
    ticker_choices = [
        questionary.Choice(
            f"{s['ticker']:10s}  {s['company']}" + (f"  [{s['sector']}]" if s.get("sector") else ""),
            value=s["ticker"],
        )
        for s in filtered
    ]

    selected = questionary.checkbox(
        f"Select tickers to analyze ({len(filtered)} available):",
        choices=ticker_choices,
        instruction="\n- Press Space to select/unselect\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one ticker.",
        style=questionary.Style([
            ("checkbox-selected", "fg:cyan"),
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "noinherit"),
            ("pointer", "noinherit"),
        ]),
    ).ask()

    if not selected:
        console.print("\n[red]No tickers selected. Exiting...[/red]")
        exit(1)

    return selected


def browse_and_select_tickers() -> list[str]:
    """Full flow: choose a list → fetch → browse → select tickers.

    Returns a list of selected ticker symbol strings.
    """
    list_name = select_stock_list()

    if list_name == "Enter ticker manually":
        from cli.utils import get_ticker as manual_get_ticker
        return [manual_get_ticker()]

    stocks = fetch_stock_list(list_name)
    if not stocks:
        console.print("[yellow]No stocks found. Falling back to manual entry.[/yellow]")
        from cli.utils import get_ticker as manual_get_ticker
        return [manual_get_ticker()]

    display_stock_table(stocks, list_name)
    return select_tickers_from_list(stocks)
