# TradingAgents/graph/propagation.py

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)

logger = logging.getLogger(__name__)


def _last_trading_day(date_str: str) -> str:
    """Roll back to the most recent trading day.

    Skips weekends and market holidays by checking for actual OHLCV
    data. Tries up to 10 days back to handle long holiday weekends.
    """
    try:
        from tradingagents.dataflows.stockstats_utils import load_ohlcv
        import pandas as pd

        dt = datetime.strptime(date_str, "%Y-%m-%d")

        # Use SPY as a reference — it trades every market day
        spy_data = load_ohlcv("SPY", date_str)

        for _ in range(10):
            check = dt.strftime("%Y-%m-%d")
            check_ts = pd.to_datetime(check)
            if not spy_data[spy_data["Date"] == check_ts].empty:
                if check != date_str:
                    logger.info(f"Adjusted analysis date from {date_str} to {check} (last trading day)")
                return check
            dt -= timedelta(days=1)

        # Fallback: just skip weekends
        logger.warning(f"Could not find trading day near {date_str}, falling back to weekend skip")
    except Exception as e:
        logger.warning(f"Could not verify trading day for {date_str}: {e}")

    # Simple weekend fallback
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    while dt.weekday() >= 5:
        dt -= timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph."""
        trade_date = _last_trading_day(str(trade_date))
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": trade_date,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
