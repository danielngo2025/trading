# TradingAgents/graph/setup.py

import time
from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents import perf_logger

from .conditional_logic import ConditionalLogic


def _timed_node(name: str, node_fn):
    """Wrap a graph node function to log its execution time.

    NOTE: We intentionally do NOT use @wraps here. LangGraph inspects
    the callable's signature via inspect.signature(follow_wrapped=True).
    If @wraps copies __wrapped__, LangGraph follows it to the original
    which may be a functools.partial or class that breaks inspection.
    A plain wrapper with a (state) signature keeps LangGraph happy.
    """
    def wrapper(state):
        t0 = time.time()
        result = node_fn(state)
        elapsed = time.time() - t0
        perf_logger.log_time(name, "node_run", elapsed)
        return result
    return wrapper


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        portfolio_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.portfolio_manager_memory = portfolio_manager_memory
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = _timed_node(
                "Market Analyst", create_market_analyst(self.quick_thinking_llm)
            )
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = _timed_node(
                "Social Analyst", create_social_media_analyst(self.quick_thinking_llm)
            )
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = _timed_node(
                "News Analyst", create_news_analyst(self.quick_thinking_llm)
            )
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = _timed_node(
                "Fundamentals Analyst", create_fundamentals_analyst(self.quick_thinking_llm)
            )
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create researcher and manager nodes
        bull_researcher_node = _timed_node(
            "Bull Researcher", create_bull_researcher(self.quick_thinking_llm, self.bull_memory)
        )
        bear_researcher_node = _timed_node(
            "Bear Researcher", create_bear_researcher(self.quick_thinking_llm, self.bear_memory)
        )
        research_manager_node = _timed_node(
            "Research Manager", create_research_manager(self.deep_thinking_llm, self.invest_judge_memory)
        )
        trader_node = _timed_node(
            "Trader", create_trader(self.quick_thinking_llm, self.trader_memory)
        )

        # Create risk analysis nodes
        aggressive_analyst = _timed_node(
            "Aggressive Analyst", create_aggressive_debator(self.quick_thinking_llm)
        )
        neutral_analyst = _timed_node(
            "Neutral Analyst", create_neutral_debator(self.quick_thinking_llm)
        )
        conservative_analyst = _timed_node(
            "Conservative Analyst", create_conservative_debator(self.quick_thinking_llm)
        )
        portfolio_manager_node = _timed_node(
            "Portfolio Manager", create_portfolio_manager(self.deep_thinking_llm, self.portfolio_manager_memory)
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph (each with its own tool loop)
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Single message-clear node at the merge point (after all parallel
        # analysts converge). Per-analyst clears can't work in parallel
        # because RemoveMessage IDs from one branch don't exist in another.
        workflow.add_node("Msg Clear Analysts", create_msg_delete())

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Fan out: START → all analysts in parallel
        for analyst_type in selected_analysts:
            workflow.add_edge(START, f"{analyst_type.capitalize()} Analyst")

        # Each analyst has its own tool loop; when done, fan in to merge clear
        for analyst_type in selected_analysts:
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"

            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, "Msg Clear Analysts"],
            )
            workflow.add_edge(current_tools, current_analyst)

        # Single merge point: clear messages → Bull Researcher
        workflow.add_edge("Msg Clear Analysts", "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        # Compile and return
        return workflow.compile()
