from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. You must retrieve stock data and then analyze exactly these 5 technical indicators:

1. rsi — RSI: Measures momentum to flag overbought/oversold conditions. Use 70/30 thresholds and watch for divergence.
2. macd — MACD: Computes momentum via EMA differences. Look for crossovers and divergence as trend change signals.
3. close_50_sma — 50 SMA: Medium-term trend direction and dynamic support/resistance.
4. boll_ub — Bollinger Upper Band: Signals overbought conditions and breakout zones.
5. boll_lb — Bollinger Lower Band: Signals oversold conditions and potential reversals.

Steps:
1. First call get_stock_data to retrieve the price CSV.
2. Then call get_indicators for each of the 5 indicators above. Use the exact names shown (rsi, macd, close_50_sma, boll_ub, boll_lb). Call all 5 — no more, no less.
3. After receiving all indicator data, write a detailed report analyzing the trends, with specific actionable insights and supporting evidence."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
