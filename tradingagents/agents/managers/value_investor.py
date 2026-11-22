from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction


def create_value_investor(llm):
    """Long-term owner's view in the Buffett tradition, independent of the trading debate."""

    def value_investor_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        prompt = f"""You are a long-term value investor in the tradition of Warren Buffett and Benjamin Graham. Judge this stock as part-ownership of a business you would hold for 5+ years, not as a trade.

{instrument_context}
Analysis date: {state["trade_date"]}.

Apply these principles, in this order:
1. **Circle of competence**: Can the business and how it makes money be understood from the reports? If not, say so and lean to Hold.
2. **Economic moat**: Is there a durable competitive advantage (switching costs, network effects, brand, cost advantage, scale) that protects returns on capital? Cite evidence, not adjectives.
3. **Management and capital allocation**: Buybacks, dividends, acquisitions, debt use. Sensible or value-destroying?
4. **Don't lose money**: Balance-sheet risk, leverage, goodwill, refinancing needs. Could a bad stretch force dilution or distress?
5. **Value and margin of safety**: Estimate a rough intrinsic value range per share from owner earnings or free cash flow and a conservative multiple. Show the arithmetic in one line. Compare it to the current price.
6. **Be greedy when others are fearful**: Is the market's mood (sentiment, news, recent price fall) pushing the price away from the business's value, in either direction?

Ignore chart signals (RSI, MACD, moving averages, Bollinger bands): price trends are not business value. Use the technical report only for the current price.

Rating scale, from a long-term owner's view (long-only, never propose shorting):
- **Buy**: wonderful or good business at a clear margin of safety
- **Overweight**: good business, fair-to-cheap price
- **Hold**: fair price, or moat unclear
- **Underweight**: price above value, or moat eroding
- **Sell**: no moat, or balance-sheet risk that could cause permanent loss

Output (strict, at most 350 words, tables over prose):
1. First line exactly: **Long-term rating: <Buy|Overweight|Hold|Underweight|Sell>** then a one-sentence why.
2. **Owner's checklist** table | Principle | Evidence | Verdict | with one row per principle above.
3. **Value estimate**: one line with the arithmetic, the range per share, the current price, and the margin of safety as a %.
4. **Would Buffett buy at**: a price below which the margin of safety would be adequate, or "No price" if the business fails the moat or balance-sheet test.
5. At most 2 bullets on where this view disagrees with a short-term trader's, and why.

Reports:
Fundamentals: {state["fundamentals_report"]}
News: {state["news_report"]}
Sentiment: {state["sentiment_report"]}
Technical (for the current price only): {state["market_report"]}{get_language_instruction()}"""

        response = llm.invoke(prompt)
        return {"value_investor_report": response.content}

    return value_investor_node
