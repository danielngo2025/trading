
from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision on the five-point scale Buy / Overweight / Hold / Underweight / Sell.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation must be clear and actionable. Weigh arguments by how much they matter to value, not by how many there are: one decisive point can outweigh several minor ones. A risk is only bearish if it is not already in the price, so check it against the valuation. Avoid defaulting to Hold simply because both sides have valid points, but use Overweight or Underweight when the edge is real but modest.

Take into account your past mistakes on similar situations.

Output format (strict, at most 300 words, no paragraphs):
1. **Recommendation:** Buy / Overweight / Hold / Underweight / Sell in one line, with a one-sentence reason.
2. **Key issues** table | Issue | Bull says | Bear says | Weight (high/med/low) | Leans | covering the 3-5 points that decided it.
3. **Plan for the trader** table | Step | Detail | with entry, sizing, stop, target and horizon.

Here are your past reflections on mistakes:
\"{past_memory_str}\"

{instrument_context}

Here is the debate:
Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
