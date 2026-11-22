import unittest

from web.api.runner import PHASES, _phase_timings

REPORTS = {k: "x" for k in ("market_report", "sentiment_report", "news_report", "fundamentals_report")}


def marks_for(states):
    marks = {}
    for t, state in states:
        for key, _, done in PHASES:
            if key not in marks and done(state):
                marks[key] = t
    return marks


class PhaseTimingTests(unittest.TestCase):
    def test_full_run_splits_into_consecutive_phases(self):
        debate = {"count": 2, "judge_decision": ""}
        states = [
            (100, {"fundamentals_report": "reused"}),  # initial state is streamed too
            (123, REPORTS),
            (140, {**REPORTS, "investment_debate_state": debate}),
            (173, {**REPORTS, "investment_debate_state": {**debate, "judge_decision": "Hold"}}),
            (183, {**REPORTS, "investment_debate_state": {**debate, "judge_decision": "Hold"}, "trader_investment_plan": "p"}),
            (202, {**REPORTS, "investment_debate_state": {**debate, "judge_decision": "Hold"}, "trader_investment_plan": "p",
                   "risk_debate_state": {"count": 3}}),
            (234, {**REPORTS, "investment_debate_state": {**debate, "judge_decision": "Hold"}, "trader_investment_plan": "p",
                   "risk_debate_state": {"count": 3}, "final_trade_decision": "Hold"}),
        ]
        marks = marks_for(states) | {"rating": 236}
        timings = _phase_timings(100, marks)
        self.assertEqual([t["key"] for t in timings],
                         ["analysts", "debate", "research_manager", "trader", "risk", "portfolio_manager", "rating"])
        self.assertEqual([t["seconds"] for t in timings], [23, 17, 33, 10, 19, 32, 2])

    def test_quick_mode_has_no_risk_phase(self):
        timings = _phase_timings(0, {"analysts": 20, "debate": 35, "research_manager": 60, "trader": 70,
                                     "portfolio_manager": 100, "rating": 101})
        self.assertNotIn("risk", [t["key"] for t in timings])
        self.assertEqual(timings[-2]["seconds"], 30)


if __name__ == "__main__":
    unittest.main()
