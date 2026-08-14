from mcp_prism.evaluation import CasePrediction, EvaluationCase, quality_gate, score_cases


def test_quality_metrics_and_gate():
    cases = [
        EvaluationCase("1", "private", "weather", frozenset({"weather.get"}), {"weather.get": {"city": "Seoul"}}),
        EvaluationCase("2", "private", "hello", frozenset(), {}, no_tool=True, category="no_tool"),
    ]
    predictions = [
        CasePrediction(("weather.get",), ("weather.get",), {"weather.get": {"city": " seoul "}}, True),
        CasePrediction((), (), {}, True),
    ]
    scores = score_cases(cases, predictions)
    assert scores["wrong_tool_call_rate"] == 0.0
    assert all(value == 1.0 for key, value in scores.items() if key != "wrong_tool_call_rate")
    assert quality_gate(scores, scores) == (True, ())

