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
    assert all(value == 1.0 for value in scores.values())
    assert quality_gate(scores, scores) == (True, ())

