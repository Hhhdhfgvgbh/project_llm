from app.config.schemas import AggregationConfig
from app.core.aggregation import AggregationEngine


def test_majority_vote_groups_similar_answers() -> None:
    engine = AggregationEngine()
    responses = [
        "The answer is Paris.",
        "the answer is paris",
        "I think it might be Berlin.",
    ]
    result = engine.aggregate(responses, AggregationConfig(type="majority_vote"))
    assert "Paris" in result.output


def test_custom_template_uses_synthesis_callback() -> None:
    calls: list[tuple[str, str]] = []

    def synth(model: str, prompt: str) -> str:
        calls.append((model, prompt))
        return "merged"

    engine = AggregationEngine(synthesis_callback=synth)
    config = AggregationConfig(
        type="custom_template",
        synthesis_model="llama3_q5",
        template="A={{model_1}} | B={{model_2}}",
    )

    result = engine.aggregate(["one", "two"], config)
    assert result.output == "merged"
    assert calls[0][0] == "llama3_q5"
    assert "A=one | B=two" in calls[0][1]
