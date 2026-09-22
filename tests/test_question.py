import pytest

from parallel_decider import BooleanQuestion


def test_boolean_question_accepts_valid_values() -> None:
    question = BooleanQuestion(
        name="needs_browser",
        hypothesis="This task requires browser access.",
    )

    assert question.name == "needs_browser"
    assert question.hypothesis == "This task requires browser access."


def test_boolean_question_rejects_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        BooleanQuestion(
            name="",
            hypothesis="This task requires browser access.",
        )


def test_boolean_question_rejects_empty_hypothesis() -> None:
    with pytest.raises(
        ValueError,
        match="hypothesis must not be empty",
    ):
        BooleanQuestion(
            name="needs_browser",
            hypothesis="",
        )