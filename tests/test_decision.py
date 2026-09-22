import math

import pytest

from parallel_decider import BooleanDecision


def test_boolean_decision_accepts_valid_probability() -> None:
    decision = BooleanDecision(
        name="needs_browser",
        probability=0.75,
    )

    assert decision.name == "needs_browser"
    assert decision.probability == 0.75


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01],
)
def test_boolean_decision_rejects_out_of_range_probability(
    probability: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="probability must be between",
    ):
        BooleanDecision(
            name="test",
            probability=probability,
        )


@pytest.mark.parametrize(
    "probability",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_boolean_decision_rejects_non_finite_probability(
    probability: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="probability must be finite",
    ):
        BooleanDecision(
            name="test",
            probability=probability,
        )


def test_boolean_decision_rejects_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        BooleanDecision(
            name="",
            probability=0.5,
        )
