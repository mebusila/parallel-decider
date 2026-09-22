from typing import Protocol, Sequence

from .decision import BooleanDecision
from .question import BooleanQuestion


class DecisionBackend(Protocol):
    """Backend capable of evaluating boolean questions against a state."""

    def decide(
        self,
        state: str,
        questions: Sequence[BooleanQuestion],
    ) -> list[BooleanDecision]:
        ...
