"""Backend protocol for boolean decision evaluation.

This module defines the interface implemented by decision backends in
``parallel_decider``. A backend receives a natural-language state together
with one or more boolean questions and returns a corresponding list of
``BooleanDecision`` results.

The protocol allows different inference strategies, such as NLI-based
classification or learned routing models, to expose a consistent API.
"""

from typing import Protocol, Sequence

from .decision import BooleanDecision
from .question import BooleanQuestion


class DecisionBackend(Protocol):
    """Protocol implemented by boolean decision backends.

    Implementations evaluate a shared natural-language state against a
    sequence of boolean questions and return one decision per question.
    """

    def decide(
        self,
        state: str,
        questions: Sequence[BooleanQuestion],
    ) -> list[BooleanDecision]:
        """Evaluate boolean questions against a shared state.

        Args:
            state: Natural-language description of the current task or state.
            questions: Boolean questions to evaluate against the state.

        Returns:
            A list of decisions corresponding to the supplied questions.
        """
        ...
