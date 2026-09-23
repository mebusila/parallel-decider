"""Boolean question type used by decision backends.

This module defines ``BooleanQuestion``, an immutable description of a
boolean decision to evaluate against a natural-language state.

Each question has a stable name used to identify the resulting decision and a
natural-language hypothesis that can be evaluated by backends such as the NLI
classifier.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BooleanQuestion:
    """Boolean question evaluated against an input state.

    Attributes:
        name: Stable identifier used for the corresponding decision result.
        hypothesis: Natural-language hypothesis describing the condition that
            should be evaluated against the state.
    """

    name: str
    hypothesis: str

    def __post_init__(self) -> None:
        """Validate the question fields after initialization.

        Raises:
            ValueError: If ``name`` or ``hypothesis`` is empty.
        """
        if not self.name:
            raise ValueError(
                "name must not be empty"
            )

        if not self.hypothesis:
            raise ValueError(
                "hypothesis must not be empty"
            )
