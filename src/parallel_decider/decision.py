"""Boolean decision result type.

This module defines the immutable ``BooleanDecision`` value object returned by
routing backends and the high-level ``ParallelDecider`` API.

A decision contains a capability name together with a probability in the closed
interval ``[0.0, 1.0]``.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BooleanDecision:
    """Boolean routing decision with an associated probability.

    Attributes:
        name: Name of the capability or decision being evaluated.
        probability: Probability assigned to the positive outcome. The value
            must be finite and lie between ``0.0`` and ``1.0`` inclusive.
    """

    name: str
    probability: float

    def __post_init__(self) -> None:
        """Validate the decision fields after initialization.

        Raises:
            ValueError: If ``name`` is empty, ``probability`` is not finite,
                or ``probability`` lies outside the interval ``[0.0, 1.0]``.
        """
        if not self.name:
            raise ValueError("name must not be empty")

        if not math.isfinite(self.probability):
            raise ValueError("probability must be finite")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "probability must be between 0.0 and 1.0"
            )
