from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BooleanDecision:
    """A boolean decision with an associated probability."""

    name: str
    probability: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")

        if not math.isfinite(self.probability):
            raise ValueError("probability must be finite")

        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0.0 and 1.0")