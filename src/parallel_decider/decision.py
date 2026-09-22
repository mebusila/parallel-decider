from dataclasses import dataclass


@dataclass(frozen=True)
class BooleanDecision:
    """A boolean decision with an associated probability."""

    name: str
    probability: float
