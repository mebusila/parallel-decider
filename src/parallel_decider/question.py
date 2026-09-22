from dataclasses import dataclass


@dataclass(frozen=True)
class BooleanQuestion:
    """A boolean question evaluated against some input state."""

    name: str
    hypothesis: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")

        if not self.hypothesis:
            raise ValueError("hypothesis must not be empty")
