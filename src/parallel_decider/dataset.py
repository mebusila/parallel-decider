"""Dataset utilities for routing experiments.

This module defines the supported routing capability labels and provides a
small JSONL loader used by the training and evaluation scripts.

Each dataset row contains:

- a natural-language ``state`` describing the task,
- a complete mapping of routing labels to binary values.

The loader validates that every example contains exactly the expected routing
labels and that all label values are binary.
"""

import json
from dataclasses import dataclass
from pathlib import Path


ROUTING_LABELS = (
    "needs_filesystem",
    "needs_git",
    "needs_shell",
    "needs_browser",
    "needs_network",
    "needs_database",
    "needs_email",
    "needs_calendar",
)


@dataclass(frozen=True)
class RoutingExample:
    """Single labeled routing example.

    Attributes:
        state: Natural-language description of the task or agent state.
        labels: Mapping from routing capability names to binary labels, where
            ``1`` means the capability is required and ``0`` means it is not.
    """

    state: str
    labels: dict[str, int]


def load_routing_dataset(
    path: str | Path,
) -> list[RoutingExample]:
    """Load and validate a routing dataset from a JSONL file.

    Blank lines are ignored. Every non-empty row must contain a ``state`` and
    a ``labels`` mapping with exactly the capabilities defined by
    ``ROUTING_LABELS``.

    Args:
        path: Path to the JSONL dataset.

    Returns:
        A list of validated routing examples in file order.

    Raises:
        ValueError: If an example has missing or unexpected labels, or if any
            label value is not binary.
        KeyError: If a dataset row does not contain the expected ``state`` or
            ``labels`` field.
        json.JSONDecodeError: If a non-empty line is not valid JSON.
    """
    dataset_path = Path(path)

    examples: list[RoutingExample] = []

    for line_number, line in enumerate(
        dataset_path.read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        row = json.loads(line)
        labels = row["labels"]

        if set(labels) != set(ROUTING_LABELS):
            raise ValueError(
                f"Invalid labels on line {line_number}"
            )

        if any(
            value not in (0, 1)
            for value in labels.values()
        ):
            raise ValueError(
                f"Labels must be binary on line {line_number}"
            )

        examples.append(
            RoutingExample(
                state=row["state"],
                labels=labels,
            )
        )

    return examples
