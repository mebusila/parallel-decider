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
    state: str
    labels: dict[str, int]


def load_routing_dataset(path: str | Path) -> list[RoutingExample]:
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

        if any(value not in (0, 1) for value in labels.values()):
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