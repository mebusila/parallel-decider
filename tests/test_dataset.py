from pathlib import Path

from parallel_decider.dataset import (
    ROUTING_LABELS,
    load_routing_dataset,
)


def test_load_routing_dataset() -> None:
    path = Path("data/routing_v0.jsonl")

    examples = load_routing_dataset(path)

    assert len(examples) == 10

    for example in examples:
        assert set(example.labels) == set(ROUTING_LABELS)
        assert all(
            value in (0, 1)
            for value in example.labels.values()
        )