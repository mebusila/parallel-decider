import torch
import pytest

from parallel_decider.checkpoint import RouterCheckpoint, save_checkpoint
from parallel_decider.parallel_decider import (
    ParallelDecider,
)
from parallel_decider.projection import (
    RoutingProjection,
)
from parallel_decider.two_tower import (
    TwoTowerDecisionHead,
)


class FakeEncoder:
    def encode(
        self,
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
    ):
        if isinstance(texts, str):
            return torch.ones(4)

        return torch.ones(
            len(texts),
            4,
        )


def test_decide_returns_boolean_decisions():
    encoder = FakeEncoder()

    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    decider = ParallelDecider(
        encoder=encoder,
        projection=projection,
        head=head,
        routing_hypotheses={
            "needs_shell": ("This task requires shell execution."),
            "needs_git": ("This task requires Git access."),
        },
        device="cpu",
    )

    decisions = decider.decide("Run the tests.")

    assert len(decisions) == 2

    assert {decision.name for decision in decisions} == {
        "needs_shell",
        "needs_git",
    }

    for decision in decisions:
        assert 0.0 <= decision.probability <= 1.0


def test_decide_rejects_empty_state():
    encoder = FakeEncoder()

    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    decider = ParallelDecider(
        encoder=encoder,
        projection=projection,
        head=head,
        routing_hypotheses={
            "needs_shell": ("This task requires shell execution."),
        },
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="state must not be empty",
    ):
        decider.decide("   ")


def test_decide_many_returns_one_result_per_state():
    encoder = FakeEncoder()

    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    decider = ParallelDecider(
        encoder=encoder,
        projection=projection,
        head=head,
        routing_hypotheses={
            "needs_shell": ("This task requires shell execution."),
            "needs_git": ("This task requires Git access."),
        },
        device="cpu",
    )

    results = decider.decide_many(
        [
            "Run the tests.",
            "Inspect repository history.",
        ]
    )

    assert len(results) == 2
    assert all(len(decisions) == 2 for decisions in results)


def test_decide_many_rejects_empty_state():
    encoder = FakeEncoder()

    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    decider = ParallelDecider(
        encoder=encoder,
        projection=projection,
        head=head,
        routing_hypotheses={
            "needs_shell": ("This task requires shell execution."),
        },
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="states must not contain empty values",
    ):
        decider.decide_many(
            [
                "Run tests.",
                "   ",
            ]
        )


def test_decide_many_empty_list_returns_empty_list():
    encoder = FakeEncoder()

    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    decider = ParallelDecider(
        encoder=encoder,
        projection=projection,
        head=head,
        routing_hypotheses={
            "needs_shell": ("This task requires shell execution."),
        },
        device="cpu",
    )

    assert decider.decide_many([]) == []


def test_public_api_imports():
    from parallel_decider import (
        BooleanDecision,
        BooleanQuestion,
        DecisionBackend,
        NLIBackend,
        ParallelDecider,
    )

    assert BooleanDecision is not None
    assert BooleanQuestion is not None
    assert DecisionBackend is not None
    assert NLIBackend is not None
    assert ParallelDecider is not None


def test_from_checkpoint_loads_and_decides(
    tmp_path,
    monkeypatch,
):
    from parallel_decider.checkpoint import (
        RouterCheckpoint,
        save_checkpoint,
    )
    from parallel_decider.parallel_decider import (
        ParallelDecider,
    )

    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    checkpoint = RouterCheckpoint(
        format_version=1,
        router_version="test-router",
        encoder_model_name="fake-encoder",
        embedding_dim=4,
        projection_dim=2,
        hidden_dim=4,
        routing_hypotheses={
            "needs_shell": ("This task requires shell execution."),
        },
        training_dataset="test-dataset",
        training_examples=1,
        training_steps=10,
        training_seed=42,
        threshold=0.5,
        projection_state_dict={
            key: value.detach().cpu() for key, value in projection.state_dict().items()
        },
        head_state_dict={
            key: value.detach().cpu() for key, value in head.state_dict().items()
        },
    )

    checkpoint_path = tmp_path / "router.pt"

    save_checkpoint(
        checkpoint,
        checkpoint_path,
    )

    monkeypatch.setattr(
        "parallel_decider.parallel_decider." "SentenceTransformer",
        lambda *args, **kwargs: FakeEncoder(),
    )

    decider = ParallelDecider.from_checkpoint(
        checkpoint_path,
        device="cpu",
    )

    decisions = decider.decide("Run the tests.")

    assert len(decisions) == 1
    assert decisions[0].name == "needs_shell"
    assert 0.0 <= decisions[0].probability <= 1.0


def test_active_decisions_respects_threshold():
    encoder = FakeEncoder()

    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    decider = ParallelDecider(
        encoder=encoder,
        projection=projection,
        head=head,
        routing_hypotheses={
            "needs_shell": ("This task requires shell execution."),
        },
        device="cpu",
        threshold=1.1,
    )

    assert decider.active_decisions("Run tests.") == []


def test_from_checkpoint_exposes_metadata(
    tmp_path,
    monkeypatch,
):
    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    checkpoint = RouterCheckpoint(
        format_version=1,
        router_version="test-router",
        encoder_model_name="fake-encoder",
        embedding_dim=4,
        projection_dim=2,
        hidden_dim=4,
        routing_hypotheses={
            "needs_shell": ("This task requires shell execution."),
        },
        training_dataset="test-dataset",
        training_examples=1,
        training_steps=10,
        training_seed=42,
        threshold=0.75,
        projection_state_dict={
            key: value.detach().cpu() for key, value in projection.state_dict().items()
        },
        head_state_dict={
            key: value.detach().cpu() for key, value in head.state_dict().items()
        },
    )

    path = tmp_path / "router.pt"

    save_checkpoint(
        checkpoint,
        path,
    )

    monkeypatch.setattr(
        "parallel_decider.parallel_decider." "SentenceTransformer",
        lambda *args, **kwargs: FakeEncoder(),
    )

    decider = ParallelDecider.from_checkpoint(
        path,
        device="cpu",
    )

    assert decider.threshold == 0.75
    assert decider.router_version == "test-router"


def test_metadata():
    encoder = FakeEncoder()

    projection = RoutingProjection(
        input_dim=4,
        output_dim=2,
    )

    head = TwoTowerDecisionHead(
        embedding_dim=2,
        hidden_dim=4,
    )

    decider = ParallelDecider(
        encoder=encoder,
        projection=projection,
        head=head,
        routing_hypotheses={
            "needs_shell": "This task requires shell execution.",
        },
        device="cpu",
        threshold=0.6,
        router_version="test-router",
        encoder_model_name="fake-encoder",
    )

    assert decider.metadata == {
        "router_version": "test-router",
        "encoder_model_name": "fake-encoder",
        "threshold": 0.6,
        "device": "cpu",
        "capabilities": [
            "needs_shell",
        ],
    }
