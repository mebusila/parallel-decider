import torch

from parallel_decider.checkpoint import (
    RouterCheckpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path):
    checkpoint = RouterCheckpoint(
        format_version=1,
        router_version="test-router",
        encoder_model_name="test-encoder",
        embedding_dim=384,
        projection_dim=128,
        hidden_dim=128,
        routing_hypotheses={
            "needs_shell": "This task requires shell execution.",
        },
        training_dataset="test-dataset",
        training_examples=10,
        training_steps=100,
        training_seed=42,
        threshold=0.5,
        projection_state_dict={
            "weight": torch.tensor([1.0, 2.0]),
        },
        head_state_dict={
            "bias": torch.tensor([0.5]),
        },
    )

    path = tmp_path / "router.pt"

    save_checkpoint(
        checkpoint,
        path,
    )

    loaded = load_checkpoint(path)

    assert loaded.encoder_model_name == "test-encoder"
    assert loaded.embedding_dim == 384
    assert loaded.projection_dim == 128
    assert loaded.hidden_dim == 128

    assert loaded.routing_hypotheses == {
        "needs_shell": "This task requires shell execution.",
    }

    assert torch.equal(
        loaded.projection_state_dict["weight"],
        torch.tensor([1.0, 2.0]),
    )

    assert torch.equal(
        loaded.head_state_dict["bias"],
        torch.tensor([0.5]),
    )

    assert loaded.format_version == 1
    assert loaded.router_version == "test-router"
    assert loaded.training_dataset == "test-dataset"
    assert loaded.training_examples == 10
    assert loaded.training_steps == 100
    assert loaded.training_seed == 42
    assert loaded.threshold == 0.5
