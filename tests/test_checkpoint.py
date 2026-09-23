import torch

from parallel_decider.checkpoint import (
    RouterCheckpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_roundtrip(tmp_path):
    checkpoint = RouterCheckpoint(
        encoder_model_name="test-encoder",
        embedding_dim=384,
        projection_dim=128,
        hidden_dim=128,
        routing_hypotheses={
            "needs_shell": "This task requires shell execution.",
        },
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
