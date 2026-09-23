"""Checkpoint serialization for trained routing models.

This module defines the persistent checkpoint format used by
``parallel_decider``. A checkpoint stores the routing architecture metadata,
training provenance, decision threshold, and learned weights required to
reconstruct a trained router.

The sentence encoder itself is not stored in the checkpoint. Instead, its
model name is recorded so it can be loaded separately when the router is
restored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RouterCheckpoint:
    """Serializable description of a trained routing model.

    Attributes:
        format_version: Version of the checkpoint serialization format.
        router_version: Human-readable version of the trained router.
        encoder_model_name: Sentence-transformer model used to encode states
            and capability hypotheses.
        embedding_dim: Dimensionality produced by the sentence encoder.
        projection_dim: Dimensionality of the learned routing projection.
        hidden_dim: Hidden dimension used by the two-tower decision head.
        routing_hypotheses: Mapping from capability names to natural-language
            hypotheses.
        training_dataset: Name of the dataset used to train the router.
        training_examples: Number of training examples used.
        training_steps: Number of optimizer updates performed during training.
        training_seed: Random seed used during training.
        threshold: Probability threshold used to mark a decision as active.
        projection_state_dict: Learned parameters of the routing projection.
        head_state_dict: Learned parameters of the decision head.
    """

    format_version: int
    router_version: str

    encoder_model_name: str
    embedding_dim: int
    projection_dim: int
    hidden_dim: int

    routing_hypotheses: dict[str, str]

    training_dataset: str
    training_examples: int
    training_steps: int
    training_seed: int

    threshold: float

    projection_state_dict: dict[str, torch.Tensor]
    head_state_dict: dict[str, torch.Tensor]


def save_checkpoint(
    checkpoint: RouterCheckpoint,
    path: str | Path,
) -> None:
    """Serialize a routing checkpoint to disk.

    Parent directories are created automatically when they do not already
    exist.

    Args:
        checkpoint: Checkpoint metadata and model parameters to persist.
        path: Destination file path.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Store plain Python values and state dictionaries so loading remains
    # independent from the original model objects.
    torch.save(
        {
            "format_version": checkpoint.format_version,
            "router_version": checkpoint.router_version,
            "encoder_model_name": checkpoint.encoder_model_name,
            "embedding_dim": checkpoint.embedding_dim,
            "projection_dim": checkpoint.projection_dim,
            "hidden_dim": checkpoint.hidden_dim,
            "routing_hypotheses": checkpoint.routing_hypotheses,
            "training_dataset": checkpoint.training_dataset,
            "training_examples": checkpoint.training_examples,
            "training_steps": checkpoint.training_steps,
            "training_seed": checkpoint.training_seed,
            "threshold": checkpoint.threshold,
            "projection_state_dict": checkpoint.projection_state_dict,
            "head_state_dict": checkpoint.head_state_dict,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
) -> RouterCheckpoint:
    """Load a routing checkpoint from disk.

    Checkpoint tensors are loaded onto CPU. They can later be moved to the
    desired runtime device when the router is reconstructed.

    Args:
        path: Path to a serialized routing checkpoint.

    Returns:
        The reconstructed ``RouterCheckpoint`` instance.
    """
    path = Path(path)

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    return RouterCheckpoint(
        format_version=data["format_version"],
        router_version=data["router_version"],
        encoder_model_name=data["encoder_model_name"],
        embedding_dim=data["embedding_dim"],
        projection_dim=data["projection_dim"],
        hidden_dim=data["hidden_dim"],
        routing_hypotheses=data["routing_hypotheses"],
        training_dataset=data["training_dataset"],
        training_examples=data["training_examples"],
        training_steps=data["training_steps"],
        training_seed=data["training_seed"],
        threshold=data["threshold"],
        projection_state_dict=data["projection_state_dict"],
        head_state_dict=data["head_state_dict"],
    )
