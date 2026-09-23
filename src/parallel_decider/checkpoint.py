from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RouterCheckpoint:
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
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
