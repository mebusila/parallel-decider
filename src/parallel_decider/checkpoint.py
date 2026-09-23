from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RouterCheckpoint:
    encoder_model_name: str
    embedding_dim: int
    projection_dim: int
    hidden_dim: int
    routing_hypotheses: dict[str, str]
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
            "encoder_model_name": checkpoint.encoder_model_name,
            "embedding_dim": checkpoint.embedding_dim,
            "projection_dim": checkpoint.projection_dim,
            "hidden_dim": checkpoint.hidden_dim,
            "routing_hypotheses": checkpoint.routing_hypotheses,
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
        encoder_model_name=data["encoder_model_name"],
        embedding_dim=data["embedding_dim"],
        projection_dim=data["projection_dim"],
        hidden_dim=data["hidden_dim"],
        routing_hypotheses=data["routing_hypotheses"],
        projection_state_dict=data["projection_state_dict"],
        head_state_dict=data["head_state_dict"],
    )
