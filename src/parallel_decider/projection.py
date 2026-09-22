import torch
from torch import nn


class RoutingProjection(nn.Module):
    """Project generic sentence embeddings into a routing-specific space."""

    def __init__(
        self,
        input_dim: int = 384,
        output_dim: int = 128,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(embeddings)