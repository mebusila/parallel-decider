"""Learned projection from encoder space into routing space.

This module defines ``RoutingProjection``, a small neural network that maps
generic sentence embeddings into a lower-dimensional representation optimized
for routing decisions.

The projection allows the router to specialize a frozen sentence encoder
without fine-tuning the encoder itself.
"""

import torch
from torch import nn


class RoutingProjection(nn.Module):
    """Project generic sentence embeddings into a routing-specific space.

    The projection applies a linear transformation followed by ReLU activation
    and layer normalization.

    Args:
        input_dim: Dimensionality of the input sentence embeddings.
        output_dim: Dimensionality of the routing-specific representation.
    """

    def __init__(
        self,
        input_dim: int = 384,
        output_dim: int = 128,
    ) -> None:
        """Initialize the routing projection network."""
        super().__init__()

        # Keep the routing adapter intentionally small so most computation
        # remains in the shared frozen sentence encoder.
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Project sentence embeddings into routing space.

        Args:
            embeddings: Input tensor with the final dimension equal to
                ``input_dim``.

        Returns:
            Tensor with the same leading dimensions and a final dimension
            equal to ``output_dim``.
        """
        return self.network(embeddings)
