"""Two-tower decision head for routing classification.

This module defines ``TwoTowerDecisionHead``, a lightweight neural classifier
that scores one state embedding against multiple capability-question
embeddings.

For each state/question pair, the head constructs a feature vector from:

- the state embedding,
- the question embedding,
- their element-wise absolute difference,
- their element-wise product.

These pairwise features are then passed through a small feed-forward network to
produce one binary-decision logit per capability.
"""

import torch
from torch import nn


class TwoTowerDecisionHead(nn.Module):
    """Score state/question embedding pairs as binary routing decisions.

    The input embeddings are assumed to already be in the same routing space.
    A single state embedding is compared against multiple question embeddings
    in parallel.

    Args:
        embedding_dim: Dimensionality of the state and question embeddings.
        hidden_dim: Hidden dimension of the feed-forward decision network.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize the pairwise decision network."""
        super().__init__()

        # Four embedding-sized feature blocks are concatenated for every
        # state/question pair.
        input_dim = embedding_dim * 4

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        question_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score one state against multiple routing questions.

        Args:
            state_embedding: One-dimensional tensor with shape
                ``[embedding_dim]``.
            question_embeddings: Two-dimensional tensor with shape
                ``[num_questions, embedding_dim]``.

        Returns:
            One logit per question with shape ``[num_questions]``.

        Raises:
            ValueError: If ``state_embedding`` is not one-dimensional or
                ``question_embeddings`` is not two-dimensional.
        """
        if state_embedding.ndim != 1:
            raise ValueError(
                "state_embedding must have shape [embedding_dim]"
            )

        if question_embeddings.ndim != 2:
            raise ValueError(
                "question_embeddings must have shape "
                "[num_questions, embedding_dim]"
            )

        # Broadcast the single state embedding so all question pairs can be
        # evaluated together.
        state = (
            state_embedding
            .unsqueeze(0)
            .expand_as(question_embeddings)
        )

        # Pairwise comparison features let the head use both the original
        # representations and simple similarity/difference interactions.
        features = torch.cat(
            [
                state,
                question_embeddings,
                torch.abs(
                    state
                    - question_embeddings
                ),
                state * question_embeddings,
            ],
            dim=-1,
        )

        logits = (
            self.network(features)
            .squeeze(-1)
        )

        return logits
