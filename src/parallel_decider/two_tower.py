import torch
from torch import nn


class TwoTowerDecisionHead(nn.Module):
    """Score state/question embedding pairs as binary decisions."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

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
        if state_embedding.ndim != 1:
            raise ValueError("state_embedding must have shape [embedding_dim]")

        if question_embeddings.ndim != 2:
            raise ValueError(
                "question_embeddings must have shape "
                "[num_questions, embedding_dim]"
            )

        state = state_embedding.unsqueeze(0).expand_as(
            question_embeddings
        )

        features = torch.cat(
            [
                state,
                question_embeddings,
                torch.abs(state - question_embeddings),
                state * question_embeddings,
            ],
            dim=-1,
        )

        logits = self.network(features).squeeze(-1)

        return logits