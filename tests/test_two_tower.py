import torch

from parallel_decider.two_tower import TwoTowerDecisionHead


def test_two_tower_head_output_shape() -> None:
    head = TwoTowerDecisionHead(
        embedding_dim=384,
        hidden_dim=64,
    )

    state = torch.randn(384)
    questions = torch.randn(6, 384)

    logits = head(
        state_embedding=state,
        question_embeddings=questions,
    )

    assert logits.shape == (6,)