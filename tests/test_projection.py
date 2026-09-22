import torch

from parallel_decider.projection import RoutingProjection


def test_routing_projection_shape() -> None:
    projection = RoutingProjection(
        input_dim=384,
        output_dim=128,
    )

    embeddings = torch.randn(8, 384)

    result = projection(embeddings)

    assert result.shape == (8, 128)