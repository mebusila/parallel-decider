from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from parallel_decider.checkpoint import (
    RouterCheckpoint,
    save_checkpoint,
)
from parallel_decider.dataset import (
    ROUTING_LABELS,
    load_routing_dataset,
)
from parallel_decider.projection import RoutingProjection
from parallel_decider.two_tower import TwoTowerDecisionHead

SEED = 42
MODEL_NAME = "BAAI/bge-base-en-v1.5"

TRAINING_PATH = Path("data/routing_v3_training.jsonl")

CHECKPOINT_PATH = Path("models/router-bge-base-v3.pt")

PROJECTION_DIM = 128
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
MAX_STEPS = 20_000


ROUTING_HYPOTHESES = {
    "needs_filesystem": "This task requires filesystem access.",
    "needs_git": "This task requires Git access.",
    "needs_shell": "This task requires shell execution.",
    "needs_browser": "This task requires browser access.",
    "needs_network": "This task requires network access.",
    "needs_database": "This task requires database access.",
    "needs_email": "This task requires email access.",
    "needs_calendar": "This task requires calendar access.",
}


def reset_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def main() -> None:
    reset_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    examples = load_routing_dataset(TRAINING_PATH)

    print(
        "Training examples:",
        len(examples),
    )

    encoder = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    embedding_dim = encoder.get_embedding_dimension()

    states = [example.state for example in examples]

    questions = [ROUTING_HYPOTHESES[label] for label in ROUTING_LABELS]

    state_embeddings = (
        encoder.encode(
            states,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        .detach()
        .clone()
        .float()
    )

    question_embeddings = (
        encoder.encode(
            questions,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        .detach()
        .clone()
        .float()
    )

    targets = torch.tensor(
        [[example.labels[label] for label in ROUTING_LABELS] for example in examples],
        dtype=torch.float32,
        device=device,
    )

    projection = RoutingProjection(
        input_dim=embedding_dim,
        output_dim=PROJECTION_DIM,
    ).to(device)

    head = TwoTowerDecisionHead(
        embedding_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(projection.parameters()) + list(head.parameters()),
        lr=LEARNING_RATE,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    step = 0

    progress = tqdm(
        total=MAX_STEPS,
        desc="Training router",
    )

    while step < MAX_STEPS:
        for state_embedding, target in zip(
            state_embeddings,
            targets,
        ):
            if step >= MAX_STEPS:
                break

            optimizer.zero_grad()

            projected_state = projection(state_embedding)

            projected_questions = projection(question_embeddings)

            logits = head(
                state_embedding=projected_state,
                question_embeddings=projected_questions,
            )

            loss = criterion(
                logits,
                target,
            )

            loss.backward()
            optimizer.step()

            step += 1
            progress.update(1)

    progress.close()

    checkpoint = RouterCheckpoint(
        encoder_model_name=MODEL_NAME,
        embedding_dim=embedding_dim,
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
        routing_hypotheses=ROUTING_HYPOTHESES,
        projection_state_dict={
            key: value.detach().cpu() for key, value in projection.state_dict().items()
        },
        head_state_dict={
            key: value.detach().cpu() for key, value in head.state_dict().items()
        },
    )

    save_checkpoint(
        checkpoint,
        CHECKPOINT_PATH,
    )

    print(f"Saved checkpoint to " f"{CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
