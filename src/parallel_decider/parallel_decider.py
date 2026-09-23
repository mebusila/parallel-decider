from __future__ import annotations

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from parallel_decider.checkpoint import load_checkpoint
from parallel_decider.decision import BooleanDecision
from parallel_decider.projection import RoutingProjection
from parallel_decider.two_tower import TwoTowerDecisionHead


class ParallelDecider:
    def __init__(
        self,
        encoder: SentenceTransformer,
        projection: RoutingProjection,
        head: TwoTowerDecisionHead,
        routing_hypotheses: dict[str, str],
        device: str,
        threshold: float = 0.5,
        router_version: str | None = None,
        encoder_model_name: str | None = None,
    ) -> None:
        self.encoder = encoder
        self.projection = projection
        self.head = head
        self.routing_hypotheses = routing_hypotheses
        self.device = device
        self.threshold = threshold
        self.router_version = router_version
        self.encoder_model_name = encoder_model_name

        self._question_names = list(routing_hypotheses.keys())

        self._question_embeddings = self._encode_questions()

        self.projection.eval()
        self.head.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | None = None,
    ) -> "ParallelDecider":
        checkpoint = load_checkpoint(path)

        resolved_device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        encoder = SentenceTransformer(
            checkpoint.encoder_model_name,
            device=resolved_device,
        )

        projection = RoutingProjection(
            input_dim=checkpoint.embedding_dim,
            output_dim=checkpoint.projection_dim,
        ).to(resolved_device)

        head = TwoTowerDecisionHead(
            embedding_dim=checkpoint.projection_dim,
            hidden_dim=checkpoint.hidden_dim,
        ).to(resolved_device)

        projection.load_state_dict(checkpoint.projection_state_dict)

        head.load_state_dict(checkpoint.head_state_dict)

        return cls(
            encoder=encoder,
            projection=projection,
            head=head,
            routing_hypotheses=checkpoint.routing_hypotheses,
            device=resolved_device,
            threshold=checkpoint.threshold,
            router_version=checkpoint.router_version,
            encoder_model_name=checkpoint.encoder_model_name,
        )

    def _encode_questions(
        self,
    ) -> torch.Tensor:
        texts = [self.routing_hypotheses[name] for name in self._question_names]

        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        return embeddings.detach().clone().float()

    def decide(
        self,
        state: str,
    ) -> list[BooleanDecision]:
        if not state.strip():
            raise ValueError("state must not be empty")

        state_embedding = self.encoder.encode(
            state,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        state_embedding = state_embedding.detach().clone().float()

        with torch.no_grad():
            projected_state = self.projection(state_embedding)

            projected_questions = self.projection(self._question_embeddings)

            logits = self.head(
                state_embedding=projected_state,
                question_embeddings=projected_questions,
            )

            probabilities = torch.sigmoid(logits).detach().cpu()

        return [
            BooleanDecision(
                name=name,
                probability=float(probability),
            )
            for name, probability in zip(
                self._question_names,
                probabilities,
            )
        ]

    def active_decisions(
        self,
        state: str,
    ) -> list[BooleanDecision]:
        return [
            decision
            for decision in self.decide(state)
            if decision.probability >= self.threshold
        ]

    def decide_many(
        self,
        states: list[str],
    ) -> list[list[BooleanDecision]]:
        if not states:
            return []

        if any(not state.strip() for state in states):
            raise ValueError("states must not contain empty values")

        state_embeddings = self.encoder.encode(
            states,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        state_embeddings = state_embeddings.detach().clone().float()

        with torch.no_grad():
            projected_questions = self.projection(self._question_embeddings)

            results = []

            for state_embedding in state_embeddings:
                projected_state = self.projection(state_embedding)

                logits = self.head(
                    state_embedding=projected_state,
                    question_embeddings=projected_questions,
                )

                probabilities = torch.sigmoid(logits).detach().cpu()

                results.append(
                    [
                        BooleanDecision(
                            name=name,
                            probability=float(probability),
                        )
                        for name, probability in zip(
                            self._question_names,
                            probabilities,
                        )
                    ]
                )

        return results

    def active_decisions_many(
        self,
        states: list[str],
    ) -> list[list[BooleanDecision]]:
        return [
            [
                decision
                for decision in decisions
                if decision.probability >= self.threshold
            ]
            for decisions in self.decide_many(states)
        ]

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "router_version": self.router_version,
            "encoder_model_name": self.encoder_model_name,
            "threshold": self.threshold,
            "device": self.device,
            "capabilities": list(self._question_names),
        }
