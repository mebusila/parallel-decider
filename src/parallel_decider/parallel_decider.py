"""High-level parallel routing API.

This module exposes :class:`ParallelDecider`, the main user-facing interface
for checkpoint-backed capability routing.

A ``ParallelDecider`` combines:

- a frozen sentence encoder,
- a learned routing projection,
- a two-tower decision head,
- a fixed set of natural-language capability hypotheses.

The state is encoded once and then evaluated against all configured
capabilities. Static capability hypotheses are embedded once during
initialization and reused across subsequent routing requests.
"""

from __future__ import annotations

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from parallel_decider.checkpoint import load_checkpoint
from parallel_decider.decision import BooleanDecision
from parallel_decider.projection import RoutingProjection
from parallel_decider.two_tower import TwoTowerDecisionHead


class ParallelDecider:
    """Evaluate multiple routing decisions from a shared state representation.

    The decider encodes a natural-language state once, projects it into the
    learned routing space, and evaluates it against all configured capability
    hypotheses.

    Args:
        encoder: Sentence encoder used for states and capability hypotheses.
        projection: Learned projection from encoder space into routing space.
        head: Two-tower decision head used to score state/question pairs.
        routing_hypotheses: Mapping from capability names to natural-language
            routing hypotheses.
        device: Torch device used for inference.
        threshold: Probability threshold used to mark a capability as active.
        router_version: Optional human-readable router version.
        encoder_model_name: Optional identifier of the sentence encoder model.
    """

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

        self._question_names = list(
            routing_hypotheses.keys()
        )

        # Capability hypotheses are static, so their embeddings are computed
        # once and reused across all later routing calls.
        self._question_embeddings = (
            self._encode_questions()
        )

        self.projection.eval()
        self.head.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | None = None,
    ) -> "ParallelDecider":
        """Restore a trained router from a serialized checkpoint.

        The checkpoint provides the architecture metadata, encoder identifier,
        routing hypotheses, threshold, and learned routing weights required to
        reconstruct the model.

        Args:
            path: Path to a serialized router checkpoint.
            device: Torch device used for inference. If omitted, CUDA is used
                when available; otherwise CPU is selected.

        Returns:
            A fully initialized ``ParallelDecider`` ready for inference.
        """
        checkpoint = load_checkpoint(
            path
        )

        resolved_device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
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

        projection.load_state_dict(
            checkpoint.projection_state_dict
        )

        head.load_state_dict(
            checkpoint.head_state_dict
        )

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
        """Encode all configured capability hypotheses once.

        Returns:
            A tensor containing one sentence embedding per configured routing
            hypothesis.
        """
        texts = [
            self.routing_hypotheses[name]
            for name in self._question_names
        ]

        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        return (
            embeddings
            .detach()
            .clone()
            .float()
        )

    def decide(
        self,
        state: str,
    ) -> list[BooleanDecision]:
        """Evaluate all configured capabilities for a single state.

        The state is encoded once, projected into routing space, and scored
        against every precomputed capability hypothesis.

        Args:
            state: Natural-language description of the current task or state.

        Returns:
            One ``BooleanDecision`` per configured capability, preserving the
            capability order from ``routing_hypotheses``.

        Raises:
            ValueError: If ``state`` is empty or contains only whitespace.
        """
        if not state.strip():
            raise ValueError(
                "state must not be empty"
            )

        state_embedding = self.encoder.encode(
            state,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        state_embedding = (
            state_embedding
            .detach()
            .clone()
            .float()
        )

        with torch.no_grad():
            projected_state = self.projection(
                state_embedding
            )

            # All capability embeddings share the same learned projection.
            projected_questions = self.projection(
                self._question_embeddings
            )

            logits = self.head(
                state_embedding=projected_state,
                question_embeddings=projected_questions,
            )

            probabilities = (
                torch.sigmoid(logits)
                .detach()
                .cpu()
            )

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
        """Return only capabilities whose probability meets the threshold.

        Args:
            state: Natural-language description of the current task or state.

        Returns:
            Decisions with probabilities greater than or equal to the
            checkpoint threshold.
        """
        return [
            decision
            for decision in self.decide(state)
            if decision.probability >= self.threshold
        ]

    def decide_many(
        self,
        states: list[str],
    ) -> list[list[BooleanDecision]]:
        """Evaluate multiple states in one encoder batch.

        State embeddings are produced together by the sentence encoder. Each
        resulting embedding is then evaluated independently against the same
        projected capability hypotheses.

        Args:
            states: Natural-language states or tasks to evaluate.

        Returns:
            One decision list per input state. Each inner list contains one
            ``BooleanDecision`` per configured capability.

        Raises:
            ValueError: If any state is empty or contains only whitespace.
        """
        if not states:
            return []

        if any(
            not state.strip()
            for state in states
        ):
            raise ValueError(
                "states must not contain empty values"
            )

        state_embeddings = self.encoder.encode(
            states,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        state_embeddings = (
            state_embeddings
            .detach()
            .clone()
            .float()
        )

        with torch.no_grad():
            # Project static question embeddings once for the entire batch.
            projected_questions = self.projection(
                self._question_embeddings
            )

            results = []

            for state_embedding in state_embeddings:
                projected_state = self.projection(
                    state_embedding
                )

                logits = self.head(
                    state_embedding=projected_state,
                    question_embeddings=projected_questions,
                )

                probabilities = (
                    torch.sigmoid(logits)
                    .detach()
                    .cpu()
                )

                results.append(
                    [
                        BooleanDecision(
                            name=name,
                            probability=float(
                                probability
                            ),
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
        """Return thresholded capability decisions for multiple states.

        Args:
            states: Natural-language states or tasks to evaluate.

        Returns:
            One list of active decisions per input state.
        """
        return [
            [
                decision
                for decision in decisions
                if (
                    decision.probability
                    >= self.threshold
                )
            ]
            for decisions in self.decide_many(
                states
            )
        ]

    @property
    def metadata(
        self,
    ) -> dict[str, object]:
        """Return runtime and checkpoint metadata for this router.

        Returns:
            A dictionary containing router version, encoder identifier,
            threshold, execution device, and configured capabilities.
        """
        return {
            "router_version": self.router_version,
            "encoder_model_name": self.encoder_model_name,
            "threshold": self.threshold,
            "device": self.device,
            "capabilities": list(
                self._question_names
            ),
        }
