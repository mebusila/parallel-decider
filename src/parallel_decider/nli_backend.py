"""Natural-language-inference backend for boolean decisions.

This module implements a ``DecisionBackend``-compatible classifier using a
sequence-classification model trained for natural-language inference (NLI).

Each boolean question is represented by a natural-language hypothesis. The
shared state is paired with every hypothesis, all pairs are evaluated in one
batch, and the model's entailment probability is returned as the probability
of the corresponding boolean decision.
"""

from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .decision import BooleanDecision
from .question import BooleanQuestion


class NLIBackend:
    """Evaluate boolean hypotheses with a natural-language-inference model.

    The backend treats the supplied state as an NLI premise and each
    ``BooleanQuestion.hypothesis`` as a hypothesis. The entailment probability
    produced by the model is interpreted as the probability that the boolean
    decision is true.

    Args:
        model_name: Hugging Face model identifier for an NLI-compatible
            sequence-classification model.
        device: Torch device on which inference should run. If omitted, CUDA
            is selected when available; otherwise CPU is used.
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device: str | None = None,
    ) -> None:
        """Initialize the tokenizer and NLI classification model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name)
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def decide(
        self,
        state: str,
        questions: Sequence[BooleanQuestion],
    ) -> list[BooleanDecision]:
        """Evaluate boolean questions against a shared natural-language state.

        All state/hypothesis pairs are tokenized and evaluated together in a
        single model batch.

        Args:
            state: Natural-language premise shared by all decisions.
            questions: Boolean questions whose hypotheses should be evaluated.

        Returns:
            One ``BooleanDecision`` per supplied question, preserving the
            original question order. An empty question sequence returns an
            empty list.
        """
        if not questions:
            return []

        # The state is duplicated only at the tokenizer/model input level so
        # every question can be evaluated as an independent NLI pair.
        states = [
            state
        ] * len(questions)

        hypotheses = [
            question.hypothesis
            for question in questions
        ]

        inputs = self.tokenizer(
            states,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        # NLI models may assign different numeric IDs to their labels, so use
        # the model configuration rather than assuming a fixed class index.
        entailment_id = (
            self.model.config.label2id[
                "entailment"
            ]
        )

        return [
            BooleanDecision(
                name=question.name,
                probability=probabilities[
                    index,
                    entailment_id,
                ].item(),
            )
            for index, question in enumerate(
                questions
            )
        ]
