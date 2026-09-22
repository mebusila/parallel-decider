from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .decision import BooleanDecision
from .question import BooleanQuestion


class NLIBackend:
    """Evaluate boolean hypotheses using a natural-language-inference model."""

    def __init__(
        self,
        model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()

    def decide(
        self,
        state: str,
        questions: Sequence[BooleanQuestion],
    ) -> list[BooleanDecision]:
        decisions: list[BooleanDecision] = []

        for question in questions:
            inputs = self.tokenizer(
                state,
                question.hypothesis,
                return_tensors="pt",
                truncation=True,
            )

            with torch.inference_mode():
                outputs = self.model(**inputs)

            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

            entailment_id = self.model.config.label2id["entailment"]
            probability = probabilities[entailment_id].item()

            decisions.append(
                BooleanDecision(
                    name=question.name,
                    probability=probability,
                )
            )

        return decisions