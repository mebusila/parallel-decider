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
        if not questions:
            return []

        states = [state] * len(questions)
        hypotheses = [question.hypothesis for question in questions]

        inputs = self.tokenizer(
            states,
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.inference_mode():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)

        entailment_id = self.model.config.label2id["entailment"]

        return [
            BooleanDecision(
                name=question.name,
                probability=probabilities[index, entailment_id].item(),
            )
            for index, question in enumerate(questions)
        ]