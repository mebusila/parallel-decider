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
        device: str | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model.to(self.device)
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

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

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