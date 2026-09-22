from parallel_decider import (
    BooleanDecision,
    BooleanQuestion,
    DecisionBackend,
)


class FakeBackend:
    def decide(
        self,
        state: str,
        questions: list[BooleanQuestion],
    ) -> list[BooleanDecision]:
        return [
            BooleanDecision(
                name=question.name,
                probability=0.5,
            )
            for question in questions
        ]


def test_fake_backend_matches_protocol() -> None:
    backend: DecisionBackend = FakeBackend()

    questions = [
        BooleanQuestion(
            name="needs_git",
            hypothesis="This task requires Git access.",
        )
    ]

    result = backend.decide(
        state="Find which commit introduced this bug.",
        questions=questions,
    )

    assert result == [
        BooleanDecision(
            name="needs_git",
            probability=0.5,
        )
    ]