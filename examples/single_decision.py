from parallel_decider import BooleanQuestion, NLIBackend

backend = NLIBackend()

questions = [
    BooleanQuestion(
        name="needs_git",
        hypothesis="This task requires access to Git.",
    )
]

result = backend.decide(
    state="Find which commit introduced the authentication bug.",
    questions=questions,
)

for decision in result:
    print(f"{decision.name}: {decision.probability:.3f}")
