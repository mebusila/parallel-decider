from __future__ import annotations

import sys

from parallel_decider.cli import main
from parallel_decider.decision import BooleanDecision


class FakeDecider:
    threshold = 0.5

    @classmethod
    def from_checkpoint(
        cls,
        path,
        device=None,
    ):
        return cls()

    def decide(
        self,
        state: str,
    ):
        return [
            BooleanDecision(
                name="needs_shell",
                probability=0.9,
            ),
            BooleanDecision(
                name="needs_git",
                probability=0.1,
            ),
        ]

    def active_decisions(
        self,
        state: str,
    ):
        return [
            BooleanDecision(
                name="needs_shell",
                probability=0.9,
            )
        ]
def test_cli_prints_all_decisions(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "parallel_decider.cli.ParallelDecider",
        FakeDecider,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parallel-decider",
            "Run the tests.",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "needs_shell" in output
    assert "0.900" in output
    assert "needs_git" in output
    assert "0.100" in output

def test_cli_active_only(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "parallel_decider.cli.ParallelDecider",
        FakeDecider,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parallel-decider",
            "--active-only",
            "Run the tests.",
        ],
    )

    main()

    output = capsys.readouterr().out

    assert "needs_shell" in output
    assert "needs_git" not in output
