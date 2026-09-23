"""Command-line interface for the parallel routing engine.

This module exposes the ``parallel-decider`` command. It loads a trained
routing checkpoint, evaluates a natural-language state, and prints the
resulting capability probabilities.

The CLI can either display all configured decisions or only the capabilities
whose probabilities meet the threshold stored in the checkpoint.
"""

from __future__ import annotations

import argparse

from parallel_decider import ParallelDecider


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        A configured ``ArgumentParser`` for the ``parallel-decider`` command.
    """
    parser = argparse.ArgumentParser(
        prog="parallel-decider",
        description=(
            "Evaluate parallel routing decisions "
            "for a natural-language state."
        ),
    )

    parser.add_argument(
        "state",
        help="State or task to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        default="models/router-bge-base-v3.pt",
        help=(
            "Path to the router checkpoint. "
            "Default: models/router-bge-base-v3.pt"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device, for example cpu or cuda. "
            "Defaults to automatic selection."
        ),
    )

    parser.add_argument(
        "--active-only",
        action="store_true",
        help=(
            "Show only capabilities whose probability "
            "meets the checkpoint threshold."
        ),
    )

    return parser


def main() -> None:
    """Run the ``parallel-decider`` command-line application."""
    parser = build_parser()
    args = parser.parse_args()

    # The checkpoint contains both the learned routing weights and the metadata
    # required to reconstruct the configured router.
    decider = ParallelDecider.from_checkpoint(
        args.checkpoint,
        device=args.device,
    )

    if args.active_only:
        decisions = decider.active_decisions(
            args.state
        )
    else:
        decisions = decider.decide(
            args.state
        )

    if not decisions:
        print("No active capabilities.")
        return

    for decision in decisions:
        active = (
            decision.probability
            >= decider.threshold
        )

        # Active decisions are marked explicitly when all capabilities are
        # displayed, making the checkpoint threshold visible in CLI output.
        marker = "*" if active else " "

        print(
            f"{marker} "
            f"{decision.name:<20} "
            f"{decision.probability:.3f}"
        )


if __name__ == "__main__":
    main()
