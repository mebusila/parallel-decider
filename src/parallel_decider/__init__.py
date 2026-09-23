"""Public API for the ``parallel_decider`` package.

The package exposes the core decision types, backend protocol, NLI baseline,
and the checkpoint-backed ``ParallelDecider`` routing interface.
"""

from .backend import DecisionBackend
from .decision import BooleanDecision
from .nli_backend import NLIBackend
from .parallel_decider import ParallelDecider
from .question import BooleanQuestion


__all__ = [
    "BooleanDecision",
    "BooleanQuestion",
    "DecisionBackend",
    "NLIBackend",
    "ParallelDecider",
]
