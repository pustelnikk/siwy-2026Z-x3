"""The battery. Each check module exposes ``NAME``, ``run(ctx) -> list[dict]`` and
``n_rounds(ctx) -> int`` (how many times it recomputes all maps; used for progress/ETA)."""

from __future__ import annotations

from collections.abc import Callable

from audiosanity.checks import agreement, invariance, loudness, randomization, shift
from audiosanity.checks.base import CheckContext

_MODULES = (randomization, shift, invariance, loudness, agreement)
CHECKS: dict[str, Callable[[CheckContext], list[dict]]] = {m.NAME: m.run for m in _MODULES}
ROUNDS: dict[str, Callable[[CheckContext], int]] = {m.NAME: m.n_rounds for m in _MODULES}

__all__ = ["CHECKS", "ROUNDS", "CheckContext"]
