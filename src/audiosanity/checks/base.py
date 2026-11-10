"""Shared machinery for checks.

Every check is a function ``run(ctx) -> list[dict]`` returning rows in one long format:
``check, method, condition, clip, normalization, measure, value`` (+ optional extra columns such as
``pred_change``). Each check also emits rows with ``condition="control"``: the same comparison made
between two *different* clips, i.e. the similarity expected by chance on this data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from audiosanity.attribution import Explanation, explain
from audiosanity.data import Clip
from audiosanity.measures import similarity
from audiosanity.models.base import AudioClassifier

log = logging.getLogger(__name__)


@dataclass
class CheckContext:
    model: AudioClassifier
    clips: list[Clip]
    windows: np.ndarray  # (n_clips, n_samples) centre windows - the analysed audio
    cfg: object  # full experiment config
    originals: dict[str, Explanation] = field(default_factory=dict)  # method -> maps of unmodified input
    # Progress over "rounds" (one round = maps of all clips with all methods for one condition).
    total_rounds: int = 0
    done_rounds: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def progress(self, label: str) -> None:
        """Log one finished round with the elapsed time and a linear estimate of the time left."""
        self.done_rounds += 1
        elapsed = time.perf_counter() - self.start_time
        left = elapsed / self.done_rounds * max(0, self.total_rounds - self.done_rounds)
        log.info(
            "[%d/%d] %-28s elapsed %5.1f min, ~%5.1f min left",
            self.done_rounds,
            self.total_rounds,
            label,
            elapsed / 60,
            left / 60,
        )

    @property
    def methods(self) -> list[str]:
        return list(self.originals)

    def explain(
        self,
        waves: np.ndarray,
        method: str,
        model: AudioClassifier | None = None,
        targets: np.ndarray | None = None,
    ) -> Explanation:
        return explain(model or self.model, waves, method, self.cfg.attribution, targets, self.cfg.seed)

    def compare(
        self, ref: np.ndarray, other: np.ndarray, measures: list[str] | None = None, **meta
    ) -> list[dict]:
        """Clip-wise similarity of two aligned stacks of maps ``(n, T, F)`` for every norm x measure."""
        extra = meta.pop("per_clip", {})
        rows = []
        for i in range(len(ref)):
            for norm in self.cfg.checks.normalizations:
                for measure in measures or self.cfg.checks.measures:
                    row = dict(meta, clip=i, normalization=norm, measure=measure)
                    row["value"] = similarity(ref[i], other[i], norm, measure)
                    row.update({k: v[i] for k, v in extra.items()})
                    rows.append(row)
        return rows

    def control(
        self, ref: np.ndarray, other: np.ndarray | None = None, condition: str = "control", **meta
    ) -> list[dict]:
        """Random control: compare clip ``i`` with clip ``i + 1`` (cyclically)."""
        other = ref if other is None else other
        return self.compare(ref, np.roll(other, -1, axis=0), condition=condition, **meta)
