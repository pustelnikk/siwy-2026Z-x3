"""Time-shift equivariance (after Kindermans et al., 2017, adapted to audio).

The analysis window is moved by ``k`` grid frames inside the (longer) clip. The map of the shifted
window, moved back by ``k`` frames, should match the original map on the overlapping part. Windows
are cut from the clip centre so that no padding enters the model. The change of the model's own
score is stored next to each similarity: a map change is only a violation if the model did not
change its decision.
"""

from __future__ import annotations

import numpy as np

from audiosanity.checks.base import CheckContext
from audiosanity.data import center_window

NAME = "shift"


def overlap(ref: np.ndarray, shifted: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Aligned overlapping parts of ``(n, T, F)`` maps when ``shifted`` starts ``k`` frames later."""
    if k >= 0:
        return ref[:, k:], shifted[:, : shifted.shape[1] - k]
    return ref[:, :k], shifted[:, -k:]


def n_rounds(ctx: CheckContext) -> int:
    return len(ctx.cfg.checks.shift.shifts)


def run(ctx: CheckContext) -> list[dict]:
    n_samples = ctx.windows.shape[1]
    hop = ctx.model.hop_length
    rows = []
    for k in ctx.cfg.checks.shift.shifts:
        waves = np.stack([center_window(c.wave, n_samples, offset=k * hop) for c in ctx.clips])
        for method, orig in ctx.originals.items():
            shifted = ctx.explain(waves, method, targets=orig.targets)
            ref, moved = overlap(orig.maps, shifted.maps, k)
            meta = dict(check=NAME, method=method, shift=k)
            rows += ctx.compare(
                ref,
                moved,
                condition=f"shift_{k:+d}",
                per_clip={"pred_change": np.abs(shifted.scores - orig.scores)},
                **meta,
            )
            rows += ctx.control(ref, moved, **meta)
        ctx.progress(f"{NAME} {k:+d}")
    return rows
