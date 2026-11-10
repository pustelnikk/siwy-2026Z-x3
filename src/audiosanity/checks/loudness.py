"""Loudness confound: does the map just follow the input energy?

Two references on the map grid: the log-magnitude spectrogram (cell by cell) and the onset strength
(spectral flux, compared with the time profile of the map). High correlation, well above the
cross-clip control, means the map tracks the input rather than the decision.
Only correlation measures are meaningful here, so SSIM/top-k are not used.
"""

from __future__ import annotations

import numpy as np

from audiosanity.checks.base import CheckContext

NAME = "loudness"
CORRELATION_MEASURES = ("spearman", "pearson")


def onset_strength(reference: np.ndarray) -> np.ndarray:
    """Spectral flux of ``(n, T, F)`` log-magnitude -> ``(n, T, 1)``."""
    flux = np.maximum(np.diff(reference, axis=1), 0).sum(axis=2, keepdims=True)
    return np.concatenate([np.zeros_like(flux[:, :1]), flux], axis=1)


def time_profile(maps: np.ndarray) -> np.ndarray:
    return np.abs(maps).sum(axis=2, keepdims=True)


def n_rounds(ctx: CheckContext) -> int:
    return 0  # reuses the original maps, no new attributions


def run(ctx: CheckContext) -> list[dict]:
    measures = [m for m in ctx.cfg.checks.measures if m in CORRELATION_MEASURES] or ["spearman"]
    reference = ctx.model.reference(ctx.windows)
    onsets = onset_strength(reference)
    rows = []
    for method, orig in ctx.originals.items():
        meta = dict(check=NAME, method=method, measures=measures)
        rows += ctx.compare(orig.maps, reference, condition="log_magnitude", **meta)
        rows += ctx.control(orig.maps, reference, condition="control_log_magnitude", **meta)
        profile = time_profile(orig.maps)
        rows += ctx.compare(profile, onsets, condition="onset_strength", **meta)
        rows += ctx.control(profile, onsets, condition="control_onset_strength", **meta)
    return rows
