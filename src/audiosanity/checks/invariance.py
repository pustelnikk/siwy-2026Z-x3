"""Input invariance: gain, resampling, codec, and the exact input shift of Kindermans et al. (2017).

Two kinds of conditions:

* **Audio transformations** (``gain_*``, ``resample_*``, ``codec_*``) keep the length and sample rate of
  the window, so maps stay on the same grid and can be compared cell by cell. The model sees a
  different input, so its score may move a little; the change is stored as ``pred_change`` so that map
  changes can be set against prediction changes.
* **Exact input shift** (``input_shift_*``, after Kindermans et al., 2017, "The (Un)reliability of
  saliency methods"): the model input is shifted by a constant and the first layer's bias compensates
  it (``AudioClassifier.input_shifted``). The two models compute the same function, so any map change
  is caused by the attribution method alone. In the paper the gradient passes, gradient x input fails,
  and Integrated Gradients passes only with a baseline that moves with the input
  (``attribution.ig_baseline=silence`` here). Skipped for models without ``input_shifted``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from audiosanity.checks.base import CheckContext
from audiosanity.transforms import codec_roundtrip, gain, resample_roundtrip

NAME = "invariance"

log = logging.getLogger(__name__)


def transformations(ctx: CheckContext) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    cfg, sr = ctx.cfg.checks.invariance, ctx.model.sample_rate
    out = {}
    for db in cfg.gains_db:
        out[f"gain_{db:+g}dB"] = lambda w, db=db: gain(w, db)
    for rate in cfg.resample_rates:
        out[f"resample_{rate}"] = lambda w, rate=rate: resample_roundtrip(w, sr, rate)
    for codec in cfg.codecs:
        out[f"codec_{codec}"] = lambda w, codec=codec: codec_roundtrip(w, sr, codec)
    return out


def input_shifts(ctx: CheckContext) -> list[float]:
    shifts = list(ctx.cfg.checks.invariance.input_shifts)
    if shifts and not ctx.model.supports_input_shift:
        return []
    return shifts


def n_rounds(ctx: CheckContext) -> int:
    return len(transformations(ctx)) + len(input_shifts(ctx))


def run(ctx: CheckContext) -> list[dict]:
    rows = []
    for method, orig in ctx.originals.items():
        rows += ctx.control(orig.maps, check=NAME, method=method)
    for name, transform in transformations(ctx).items():
        waves = np.stack([transform(w) for w in ctx.windows])
        for method, orig in ctx.originals.items():
            new = ctx.explain(waves, method, targets=orig.targets)
            rows += ctx.compare(
                orig.maps,
                new.maps,
                check=NAME,
                method=method,
                condition=name,
                per_clip={"pred_change": np.abs(new.scores - orig.scores)},
            )
        ctx.progress(f"{NAME} {name}")

    if ctx.cfg.checks.invariance.input_shifts and not input_shifts(ctx):
        log.warning("input-shift check skipped: %s has no input_shifted()", type(ctx.model).__name__)
    for offset in input_shifts(ctx):
        shifted_model = ctx.model.input_shifted(offset)
        name = f"input_shift_{offset:+g}"
        for method, orig in ctx.originals.items():
            new = ctx.explain(ctx.windows, method, model=shifted_model, targets=orig.targets)
            rows += ctx.compare(
                orig.maps,
                new.maps,
                check=NAME,
                method=method,
                condition=name,
                per_clip={"pred_change": np.abs(new.scores - orig.scores)},  # ~0 by construction
            )
        del shifted_model
        ctx.progress(f"{NAME} {name}")
    return rows
