"""Gain, resampling and codec invariance.

Each transformation keeps the length and sample rate of the window, so maps stay on the same grid
and can be compared cell by cell. As in the shift check, the change of the model's score is stored
so that map changes can be set against prediction changes.
"""

from __future__ import annotations

import numpy as np

from audiosanity.checks.base import CheckContext
from audiosanity.transforms import codec_roundtrip, gain, resample_roundtrip

NAME = "invariance"


def transformations(ctx: CheckContext) -> dict[str, callable]:
    cfg, sr = ctx.cfg.checks.invariance, ctx.model.sample_rate
    out = {}
    for db in cfg.gains_db:
        out[f"gain_{db:+g}dB"] = lambda w, db=db: gain(w, db)
    for rate in cfg.resample_rates:
        out[f"resample_{rate}"] = lambda w, rate=rate: resample_roundtrip(w, sr, rate)
    for codec in cfg.codecs:
        out[f"codec_{codec}"] = lambda w, codec=codec: codec_roundtrip(w, sr, codec)
    return out


def n_rounds(ctx: CheckContext) -> int:
    return len(transformations(ctx))


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
    return rows
