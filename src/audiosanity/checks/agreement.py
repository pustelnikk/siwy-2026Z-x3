"""Agreement between attribution methods (the disagreement problem, Krishna et al., 2022).

For every pair of methods the maps of the same clip are compared; the control compares the maps of
different clips. Agreement at control level means the methods do not share any clip-specific
evidence.
"""

from __future__ import annotations

from itertools import combinations

from audiosanity.checks.base import CheckContext

NAME = "agreement"


def n_rounds(ctx: CheckContext) -> int:
    return 0  # reuses the original maps, no new attributions


def run(ctx: CheckContext) -> list[dict]:
    rows = []
    for a, b in combinations(ctx.methods, 2):
        meta = dict(check=NAME, method=f"{a}|{b}")
        maps_a, maps_b = ctx.originals[a].maps, ctx.originals[b].maps
        rows += ctx.compare(maps_a, maps_b, condition="same_clip", **meta)
        rows += ctx.control(maps_a, maps_b, **meta)
    return rows
