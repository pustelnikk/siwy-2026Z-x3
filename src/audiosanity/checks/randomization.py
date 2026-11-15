"""Model randomization (Adebayo et al., 2018, "Sanity Checks for Saliency Maps").

Two variants, chosen with ``checks.randomization.modes``:

* ``cascading``: layer groups are randomized cumulatively from the output towards the input. Rows have
  ``check="randomization"``.
* ``independent``: one group at a time is randomized while all others keep their trained weights.
  Rows have ``check="randomization_independent"``, so the two curves get separate figures.

After each step the maps are recomputed (for the *original* target class) and compared with the maps of
the trained model. A map that stays similar does not depend on the learned weights.

Deviation from the paper: Adebayo et al. re-initialise weights from a truncated normal distribution
(mean 0, std 0.01 for their CNN; a uniform distribution gave identical results). Here every parameter
tensor is replaced by Gaussian noise with that tensor's own mean and standard
deviation (``randomize_``). This needs no knowledge of how each architecture was initialised and keeps
activations on the trained scale; it should be reported as such.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from audiosanity.checks.base import CheckContext

NAME = "randomization"
MODES = {"cascading": NAME, "independent": f"{NAME}_independent"}


@torch.no_grad()
def randomize_(modules, generator: torch.Generator) -> None:
    """Replace every parameter by Gaussian noise with the parameter tensor's own mean and std."""
    for module in modules:
        for p in module.parameters():
            std = p.std() if p.numel() > 1 else p.abs()
            noise = torch.randn(p.shape, generator=generator).to(p.device, p.dtype)
            p.copy_(noise * std + p.mean())


@contextmanager
def randomized(modules, generator: torch.Generator) -> Iterator[None]:
    """Randomize ``modules`` inside the block and restore their trained weights afterwards."""
    saved = [p.detach().clone() for m in modules for p in m.parameters()]
    randomize_(modules, generator)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, value in zip((p for m in modules for p in m.parameters()), saved, strict=True):
                p.copy_(value)


def selected_groups(model, cfg) -> list:
    groups = model.layer_groups()
    max_groups = cfg.checks.randomization.max_groups
    return groups if max_groups is None else groups[:max_groups]


def modes(ctx: CheckContext) -> list[str]:
    selected = list(ctx.cfg.checks.randomization.modes)
    unknown = set(selected) - set(MODES)
    if unknown:
        raise ValueError(f"unknown randomization modes {sorted(unknown)}; available: {sorted(MODES)}")
    return selected


def n_rounds(ctx: CheckContext) -> int:
    return len(selected_groups(ctx.model, ctx.cfg)) * len(modes(ctx))


def _recompute(ctx: CheckContext, model, check: str, step: int, name: str) -> list[dict]:
    rows = []
    for method, orig in ctx.originals.items():
        rand = ctx.explain(ctx.windows, method, model=model, targets=orig.targets)
        rows += ctx.compare(
            orig.maps, rand.maps, check=check, method=method, condition=f"{step:02d}_{name}", step=step
        )
    ctx.progress(f"{check} {step:02d}_{name}")
    return rows


def run(ctx: CheckContext) -> list[dict]:
    rows = []
    for mode in modes(ctx):
        check = MODES[mode]
        model = copy.deepcopy(ctx.model)
        groups = selected_groups(model, ctx.cfg)
        generator = torch.Generator().manual_seed(ctx.cfg.seed)
        for method, orig in ctx.originals.items():
            rows += ctx.control(orig.maps, check=check, method=method, step=0)
        for step, (name, modules) in enumerate(groups, start=1):
            if mode == "cascading":
                randomize_(modules, generator)
                rows += _recompute(ctx, model, check, step, name)
            else:
                with randomized(modules, generator):
                    rows += _recompute(ctx, model, check, step, name)
        del model
    return rows
