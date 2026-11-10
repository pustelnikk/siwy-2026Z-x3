"""Cascading model randomization (Adebayo et al., 2018).

Layer groups are randomized cumulatively from the output towards the input; after each step the
maps are recomputed (for the *original* target class) and compared with the maps of the trained
model. A map that stays similar does not depend on the learned weights.
"""

from __future__ import annotations

import copy

import torch

from audiosanity.checks.base import CheckContext

NAME = "randomization"


@torch.no_grad()
def randomize_(modules, generator: torch.Generator) -> None:
    """Replace every parameter by Gaussian noise with the parameter tensor's own mean and std."""
    for module in modules:
        for p in module.parameters():
            std = p.std() if p.numel() > 1 else p.abs()
            noise = torch.randn(p.shape, generator=generator).to(p.device, p.dtype)
            p.copy_(noise * std + p.mean())


def selected_groups(model, cfg) -> list:
    groups = model.layer_groups()
    max_groups = cfg.checks.randomization.max_groups
    return groups if max_groups is None else groups[:max_groups]


def n_rounds(ctx: CheckContext) -> int:
    return len(selected_groups(ctx.model, ctx.cfg))


def run(ctx: CheckContext) -> list[dict]:
    model = copy.deepcopy(ctx.model)
    groups = selected_groups(model, ctx.cfg)
    generator = torch.Generator().manual_seed(ctx.cfg.seed)

    rows = []
    for method, orig in ctx.originals.items():
        rows += ctx.control(orig.maps, check=NAME, method=method, step=0)
    for step, (name, modules) in enumerate(groups, start=1):
        randomize_(modules, generator)
        for method, orig in ctx.originals.items():
            rand = ctx.explain(ctx.windows, method, model=model, targets=orig.targets)
            rows += ctx.compare(
                orig.maps,
                rand.maps,
                check=NAME,
                method=method,
                condition=f"{step:02d}_{name}",
                step=step,
            )
        ctx.progress(f"{NAME} {step:02d}_{name}")
    return rows
