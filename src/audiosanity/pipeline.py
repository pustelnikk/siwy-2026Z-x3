"""Experiment runner: config -> model + clips -> original maps -> checks -> results directory.

A run directory contains:
    config.yaml      resolved configuration (re-run with `audiosanity run <run>/config.yaml`)
    predictions.csv  clips, explained classes and scores
    maps.npz         original attribution maps per method (n_clips, T, F)
    results.csv      every clip-level similarity (long format)
    summary.csv      means with bootstrap intervals per check/method/condition/normalisation/measure
    figures/         <check>.png (primary normalisation/measure) and <check>_all.png per check
    tb/              TensorBoard logs
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from audiosanity.attribution import is_available
from audiosanity.checks import CHECKS, ROUNDS, CheckContext
from audiosanity.data import center_window, load_clips
from audiosanity.measures import bootstrap_ci
from audiosanity.models import load_model
from audiosanity.plotting import plot_all

log = logging.getLogger(__name__)

GROUP_COLS = ["model", "check", "method", "condition", "normalization", "measure"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds third-party code using the global RNG
    torch.manual_seed(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def summarize(results: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for key, group in results.groupby(GROUP_COLS, sort=False):
        mean, lo, hi = bootstrap_ci(group["value"].to_numpy(), n_boot=n_boot, seed=seed)
        row = dict(zip(GROUP_COLS, key, strict=True), mean=mean, ci_low=lo, ci_high=hi, n=len(group))
        if "pred_change" in group and group["pred_change"].notna().any():
            row["pred_change"] = group["pred_change"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def run_checks(ctx: CheckContext, model_name: str, run_dir: Path) -> pd.DataFrame:
    """Run the configured checks; ``results.csv`` is rewritten after each one (partial results)."""
    frames = []
    for name in ctx.cfg.checks.run:
        start = time.perf_counter()
        frame = pd.DataFrame(CHECKS[name](ctx))
        frame.insert(0, "model", model_name)
        frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(run_dir / "results.csv", index=False)
        log.info(
            "check %-13s done in %.1f min (results.csv updated)", name, (time.perf_counter() - start) / 60
        )
    return pd.concat(frames, ignore_index=True)


def run_experiment(cfg) -> Path:
    seed_everything(cfg.seed)
    run_dir = Path(cfg.output_dir) / f"{cfg.name}_{datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "figures").mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    log.info("Run directory: %s", run_dir)

    device = resolve_device(cfg.device)
    model = load_model(cfg.model.arch, cfg.model.checkpoint, device)
    model_name = f"{cfg.model.arch}:{cfg.model.checkpoint}"
    window = int(cfg.data.clip_seconds * model.sample_rate)
    clips = load_clips(cfg.data, model.sample_rate, window, cfg.seed)
    windows = np.stack([center_window(c.wave, window) for c in clips])

    unknown = set(cfg.checks.run) - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown checks {sorted(unknown)}; available: {sorted(CHECKS)}")
    ctx = CheckContext(model=model, clips=clips, windows=windows, cfg=cfg)
    ctx.total_rounds = 1 + sum(ROUNDS[name](ctx) for name in cfg.checks.run)
    unavailable = []
    for method in cfg.attribution.methods:
        if not is_available(method, model):
            log.warning("method %s is unavailable for %s", method, cfg.model.arch)
            unavailable.append(method)
            continue
        ctx.originals[method] = ctx.explain(windows, method)
    ctx.progress("original maps")
    np.savez_compressed(run_dir / "maps.npz", **{m: e.maps for m, e in ctx.originals.items()})

    first = next(iter(ctx.originals.values()))
    pd.DataFrame(
        {
            "clip": range(len(clips)),
            "file": [c.name for c in clips],
            "label": [c.label for c in clips],
            "explained_class": [model.labels[t] for t in first.targets],
            "score": first.scores,
        }
    ).to_csv(run_dir / "predictions.csv", index=False)

    results = run_checks(ctx, model_name, run_dir)
    summary = summarize(results, cfg.n_bootstrap, cfg.seed)
    if unavailable:
        summary = pd.concat(
            [summary, pd.DataFrame({"model": model_name, "method": unavailable, "status": "unavailable"})]
        )
    summary.to_csv(run_dir / "summary.csv", index=False)

    log_tensorboard(run_dir / "tb", cfg, summary)
    plot_all(summary, run_dir / "figures", cfg.plot.normalization, cfg.plot.measure)
    return run_dir


def log_tensorboard(log_dir: Path, cfg, summary: pd.DataFrame) -> None:
    with SummaryWriter(log_dir) as writer:
        writer.add_text("config", f"```\n{OmegaConf.to_yaml(cfg)}\n```")
        for (check, method, norm, measure), group in summary.dropna(subset=["mean"]).groupby(
            ["check", "method", "normalization", "measure"], sort=False
        ):
            tag = f"{check}/{method}/{norm}/{measure}"
            conditions = group[~group["condition"].str.startswith("control")]
            for step, value in enumerate(conditions["mean"]):
                writer.add_scalar(tag, value, step)
