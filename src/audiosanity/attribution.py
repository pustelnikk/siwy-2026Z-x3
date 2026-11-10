"""Attribution methods and the ``explain`` entry point that maps waveforms to attribution grids.

To add a method, write a function ``(model, inputs, targets, cfg) -> attributions`` (same shape as
``inputs``) and register it in ``METHODS``. A method that cannot be applied to an architecture
must be listed in ``UNAVAILABLE`` - it is then reported as unavailable, never as failing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from captum.attr import InputXGradient, IntegratedGradients, NoiseTunnel, Saliency

from audiosanity.models.base import AudioClassifier

AttributionFn = Callable[[AudioClassifier, torch.Tensor, torch.Tensor, object], torch.Tensor]


def _saliency(model, inputs, targets, cfg):
    return Saliency(model).attribute(inputs, target=targets, abs=False)


def _input_x_gradient(model, inputs, targets, cfg):
    return InputXGradient(model).attribute(inputs, target=targets)


def _integrated_gradients(model, inputs, targets, cfg):
    return IntegratedGradients(model).attribute(
        inputs,
        baselines=torch.zeros_like(inputs),
        target=targets,
        n_steps=cfg.ig_steps,
        internal_batch_size=cfg.internal_batch_size,
    )


def _smoothgrad(model, inputs, targets, cfg):
    stdev = float(cfg.smoothgrad_noise * (inputs.max() - inputs.min()))
    return NoiseTunnel(Saliency(model)).attribute(
        inputs,
        nt_type="smoothgrad",
        nt_samples=cfg.smoothgrad_samples,
        # captum runs (clips in batch) x nt_samples_batch_size inputs at once
        nt_samples_batch_size=max(1, cfg.internal_batch_size // len(inputs)),
        stdevs=stdev,
        target=targets,
        abs=False,
    )


METHODS: dict[str, AttributionFn] = {
    "saliency": _saliency,
    "input_x_gradient": _input_x_gradient,
    "integrated_gradients": _integrated_gradients,
    "smoothgrad": _smoothgrad,
}

# (method, architecture class name) pairs that cannot be computed, e.g. Grad-CAM on a waveform model.
UNAVAILABLE: set[tuple[str, str]] = set()


def is_available(method: str, model: AudioClassifier) -> bool:
    return (method, type(model).__name__) not in UNAVAILABLE


@dataclass
class Explanation:
    maps: np.ndarray  # (B, T, F) signed attributions on the model grid
    targets: np.ndarray  # (B,) explained class indices
    scores: np.ndarray  # (B,) model score of the explained class


def explain(
    model: AudioClassifier,
    waves: np.ndarray,
    method: str,
    cfg,
    targets: np.ndarray | None = None,
    seed: int = 0,
) -> Explanation:
    """Compute attribution grids for a batch of equal-length waveforms ``(B, N)``.

    If ``targets`` is None the top predicted class of each clip is explained.
    """
    if method not in METHODS:
        raise ValueError(f"unknown attribution method {method!r}; available: {sorted(METHODS)}")
    fn = METHODS[method]
    maps, out_targets, scores = [], [], []
    for start in range(0, len(waves), cfg.batch_size):
        batch = np.asarray(waves[start : start + cfg.batch_size], dtype=np.float32)
        inputs = model.features(batch)
        probs = model.predict_proba(batch)
        if targets is None:
            batch_targets = probs.argmax(axis=1)
        else:
            batch_targets = np.asarray(targets[start : start + cfg.batch_size])
        torch.manual_seed(seed + start)  # reproducible noise for stochastic methods
        attr = fn(model, inputs, torch.as_tensor(batch_targets, device=model.device), cfg)
        maps.append(model.to_grid(attr, batch.shape[-1]))
        out_targets.append(batch_targets)
        scores.append(probs[np.arange(len(batch)), batch_targets])
    return Explanation(np.concatenate(maps), np.concatenate(out_targets), np.concatenate(scores))
