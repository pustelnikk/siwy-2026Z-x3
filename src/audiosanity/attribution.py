"""Attribution methods and the ``explain`` entry point that maps waveforms to attribution grids.

To add a method, write a function ``(model, inputs, targets, cfg, baselines) -> attributions`` (same
shape as ``inputs``) and register it in ``METHODS``. ``baselines`` is only computed for methods listed in
``BASELINE_METHODS`` (``None`` otherwise). A method that cannot be applied to an architecture
must be listed in ``UNAVAILABLE`` - it is then reported as unavailable, never as failing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from captum.attr import InputXGradient, IntegratedGradients, NoiseTunnel, Saliency

from audiosanity.models.base import AudioClassifier

AttributionFn = Callable[
    [AudioClassifier, torch.Tensor, torch.Tensor, object, torch.Tensor | None], torch.Tensor
]

# Reference input of Integrated Gradients. The verdict of input invariance depends on it
# (Kindermans et al., 2017), so it is a config choice (``attribution.ig_baseline``):
#   zero     - zeros in model-input space; for AST that is the dataset-mean log-mel, *not* silence
#   silence  - the model input of an all-zero waveform, i.e. true silence through the model's own front
#              end; it moves with a constant input shift, like the "black image" baseline of Kindermans et al.
#   mean     - per-clip mean of the model input
BASELINES = ("zero", "silence", "mean")


def make_baseline(model: AudioClassifier, waves: np.ndarray, inputs: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros_like(inputs)
    if kind == "silence":
        return model.features(np.zeros_like(waves))
    if kind == "mean":
        return inputs.mean(dim=tuple(range(1, inputs.ndim)), keepdim=True).expand_as(inputs).contiguous()
    raise ValueError(f"unknown IG baseline {kind!r}; available: {list(BASELINES)}")


def _saliency(model, inputs, targets, cfg, baselines=None):
    return Saliency(model).attribute(inputs, target=targets, abs=False)


def _input_x_gradient(model, inputs, targets, cfg, baselines=None):
    return InputXGradient(model).attribute(inputs, target=targets)


def _integrated_gradients(model, inputs, targets, cfg, baselines=None):
    return IntegratedGradients(model).attribute(
        inputs,
        baselines=baselines,
        target=targets,
        n_steps=cfg.ig_steps,
        internal_batch_size=cfg.internal_batch_size,
    )


def _smoothgrad(model, inputs, targets, cfg, baselines=None):
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

# Methods that take a reference input; ``explain`` builds it with ``make_baseline``.
BASELINE_METHODS = {"integrated_gradients"}

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
        baselines = None
        if method in BASELINE_METHODS:
            baselines = make_baseline(model, batch, inputs, cfg.ig_baseline)
        torch.manual_seed(seed + start)  # reproducible noise for stochastic methods
        attr = fn(model, inputs, torch.as_tensor(batch_targets, device=model.device), cfg, baselines)
        maps.append(model.to_grid(attr, batch.shape[-1]))
        out_targets.append(batch_targets)
        scores.append(probs[np.arange(len(batch)), batch_targets])
    return Explanation(np.concatenate(maps), np.concatenate(out_targets), np.concatenate(scores))
