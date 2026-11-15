"""Common interface every audio classifier must implement to be run through the battery."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class AudioClassifier(nn.Module, ABC):
    """Wraps a classifier so that the battery can treat all architectures the same way.

    Pipeline: ``waves --features()--> model input --forward()--> logits``.
    Attributions are computed with respect to the model input (spectrogram or waveform) and then
    projected by ``to_grid()`` onto a time-frequency grid with frames of ``hop_length`` samples.
    Frame ``k`` of the grid starts at sample ``k * hop_length``, so shifting the audio by
    ``k * hop_length`` samples shifts the grid by exactly ``k`` frames.
    """

    sample_rate: int = 16_000
    hop_length: int = 160
    labels: list[str]
    # Multi-label taggers (e.g. AudioSet) use sigmoid scores, single-label classifiers softmax.
    multi_label: bool = False
    # True if ``input_shifted`` is implemented (exact input-invariance check).
    supports_input_shift: bool = False

    @abstractmethod
    def features(self, waves: np.ndarray) -> torch.Tensor:
        """Turn a batch of waveforms ``(B, N)`` into the model input (on the model device)."""

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits ``(B, n_classes)`` for model inputs produced by ``features()``."""

    @abstractmethod
    def to_grid(self, attributions: torch.Tensor, n_samples: int) -> np.ndarray:
        """Project input-space attributions onto the ``(B, T, F)`` grid (signed values)."""

    @abstractmethod
    def reference(self, waves: np.ndarray) -> np.ndarray:
        """Log-magnitude of the input on the same ``(B, T, F)`` grid (for the loudness confound)."""

    @abstractmethod
    def layer_groups(self) -> list[tuple[str, list[nn.Module]]]:
        """Layer groups ordered from the output towards the input (for cascading randomization)."""

    def input_shifted(self, offset: float) -> AudioClassifier:
        """Copy whose model input is shifted by ``offset`` and whose first layer compensates the shift.

        The copy computes exactly the same function of the audio as ``self`` (Kindermans et al., 2017),
        so an attribution method that is input invariant must give it the same map.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support the input-shift check")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def predict_proba(self, waves: np.ndarray) -> np.ndarray:
        logits = self(self.features(waves))
        scores = torch.sigmoid(logits) if self.multi_label else torch.softmax(logits, dim=-1)
        return scores.cpu().numpy()
