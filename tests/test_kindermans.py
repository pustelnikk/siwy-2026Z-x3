"""Exact input invariance (Kindermans et al., 2017) and the IG baseline it depends on."""

import numpy as np
import pytest
import torch

from audiosanity.attribution import BASELINES, explain, make_baseline
from audiosanity.config import AttributionConfig
from audiosanity.data import center_window
from audiosanity.models import ASTClassifier, Wav2Vec2Classifier

SR = 16_000
OFFSET = 0.7


@pytest.fixture
def ast_pair():
    model = ASTClassifier.tiny().requires_grad_(False).eval()
    return model, model.input_shifted(OFFSET)


@pytest.fixture
def waves(clips):
    return np.stack([center_window(c.wave, SR) for c in clips])


def _close(a, b):
    return np.allclose(a, b, rtol=1e-3, atol=1e-4 * np.abs(a).max())


def test_shifted_model_computes_the_same_function(ast_pair, waves):
    model, shifted = ast_pair
    np.testing.assert_allclose(
        shifted.features(waves).cpu(), (model.features(waves) + OFFSET).cpu(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(shifted.predict_proba(waves), model.predict_proba(waves), atol=1e-5)
    assert model.input_offset == 0.0  # the original is untouched


@pytest.mark.parametrize(
    ("method", "baseline", "invariant"),
    [
        ("saliency", "zero", True),  # gradient: input invariant
        ("input_x_gradient", "zero", False),  # x * grad changes with x
        ("integrated_gradients", "zero", False),  # fixed baseline does not move with the input
        ("integrated_gradients", "silence", True),  # baseline built by the shifted front end moves too
    ],
)
def test_kindermans_verdicts(ast_pair, waves, method, baseline, invariant):
    model, shifted = ast_pair
    cfg = AttributionConfig(batch_size=3, ig_steps=8, ig_baseline=baseline)
    targets = np.array([0, 1, 2])
    a = explain(model, waves, method, cfg, targets=targets).maps
    b = explain(shifted, waves, method, cfg, targets=targets).maps
    assert _close(a, b) is invariant


def test_baselines(tiny_model, waves):
    inputs = tiny_model.features(waves)
    for kind in BASELINES:
        base = make_baseline(tiny_model, waves, inputs, kind)
        assert base.shape == inputs.shape and torch.isfinite(base).all()
    assert torch.equal(make_baseline(tiny_model, waves, inputs, "zero"), torch.zeros_like(inputs))
    with pytest.raises(ValueError):
        make_baseline(tiny_model, waves, inputs, "nope")


def test_unsupported_model_raises():
    model = Wav2Vec2Classifier.tiny()
    assert not model.supports_input_shift
    with pytest.raises(NotImplementedError):
        model.input_shifted(1.0)
