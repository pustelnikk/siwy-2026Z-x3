import copy

import numpy as np
import torch

from audiosanity.attribution import METHODS, explain
from audiosanity.checks.randomization import randomize_
from audiosanity.config import AttributionConfig
from audiosanity.data import center_window

SR = 16_000
CFG = AttributionConfig(batch_size=2, ig_steps=4, smoothgrad_samples=4)


def test_grid_shape_and_reference(tiny_model, clips):
    waves = np.stack([center_window(c.wave, SR) for c in clips])
    exp = explain(tiny_model, waves, "saliency", CFG)
    ref = tiny_model.reference(waves)
    assert exp.maps.ndim == 3 and exp.maps.shape[0] == len(clips)
    assert exp.maps.shape == ref.shape
    # one frame per hop: about 100 frames per second
    assert abs(exp.maps.shape[1] - SR // tiny_model.hop_length) <= 3


def test_all_methods_run_and_fixed_targets_are_used(tiny_model, clips):
    waves = np.stack([center_window(c.wave, SR) for c in clips])
    targets = np.array([1, 2, 3])
    for method in METHODS:
        exp = explain(tiny_model, waves, method, CFG, targets=targets)
        np.testing.assert_array_equal(exp.targets, targets)
        assert np.isfinite(exp.maps).all()
        assert np.abs(exp.maps).sum() > 0


def test_layer_groups_cover_every_parameter(tiny_model):
    grouped = {id(p) for _, mods in tiny_model.layer_groups() for m in mods for p in m.parameters()}
    # masked_spec_embed (wav2vec2) is only used for masking during pre-training
    used = {id(p) for n, p in tiny_model.named_parameters() if "masked_spec_embed" not in n}
    assert grouped == used


def test_randomize_changes_only_selected_group(tiny_model):
    model = copy.deepcopy(tiny_model)
    groups = model.layer_groups()
    head, rest = groups[0][1], [m for _, mods in groups[1:] for m in mods]
    rest_before = [p.clone() for m in rest for p in m.parameters()]
    head_before = [p.clone() for m in head for p in m.parameters()]

    randomize_(head, torch.Generator().manual_seed(0))

    head_after = [p for m in head for p in m.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(head_before, head_after, strict=True))
    rest_after = [p for m in rest for p in m.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(rest_before, rest_after, strict=True))
