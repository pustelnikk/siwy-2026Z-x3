import numpy as np
import pytest

from audiosanity.measures import MEASURES, bootstrap_ci, normalize, similarity

rng = np.random.default_rng(0)
MAP = rng.standard_normal((40, 16))


@pytest.mark.parametrize("measure", list(MEASURES))
def test_identical_maps_are_maximally_similar(measure):
    assert similarity(MAP, MAP, "abs", measure) == pytest.approx(1.0)


@pytest.mark.parametrize("measure", ["spearman", "pearson"])
def test_independent_maps_are_near_zero(measure):
    other = rng.standard_normal(MAP.shape)
    assert abs(similarity(MAP, other, "raw", measure)) < 0.2


def test_normalizations():
    assert normalize(MAP, "abs").min() >= 0
    assert normalize(MAP, "abs_max").max() == pytest.approx(1.0)
    assert np.abs(normalize(MAP, "max_abs")).max() == pytest.approx(1.0)
    np.testing.assert_array_equal(normalize(MAP, "raw"), MAP)
    with pytest.raises(ValueError):
        normalize(MAP, "nope")


def test_rank_measures_ignore_scale_but_ssim_does_not():
    assert similarity(MAP, 5 * MAP, "raw", "spearman") == pytest.approx(1.0)
    assert similarity(MAP, 5 * MAP, "raw", "ssim") < 0.99
    assert similarity(MAP, 5 * MAP, "abs_max", "ssim") == pytest.approx(1.0)


def test_ssim_undefined_for_time_only_maps():
    assert np.isnan(similarity(MAP[:, :1], MAP[:, :1], "abs", "ssim"))


def test_bootstrap_ci_contains_mean_and_ignores_nan():
    values = np.r_[rng.normal(0.5, 0.1, 200), np.nan]
    mean, lo, hi = bootstrap_ci(values, n_boot=500)
    assert lo < mean < hi
    assert mean == pytest.approx(np.nanmean(values))
