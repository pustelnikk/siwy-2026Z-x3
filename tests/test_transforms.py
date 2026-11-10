import numpy as np
import pytest

from audiosanity.data import center_window
from audiosanity.transforms import align, codec_roundtrip, gain, resample_roundtrip

SR = 16_000
rng = np.random.default_rng(0)
WAVE = (0.2 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR) + 0.05 * rng.standard_normal(SR)).astype(
    np.float32
)


def test_gain():
    np.testing.assert_allclose(gain(WAVE, 20.0), 10 * WAVE, rtol=1e-5)


def test_resample_roundtrip_keeps_length_and_low_band():
    out = resample_roundtrip(WAVE, SR, 8000)
    assert out.shape == WAVE.shape
    assert np.corrcoef(out, WAVE)[0, 1] > 0.9  # the 440 Hz tone survives


@pytest.mark.parametrize("codec", ["mp3", "ogg"])
def test_codec_roundtrip_is_aligned(codec):
    out = codec_roundtrip(WAVE, SR, codec)
    assert out.shape == WAVE.shape
    assert np.corrcoef(out, WAVE)[0, 1] > 0.9


def test_align_recovers_delay():
    delayed = np.concatenate([np.zeros(300, np.float32), WAVE])
    np.testing.assert_allclose(align(delayed, WAVE), WAVE, atol=1e-6)


def test_center_window_offsets():
    x = np.arange(100)
    np.testing.assert_array_equal(center_window(x, 10), np.arange(45, 55))
    np.testing.assert_array_equal(center_window(x, 10, offset=5), np.arange(50, 60))
    with pytest.raises(ValueError):
        center_window(x, 10, offset=60)
