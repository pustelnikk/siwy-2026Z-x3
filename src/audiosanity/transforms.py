"""Audio transformations the model should (mostly) ignore. All keep length and sample rate."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from scipy.signal import correlate

from audiosanity.data import resample

CODEC_FORMATS = {"mp3": ("MP3", None), "ogg": ("OGG", "VORBIS"), "flac": ("FLAC", None)}


def gain(wave: np.ndarray, db: float) -> np.ndarray:
    return (wave * 10 ** (db / 20)).astype(np.float32)


def resample_roundtrip(wave: np.ndarray, sample_rate: int, via_rate: int) -> np.ndarray:
    """Down/up-sample through ``via_rate`` and back (removes content above ``via_rate / 2``)."""
    out = resample(resample(wave, sample_rate, via_rate), via_rate, sample_rate)
    return _fit_length(out, len(wave))


def codec_roundtrip(wave: np.ndarray, sample_rate: int, codec: str) -> np.ndarray:
    """Encode and decode with a lossy codec, then re-align to undo encoder delay."""
    if codec not in CODEC_FORMATS:
        raise ValueError(f"unknown codec {codec!r}; available: {sorted(CODEC_FORMATS)}")
    fmt, subtype = CODEC_FORMATS[codec]
    buf = io.BytesIO()
    sf.write(buf, np.clip(wave, -1, 1), sample_rate, format=fmt, subtype=subtype)
    buf.seek(0)
    decoded, _ = sf.read(buf, dtype="float32")
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)
    return align(decoded, wave)


def align(x: np.ndarray, ref: np.ndarray, max_lag: int = 4096) -> np.ndarray:
    """Shift ``x`` so that it best matches ``ref`` (cross-correlation) and cut it to ``len(ref)``."""
    corr = correlate(x, ref, mode="full", method="fft")
    lags = np.arange(-len(ref) + 1, len(x))
    keep = np.abs(lags) <= max_lag
    lag = int(lags[keep][np.argmax(corr[keep])])
    x = x[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, dtype=x.dtype), x])
    return _fit_length(x, len(ref))


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    x = x[:n]
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return x.astype(np.float32)
