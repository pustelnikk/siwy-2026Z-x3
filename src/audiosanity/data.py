"""Datasets: download, clip selection and windowing.

Raw data is treated as immutable: it is only ever read from ``data/raw``.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


@dataclass
class Clip:
    """A full-length mono recording at the model sample rate."""

    name: str
    label: str
    wave: np.ndarray  # float32, shape (n_samples,)


def resample(wave: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wave.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(wave, sr_out // g, sr_in // g).astype(np.float32)


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(wave.mean(axis=1), sr, sample_rate)


def download_esc50(raw_dir: str | Path = "data/raw") -> Path:
    """Download and unpack ESC-50 into ``raw_dir/ESC-50-master`` (skipped if already present)."""
    raw_dir = Path(raw_dir)
    target = raw_dir / "ESC-50-master"
    if (target / "meta" / "esc50.csv").exists():
        log.info("ESC-50 already present in %s", target)
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "esc50.zip"
    log.info("Downloading %s (~600 MB)", ESC50_URL)
    with urllib.request.urlopen(ESC50_URL) as response, open(archive, "wb") as f:
        shutil.copyfileobj(response, f)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()
    return target


def center_window(wave: np.ndarray, n_samples: int, offset: int = 0) -> np.ndarray:
    """Cut ``n_samples`` from the centre of ``wave``, moved right by ``offset`` samples."""
    start = (len(wave) - n_samples) // 2 + offset
    if start < 0 or start + n_samples > len(wave):
        raise ValueError(f"window (offset {offset}) does not fit in a clip of {len(wave)} samples")
    return wave[start : start + n_samples]


def load_esc50_clips(
    root: str | Path,
    n_clips: int,
    sample_rate: int,
    window_samples: int,
    min_rms: float = 1e-3,
    seed: int = 0,
) -> list[Clip]:
    """Pick ``n_clips`` random ESC-50 clips whose centre window is not silent."""
    root = Path(root)
    meta_path = root / "meta" / "esc50.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found - run `just data` (audiosanity download) first")
    meta = pd.read_csv(meta_path).sample(frac=1.0, random_state=seed)
    clips: list[Clip] = []
    for row in meta.itertuples():
        wave = load_audio(root / "audio" / row.filename, sample_rate)
        window = center_window(wave, window_samples)
        if np.sqrt(np.mean(window**2)) < min_rms:
            continue
        clips.append(Clip(name=row.filename, label=row.category, wave=wave))
        if len(clips) == n_clips:
            break
    log.info("Selected %d ESC-50 clips", len(clips))
    return clips


def load_clips(cfg, sample_rate: int, window_samples: int, seed: int) -> list[Clip]:
    if cfg.dataset == "esc50":
        return load_esc50_clips(cfg.root, cfg.n_clips, sample_rate, window_samples, cfg.min_rms, seed)
    raise ValueError(f"unknown dataset {cfg.dataset!r}")
