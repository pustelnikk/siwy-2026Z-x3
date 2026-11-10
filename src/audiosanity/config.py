"""Experiment configuration.

The schema lives here; concrete values live in YAML files under ``configs/``.
Any value can be overridden from the command line with dotted keys, e.g.
``data.n_clips=10 attribution.methods=[saliency]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class DataConfig:
    dataset: str = "esc50"
    root: str = "data/raw/ESC-50-master"
    n_clips: int = 50
    # Length of the analysed window, cut from the centre of each clip (padding confounds shifts).
    clip_seconds: float = 4.0
    # Windows whose RMS is below this value are skipped (ESC-50 contains near-silent clips).
    min_rms: float = 1e-3


@dataclass
class ModelConfig:
    arch: str = "ast"  # key in audiosanity.models.REGISTRY
    checkpoint: str = "MIT/ast-finetuned-audioset-10-10-0.4593"


@dataclass
class AttributionConfig:
    methods: list[str] = field(
        default_factory=lambda: ["saliency", "input_x_gradient", "integrated_gradients", "smoothgrad"]
    )
    # Clips explained together.
    batch_size: int = 4
    # Max model inputs in one forward+backward pass inside IG / SmoothGrad (they expand every clip into
    # many inputs). AST needs ~0.45 GB of VRAM per input, so 8 -> ~4 GB. If the GPU memory fills up,
    # Windows silently spills into system RAM and everything becomes several times slower.
    internal_batch_size: int = 8
    ig_steps: int = 16
    smoothgrad_samples: int = 16
    # Noise std as a fraction of the input range (Smilkov et al. use 0.1-0.2).
    smoothgrad_noise: float = 0.15


@dataclass
class RandomizationConfig:
    # How many layer groups (counted from the output) to randomize; null = all of them.
    max_groups: int | None = None


@dataclass
class ShiftConfig:
    # Shifts in grid frames (one frame = model hop, 10 ms for 16 kHz models).
    shifts: list[int] = field(default_factory=lambda: [-25, -10, -3, 3, 10, 25])


@dataclass
class InvarianceConfig:
    gains_db: list[float] = field(default_factory=lambda: [-12.0, -6.0, 6.0])
    resample_rates: list[int] = field(default_factory=lambda: [8000, 22050])
    codecs: list[str] = field(default_factory=lambda: ["mp3", "ogg"])


@dataclass
class ChecksConfig:
    run: list[str] = field(
        default_factory=lambda: ["randomization", "shift", "invariance", "loudness", "agreement"]
    )
    normalizations: list[str] = field(default_factory=lambda: ["raw", "abs", "abs_max"])
    measures: list[str] = field(default_factory=lambda: ["spearman", "pearson", "ssim", "topk"])
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    shift: ShiftConfig = field(default_factory=ShiftConfig)
    invariance: InvarianceConfig = field(default_factory=InvarianceConfig)


@dataclass
class PlotConfig:
    # Combination shown in the main <check>.png figures; <check>_all.png shows every combination.
    normalization: str = "abs"
    measure: str = "spearman"


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda
    output_dir: str = "results"
    n_bootstrap: int = 1000
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    checks: ChecksConfig = field(default_factory=ChecksConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    """Merge defaults <- YAML file <- command-line overrides and validate against the schema."""
    cfg = OmegaConf.structured(ExperimentConfig)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
