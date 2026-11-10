import numpy as np
import pytest
import torch

from audiosanity.data import Clip
from audiosanity.models import ASTClassifier, Wav2Vec2Classifier

SR = 16_000


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)  # noqa: NPY002


def make_clips(n: int = 3, seconds: float = 1.5) -> list[Clip]:
    """Noise bursts at different times - cheap synthetic stand-ins for real recordings."""
    rng = np.random.default_rng(0)
    t = int(seconds * SR)
    clips = []
    for i in range(n):
        wave = 0.01 * rng.standard_normal(t)
        start = (i + 1) * t // (n + 2)
        wave[start : start + SR // 5] += 0.3 * rng.standard_normal(SR // 5)
        clips.append(Clip(name=f"clip{i}.wav", label=f"class{i}", wave=wave.astype(np.float32)))
    return clips


@pytest.fixture
def clips():
    return make_clips()


@pytest.fixture(params=["ast", "wav2vec2"])
def tiny_model(request):
    model = ASTClassifier.tiny() if request.param == "ast" else Wav2Vec2Classifier.tiny()
    return model.requires_grad_(False).eval()
