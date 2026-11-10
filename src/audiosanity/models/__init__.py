"""Model registry. To add an architecture, implement ``AudioClassifier`` and register it here."""

from __future__ import annotations

from audiosanity.models.base import AudioClassifier
from audiosanity.models.hf import ASTClassifier, Wav2Vec2Classifier

REGISTRY: dict[str, type[AudioClassifier]] = {
    "ast": ASTClassifier,
    "wav2vec2": Wav2Vec2Classifier,
}


def load_model(arch: str, checkpoint: str, device: str = "cpu") -> AudioClassifier:
    if arch not in REGISTRY:
        raise ValueError(f"unknown architecture {arch!r}; available: {sorted(REGISTRY)}")
    model = REGISTRY[arch].from_pretrained(checkpoint)
    # Attributions only need gradients w.r.t. the input.
    model.requires_grad_(False)
    return model.to(device).eval()


__all__ = ["REGISTRY", "AudioClassifier", "ASTClassifier", "Wav2Vec2Classifier", "load_model"]
