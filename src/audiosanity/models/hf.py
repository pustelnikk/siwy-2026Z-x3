"""Hugging Face audio classifiers: Audio Spectrogram Transformer and wav2vec 2.0."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import (
    ASTConfig,
    ASTFeatureExtractor,
    ASTForAudioClassification,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

from audiosanity.models.base import AudioClassifier

EPS = 1e-10


def _frame_sum(x: torch.Tensor, hop: int) -> torch.Tensor:
    """Sum ``(B, N)`` samples into non-overlapping frames of ``hop`` samples -> ``(B, N // hop)``."""
    n_frames = x.shape[-1] // hop
    return x[..., : n_frames * hop].reshape(*x.shape[:-1], n_frames, hop).sum(-1)


class ASTClassifier(AudioClassifier):
    """Audio Spectrogram Transformer; attributions live on its 128-bin log-mel filterbank.

    The filterbank uses 25 ms windows with a 10 ms hop (Kaldi, ``snip_edges=True``), so frame ``k``
    starts at sample ``160 k``. The model always sees ``max_length`` frames (zero-padded); the
    padding is cut away in ``to_grid``.
    """

    window_length = 400

    def __init__(self, model: ASTForAudioClassification, extractor: ASTFeatureExtractor | None = None):
        super().__init__()
        self.model = model.eval()
        self.extractor = extractor or ASTFeatureExtractor(
            num_mel_bins=model.config.num_mel_bins, max_length=model.config.max_length
        )
        self.labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        self.multi_label = model.config.num_labels > 50  # AudioSet taggers

    @classmethod
    def from_pretrained(cls, checkpoint: str) -> ASTClassifier:
        return cls(
            ASTForAudioClassification.from_pretrained(checkpoint),
            ASTFeatureExtractor.from_pretrained(checkpoint),
        )

    @classmethod
    def tiny(cls, num_labels: int = 5, max_length: int = 128) -> ASTClassifier:
        """Small randomly initialised model for tests (no download)."""
        config = ASTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_length=max_length,
            num_labels=num_labels,
        )
        return cls(ASTForAudioClassification(config))

    def n_frames(self, n_samples: int) -> int:
        return 1 + (n_samples - self.window_length) // self.hop_length

    def features(self, waves: np.ndarray) -> torch.Tensor:
        out = self.extractor(list(waves), sampling_rate=self.sample_rate, return_tensors="pt")
        return out["input_values"].to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=inputs).logits

    def to_grid(self, attributions: torch.Tensor, n_samples: int) -> np.ndarray:
        return attributions[:, : self.n_frames(n_samples), :].detach().float().cpu().numpy()

    def reference(self, waves: np.ndarray) -> np.ndarray:
        # The extractor output is the (normalised) log-mel spectrogram itself.
        return self.to_grid(self.features(waves), waves.shape[-1])

    def layer_groups(self) -> list[tuple[str, list[nn.Module]]]:
        ast = self.model.audio_spectrogram_transformer
        groups: list[tuple[str, list[nn.Module]]] = [("head", [self.model.classifier, ast.layernorm])]
        # transformers>=5 keeps the blocks in `ast.layers`, older versions in `ast.encoder.layer`
        blocks = ast.layers if hasattr(ast, "layers") else ast.encoder.layer
        for i in reversed(range(len(blocks))):
            groups.append((f"block{i}", [blocks[i]]))
        groups.append(("embeddings", [ast.embeddings]))
        return groups


class Wav2Vec2Classifier(AudioClassifier):
    """wav2vec 2.0 with a sequence-classification head; attributions live on the raw waveform."""

    def __init__(
        self, model: Wav2Vec2ForSequenceClassification, extractor: Wav2Vec2FeatureExtractor | None = None
    ):
        super().__init__()
        self.model = model.eval()
        self.extractor = extractor or Wav2Vec2FeatureExtractor(do_normalize=True)
        self.labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @classmethod
    def from_pretrained(cls, checkpoint: str) -> Wav2Vec2Classifier:
        return cls(
            Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint),
            Wav2Vec2FeatureExtractor.from_pretrained(checkpoint),
        )

    @classmethod
    def tiny(cls, num_labels: int = 5) -> Wav2Vec2Classifier:
        """Small randomly initialised model for tests (no download)."""
        config = Wav2Vec2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            conv_dim=(32, 32, 32),
            conv_kernel=(10, 3, 3),
            conv_stride=(5, 2, 2),
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=4,
            classifier_proj_size=16,
            num_labels=num_labels,
        )
        return cls(Wav2Vec2ForSequenceClassification(config))

    def features(self, waves: np.ndarray) -> torch.Tensor:
        out = self.extractor(list(waves), sampling_rate=self.sample_rate, return_tensors="pt")
        return out["input_values"].to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=inputs).logits

    def to_grid(self, attributions: torch.Tensor, n_samples: int) -> np.ndarray:
        frames = _frame_sum(attributions.detach().float(), self.hop_length)
        return frames.unsqueeze(-1).cpu().numpy()

    def reference(self, waves: np.ndarray) -> np.ndarray:
        energy = _frame_sum(torch.as_tensor(np.asarray(waves, dtype=np.float32)) ** 2, self.hop_length)
        return (10 * torch.log10(energy / self.hop_length + EPS)).unsqueeze(-1).numpy()

    def layer_groups(self) -> list[tuple[str, list[nn.Module]]]:
        w2v = self.model.wav2vec2
        groups: list[tuple[str, list[nn.Module]]] = [("head", [self.model.classifier, self.model.projector])]
        for i in reversed(range(len(w2v.encoder.layers))):
            groups.append((f"block{i}", [w2v.encoder.layers[i]]))
        groups.append(
            ("positional", [w2v.encoder.pos_conv_embed, w2v.encoder.layer_norm, w2v.feature_projection])
        )
        groups.append(("conv_encoder", [w2v.feature_extractor]))
        return groups
