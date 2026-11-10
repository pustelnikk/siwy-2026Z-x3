import numpy as np
import pandas as pd
import pytest

from audiosanity import pipeline
from audiosanity.checks.shift import overlap
from audiosanity.config import load_config
from audiosanity.models import ASTClassifier, Wav2Vec2Classifier
from tests.conftest import make_clips


def test_overlap_alignment():
    maps = np.arange(10)[None, :, None].astype(float)
    ref, moved = overlap(maps, maps + 3, 3)  # shifted window starts 3 frames later
    np.testing.assert_array_equal(ref, moved)
    ref, moved = overlap(maps, maps - 2, -2)
    np.testing.assert_array_equal(ref, moved)


@pytest.mark.parametrize("arch", ["ast", "wav2vec2"])
def test_end_to_end(tmp_path, monkeypatch, arch):
    tiny = {"ast": ASTClassifier.tiny, "wav2vec2": Wav2Vec2Classifier.tiny}[arch]
    monkeypatch.setattr(pipeline, "load_model", lambda *a, **k: tiny().requires_grad_(False))
    monkeypatch.setattr(pipeline, "load_clips", lambda *a, **k: make_clips(3))
    cfg = load_config(
        "configs/smoke.yaml",
        [
            f"output_dir={tmp_path.as_posix()}",
            f"model.arch={arch}",
            "device=cpu",
            "data.clip_seconds=1.0",
            "n_bootstrap=50",
        ],
    )
    run_dir = pipeline.run_experiment(cfg)

    summary = pd.read_csv(run_dir / "summary.csv")
    assert set(summary["check"]) == set(cfg.checks.run)
    for check in cfg.checks.run:
        assert (run_dir / "figures" / f"{check}.png").exists()
        assert (run_dir / "figures" / f"{check}_all.png").exists()
        conditions = summary.loc[summary["check"] == check, "condition"]
        assert conditions.str.startswith("control").any(), f"{check} has no random control"
    results = pd.read_csv(run_dir / "results.csv")
    assert "pred_change" in results.columns
    assert (run_dir / "maps.npz").exists()
    assert (run_dir / "config.yaml").exists()
