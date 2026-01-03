from __future__ import annotations

from pathlib import Path

import pytest


def test_spacy_detector_reuses_cached_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spacy = pytest.importorskip("spacy")

    model_dir = tmp_path / "spacy_blank_en"
    nlp = spacy.blank("en")
    nlp.to_disk(model_dir)

    monkeypatch.setenv("SPACY_EN_MODEL", str(model_dir))

    from deid_pipeline.pii.detectors.legacy.spacy_detector import SpacyDetector

    d1 = SpacyDetector(lang="en")
    d2 = SpacyDetector(lang="en")

    assert d1.nlp is d2.nlp
    assert d1.nlp.pipe_names.count("entity_ruler") == 1
    assert getattr(d1.nlp, "_edge_deid_rules_fingerprint", None)
