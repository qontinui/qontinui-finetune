"""Unit tests for ``prm.infer.PRMInferencer``.

Avoids any real model download by monkey-patching the HF loaders with tiny
fake classes. Skipped when torch is not importable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Test fixtures: fake backbone + fake processor
# ---------------------------------------------------------------------------


class _FakeConfig:
    projection_dim = 8


class _FakeBackboneOutput:
    """Mimics CLIPModel output with image_embeds + text_embeds."""

    def __init__(self, image_embeds: Any, text_embeds: Any) -> None:
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds
        self.last_hidden_state = None
        self.pooler_output = None


class _FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        # One trainable param so .parameters() doesn't return empty.
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, **inputs: Any) -> Any:
        batch = inputs["pixel_values"].shape[0]
        image = torch.ones(batch, 8) * 0.5
        text = torch.ones(batch, 8) * 0.5
        return _FakeBackboneOutput(image, text)

    def to(self, device: Any) -> _FakeBackbone:  # noqa: D401 — torch protocol
        return self


class _FakeProcessor:
    def __call__(
        self,
        text: list[str] | None = None,
        images: list[Any] | None = None,
        return_tensors: str = "pt",
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int = 64,
    ) -> dict[str, Any]:
        n = len(images) if images is not None else len(text or [])
        return {
            "pixel_values": torch.zeros(n, 3, 4, 4),
            "input_ids": torch.zeros(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }

    def save_pretrained(self, path: str) -> None:  # pragma: no cover — unused here
        Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixture: build a checkpoint on disk backed by the fake backbone
# ---------------------------------------------------------------------------


def _patch_transformers_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch HF loaders across every module that imports them.

    ``from transformers import AutoModel`` rebinds the name at the importing
    module's globals at import time, so monkeypatching the ``transformers``
    module alone is not enough — we also patch the entry on
    ``transformers.models.auto.modeling_auto``/``processing_auto`` where the
    real classes live, so any lazy import resolves to the fake.
    """
    import transformers

    fake_model_cls = type(
        "_AM", (), {"from_pretrained": staticmethod(lambda _n, **_kw: _FakeBackbone())}
    )
    fake_proc_cls = type(
        "_AP", (), {"from_pretrained": staticmethod(lambda _n, **_kw: _FakeProcessor())}
    )
    monkeypatch.setattr(transformers, "AutoModel", fake_model_cls, raising=False)
    monkeypatch.setattr(transformers, "AutoProcessor", fake_proc_cls, raising=False)


@pytest.fixture
def fake_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a PRM with the fake backbone and save its checkpoint to disk."""
    _patch_transformers_loaders(monkeypatch)

    from prm.model import GroundingPRM

    prm = GroundingPRM(backbone_name="fake/ckpt", freeze_backbone=True, hidden_dim=8)
    ckpt_path = tmp_path / "prm_checkpoint.pt"
    torch.save(prm.state_dict(), ckpt_path)
    return ckpt_path


@pytest.fixture(autouse=True)
def _patch_for_each_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure patches are also active when PRMInferencer loads the checkpoint
    inside the test body (not just the fixture setup)."""
    _patch_transformers_loaders(monkeypatch)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_inferencer_score_batch_returns_expected_shape(
    fake_checkpoint: Path,
) -> None:
    from prm.infer import PRMInferencer

    inf = PRMInferencer(fake_checkpoint, device="cpu")

    images = [object(), object(), object()]  # fake processor ignores contents
    instructions = ["click the target", "type 'x'", "click the target"]
    groundings = ["<point>0.5 0.5</point>"] * 3

    scores = inf.score_batch(images, instructions, groundings)
    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_inferencer_score_one_returns_float(fake_checkpoint: Path) -> None:
    from prm.infer import PRMInferencer

    inf = PRMInferencer(fake_checkpoint, device="cpu")
    score = inf.score_one(object(), "click", "<point>0.1 0.2</point>")
    assert isinstance(score, float)


def test_inferencer_empty_batch_returns_empty_list(fake_checkpoint: Path) -> None:
    from prm.infer import PRMInferencer

    inf = PRMInferencer(fake_checkpoint, device="cpu")
    assert inf.score_batch([], [], []) == []


def test_inferencer_missing_checkpoint_raises(tmp_path: Path) -> None:
    from prm.infer import PRMInferencer

    with pytest.raises(FileNotFoundError, match="PRM checkpoint not found"):
        PRMInferencer(tmp_path / "does_not_exist.pt", device="cpu")


def test_inferencer_length_mismatch_raises(fake_checkpoint: Path) -> None:
    from prm.infer import PRMInferencer

    inf = PRMInferencer(fake_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="equal length"):
        inf.score_batch([object()], ["a", "b"], ["<point>0 0</point>"])
