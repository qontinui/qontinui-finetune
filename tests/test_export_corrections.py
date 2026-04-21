"""Tests for scripts/export_corrections_to_vlm_sft.py.

Covers the four cases called out in the VGA plan milestone (c) phase 1:

1. ``--include-private false`` emits only public entries.
2. Stratified split puts each domain in at least one split.
3. Missing image file → entry skipped with a warning.
4. Sample shape matches the expected ``messages`` structure the trainer
   (``scripts/finetune_grounding_lora.py``) consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from scripts import export_corrections_to_vlm_sft as exporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_png(path: Path, width: int = 200, height: int = 100) -> None:
    """Write a minimal solid-colour PNG to *path* for dimension reads."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - Pillow is a hard dep of the trainer
        pytest.skip("Pillow not available")
    img = Image.new("RGB", (width, height), color=(127, 127, 127))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _make_entry(
    *,
    image_path: Path,
    prompt: str,
    target_process: str,
    private: bool,
    bbox: dict[str, int] | None = None,
    test_reserved: bool = False,
    source: str = "builder",
) -> dict[str, Any]:
    """Build a JSONL entry matching ``CorrectionLogger.append`` output."""
    return {
        "ts": "2026-04-21T12:00:00+00:00",
        "state_machine_id": str(uuid4()),
        "image_sha": "deadbeef",
        "image_path": str(image_path),
        "prompt": prompt,
        "corrected_bbox": bbox or {"x": 10, "y": 20, "w": 40, "h": 30},
        "source": source,
        "target_process": target_process,
        "private": private,
        "test_reserved": test_reserved,
    }


@pytest.fixture
def corrections_dir(tmp_path: Path) -> Path:
    """Create a temp corrections dir with 3 entries across 2 target processes.

    - Entry A: notepad++.exe, PUBLIC
    - Entry B: notepad++.exe, PUBLIC
    - Entry C: obs64.exe, PRIVATE
    """
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    images_dir = corrections / "images"
    images_dir.mkdir()

    img_a = images_dir / "a.png"
    img_b = images_dir / "b.png"
    img_c = images_dir / "c.png"
    _make_png(img_a)
    _make_png(img_b)
    _make_png(img_c)

    entries = [
        _make_entry(
            image_path=img_a, prompt="Save button",
            target_process="notepad++.exe", private=False,
        ),
        _make_entry(
            image_path=img_b, prompt="Open menu",
            target_process="notepad++.exe", private=False,
        ),
        _make_entry(
            image_path=img_c, prompt="Scene panel",
            target_process="obs64.exe", private=True,
        ),
    ]

    jsonl = corrections / "corrections.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")))
            f.write("\n")

    return corrections


# ---------------------------------------------------------------------------
# Test 1: include-private=false filters out private entries
# ---------------------------------------------------------------------------


def test_include_private_false_excludes_private_entries(
    corrections_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    summary = exporter.export_corrections(
        corrections_jsonl=corrections_dir / "corrections.jsonl",
        output_dir=output_dir,
        include_private=False,
        split=(0.8, 0.1, 0.1),
        seed=42,
    )

    assert summary["total_samples"] == 2
    # The private obs64.exe entry must not leak into any split.
    assert "obs64.exe" not in summary["per_target_process"]
    assert summary["per_target_process"]["notepad++.exe"] == 2
    assert summary["excluded_private"] == 1


def test_include_private_true_keeps_private_entries(
    corrections_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out_private"
    summary = exporter.export_corrections(
        corrections_jsonl=corrections_dir / "corrections.jsonl",
        output_dir=output_dir,
        include_private=True,
        split=(0.8, 0.1, 0.1),
        seed=42,
    )

    # With include_private=True, the obs64 entry is kept.
    assert summary["total_samples"] == 3
    assert summary["per_target_process"]["obs64.exe"] == 1


# ---------------------------------------------------------------------------
# Test 2: Stratified split puts each domain in at least one split
# ---------------------------------------------------------------------------


def test_stratified_split_covers_every_domain(tmp_path: Path) -> None:
    """Bigger fixture — 6 public samples across 2 domains — for split test."""
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    images_dir = corrections / "images"
    images_dir.mkdir()

    entries: list[dict[str, Any]] = []
    for i in range(3):
        img = images_dir / f"np_{i}.png"
        _make_png(img)
        entries.append(
            _make_entry(
                image_path=img, prompt=f"Element {i}",
                target_process="notepad++.exe", private=False,
            )
        )
    for i in range(3):
        img = images_dir / f"obs_{i}.png"
        _make_png(img)
        entries.append(
            _make_entry(
                image_path=img, prompt=f"Element {i}",
                target_process="obs64.exe", private=False,
            )
        )

    jsonl = corrections / "corrections.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")))
            f.write("\n")

    output_dir = tmp_path / "out"
    summary = exporter.export_corrections(
        corrections_jsonl=jsonl,
        output_dir=output_dir,
        include_private=False,
        split=(0.5, 0.25, 0.25),
        seed=42,
    )

    # Every domain should show up with non-zero counts in at least one
    # split, and their totals must match what was ingested.
    per_domain = summary["per_domain_per_split"]
    assert set(per_domain.keys()) == {"notepad++.exe", "obs64.exe"}
    for domain, counts in per_domain.items():
        total = counts["train"] + counts["val"] + counts["test"]
        assert total == 3, f"{domain} split counts don't sum to 3: {counts}"
        # Must appear in at least one split.
        assert total > 0


def test_test_reserved_entries_always_go_to_test(tmp_path: Path) -> None:
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    images_dir = corrections / "images"
    images_dir.mkdir()

    entries: list[dict[str, Any]] = []
    # 4 regular notepad++ samples.
    for i in range(4):
        img = images_dir / f"np_{i}.png"
        _make_png(img)
        entries.append(
            _make_entry(
                image_path=img, prompt=f"E{i}",
                target_process="notepad++.exe", private=False,
            )
        )
    # 1 reserved-for-test entry.
    reserved_img = images_dir / "reserved.png"
    _make_png(reserved_img)
    entries.append(
        _make_entry(
            image_path=reserved_img, prompt="Reserved holdout",
            target_process="notepad++.exe", private=False,
            test_reserved=True,
        )
    )

    jsonl = corrections / "corrections.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")))
            f.write("\n")

    output_dir = tmp_path / "out_reserved"
    summary = exporter.export_corrections(
        corrections_jsonl=jsonl,
        output_dir=output_dir,
        # Use 100% train split so the only way a sample lands in test is
        # via the reserved flag.
        include_private=False,
        split=(1.0, 0.0, 0.0),
        seed=42,
    )

    assert summary["per_split"]["test"] >= 1, (
        "reserved entry did not route to test split"
    )


# ---------------------------------------------------------------------------
# Test 3: Missing image file → entry skipped with warning
# ---------------------------------------------------------------------------


def test_missing_image_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    images_dir = corrections / "images"
    images_dir.mkdir()

    img_present = images_dir / "present.png"
    _make_png(img_present)
    img_missing = images_dir / "missing.png"  # note: never created

    entries = [
        _make_entry(
            image_path=img_present, prompt="Exists",
            target_process="notepad++.exe", private=False,
        ),
        _make_entry(
            image_path=img_missing, prompt="Ghost",
            target_process="notepad++.exe", private=False,
        ),
    ]

    jsonl = corrections / "corrections.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")))
            f.write("\n")

    output_dir = tmp_path / "out_missing"
    with caplog.at_level("WARNING", logger=exporter.__name__):
        summary = exporter.export_corrections(
            corrections_jsonl=jsonl,
            output_dir=output_dir,
            include_private=False,
            split=(0.8, 0.1, 0.1),
            seed=42,
        )

    assert summary["total_samples"] == 1
    assert summary["excluded_missing_image"] == 1
    assert any(
        "Image not found" in record.getMessage() for record in caplog.records
    ), "Expected warning about missing image"


# ---------------------------------------------------------------------------
# Test 4: Sample shape matches the trainer's expected `messages` structure
# ---------------------------------------------------------------------------


def test_sample_shape_matches_trainer_expectation(
    corrections_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out_shape"
    exporter.export_corrections(
        corrections_jsonl=corrections_dir / "corrections.jsonl",
        output_dir=output_dir,
        include_private=False,
        # 100% train so every sample lands in vlm_train.jsonl.
        split=(1.0, 0.0, 0.0),
        seed=42,
    )

    train_path = output_dir / "vlm_train.jsonl"
    assert train_path.exists()

    lines = train_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    for line in lines:
        sample = json.loads(line)
        assert "messages" in sample
        messages = sample["messages"]
        assert len(messages) == 2

        user_msg, asst_msg = messages
        assert user_msg["role"] == "user"
        assert asst_msg["role"] == "assistant"

        # User content must be a two-part list: image then text.
        user_content = user_msg["content"]
        assert isinstance(user_content, list)
        assert len(user_content) == 2

        img_part = next(p for p in user_content if p.get("type") == "image")
        text_part = next(p for p in user_content if p.get("type") == "text")

        assert img_part["image"].startswith("file:///")
        assert "Locate the following element" in text_part["text"]
        assert "<point>" in asst_msg["content"]
        assert "</point>" in asst_msg["content"]

        # Parse the assistant point: must be two ints in [0, 1000].
        import re
        m = re.search(r"<point>\s*(\d+)\s+(\d+)\s*</point>", asst_msg["content"])
        assert m, f"Could not parse point from {asst_msg['content']!r}"
        nx, ny = int(m.group(1)), int(m.group(2))
        assert 0 <= nx <= 1000
        assert 0 <= ny <= 1000


# ---------------------------------------------------------------------------
# Auxiliary: summary.json is emitted and well-formed
# ---------------------------------------------------------------------------


def test_summary_json_is_emitted(corrections_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out_summary"
    exporter.export_corrections(
        corrections_jsonl=corrections_dir / "corrections.jsonl",
        output_dir=output_dir,
        include_private=False,
        split=(0.8, 0.1, 0.1),
        seed=42,
    )
    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "total_samples" in data
    assert "per_target_process" in data
    assert "per_split" in data
    assert "excluded_private" in data
    assert "excluded_missing_image" in data
