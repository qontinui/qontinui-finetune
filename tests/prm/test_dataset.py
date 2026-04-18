"""Unit tests for ``prm.dataset`` — label policy & success_source filtering.

These tests only touch the label-derivation logic; they do not require torch
or transformers, so they run in any environment that has pytest + Python 3.12.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prm.dataset import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    LABEL_NEG,
    LABEL_POS,
    PRMDataset,
    record_to_example,
)


def _make_record(
    success_source: str | None,
    success: bool | None,
    *,
    image_path: str = "images/deadbeef.png",
    bbox: list[int] | None = None,
    action_type: str = "click",
    typed_text: str | None = None,
) -> dict:
    """Build a minimal GroundingRecord dict with the fields ``record_to_example``
    actually reads."""
    action: dict = {"type": action_type}
    if success is not None:
        action["success"] = success
    if success_source is not None:
        action["success_source"] = success_source
    if bbox is not None:
        action["target_bbox"] = bbox
    if typed_text is not None:
        action["typed_text"] = typed_text
    return {
        "image_hash": "deadbeef",
        "image_path": image_path,
        "viewport_width": 1000,
        "viewport_height": 500,
        "elements": [],
        "action": action,
        "source": "dynamic",
        "timestamp": "2026-04-17T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# record_to_example — one-off label/confidence assertions
# ---------------------------------------------------------------------------


class TestLabelPolicy:
    def test_wsm_success_true(self) -> None:
        ex = record_to_example(_make_record("wsm", True))
        assert ex is not None
        assert ex.label == LABEL_POS
        assert ex.confidence == CONFIDENCE_HIGH
        assert ex.success_source == "wsm"

    def test_wsm_success_false(self) -> None:
        ex = record_to_example(_make_record("wsm", False))
        assert ex is not None
        assert ex.label == LABEL_NEG
        assert ex.confidence == CONFIDENCE_HIGH

    def test_wsm_success_missing_defaults_positive(self) -> None:
        """A WSM-stamped record without a success flag is trusted as a pass."""
        ex = record_to_example(_make_record("wsm", None))
        assert ex is not None
        assert ex.label == LABEL_POS
        assert ex.confidence == CONFIDENCE_HIGH

    def test_pixel_diff_success_true(self) -> None:
        ex = record_to_example(_make_record("pixel_diff", True))
        assert ex is not None
        assert ex.label == LABEL_POS
        assert ex.confidence == CONFIDENCE_HIGH

    def test_pixel_diff_success_false(self) -> None:
        ex = record_to_example(_make_record("pixel_diff", False))
        assert ex is not None
        assert ex.label == LABEL_NEG
        assert ex.confidence == CONFIDENCE_HIGH

    def test_record_flag_is_downweighted(self) -> None:
        ex = record_to_example(_make_record("record_flag", True))
        assert ex is not None
        assert ex.label == LABEL_POS
        assert ex.confidence == CONFIDENCE_LOW

    def test_record_flag_failure_is_downweighted(self) -> None:
        ex = record_to_example(_make_record("record_flag", False))
        assert ex is not None
        assert ex.label == LABEL_NEG
        assert ex.confidence == CONFIDENCE_LOW

    def test_none_success_source_is_skipped(self) -> None:
        assert record_to_example(_make_record(None, True)) is None

    def test_unknown_success_source_is_skipped(self) -> None:
        assert record_to_example(_make_record("some_novel_label", True)) is None

    def test_static_record_without_action_is_skipped(self) -> None:
        rec = _make_record("wsm", True)
        rec.pop("action")
        assert record_to_example(rec) is None


# ---------------------------------------------------------------------------
# Grounding string + instruction formatting
# ---------------------------------------------------------------------------


class TestGroundingSerialisation:
    def test_bbox_normalises_to_point_string(self) -> None:
        ex = record_to_example(
            _make_record("wsm", True, bbox=[500, 250, 100, 50])
        )
        assert ex is not None
        # Center: (550, 275) / (1000, 500) → (0.55, 0.55)
        assert ex.predicted_grounding == "<point>0.55 0.55</point>"

    def test_missing_bbox_falls_back_to_center(self) -> None:
        ex = record_to_example(_make_record("wsm", True, bbox=None))
        assert ex is not None
        assert ex.predicted_grounding == "<point>0.50 0.50</point>"

    def test_click_instruction(self) -> None:
        ex = record_to_example(_make_record("wsm", True, action_type="click"))
        assert ex is not None
        assert ex.instruction == "click the target"

    def test_type_instruction_with_text(self) -> None:
        ex = record_to_example(
            _make_record("wsm", True, action_type="type", typed_text="hello")
        )
        assert ex is not None
        assert ex.instruction == "type 'hello'"

    def test_type_instruction_without_text(self) -> None:
        ex = record_to_example(
            _make_record("wsm", True, action_type="type")
        )
        assert ex is not None
        assert ex.instruction == "type the text"


# ---------------------------------------------------------------------------
# PRMDataset — from_records + disk-backed
# ---------------------------------------------------------------------------


class TestPRMDataset:
    def test_from_records_skips_none_and_static(self) -> None:
        recs = [
            _make_record("wsm", True),
            _make_record("pixel_diff", False),
            _make_record("record_flag", True),
            _make_record(None, True),            # skip
            _make_record("unknown_src", True),   # skip
        ]
        # Add a static record (no action at all)
        static = _make_record("wsm", True)
        static.pop("action")
        recs.append(static)

        ds = PRMDataset.from_records(recs)
        assert len(ds) == 3
        sources = [ex.success_source for ex in ds.examples]
        assert sources == ["wsm", "pixel_diff", "record_flag"]

    def test_from_records_preserves_labels_and_confidence(self) -> None:
        ds = PRMDataset.from_records(
            [
                _make_record("wsm", True),
                _make_record("pixel_diff", False),
                _make_record("record_flag", True),
            ]
        )
        pairs = [(ex.success_source, ex.label, ex.confidence) for ex in ds.examples]
        assert pairs == [
            ("wsm", LABEL_POS, CONFIDENCE_HIGH),
            ("pixel_diff", LABEL_NEG, CONFIDENCE_HIGH),
            ("record_flag", LABEL_POS, CONFIDENCE_LOW),
        ]

    def test_reads_jsonl_from_disk(self, tmp_path: Path) -> None:
        recs = [
            _make_record("wsm", True),
            _make_record("pixel_diff", False),
            _make_record(None, True),  # should be skipped
        ]
        jsonl = tmp_path / "grounding.jsonl"
        with open(jsonl, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        ds = PRMDataset(jsonl)
        assert len(ds) == 2

    def test_empty_file_produces_empty_dataset(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "grounding.jsonl"
        jsonl.write_text("", encoding="utf-8")
        ds = PRMDataset(jsonl)
        assert len(ds) == 0

    def test_missing_image_path_is_skipped(self) -> None:
        rec = _make_record("wsm", True)
        rec["image_path"] = ""
        assert record_to_example(rec) is None


# ---------------------------------------------------------------------------
# Indexing / protocol
# ---------------------------------------------------------------------------


def test_dataset_supports_indexing() -> None:
    ds = PRMDataset.from_records([_make_record("wsm", True)])
    ex = ds[0]
    assert ex.success_source == "wsm"
    with pytest.raises(IndexError):
        _ = ds[1]
