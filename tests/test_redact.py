"""Tests for qontinui_finetune.redact and the exporter's centralized mode.

Covers the four cases called out in VGA follow-up plan §Phase 2 Item 3:

1. ``preserve_ui_labels=True`` preserves short UI chrome strings while
   masking longer body text.
2. ``preserve_ui_labels=False`` masks every detected text region.
3. :meth:`RedactionPolicy.policy_hash` is stable across runs.
4. The exporter refuses a centralized export when a private entry is
   reachable — even via the ``include_private=True`` flag.

OCR-dependent assertions are gated behind a backend probe. If no OCR
backend is installed, those tests are skipped but the policy-hash,
argparse, and manifest tests still run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from qontinui_finetune.redact import RedactionPolicy, redact_image

# ---------------------------------------------------------------------------
# OCR backend probe
# ---------------------------------------------------------------------------


def _ocr_backend_available() -> bool:
    """Check whether at least one OCR backend can actually run.

    Uses the module's private selector — if it raises RuntimeError, no
    backend works. We catch broadly because the selector probes binary
    availability for pytesseract and can fail with several error shapes.
    """
    try:
        from qontinui_finetune.redact import _select_backend
    except Exception:
        return False
    try:
        _select_backend()
        return True
    except Exception:
        return False


OCR_AVAILABLE = _ocr_backend_available()
ocr_required = pytest.mark.skipif(
    not OCR_AVAILABLE, reason="No OCR backend available (easyocr/tesseract)"
)


# ---------------------------------------------------------------------------
# Synthetic image fixture
# ---------------------------------------------------------------------------


def _pick_font(size: int):
    """Find a readable TrueType font across Windows / Linux / macOS.

    OCR accuracy on bitmap fonts is miserable, so we prefer any TTF on
    disk. Falls back to PIL's default bitmap font only if nothing else
    resolves — in that case OCR-dependent tests will likely be skipped
    by the backend probe anyway.
    """
    from PIL import ImageFont  # type: ignore[import-not-found]

    candidates = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        # macOS
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


@pytest.fixture
def synthetic_image(tmp_path: Path) -> tuple[Path, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Render a 512x256 image with a short "Save" label and a long body.

    Returns ``(image_path, save_bbox, body_bbox)`` where the bboxes are
    approximate pixel regions so the test can sample the redacted output.
    """
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]

    width, height = 512, 256
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    short_font = _pick_font(size=28)
    long_font = _pick_font(size=20)

    # Short UI label in upper-left.
    save_xy = (20, 20)
    draw.text(save_xy, "Save", fill=(0, 0, 0), font=short_font)
    # Approximate region occupied by "Save" at size 28.
    save_bbox = (save_xy[0], save_xy[1], save_xy[0] + 80, save_xy[1] + 40)

    # Long body text lower down.
    body_xy = (20, 120)
    draw.text(
        body_xy,
        "The quick brown fox jumps over the lazy dog",
        fill=(0, 0, 0),
        font=long_font,
    )
    body_bbox = (body_xy[0], body_xy[1], body_xy[0] + 470, body_xy[1] + 32)

    image_path = tmp_path / "synthetic.png"
    img.save(image_path)
    return image_path, save_bbox, body_bbox


def _region_is_blacked_out(
    image_path: Path, bbox: tuple[int, int, int, int], threshold: float = 0.5
) -> bool:
    """Sample *bbox* and return True when at least *threshold* of pixels are near-black.

    We don't require 100% because redaction uses OCR-reported bounds
    which may not cover every rendered anti-aliased pixel. A majority
    check is the right invariant: "this text is not readable anymore".
    """
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(rgb.width, x2)
        y2 = min(rgb.height, y2)
        if x2 <= x1 or y2 <= y1:
            return False
        region = rgb.crop((x1, y1, x2, y2))
    pixels = list(region.getdata())
    if not pixels:
        return False
    black = sum(1 for r, g, b in pixels if r < 32 and g < 32 and b < 32)
    return (black / len(pixels)) >= threshold


# ---------------------------------------------------------------------------
# Test 1: preserve_ui_labels keeps short labels, masks long text
# ---------------------------------------------------------------------------


@ocr_required
def test_preserve_ui_labels_true_keeps_short_masks_long(
    synthetic_image: tuple[Path, tuple[int, int, int, int], tuple[int, int, int, int]],
    tmp_path: Path,
) -> None:
    image_path, save_bbox, body_bbox = synthetic_image
    out_path = tmp_path / "redacted.png"

    redact_image(
        image_path=image_path,
        policy=RedactionPolicy(preserve_ui_labels=True, min_confidence=0.3),
        out_path=out_path,
    )

    assert out_path.exists()
    # "Save" region should NOT be mostly black — label preserved.
    assert not _region_is_blacked_out(out_path, save_bbox, threshold=0.5), (
        "Short 'Save' label should be preserved with preserve_ui_labels=True"
    )
    # Body region should be mostly black — long text masked.
    assert _region_is_blacked_out(out_path, body_bbox, threshold=0.3), (
        "Long body text should be blacked out with preserve_ui_labels=True"
    )


# ---------------------------------------------------------------------------
# Test 2: preserve_ui_labels=False blacks out everything
# ---------------------------------------------------------------------------


@ocr_required
def test_preserve_ui_labels_false_masks_all(
    synthetic_image: tuple[Path, tuple[int, int, int, int], tuple[int, int, int, int]],
    tmp_path: Path,
) -> None:
    image_path, save_bbox, body_bbox = synthetic_image
    out_path = tmp_path / "redacted_all.png"

    redact_image(
        image_path=image_path,
        policy=RedactionPolicy(preserve_ui_labels=False, min_confidence=0.3),
        out_path=out_path,
    )

    assert out_path.exists()
    assert _region_is_blacked_out(out_path, save_bbox, threshold=0.3), (
        "Save label should be masked when preserve_ui_labels=False"
    )
    assert _region_is_blacked_out(out_path, body_bbox, threshold=0.3), (
        "Body text should be masked when preserve_ui_labels=False"
    )


# ---------------------------------------------------------------------------
# Test 3: policy_hash is stable and sensitive to config changes
# ---------------------------------------------------------------------------


def test_policy_hash_stable_across_runs() -> None:
    p1 = RedactionPolicy()
    p2 = RedactionPolicy()
    assert p1.policy_hash() == p2.policy_hash()

    p3 = RedactionPolicy(preserve_ui_labels=True, min_confidence=0.5)
    p4 = RedactionPolicy(preserve_ui_labels=True, min_confidence=0.5)
    assert p3.policy_hash() == p4.policy_hash()

    # Different config → different hash.
    p5 = RedactionPolicy(preserve_ui_labels=False)
    assert p5.policy_hash() != p1.policy_hash()

    p6 = RedactionPolicy(min_confidence=0.7)
    assert p6.policy_hash() != p1.policy_hash()

    p7 = RedactionPolicy(max_ui_label_chars=50)
    assert p7.policy_hash() != p1.policy_hash()

    p8 = RedactionPolicy(fill=(255, 0, 0))
    assert p8.policy_hash() != p1.policy_hash()

    # Hash is hex-64 (SHA-256).
    h = p1.policy_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Test 4: Exporter refuses centralized export with private entries
# ---------------------------------------------------------------------------


def _make_png(path: Path, width: int = 64, height: int = 48) -> None:
    from PIL import Image  # type: ignore[import-not-found]

    Image.new("RGB", (width, height), color=(200, 200, 200)).save(path)


def _write_entry(
    jsonl: Path,
    *,
    image_path: Path,
    private: bool,
    target_process: str = "notepad++.exe",
    prompt: str = "Save",
) -> None:
    entry = {
        "ts": "2026-04-21T12:00:00+00:00",
        "state_machine_id": str(uuid4()),
        "image_sha": "deadbeefcafebabe",
        "image_path": str(image_path),
        "prompt": prompt,
        "corrected_bbox": {"x": 5, "y": 5, "w": 20, "h": 20},
        "source": "test",
        "target_process": target_process,
        "private": private,
        "test_reserved": False,
    }
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",", ":")))
        f.write("\n")


def test_centralized_export_refuses_include_private_via_api(tmp_path: Path) -> None:
    """Calling export_corrections(..., centralized_export=True, include_private=True) raises."""
    from scripts import export_corrections_to_vlm_sft as exporter

    corrections = tmp_path / "corrections"
    corrections.mkdir()
    jsonl = corrections / "corrections.jsonl"
    img = corrections / "img.png"
    _make_png(img)
    _write_entry(jsonl, image_path=img, private=True)

    with pytest.raises(ValueError, match="include_private"):
        exporter.export_corrections(
            corrections_jsonl=jsonl,
            output_dir=tmp_path / "out",
            include_private=True,
            split=(1.0, 0.0, 0.0),
            seed=42,
            centralized_export=True,
        )


def test_centralized_export_refuses_include_private_via_cli(tmp_path: Path) -> None:
    """Running the CLI with --centralized-export --include-private=true exits 2."""
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    jsonl = corrections / "corrections.jsonl"
    img = corrections / "img.png"
    _make_png(img)
    _write_entry(jsonl, image_path=img, private=True)

    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "export_corrections_to_vlm_sft.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--corrections-jsonl",
            str(jsonl),
            "--output-dir",
            str(tmp_path / "out"),
            "--centralized-export",
            "--include-private",
            "true",
        ],
        capture_output=True,
        text=True,
    )
    # argparse.parser.error() exits with 2.
    assert result.returncode == 2, (
        f"Expected exit 2, got {result.returncode}. stderr={result.stderr!r}"
    )
    combined = result.stderr + result.stdout
    assert "centralized-export" in combined or "include-private" in combined


@ocr_required
def test_centralized_export_produces_manifest_and_redacts(tmp_path: Path) -> None:
    """End-to-end centralized export: JSONL points at redacted images + manifest written."""
    from scripts import export_corrections_to_vlm_sft as exporter

    corrections = tmp_path / "corrections"
    corrections.mkdir()
    jsonl = corrections / "corrections.jsonl"

    # Build a synthetic image with text so redaction has something to do.
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]

    img_path = corrections / "sample.png"
    im = Image.new("RGB", (400, 200), color=(255, 255, 255))
    ImageDraw.Draw(im).text((20, 20), "Save", fill=(0, 0, 0), font=_pick_font(24))
    ImageDraw.Draw(im).text(
        (20, 100),
        "The quick brown fox jumps over the lazy dog",
        fill=(0, 0, 0),
        font=_pick_font(18),
    )
    im.save(img_path)

    _write_entry(jsonl, image_path=img_path, private=False, prompt="Save button")

    output_dir = tmp_path / "out"
    summary = exporter.export_corrections(
        corrections_jsonl=jsonl,
        output_dir=output_dir,
        include_private=False,
        split=(1.0, 0.0, 0.0),
        seed=42,
        centralized_export=True,
    )

    assert summary["redacted"] is True
    assert summary["redaction_policy_hash"] is not None
    assert summary["redacted_image_count"] == 1

    manifest = output_dir / "manifest.jsonl"
    assert manifest.exists()
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert Path(record["original_image"]) == img_path
    assert Path(record["redacted_image"]).exists()
    assert Path(record["redacted_image"]).parent.name == "redacted-images"
    assert record["policy_hash"] == summary["redaction_policy_hash"]

    # JSONL references the redacted copy, not the original.
    train = (output_dir / "vlm_train.jsonl").read_text(encoding="utf-8").strip()
    assert train, "train JSONL should not be empty"
    sample = json.loads(train.splitlines()[0])
    image_uri = next(
        p["image"]
        for msg in sample["messages"]
        for p in (msg.get("content") or [])
        if isinstance(p, dict) and p.get("type") == "image"
    )
    assert "redacted-images" in image_uri
    assert str(img_path.resolve().as_posix()).lstrip("/") not in image_uri


def test_local_only_skips_redaction(tmp_path: Path) -> None:
    """--local-only (default) path does not redact and emits no manifest."""
    from scripts import export_corrections_to_vlm_sft as exporter

    corrections = tmp_path / "corrections"
    corrections.mkdir()
    jsonl = corrections / "corrections.jsonl"
    img = corrections / "img.png"
    _make_png(img)
    _write_entry(jsonl, image_path=img, private=False)

    output_dir = tmp_path / "out_local"
    summary = exporter.export_corrections(
        corrections_jsonl=jsonl,
        output_dir=output_dir,
        include_private=False,
        split=(1.0, 0.0, 0.0),
        seed=42,
        centralized_export=False,
    )

    assert summary["redacted"] is False
    assert summary["redaction_policy_hash"] is None
    assert summary["redacted_image_count"] == 0
    assert not (output_dir / "manifest.jsonl").exists()
    assert not (output_dir / "redacted-images").exists()
