"""OCR-driven PII redaction for correction screenshots.

This module implements milestone (c) phase 2 item 3 of the VGA follow-up
plan: before any correction image leaves the dev machine for a
centralized training bundle, all OCR-detected text regions must be
redacted with configurable policy. Short UI chrome labels (e.g. "Save",
"Cancel") can optionally be preserved so the model still learns to
ground them; longer body text (email subjects, document contents) is
always masked.

Backend selection
-----------------
1. Try ``qontinui.hal`` ``EasyOCREngine`` first — it's the OCR wrapper
   already used elsewhere in the codebase and returns ``TextRegion``
   objects with ``(x, y, width, height, confidence, text)``.
2. Fall back to direct ``pytesseract.image_to_data`` if qontinui's HAL
   isn't importable or fails to initialize.
3. If neither backend is available, raise ``RuntimeError`` — the whole
   point of ``--centralized-export`` is that redaction is mandatory; we
   never silently ship un-redacted images.

Policy
------
:class:`RedactionPolicy` controls the mask behavior. Its
:meth:`RedactionPolicy.policy_hash` is a stable SHA-256 over the sorted
``asdict`` — downstream trainers stamp redacted samples with this hash
so audits can tie a bundle back to the exact policy that produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedactionPolicy:
    """Configuration for OCR-driven text-region redaction.

    Attributes:
        preserve_ui_labels: When True, OCR'd strings shorter than
            :attr:`max_ui_label_chars` stay visible — on the theory that
            short labels are UI chrome (button text, menu items) rather
            than PII. When False, every detected text region is masked.
        min_confidence: OCR confidence floor. Detections below this
            threshold are ignored (neither preserved nor redacted);
            preserving them would be unsafe (might be text we can't
            read) and redacting random low-confidence artefacts wastes
            pixels. We choose to ignore — anything under this bar is
            treated as noise.
        max_ui_label_chars: Length threshold for ``preserve_ui_labels``.
            Strings with ``len(text.strip()) < max_ui_label_chars`` are
            kept when ``preserve_ui_labels`` is True; longer strings are
            always masked.
        fill: RGB fill color for the mask rectangle. Default is black.
    """

    preserve_ui_labels: bool = True
    min_confidence: float = 0.5
    max_ui_label_chars: int = 30
    fill: tuple[int, int, int] = field(default=(0, 0, 0))

    def policy_hash(self) -> str:
        """Stable SHA-256 hex of the policy for audit trails.

        Uses ``json.dumps(asdict(self), sort_keys=True)`` so the hash is
        deterministic across Python sessions — the exporter stamps this
        into ``summary.json`` and the per-sample manifest so downstream
        training runs can verify what policy masked their inputs.
        """
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------


class _OCRBackend(Protocol):
    """Internal protocol for the two-tier OCR backend fallback.

    Each backend returns a list of ``(text, x, y, width, height,
    confidence)`` tuples. Confidence is on a 0.0-1.0 scale even though
    pytesseract reports 0-100; the backend normalizes.
    """

    def detect(
        self, image_path: Path
    ) -> list[tuple[str, int, int, int, int, float]]: ...


def _try_qontinui_hal_backend() -> _OCRBackend | None:
    """Build an OCR backend from ``qontinui.hal`` if importable.

    Returns ``None`` (not raising) if the import or construction fails
    for any reason — the caller then falls back to pytesseract.
    """
    try:
        from qontinui.hal.implementations.easyocr_engine import (  # type: ignore[import-not-found]
            EasyOCREngine,
        )
    except Exception as exc:  # pragma: no cover - exercised when HAL absent
        logger.debug("qontinui.hal EasyOCR unavailable: %s", exc)
        return None

    try:
        engine = EasyOCREngine()
    except Exception as exc:
        logger.debug("EasyOCREngine init failed: %s", exc)
        return None

    from PIL import Image  # type: ignore[import-not-found]

    class _HALBackend:
        def detect(
            self, image_path: Path
        ) -> list[tuple[str, int, int, int, int, float]]:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                # min_confidence=0.0 here — the policy filter runs upstream
                # so the caller gets the full detection list.
                regions = engine.get_text_regions(rgb, min_confidence=0.0)
            return [
                (r.text, r.x, r.y, r.width, r.height, float(r.confidence))
                for r in regions
            ]

    return _HALBackend()


def _try_pytesseract_backend() -> _OCRBackend | None:
    """Build a pytesseract fallback backend if installed and functional.

    Checks both that pytesseract imports and that the tesseract binary
    is actually on PATH — pytesseract happily imports without a binary
    but its first call raises ``TesseractNotFoundError``.
    """
    try:
        import pytesseract  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        logger.debug("pytesseract unavailable: %s", exc)
        return None

    # Probe the binary — image_to_data will raise TesseractNotFoundError
    # if the tesseract CLI is missing, so we want to surface that early.
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        logger.debug("tesseract binary not callable: %s", exc)
        return None

    from PIL import Image  # type: ignore[import-not-found]

    class _TesseractBackend:
        def detect(
            self, image_path: Path
        ) -> list[tuple[str, int, int, int, int, float]]:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                data = pytesseract.image_to_data(
                    rgb, output_type=pytesseract.Output.DICT
                )
            results: list[tuple[str, int, int, int, int, float]] = []
            n = len(data.get("text", []))
            for i in range(n):
                text = data["text"][i]
                if not text or not text.strip():
                    continue
                try:
                    conf = float(data["conf"][i])
                except (TypeError, ValueError):
                    continue
                # pytesseract reports -1 for "no confidence available"
                if conf < 0:
                    continue
                # Normalize pytesseract's 0-100 range to 0.0-1.0 so the
                # policy's min_confidence is comparable across backends.
                conf_norm = conf / 100.0
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                results.append((text, x, y, w, h, conf_norm))
            return results

    return _TesseractBackend()


def _select_backend() -> _OCRBackend:
    """Pick the best available OCR backend.

    Raises ``RuntimeError`` with an install hint when neither backend is
    available — by design, redaction failure is not silent.
    """
    backend = _try_qontinui_hal_backend()
    if backend is not None:
        logger.info("redaction backend: qontinui.hal EasyOCR")
        return backend
    backend = _try_pytesseract_backend()
    if backend is not None:
        logger.info("redaction backend: pytesseract")
        return backend
    raise RuntimeError(
        "No OCR backend available for redaction. Install one of:\n"
        "  - easyocr (preferred): pip install easyocr\n"
        "    (or ensure qontinui.hal.implementations.easyocr_engine "
        "imports cleanly)\n"
        "  - tesseract CLI + pytesseract: install tesseract-ocr from "
        "https://github.com/UB-Mannheim/tesseract/wiki and run "
        "`pip install pytesseract`\n"
        "Redaction is mandatory for --centralized-export; refusing to "
        "proceed without OCR."
    )


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def redact_image(
    image_path: Path,
    policy: RedactionPolicy,
    out_path: Path | None = None,
    *,
    backend: _OCRBackend | None = None,
) -> Path:
    """Redact text regions from *image_path* according to *policy*.

    Args:
        image_path: Source image. Must exist.
        policy: Redaction configuration.
        out_path: Destination path. Defaults to
            ``<stem>.redacted<suffix>`` alongside the source.
        backend: Optional pre-constructed backend (tests inject mocks
            here to avoid initializing a real OCR engine).

    Returns:
        The path of the written redacted image.

    Raises:
        RuntimeError: When no OCR backend is available.
        FileNotFoundError: When *image_path* doesn't exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Redaction source missing: {image_path}")

    if out_path is None:
        out_path = image_path.parent / (
            f"{image_path.stem}.redacted{image_path.suffix}"
        )

    ocr = backend if backend is not None else _select_backend()
    detections = ocr.detect(image_path)

    from PIL import Image, ImageDraw  # type: ignore[import-not-found]

    with Image.open(image_path) as src:
        img = src.convert("RGB")
    draw = ImageDraw.Draw(img)

    redacted = 0
    preserved = 0
    for text, x, y, w, h, conf in detections:
        if conf < policy.min_confidence:
            continue
        stripped = text.strip()
        if policy.preserve_ui_labels and 0 < len(stripped) < policy.max_ui_label_chars:
            preserved += 1
            continue
        # Draw the opaque rectangle over the text region.
        # PIL's rectangle is inclusive on both ends; we pass w/h directly.
        draw.rectangle(
            (x, y, x + w, y + h),
            fill=tuple(policy.fill),
        )
        redacted += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    logger.info(
        "redacted %s -> %s (redacted=%d, preserved=%d, total_detections=%d)",
        image_path,
        out_path,
        redacted,
        preserved,
        len(detections),
    )
    return out_path


__all__ = [
    "RedactionPolicy",
    "redact_image",
]
