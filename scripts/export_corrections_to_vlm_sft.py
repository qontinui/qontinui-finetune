#!/usr/bin/env python3
"""Export VGA correction log entries to vlm_sft chat-format JSONL.

Reads the append-only JSONL correction store written by
``qontinui.vga.correction_log.CorrectionLogger`` and emits train/val/test
files in the exact shape consumed by
``qontinui-finetune/scripts/finetune_grounding_lora.py``.

This is milestone (c) phase 1 of the VGA plan — the bridge from
user-correction events back into the v6+ training set.

Usage::

    python scripts/export_corrections_to_vlm_sft.py \\
        --corrections-jsonl datasets/vga-corrections/corrections.jsonl \\
        --output-dir dataset/vga-sft/ \\
        --include-private false \\
        --split 0.8,0.1,0.1 \\
        --seed 42

Output files::

    <output-dir>/vlm_train.jsonl
    <output-dir>/vlm_val.jsonl
    <output-dir>/vlm_test.jsonl       # only when test split > 0
    <output-dir>/summary.json

Stratified split rules
----------------------
- Stratification is by ``target_process`` so each domain (``notepad++.exe``,
  ``obs64.exe``, …) appears in every non-empty split when there are enough
  samples.
- Any entry with ``test_reserved: true`` in the correction log is routed
  to the test split regardless of the requested ratio. Those entries are
  held out from both train and val.
- Split ratios are parsed as three comma-separated floats; a zero test
  ratio disables the test output (no ``vlm_test.jsonl`` emitted) UNLESS
  at least one ``test_reserved`` entry exists.

Privacy
-------
By default, private entries (``--include-private false``) are excluded
from every split. Counted separately in ``summary.json`` as
``excluded_private``. This matches ``CorrectionLogger.iter_entries``'s
default behavior and plan §13 recommendation D.

Missing images
--------------
Entries whose ``image_path`` does not exist on disk are logged and
skipped. This mirrors the trainer's own behaviour
(``finetune_grounding_lora.GroundingDataset``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template — matches what the v5 trainer was prompted with.
# Keep this string in sync with the single-point grounding prompt used in
# ``qontinui/src/qontinui/vga/prompts.py``; divergence between exporter
# and runtime prompt templates is the highest-impact silent-failure mode
# here. (Plan §9 risk register: "Correction log drifts from the vlm_sft
# format the trainer expects.")
# ---------------------------------------------------------------------------

_GROUND_PROMPT_TEMPLATE = (
    "Locate the following element in the screenshot and output its "
    "position as <point>x y</point> where x and y are integers between "
    "0 and 1000 (normalized coordinates).\n\nElement: {prompt}"
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_bool(value: str) -> bool:
    """Argparse-friendly bool parser — ``"true"``/``"false"`` case-insensitive."""
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def _parse_split(value: str) -> tuple[float, float, float]:
    """Parse ``"train,val,test"`` into three floats summing to ~1.0."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected three comma-separated floats, got {value!r}"
        )
    try:
        ratios = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid split ratios {value!r}: {exc}"
        ) from exc
    total = sum(ratios)
    if total <= 0:
        raise argparse.ArgumentTypeError("Split ratios must be positive")
    # Normalize in case they don't sum to exactly 1.0.
    return (ratios[0] / total, ratios[1] / total, ratios[2] / total)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corrections-jsonl",
        type=Path,
        default=Path("datasets/vga-corrections/corrections.jsonl"),
        help="Path to the correction log JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/vga-sft"),
        help="Directory to write vlm_{train,val,test}.jsonl and summary.json",
    )
    parser.add_argument(
        "--include-private",
        type=_parse_bool,
        default=False,
        help="Include private entries (default: false)",
    )
    parser.add_argument(
        "--split",
        type=_parse_split,
        default=(0.8, 0.1, 0.1),
        help="Train,val,test ratios (comma-separated, default: 0.8,0.1,0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling within each stratum",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    # --centralized-export / --local-only are mutually exclusive. The
    # plan §13 rec D: redaction is mandatory for any bundle leaving the
    # dev machine; --local-only is the default and skips redaction on
    # the assumption the output stays on the user's laptop.
    export_scope = parser.add_mutually_exclusive_group()
    export_scope.add_argument(
        "--centralized-export",
        action="store_true",
        help=(
            "Produce a centralized training bundle — every sample's "
            "image is redacted via OCR before being referenced from the "
            "JSONL. Mutually exclusive with --local-only. Refuses to "
            "run alongside --include-private=true."
        ),
    )
    export_scope.add_argument(
        "--local-only",
        action="store_true",
        help=(
            "Produce a local bundle (default behavior) — no redaction. "
            "Mutually exclusive with --centralized-export."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Image dimensions
# ---------------------------------------------------------------------------


def _read_image_dimensions(image_path: Path) -> tuple[int, int] | None:
    """Return ``(width, height)`` of *image_path*, or ``None`` on error.

    Uses PIL lazily so this module stays importable even if Pillow is not
    installed — the trainer already depends on it, so in practice it will
    be available.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.error(
            "Pillow is required to read image dimensions: %s. "
            "Install with: pip install Pillow",
            exc,
        )
        return None

    try:
        with Image.open(image_path) as im:
            w, h = im.size
            return (w, h)
    except (OSError, ValueError) as exc:
        logger.warning("Cannot open image %s: %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# Correction → VLM sample conversion
# ---------------------------------------------------------------------------


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    """Return the pixel center of an ``{x, y, w, h}`` bbox."""
    return (bbox["x"] + bbox["w"] / 2.0, bbox["y"] + bbox["h"] / 2.0)


def _to_normalized_thousand(
    center_px: tuple[float, float],
    image_w: int,
    image_h: int,
) -> tuple[int, int]:
    """Project a pixel center to the integer [0, 1000]² grid.

    The v5 training prompt declares outputs as integers between 0 and 1000
    (see ``_GROUND_PROMPT_TEMPLATE``). We clamp defensively so a slightly
    out-of-bounds correction (user drags a bbox off-screen) never emits a
    label the model can't produce.
    """
    cx_px, cy_px = center_px
    if image_w <= 0 or image_h <= 0:
        return 0, 0
    nx = int(round(cx_px / image_w * 1000))
    ny = int(round(cy_px / image_h * 1000))
    nx = max(0, min(1000, nx))
    ny = max(0, min(1000, ny))
    return nx, ny


def _entry_to_vlm_sample(
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert one correction entry to a vlm_sft chat sample.

    Returns ``None`` (with a logged warning) when the image file is
    missing or the corrected bbox is malformed.
    """
    image_path_str = entry.get("image_path")
    prompt = entry.get("prompt")
    bbox = entry.get("corrected_bbox")
    if not image_path_str or not isinstance(prompt, str) or not isinstance(bbox, dict):
        logger.warning("Malformed correction entry, skipping: %r", entry.get("ts"))
        return None

    image_path = Path(image_path_str)
    if not image_path.exists():
        logger.warning("Image not found, skipping: %s", image_path)
        return None

    dims = _read_image_dimensions(image_path)
    if dims is None:
        return None
    image_w, image_h = dims

    try:
        center_px = _bbox_center(bbox)
    except (KeyError, TypeError) as exc:
        logger.warning("Malformed bbox %r (%s), skipping", bbox, exc)
        return None

    nx, ny = _to_normalized_thousand(center_px, image_w, image_h)

    # The trainer's GroundingDataset strips "file:///" prefixes. We emit
    # an absolute POSIX path with the "file:///" scheme so the trainer's
    # _extract_image_path() strips to an absolute path on every platform.
    abs_posix = image_path.resolve().as_posix()
    uri = f"file:///{abs_posix.lstrip('/')}"

    user_text = _GROUND_PROMPT_TEMPLATE.format(prompt=prompt)

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": uri},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": f"<point>{nx} {ny}</point>",
            },
        ]
    }


# ---------------------------------------------------------------------------
# Reading the correction log
# ---------------------------------------------------------------------------


def _iter_entries(
    corrections_jsonl: Path,
    include_private: bool,
) -> Iterable[dict[str, Any]]:
    """Yield parsed JSONL entries, applying the privacy filter.

    We read the file directly rather than going through
    :class:`qontinui.vga.correction_log.CorrectionLogger` because the
    exporter is intentionally resilient to an older schema (e.g. missing
    ``private`` field or ``test_reserved`` arriving later). Matching the
    logger's own skip-on-decode-error behavior.
    """
    if not corrections_jsonl.exists():
        logger.error("Correction log not found: %s", corrections_jsonl)
        return

    with corrections_jsonl.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: invalid JSON (%s), skipping", lineno, exc)
                continue

            if not include_private:
                if entry.get("private", True):
                    continue
                image_path = entry.get("image_path")
                if image_path and Path(f"{image_path}.private").exists():
                    continue

            yield entry


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def _stratified_split(
    samples_by_domain: dict[str, list[dict[str, Any]]],
    ratios: tuple[float, float, float],
    seed: int,
    reserved_test_by_domain: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, int]],
]:
    """Split samples per-domain, returning (train, val, test, per_domain_counts).

    ``per_domain_counts`` has shape ``{domain: {"train": n, "val": n,
    "test": n}}`` — handy for the summary file.

    Stratification preserves domain coverage: if domain D has N samples,
    D contributes ``round(N * r_train)`` to train, ``round(N * r_val)``
    to val, and the remainder to test. Any entries in
    ``reserved_test_by_domain[D]`` are appended to test after the ratio
    split.

    Edge cases:
    - If a domain has 1 sample, it goes to train (we never send a lone
      sample to val or test — that would leave domain coverage for the
      bigger split at zero).
    - If a domain has 2 samples, they go to train and val (no test
      allocation from ratio, but reserved entries still route to test).
    - When train_ratio is zero (pathological), samples fall through to
      val then test.
    """
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    per_domain: dict[str, dict[str, int]] = {}

    reserved_test_by_domain = reserved_test_by_domain or {}
    r_train, r_val, _r_test = ratios

    rng = random.Random(seed)

    all_domains = sorted(
        set(samples_by_domain.keys()) | set(reserved_test_by_domain.keys())
    )

    for domain in all_domains:
        shuffled = list(samples_by_domain.get(domain, []))
        rng.shuffle(shuffled)
        n = len(shuffled)

        if n == 0:
            # Only reserved test items for this domain.
            domain_reserved = reserved_test_by_domain.get(domain, [])
            test.extend(domain_reserved)
            per_domain[domain] = {
                "train": 0,
                "val": 0,
                "test": len(domain_reserved),
            }
            continue

        # Ratio-driven split with defensive floors.
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        # Never take more than exist.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        # Floors: if the caller asks for a non-zero ratio for a split
        # but rounding eats it to zero, give that split at least one
        # sample as long as there's something left to give.
        if r_train > 0 and n_train == 0 and n > 0:
            n_train = 1
            n_val = min(n_val, n - n_train)
        if (
            r_val > 0
            and n_val == 0
            and n - n_train > 0
            and n >= 2  # don't starve a 1-sample domain
        ):
            n_val = 1
        n_test_planned = n - n_train - n_val
        if _r_test > 0 and n_test_planned == 0 and n - n_train - n_val > 0 and n >= 3:
            # Steal one from val (not train) when possible.
            if n_val > 0:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1

        # Slice indices for the three splits.
        i_val = n_train
        i_test = n_train + n_val
        domain_train = shuffled[:i_val]
        domain_val = shuffled[i_val:i_test]
        domain_test = shuffled[i_test:]

        # Reserved-for-test entries always land in the test split.
        domain_test = list(domain_test) + reserved_test_by_domain.get(domain, [])

        train.extend(domain_train)
        val.extend(domain_val)
        test.extend(domain_test)

        per_domain[domain] = {
            "train": len(domain_train),
            "val": len(domain_val),
            "test": len(domain_test),
        }

    # Shuffle the final lists so domain ordering is not encoded in index.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test, per_domain


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, separators=(",", ":")))
            f.write("\n")


def _rewrite_sample_image_uri(sample: dict[str, Any], new_path: Path) -> None:
    """Rewrite the sample's image URI to point at *new_path*.

    Matches the URI scheme that ``_entry_to_vlm_sample`` emits so the
    trainer's ``_extract_image_path`` strips ``file:///`` cleanly on
    every OS.
    """
    abs_posix = new_path.resolve().as_posix()
    uri = f"file:///{abs_posix.lstrip('/')}"
    messages = sample.get("messages", [])
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                part["image"] = uri


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def export_corrections(
    corrections_jsonl: Path,
    output_dir: Path,
    include_private: bool,
    split: tuple[float, float, float],
    seed: int,
    centralized_export: bool = False,
    redaction_policy: Any | None = None,
) -> dict[str, Any]:
    """Run the full export. Returns the summary dict that's written to disk.

    When ``centralized_export`` is True, every sample's image is copied
    through :func:`qontinui_finetune.redact.redact_image` and the JSONL
    references the redacted copy (not the original). A ``manifest.jsonl``
    file is written alongside ``summary.json`` giving one record per
    redacted sample — ``{original_image, redacted_image, policy_hash}``
    — so downstream trainers can audit the redaction trail.
    """
    r_train, r_val, r_test = split

    # Guardrail: refuse to centralize private entries. The exporter is
    # called programmatically in tests too, so this check runs both here
    # and in the argparse layer (parser.error) to fail as early as
    # possible.
    if centralized_export and include_private:
        raise ValueError(
            "Refusing to produce a centralized export while "
            "include_private=True: private correction entries must not "
            "leave the dev machine even with redaction. Re-run with "
            "--include-private=false or drop --centralized-export."
        )

    # Lazy-import the redactor so local-only exports don't pay the OCR
    # import cost (and don't fail when OCR isn't installed).
    policy = None
    policy_hash: str | None = None
    redacted_dir: Path | None = None
    manifest_records: list[dict[str, Any]] = []
    if centralized_export:
        from qontinui_finetune.redact import RedactionPolicy, redact_image

        policy = redaction_policy or RedactionPolicy()
        policy_hash = policy.policy_hash()
        redacted_dir = output_dir / "redacted-images"
        redacted_dir.mkdir(parents=True, exist_ok=True)

    # Bucket by domain; collect reserved-for-test separately so they
    # don't inflate the ratio split.
    samples_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reserved_test_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_target_process: dict[str, int] = defaultdict(int)

    excluded_private = 0
    excluded_missing_image = 0

    # First pass: count privacy exclusions BEFORE filtering, so the
    # summary tells the truth about what the log held.
    if corrections_jsonl.exists():
        with corrections_jsonl.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                is_private = entry.get("private", True)
                if is_private and not include_private:
                    excluded_private += 1

    # Second pass: produce VLM samples for included entries.
    for entry in _iter_entries(corrections_jsonl, include_private=include_private):
        sample = _entry_to_vlm_sample(entry)
        if sample is None:
            # The entry_to_vlm_sample function has already logged why.
            # Distinguish missing-image from malformed by re-checking the
            # image path.
            image_path_str = entry.get("image_path")
            if image_path_str and not Path(image_path_str).exists():
                excluded_missing_image += 1
            continue

        # Defense-in-depth: if a private entry slipped through (e.g. the
        # guard above was bypassed in tests), re-check here. If we're in
        # centralized mode, refuse outright.
        if centralized_export and entry.get("private", False):
            raise ValueError(
                "Refusing centralized export: correction entry marked "
                f"private=True slipped through include_private filter: "
                f"{entry.get('ts')}"
            )

        # Redact the image and rewrite the sample's image URI to point
        # at the redacted copy in <output-dir>/redacted-images/.
        if centralized_export and redacted_dir is not None and policy is not None:
            from qontinui_finetune.redact import redact_image

            original_path = Path(entry["image_path"])
            # Use image_sha when available (deterministic + short);
            # otherwise hash the absolute path so retries are idempotent.
            sha_hint = (
                entry.get("image_sha")
                or hashlib.sha256(
                    str(original_path.resolve()).encode("utf-8")
                ).hexdigest()[:16]
            )
            redacted_path = redacted_dir / f"{sha_hint}.png"
            if not redacted_path.exists():
                redact_image(
                    image_path=original_path,
                    policy=policy,
                    out_path=redacted_path,
                )
            _rewrite_sample_image_uri(sample, redacted_path)
            manifest_records.append(
                {
                    "original_image": str(original_path),
                    "redacted_image": str(redacted_path),
                    "policy_hash": policy_hash,
                }
            )

        target_process = entry.get("target_process", "unknown")
        per_target_process[target_process] += 1

        if entry.get("test_reserved", False):
            reserved_test_by_domain[target_process].append(sample)
        else:
            samples_by_domain[target_process].append(sample)

    train, val, test, per_domain_counts = _stratified_split(
        samples_by_domain=dict(samples_by_domain),
        ratios=split,
        seed=seed,
        reserved_test_by_domain=dict(reserved_test_by_domain),
    )

    # Writes
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "vlm_train.jsonl", train)
    _write_jsonl(output_dir / "vlm_val.jsonl", val)

    # Only emit test file when test split > 0 or we have reserved entries.
    write_test = r_test > 0 or any(reserved_test_by_domain.values())
    if write_test:
        _write_jsonl(output_dir / "vlm_test.jsonl", test)

    total_samples = len(train) + len(val) + len(test)
    summary: dict[str, Any] = {
        "total_samples": total_samples,
        "per_target_process": dict(per_target_process),
        "per_split": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "per_domain_per_split": per_domain_counts,
        "excluded_private": excluded_private,
        "excluded_missing_image": excluded_missing_image,
        "redacted": centralized_export,
        "redaction_policy_hash": policy_hash,
        "redacted_image_count": len(manifest_records),
        "config": {
            "corrections_jsonl": str(corrections_jsonl),
            "output_dir": str(output_dir),
            "include_private": include_private,
            "split": {"train": r_train, "val": r_val, "test": r_test},
            "seed": seed,
            "centralized_export": centralized_export,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Manifest: one record per redacted sample for downstream audit.
    # Only emitted in centralized mode — nothing to audit locally.
    if centralized_export:
        with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for record in manifest_records:
                f.write(json.dumps(record, separators=(",", ":")))
                f.write("\n")

    logger.info(
        "Exported %d samples (train=%d val=%d test=%d) across %d domains",
        total_samples,
        len(train),
        len(val),
        len(test),
        len(per_target_process),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s - %(message)s",
    )

    # Enforce the PII-leak guardrail at the CLI layer: a centralized
    # export with include_private=True would route private corrections
    # off-device, defeating the whole point of the privacy flag.
    # argparse.parser.error() writes to stderr and sys.exits(2).
    if args.centralized_export and args.include_private:
        parser.error(
            "--centralized-export is incompatible with "
            "--include-private=true: private correction entries must "
            "not be shipped off the dev machine. Either drop "
            "--centralized-export (for a local bundle) or re-run with "
            "--include-private=false (to exclude private entries)."
        )

    summary = export_corrections(
        corrections_jsonl=args.corrections_jsonl,
        output_dir=args.output_dir,
        include_private=args.include_private,
        split=args.split,
        seed=args.seed,
        centralized_export=args.centralized_export,
    )

    # Pretty-print a condensed report to stderr for interactive use.
    logger.info("Summary: %s", json.dumps(summary["per_split"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
