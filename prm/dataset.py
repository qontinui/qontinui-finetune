"""``PRMDataset`` — read grounding.jsonl → PRM training examples.

The PRM is supervised by the ``success_source`` field on each
:class:`GroundingRecord.action`. See ``qontinui-train/.../grounding_record.py``
for the canonical schema.

Label policy
------------
- ``success_source == "wsm"``        → label=+1  (confidence 1.0)
- ``success_source == "pixel_diff"`` → label from ``action.success``:
      True → +1, False → -1  (confidence 1.0)
- ``success_source == "record_flag"``→ label from ``action.success``
      (True → +1, False → -1)  but **confidence=0.5**, reflecting the
      weaker signal of a user-side flag vs verified observation
- ``success_source is None``         → **skip** (no reliable label)

The confidence factor is exposed on :class:`PRMExample` and consumed by the
loss in ``train.py`` as a per-example weight.

Static records (no ``action`` at all) are also skipped — they carry no success
signal and are handled by the SFT stage.

Data-quality dependency
-----------------------
PRM quality depends on WSM (Workflow Success Monitor) wiring producing clean
``success_source="wsm"`` labels. As of Phase 3a this wiring is still being
built; until then the dataset will be dominated by ``pixel_diff`` and
``record_flag`` examples, and the RL polish that consumes this PRM should be
treated as a smoke-test rather than a real win.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Label / confidence constants — keep together so downstream callers can
# ``from prm.dataset import LABEL_POS, LABEL_NEG``.
LABEL_POS: float = 1.0
LABEL_NEG: float = -1.0
CONFIDENCE_HIGH: float = 1.0
CONFIDENCE_LOW: float = 0.5


@dataclass
class PRMExample:
    """One labeled example for PRM training.

    The ``image_path`` is kept as a string (relative or absolute) so callers
    can resolve it against the grounding.jsonl parent directory.
    """

    image_path: str
    instruction: str
    predicted_grounding: str
    label: float
    confidence: float
    success_source: str


def _instruction_from_action(action: dict[str, Any]) -> str:
    """Build a natural-language instruction string from a dynamic action.

    Examples::

        {"type": "click"}                      → "click the target"
        {"type": "type", "typed_text": "foo"}  → "type 'foo'"
        {"type": "scroll"}                     → "scroll"
    """
    atype = action.get("type", "act")
    if atype == "type":
        text = action.get("typed_text") or ""
        return f"type '{text}'" if text else "type the text"
    if atype == "click":
        return "click the target"
    return str(atype)


def _grounding_from_action(
    action: dict[str, Any],
    viewport_width: int,
    viewport_height: int,
) -> str:
    """Serialise the action target into the same ``<point>x y</point>`` format
    that the SFT model emits (see ``grounding_to_vlm.record_to_samples``).

    When ``target_bbox`` is missing, fall back to a centered point ``0.50 0.50``
    so the example still trains something (the label is driven by
    ``success_source``, not by this string). Callers that want stricter
    filtering can inspect ``PRMExample.predicted_grounding``.
    """
    bbox = action.get("target_bbox")
    if not bbox or len(bbox) < 4 or viewport_width <= 0 or viewport_height <= 0:
        return "<point>0.50 0.50</point>"
    x, y, w, h = bbox[:4]
    x_c = (x + w / 2) / viewport_width
    y_c = (y + h / 2) / viewport_height
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    return f"<point>{x_c:.2f} {y_c:.2f}</point>"


def _label_and_confidence(
    success_source: str,
    success_flag: bool | None,
) -> tuple[float, float] | None:
    """Derive (label, confidence) from ``success_source`` + ``action.success``.

    Returns ``None`` when the record should be skipped.
    """
    if success_source == "wsm":
        # WSM is the trusted label: polarity comes from success flag if
        # present, else default positive (a WSM-stamped record without a
        # success flag is considered a pass).
        label = LABEL_POS if success_flag is not False else LABEL_NEG
        return label, CONFIDENCE_HIGH

    if success_source == "pixel_diff":
        label = LABEL_POS if success_flag else LABEL_NEG
        return label, CONFIDENCE_HIGH

    if success_source == "record_flag":
        # User-side flag: downweight per spec.
        label = LABEL_POS if success_flag else LABEL_NEG
        return label, CONFIDENCE_LOW

    # Unknown / None → caller should skip.
    return None


def record_to_example(
    record: dict[str, Any],
    input_dir: Path | None = None,
) -> PRMExample | None:
    """Convert one GroundingRecord dict into a PRMExample, or ``None``.

    ``input_dir`` is used only to resolve the image path; when not provided,
    the raw ``image_path`` from the record is kept as-is.
    """
    action = record.get("action")
    if not action:
        return None  # Static records: no success signal.

    success_source = action.get("success_source")
    if not success_source:
        return None  # Spec: skip records with success_source=None.

    label_conf = _label_and_confidence(success_source, action.get("success"))
    if label_conf is None:
        return None

    label, confidence = label_conf

    image_rel = record.get("image_path", "")
    if not image_rel:
        return None
    image_path = str((input_dir / image_rel).resolve()) if input_dir else image_rel

    instruction = _instruction_from_action(action)
    grounding = _grounding_from_action(
        action,
        record.get("viewport_width", 1),
        record.get("viewport_height", 1),
    )

    return PRMExample(
        image_path=image_path,
        instruction=instruction,
        predicted_grounding=grounding,
        label=label,
        confidence=confidence,
        success_source=success_source,
    )


class PRMDataset:
    """Reads grounding.jsonl (+ rotated siblings) and yields PRMExamples.

    Implements the ``torch.utils.data.Dataset`` protocol (``__len__`` /
    ``__getitem__``) without inheriting from it — that keeps the module
    importable when torch is not installed (useful for unit tests of the
    label-policy logic).

    Parameters
    ----------
    grounding_jsonl:
        Path to ``grounding.jsonl`` (the primary file; rotated ``.1.jsonl``
        etc. are also read when present in the same directory).
    input_dir:
        Optional override for the directory used to resolve relative image
        paths. Defaults to ``grounding_jsonl.parent``.
    """

    def __init__(
        self,
        grounding_jsonl: Path,
        input_dir: Path | None = None,
    ) -> None:
        self.grounding_jsonl = Path(grounding_jsonl)
        self.input_dir = Path(input_dir) if input_dir else self.grounding_jsonl.parent
        self.examples: list[PRMExample] = []

        counts_by_source: dict[str, int] = {}
        skipped = 0
        total = 0

        jsonl_paths = [self.grounding_jsonl]
        jsonl_paths.extend(sorted(self.grounding_jsonl.parent.glob("grounding.*.jsonl")))

        for path in jsonl_paths:
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "%s:%d – invalid JSON: %s", path.name, lineno, exc
                        )
                        skipped += 1
                        continue
                    ex = record_to_example(rec, input_dir=self.input_dir)
                    if ex is None:
                        skipped += 1
                        continue
                    self.examples.append(ex)
                    counts_by_source[ex.success_source] = (
                        counts_by_source.get(ex.success_source, 0) + 1
                    )

        logger.info(
            "PRMDataset: loaded %d examples from %d records (%d skipped). By source: %s",
            len(self.examples),
            total,
            skipped,
            counts_by_source,
        )

    # Dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PRMExample:
        return self.examples[idx]

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        input_dir: Path | None = None,
    ) -> PRMDataset:
        """Build a ``PRMDataset`` from an in-memory list of record dicts.

        Mainly useful for unit tests that don't want to touch disk.
        """
        ds = cls.__new__(cls)
        ds.grounding_jsonl = Path("<memory>")
        ds.input_dir = Path(input_dir) if input_dir else Path(".")
        ds.examples = []
        for rec in records:
            ex = record_to_example(rec, input_dir=input_dir)
            if ex is not None:
                ds.examples.append(ex)
        return ds
