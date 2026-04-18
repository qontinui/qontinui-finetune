"""Process Reward Model (PRM) for grounding v2 RL polish.

The PRM scores partial grounding rollouts with a scalar reward signal. It is
trained on success-labeled grounding records (``success_source in {"wsm",
"pixel_diff"}``) and consumed by the GRPO-based RL fine-tuning pipeline in
``scripts/finetune_grounding_rl.py``.

Public API::

    from prm import GroundingPRM, PRMDataset, PRMInferencer

Training entrypoint::

    python -m prm.train --grounding-jsonl dataset/grounding.jsonl \\
        --output-dir models/grounding-prm-v1 --epochs 3 --lr 1e-4

Design notes
------------
- Backbone: a small frozen vision+language encoder (default CLIP ViT-B/32,
  ~150 M params) paired with a trainable linear head. The VLM backbone used
  for SFT is *not* reused — PRM inference needs to be cheap (runs K times per
  rollout at RL time), and reusing the 7 B grounding VLM would dominate the
  RL step cost. See ``model.py`` for the rationale.
- Label policy (see ``dataset.py``):
    * ``success_source == "wsm"``        → label=+1, confidence=1.0
    * ``success_source == "pixel_diff"`` → label=+1, confidence=1.0 when
      action.success is True; label=-1 otherwise
    * ``success_source == "record_flag"``→ downweight to 0.5 confidence
    * ``success_source is None``         → skip
- PRM quality is gated on the WSM (Workflow Success Monitor) wiring delivering
  clean ``success_source="wsm"`` labels. Without that (Phase 3a), the PRM
  effectively trains only on synthetic/static data and the RL polish won't
  move accuracy beyond SFT. Document this clearly in downstream tooling.
"""

from __future__ import annotations

from prm.dataset import PRMDataset, PRMExample
from prm.infer import PRMInferencer
from prm.model import GroundingPRM

__all__ = [
    "GroundingPRM",
    "PRMDataset",
    "PRMExample",
    "PRMInferencer",
]
