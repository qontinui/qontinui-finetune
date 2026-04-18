"""``PRMInferencer`` — lightweight batch scoring of grounding candidates.

Used by the RL rollout scorer in ``scripts/finetune_grounding_rl.py`` to
score K candidate completions per prompt. The checkpoint format matches what
``train.py`` writes::

    {
        "backbone_name": "openai/clip-vit-base-patch32",
        "hidden_dim": 512,
        "head_state": {...},      # linear head state_dict
    }

Only the linear head is stored; the backbone is loaded by identifier. This
keeps PRM checkpoints tiny (<1 MB) and avoids having to ship the huge frozen
encoder weights alongside the head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PRMInferencer:
    """Load a trained PRM checkpoint and score batches of candidates.

    Parameters
    ----------
    checkpoint_path:
        Path to ``prm_checkpoint.pt`` produced by ``prm.train``.
    device:
        Torch device string. Defaults to ``cuda`` if available, else ``cpu``.
    processor_dir:
        Optional override for the matching ``AutoProcessor`` directory. When
        ``None``, we reload it from ``backbone_name`` via HF hub.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        processor_dir: str | Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        from prm.model import GroundingPRM

        self._torch = torch
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"PRM checkpoint not found: {ckpt_path}. Train one with "
                "`python -m prm.train --grounding-jsonl ... --output-dir ...` "
                "before running the RL stage."
            )

        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        backbone_name: str = state["backbone_name"]
        hidden_dim: int = int(state["hidden_dim"])

        self.prm = GroundingPRM(
            backbone_name=backbone_name,
            freeze_backbone=True,
            hidden_dim=hidden_dim,
        )
        self.prm.load_head_state(state["head_state"])

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.prm.to(device).eval()

        proc_source = str(processor_dir) if processor_dir else backbone_name
        self.processor: Any = AutoProcessor.from_pretrained(proc_source)

        logger.info(
            "PRMInferencer ready (backbone=%s, device=%s)", backbone_name, device
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        images: list[Any],
        instructions: list[str],
        predicted_groundings: list[str],
    ) -> list[float]:
        """Score a batch of (image, instruction, predicted_grounding) tuples.

        Parameters
        ----------
        images:
            List of PIL Images (or tensors compatible with the processor).
        instructions:
            List of instruction strings, same length as ``images``.
        predicted_groundings:
            List of candidate grounding strings, same length as ``images``.

        Returns
        -------
        list[float] of length ``len(images)`` — scalar rewards.
        """
        if not (len(images) == len(instructions) == len(predicted_groundings)):
            raise ValueError(
                "score_batch: images, instructions, and predicted_groundings "
                f"must be equal length (got {len(images)}, {len(instructions)}, "
                f"{len(predicted_groundings)})."
            )
        if not images:
            return []

        torch = self._torch
        texts = [f"{i} → {g}" for i, g in zip(instructions, predicted_groundings, strict=True)]
        enc = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            reward = self.prm.forward(**enc)

        result: list[float] = reward.detach().cpu().tolist()
        return result

    def score_one(
        self,
        image: Any,
        instruction: str,
        predicted_grounding: str,
    ) -> float:
        """Convenience wrapper for scoring a single candidate."""
        return self.score_batch([image], [instruction], [predicted_grounding])[0]
