#!/usr/bin/env python3
"""GRPO-based RL polish for the grounding VLM — second stage after SFT LoRA.

READ ME FIRST
-------------
This script is a **second-stage polish** for the grounding VLM. It must run
AFTER the supervised LoRA fine-tune (``scripts/finetune_grounding_lora.py``)
and needs:

1. A **base model** — ideally the SFT adapter's merged (bf16) directory from
   the first stage. Passed via ``--sft-adapter`` (preferred) or ``--model``
   (raw HF id, useful only for smoke tests).
2. A **trained PRM checkpoint** (``--prm-checkpoint``). Train one with
   ``python -m prm.train --grounding-jsonl ...`` first. Without it the RL
   step reduces to outcome-only reward, and a warning is logged.
3. **WSM-stamped training data** (``--grounding-jsonl``) — the same
   grounding.jsonl consumed by the PRM. RL rollouts need per-step rewards and
   those come from the PRM, which in turn needs ``success_source="wsm"``
   labels to be meaningful (see ``prm/dataset.py``).

Algorithmic notes (ClawGUI GiGPO, paper-described math only — we do not
vendor their code)::

    For each prompt p, we generate K candidate completions c_1..c_K using
    model.generate(num_return_sequences=K). Each candidate is parsed into
    action steps (the VLM output format from grounding_to_vlm.py is a single
    <point> today; if multi-step completions appear later, the parser in
    ``parse_action_steps`` will fan them out). Every step gets a PRM score
    on the partial rollout up to that step. The total candidate reward is::

        R(c) = R_outcome(c) + step_reward_weight * mean_k(PRM(step_k))

    GRPO then computes group-relative advantages::

        A(c_i) = (R(c_i) - mean_j R(c_j)) / (std_j R(c_j) + eps)

    and optimises the policy under the KL-regularised objective (trl's
    ``GRPOTrainer`` handles the clipping + β-KL term).

    Reference: ClawGUI / GiGPO (ZJU-REAL), but the implementation below is a
    clean-room port expressed in trl terms.

Usage::

    python scripts/finetune_grounding_rl.py \\
        --train-data dataset/vlm_sft/vlm_train.jsonl \\
        --sft-adapter models/qontinui-grounding-v1/merged \\
        --prm-checkpoint models/grounding-prm-v1/prm_checkpoint.pt \\
        --output-dir models/qontinui-grounding-rl-v1 \\
        --epochs 1 --lr 1e-5 --num-rollouts 8 --grpo-beta 0.04 \\
        --step-reward-weight 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# trl compatibility check
# ---------------------------------------------------------------------------


def _import_grpo() -> tuple[Any, Any]:
    """Import ``GRPOTrainer`` + ``GRPOConfig``, or raise a helpful error.

    ``GRPOTrainer`` landed in trl 0.15. We pin ``trl>=0.18`` in
    ``pyproject.toml`` (see the top-level ``[tool.poetry.dependencies]``
    block) to guarantee a stable API surface.
    """
    try:
        import trl
    except ImportError as exc:
        raise RuntimeError(
            "trl is not installed. Install with `poetry install` in "
            "qontinui-finetune/ (pyproject.toml pins trl>=0.18)."
        ) from exc

    GRPOTrainer = getattr(trl, "GRPOTrainer", None)
    GRPOConfig = getattr(trl, "GRPOConfig", None)
    if GRPOTrainer is None or GRPOConfig is None:
        trl_version = getattr(trl, "__version__", "?")
        raise RuntimeError(
            f"Installed trl {trl_version} is missing GRPOTrainer/GRPOConfig. "
            "GRPOTrainer was added in trl 0.15; upgrade with "
            "`pip install -U 'trl>=0.18'`."
        )
    return GRPOTrainer, GRPOConfig


# ---------------------------------------------------------------------------
# Ground-truth parsing
# ---------------------------------------------------------------------------

_POINT_RE = re.compile(r"<point>\s*([\d.]+)\s+([\d.]+)\s*</point>")


def parse_point(text: str | None) -> tuple[float, float] | None:
    """Extract ``(x, y)`` from ``<point>x y</point>``. Mirror of
    ``qontinui_train.evaluation.grounding_eval.parse_point``."""
    if not text:
        return None
    m = _POINT_RE.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _gt_point_from_sample(sample: dict[str, Any]) -> tuple[float, float] | None:
    """Extract the ground-truth point from a VLM SFT sample record."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        return parse_point(part.get("text", ""))
            elif isinstance(content, str):
                return parse_point(content)
    return None


def _prompt_text_from_sample(sample: dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        return str(part.get("text", ""))
            elif isinstance(content, str):
                return content
    return ""


def _image_path_from_sample(sample: dict[str, Any]) -> str | None:
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        raw = str(part.get("image", ""))
                        if raw.startswith("file:///"):
                            return raw[len("file:///"):]
                        if raw.startswith("file://"):
                            return raw[len("file://"):]
                        return raw
    return None


# ---------------------------------------------------------------------------
# Outcome reward: Acc@center-style tolerance check
# ---------------------------------------------------------------------------


def outcome_reward(
    completion: str,
    gt_point: tuple[float, float] | None,
    tolerance: float,
) -> float:
    """Reward 1.0 if predicted point is within ``tolerance`` of GT, else 0.

    Mirrors ``grounding_eval.is_within_tolerance`` so training-time rewards
    stay consistent with eval-time accuracy.
    """
    if gt_point is None:
        return 0.0
    pred = parse_point(completion)
    if pred is None:
        return 0.0
    dx = pred[0] - gt_point[0]
    dy = pred[1] - gt_point[1]
    return 1.0 if math.sqrt(dx * dx + dy * dy) <= tolerance else 0.0


# ---------------------------------------------------------------------------
# Action-step parsing (GiGPO step-level reward plumbing)
# ---------------------------------------------------------------------------


def parse_action_steps(completion: str) -> list[str]:
    """Split a completion into action steps.

    The current grounding VLM output is single-shot — one ``<point>x y</point>``
    per completion (see ``grounding_to_vlm.record_to_samples``). We return
    a single-element list in that case so the GiGPO machinery still works.

    If downstream experiments start emitting multi-action trajectories
    (e.g. ``<action>click</action><point>…</point><action>type</action>…``),
    extend this parser rather than reworking the reward function.
    """
    points = _POINT_RE.findall(completion or "")
    if not points:
        # No parseable structure — treat the whole completion as one "step"
        # so the PRM still scores it.
        return [completion or ""] if completion else []
    # Preserve the raw <point>…</point> strings as the step tokens so PRM
    # scoring sees the actual emitted syntax.
    return [f"<point>{x} {y}</point>" for x, y in points]


# ---------------------------------------------------------------------------
# Reward function factory
# ---------------------------------------------------------------------------


def make_reward_fn(
    sample_index: dict[str, dict[str, Any]],
    prm_inferencer: Any | None,
    tolerance: float,
    step_reward_weight: float,
) -> Any:
    """Build the reward function consumed by ``GRPOTrainer``.

    Parameters
    ----------
    sample_index:
        Map from ``prompt_text → sample_dict``. Used to recover GT point +
        image path at reward time. GRPOTrainer only passes prompts/completions
        to the reward callback, so we precompute this index from the training
        JSONL.
    prm_inferencer:
        Optional :class:`prm.infer.PRMInferencer`. When ``None``, step reward
        is skipped and a warning is logged once.
    tolerance:
        Passed to :func:`outcome_reward`.
    step_reward_weight:
        Coefficient applied to the mean step PRM score before adding to the
        outcome reward.

    Returns
    -------
    callable
        ``reward_fn(prompts, completions, **kwargs) -> list[float]`` as
        expected by ``GRPOTrainer``.
    """
    from PIL import Image

    warned_no_prm = {"flag": False}
    warned_fallback_single_step = {"flag": False}

    def reward_fn(prompts: list[str], completions: list[str], **_: Any) -> list[float]:
        rewards: list[float] = []
        # Collect PRM-scoring batch across all completions for a single call.
        prm_batch_images: list[Any] = []
        prm_batch_instr: list[str] = []
        prm_batch_steps: list[str] = []
        prm_owner: list[int] = []  # maps each PRM input back to completion idx

        outcome_only: list[float] = []

        for idx, (prompt, completion) in enumerate(
            zip(prompts, completions, strict=True)
        ):
            sample = sample_index.get(prompt)
            if sample is None:
                outcome_only.append(0.0)
                continue

            gt = _gt_point_from_sample(sample)
            out_r = outcome_reward(completion, gt, tolerance)
            outcome_only.append(out_r)

            if prm_inferencer is None:
                if not warned_no_prm["flag"]:
                    logger.warning(
                        "PRM not loaded — using outcome-only reward. Pass "
                        "--prm-checkpoint to enable the GiGPO step-level term."
                    )
                    warned_no_prm["flag"] = True
                continue

            steps = parse_action_steps(completion)
            if len(steps) <= 1 and not warned_fallback_single_step["flag"]:
                logger.warning(
                    "Completion is single-step — GiGPO step reward reduces "
                    "to a single PRM score. This is expected for today's "
                    "single-<point> output format."
                )
                warned_fallback_single_step["flag"] = True

            img_path = _image_path_from_sample(sample)
            if not img_path:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except OSError:
                logger.debug("Cannot open image %s for PRM scoring", img_path)
                continue

            for step in steps:
                prm_batch_images.append(img)
                prm_batch_instr.append(prompt)
                prm_batch_steps.append(step)
                prm_owner.append(idx)

        # Score all PRM inputs in one batched call, then aggregate per completion.
        if prm_inferencer is not None and prm_batch_images:
            step_scores = prm_inferencer.score_batch(
                prm_batch_images, prm_batch_instr, prm_batch_steps
            )
            agg: dict[int, list[float]] = {}
            for owner_idx, score in zip(prm_owner, step_scores, strict=True):
                agg.setdefault(owner_idx, []).append(float(score))
            for idx, out_r in enumerate(outcome_only):
                step_avg = sum(agg[idx]) / len(agg[idx]) if idx in agg else 0.0
                rewards.append(out_r + step_reward_weight * step_avg)
        else:
            rewards = outcome_only

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Model + LoRA loading (layered on top of SFT adapter, not replacing it)
# ---------------------------------------------------------------------------


def load_policy_model(
    args: argparse.Namespace,
) -> tuple[Any, Any]:
    """Load the policy model for RL.

    If ``--sft-adapter`` is provided, we load the merged SFT model (bf16 HF
    checkpoint) and stack a **fresh** LoRA adapter on top for the RL polish.
    This deliberately does not stomp the SFT weights: the SFT merge lives in
    that directory untouched, and the RL adapter is a separate, additive
    delta.

    If ``--sft-adapter`` is missing, we fall back to ``--model`` (raw HF id)
    with a warning — useful for smoke tests but not a supported production
    path.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    sft_dir = Path(args.sft_adapter) if args.sft_adapter else None
    if sft_dir is None or not sft_dir.exists():
        if args.require_sft_adapter:
            raise RuntimeError(
                "--sft-adapter is required (see --require-sft-adapter). "
                f"Got: {sft_dir!r}. Run finetune_grounding_lora.py first and "
                "pass its output_dir/merged directory here."
            )
        logger.warning(
            "No --sft-adapter provided — loading raw base model %s. RL on top "
            "of an un-SFT'd base is a smoke-test configuration; accuracy "
            "will not improve. Run finetune_grounding_lora.py first and "
            "pass --sft-adapter <output_dir>/merged for real runs.",
            args.model,
        )
        base_path = args.model
    else:
        logger.info("Loading SFT-merged base from %s", sft_dir)
        base_path = str(sft_dir)

    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )
    # New LoRA adapter on top of the (already-SFT-merged) base — the SFT
    # weights are preserved; this adapter only learns the RL delta.
    model = get_peft_model(model, lora_config)  # type: ignore[assignment]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "RL LoRA parameters: %s trainable / %s total (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )

    return model, processor


# ---------------------------------------------------------------------------
# PRM loading
# ---------------------------------------------------------------------------


def load_prm(checkpoint: str | None) -> Any | None:
    if not checkpoint:
        logger.warning(
            "No --prm-checkpoint provided — RL will use outcome-only reward. "
            "Train a PRM with `python -m prm.train --grounding-jsonl ...` "
            "and re-run with --prm-checkpoint <path> to enable GiGPO step "
            "rewards."
        )
        return None

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"PRM checkpoint not found: {ckpt_path}. Train one first with "
            "`python -m prm.train --grounding-jsonl ... --output-dir ...` "
            "or pass an existing checkpoint path."
        )

    from prm.infer import PRMInferencer

    return PRMInferencer(ckpt_path)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def _load_samples(jsonl_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def run_training(args: argparse.Namespace) -> None:
    import torch

    GRPOTrainer, GRPOConfig = _import_grpo()

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prompts ─────────────────────────────────────────────────────
    logger.info("Loading RL training samples from %s", args.train_data)
    samples = _load_samples(Path(args.train_data))
    if not samples:
        raise RuntimeError(f"No samples loaded from {args.train_data}")

    # Build prompt → sample index so the reward fn can recover GT + image.
    prompt_to_sample: dict[str, dict[str, Any]] = {}
    prompts: list[str] = []
    for s in samples:
        p = _prompt_text_from_sample(s)
        if not p:
            continue
        prompt_to_sample[p] = s
        prompts.append(p)

    logger.info("Indexed %d prompts", len(prompts))

    # ── Model ────────────────────────────────────────────────────────────
    model, processor = load_policy_model(args)

    # ── PRM ──────────────────────────────────────────────────────────────
    prm = load_prm(args.prm_checkpoint)

    # ── Reward ───────────────────────────────────────────────────────────
    reward_fn = make_reward_fn(
        sample_index=prompt_to_sample,
        prm_inferencer=prm,
        tolerance=args.tolerance,
        step_reward_weight=args.step_reward_weight,
    )

    # ── GRPO config ──────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        num_generations=args.num_rollouts,
        beta=args.grpo_beta,
        bf16=True,
    )

    # trl's GRPOTrainer wants a HuggingFace Dataset. Build a minimal one from
    # the indexed prompts.
    from datasets import Dataset

    hf_ds = Dataset.from_dict({"prompt": prompts})

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=[reward_fn],
        train_dataset=hf_ds,
        processing_class=processor,
    )

    logger.info(
        "Starting GRPO RL (%d epochs, K=%d rollouts, β=%g, step_w=%g)",
        args.epochs,
        args.num_rollouts,
        args.grpo_beta,
        args.step_reward_weight,
    )
    trainer.train()

    # Save the RL LoRA adapter separately — this is the "polish" delta on top
    # of the SFT merge.
    adapter_dir = output_dir / "rl_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    logger.info("RL adapter saved to %s", adapter_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GRPO-based RL polish for the grounding VLM (second stage after "
            "SFT). Combines outcome reward (Acc@center tolerance) with PRM "
            "step reward (ClawGUI-style GiGPO)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / model — mirror finetune_grounding_lora.py
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to vlm_train.jsonl (same format as SFT).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ByteDance-Seed/UI-TARS-1.5-7B",
        help="HuggingFace model ID (used only when --sft-adapter is not set).",
    )
    parser.add_argument(
        "--sft-adapter",
        type=str,
        default=None,
        help=(
            "Path to the merged bf16 SFT model "
            "(output-dir/merged from finetune_grounding_lora.py). Strongly "
            "preferred — RL starts from an SFT'd base."
        ),
    )
    parser.add_argument(
        "--require-sft-adapter",
        action="store_true",
        help="Fail hard if --sft-adapter is missing or the path does not exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/qontinui-grounding-rl-v1",
        help="Output directory for checkpoints and the RL LoRA adapter.",
    )

    # RL-specific
    parser.add_argument(
        "--prm-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a trained PRM checkpoint (prm/train.py output). When "
            "absent, RL falls back to outcome-only reward (with a warning)."
        ),
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=8,
        help="K — number of completions sampled per prompt for GRPO.",
    )
    parser.add_argument(
        "--grpo-beta",
        type=float,
        default=0.04,
        help="β — KL penalty coefficient in GRPO.",
    )
    parser.add_argument(
        "--step-reward-weight",
        type=float,
        default=0.5,
        help="Weight for the GiGPO step-level PRM term vs outcome reward.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Acc@center tolerance for outcome reward.",
    )

    # Training knobs
    parser.add_argument("--epochs", type=int, default=1, help="RL epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="RL LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="RL LoRA alpha.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    args = parse_args(argv)
    logger.info("=== Qontinui Grounding RL (GRPO + GiGPO PRM) ===")
    logger.info("Train data      : %s", args.train_data)
    logger.info("SFT adapter     : %s", args.sft_adapter or "(none — smoke test)")
    logger.info("PRM checkpoint  : %s", args.prm_checkpoint or "(none — outcome only)")
    logger.info("Output dir      : %s", args.output_dir)
    logger.info("Epochs          : %d", args.epochs)
    logger.info("Num rollouts K  : %d", args.num_rollouts)
    logger.info("GRPO β          : %g", args.grpo_beta)
    logger.info("Step weight     : %g", args.step_reward_weight)
    logger.info("Tolerance       : %g", args.tolerance)

    run_training(args)


if __name__ == "__main__":
    main()
