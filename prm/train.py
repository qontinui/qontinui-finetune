"""Standalone PRM training entrypoint.

Usage::

    python -m prm.train \\
        --grounding-jsonl dataset/grounding.jsonl \\
        --output-dir models/grounding-prm-v1 \\
        --epochs 3 --lr 1e-4 --batch-size 16

Uses HuggingFace ``Trainer`` with a custom ``compute_loss`` that weights each
example by ``PRMExample.confidence`` (see ``dataset.py`` for the label
policy). The trained head + backbone identifier are saved to
``<output-dir>/prm_checkpoint.pt`` for consumption by :class:`PRMInferencer`.

The backbone stays frozen (see ``model.py``); only the linear head is updated,
so training is fast and cheap even on modest GPUs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_hf_dataset(
    examples: list[Any],
    processor: Any,
) -> Any:
    """Wrap the PRMExample list as a ``torch.utils.data.Dataset``.

    Declared as a local factory (rather than a module-level class) so that
    ``train.py`` can be imported without torch — the CLI parser tests in
    ``tests/scripts/`` exercise ``parse_args`` only.
    """
    from PIL import Image
    from torch.utils.data import Dataset

    class _PRMTorchDataset(Dataset):
        def __init__(self, items: list[Any]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            ex = self.items[idx]
            img = Image.open(ex.image_path).convert("RGB")
            text = f"{ex.instruction} → {ex.predicted_grounding}"
            enc = processor(
                text=[text],
                images=[img],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            # Trainer expects flat tensors per example; drop the batch dim.
            out = {k: v.squeeze(0) for k, v in enc.items()}
            out["labels"] = float(ex.label)
            out["confidence"] = float(ex.confidence)
            return out

    return _PRMTorchDataset(examples)


def _build_trainer(
    model_wrapper: Any,
    train_dataset: Any,
    output_dir: Path,
    args: argparse.Namespace,
) -> Any:
    """Construct the HuggingFace ``Trainer`` with a confidence-weighted loss."""
    import torch
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(output_dir / "hf_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    class _HeadOnlyTrainableModule(torch.nn.Module):
        """Thin ``nn.Module`` wrapper so ``Trainer`` can manage the PRM."""

        def __init__(self, prm: Any) -> None:
            super().__init__()
            self.prm = prm
            # Register the head so its params show up in .parameters().
            self.head = prm.head

        def forward(self, **inputs: Any) -> Any:
            labels = inputs.pop("labels", None)
            confidence = inputs.pop("confidence", None)
            reward = self.prm.forward(**inputs)
            if labels is None:
                return {"reward": reward}
            loss_per = (reward - labels.to(reward.dtype)) ** 2
            if confidence is not None:
                loss_per = loss_per * confidence.to(reward.dtype)
            loss = loss_per.mean()
            return {"loss": loss, "reward": reward}

    wrapped = _HeadOnlyTrainableModule(model_wrapper)
    trainer = Trainer(
        model=wrapped,
        args=training_args,
        train_dataset=train_dataset,
    )
    return trainer, wrapped


def run_training(args: argparse.Namespace) -> None:
    """Full pipeline: load dataset → build PRM → train head → checkpoint."""
    import torch
    from transformers import AutoProcessor

    from prm.dataset import PRMDataset
    from prm.model import DEFAULT_BACKBONE, GroundingPRM

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    logger.info("Loading PRM dataset from %s", args.grounding_jsonl)
    ds = PRMDataset(Path(args.grounding_jsonl))
    if len(ds) == 0:
        raise RuntimeError(
            "PRMDataset is empty — check that grounding.jsonl has records "
            "with success_source in {'wsm','pixel_diff','record_flag'}. See "
            "prm/dataset.py for the label policy."
        )
    logger.info("Loaded %d labeled examples", len(ds))

    # ── Backbone + processor ────────────────────────────────────────────
    backbone = args.backbone or DEFAULT_BACKBONE
    logger.info("Loading processor for backbone %s", backbone)
    processor = AutoProcessor.from_pretrained(backbone)

    prm = GroundingPRM(backbone_name=backbone, freeze_backbone=True)

    # ── Trainer ─────────────────────────────────────────────────────────
    torch_ds = _build_hf_dataset(ds.examples, processor)
    trainer, wrapped = _build_trainer(prm, torch_ds, output_dir, args)

    logger.info("Starting PRM training (%d epochs)…", args.epochs)
    result = trainer.train()
    logger.info("Training loss: %.4f", result.training_loss)

    # ── Checkpoint (head only + backbone identifier) ────────────────────
    ckpt_path = output_dir / "prm_checkpoint.pt"
    torch.save(prm.state_dict(), ckpt_path)
    processor.save_pretrained(str(output_dir / "processor"))
    logger.info("Saved PRM checkpoint → %s", ckpt_path)
    print(f"PRM checkpoint saved to: {ckpt_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the grounding Process Reward Model (PRM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--grounding-jsonl",
        required=True,
        type=Path,
        help="Path to grounding.jsonl (rotated *.jsonl siblings are also read).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where prm_checkpoint.pt and processor/ are written.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the head.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument(
        "--backbone",
        default=None,
        help=(
            "HuggingFace model ID for the PRM encoder. Defaults to "
            "openai/clip-vit-base-patch32."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s – %(message)s",
    )
    run_training(args)


if __name__ == "__main__":
    main()
