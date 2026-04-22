#!/usr/bin/env python3
"""
QLoRA fine-tuning script for grounding data on UI-TARS-1.5-7B (Qwen2.5-VL base).

Reads vlm_train.jsonl / vlm_val.jsonl produced by the component-render synthetic
grounding data pipeline and fine-tunes the model to predict normalised <point>
coordinates from natural-language element descriptions.

Usage:
    python scripts/finetune_grounding_lora.py \\
        --train-data dataset/vlm_sft/vlm_train.jsonl \\
        --val-data   dataset/vlm_sft/vlm_val.jsonl \\
        --model      ByteDance-Seed/UI-TARS-1.5-7B \\
        --output-dir models/qontinui-grounding-v1 \\
        --epochs 3 --lr 2e-4 --batch-size 2 --grad-accum 16 \\
        --lora-r 16 --lora-alpha 32 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100


def _extract_image_path(content: list[dict[str, Any]]) -> str | None:
    """Return the local file path from the image part of a user message."""
    for part in content:
        if part.get("type") == "image":
            raw: str = part.get("image", "")
            # Strip file:// prefix if present
            if raw.startswith("file:///"):
                return raw[8:]
            if raw.startswith("file://"):
                return raw[7:]
            return raw
    return None


def _extract_text(content: list[dict[str, Any]]) -> str:
    """Return the text part of a user or assistant message."""
    for part in content:
        if part.get("type") == "text":
            return part["text"]
    # Fallback: if content is a plain string (assistant turn)
    if isinstance(content, str):
        return content
    return ""


class GroundingDataset(Dataset):
    """
    Reads a JSONL file where every line is a chat-formatted grounding sample.

    Each sample has:
      messages[0]  – user turn with an image part and a text part
      messages[1]  – assistant turn with a <point>x y</point> string

    Samples whose image files are missing on disk are silently skipped.
    """

    def __init__(
        self,
        jsonl_path: Path,
        processor: Any,
        max_image_pixels: int | None = None,
    ) -> None:
        self.processor = processor
        self.samples: list[dict[str, Any]] = []

        skipped = 0
        with open(jsonl_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Line %d: JSON decode error – %s", lineno, exc)
                    skipped += 1
                    continue

                messages = record.get("messages", [])
                if len(messages) < 2:
                    logger.warning("Line %d: fewer than 2 messages, skipping", lineno)
                    skipped += 1
                    continue

                user_msg = messages[0]
                asst_msg = messages[1]

                user_content = user_msg.get("content", [])
                img_path = _extract_image_path(user_content)
                if img_path is None:
                    logger.warning("Line %d: no image part found, skipping", lineno)
                    skipped += 1
                    continue

                if not Path(img_path).exists():
                    logger.warning(
                        "Line %d: image not found at %r, skipping", lineno, img_path
                    )
                    skipped += 1
                    continue

                text_prompt = _extract_text(user_content)
                asst_content = asst_msg.get("content", "")
                if isinstance(asst_content, list):
                    target = _extract_text(asst_content)
                else:
                    target = str(asst_content)

                self.samples.append(
                    {
                        "img_path": img_path,
                        "prompt": text_prompt,
                        "target": target,
                    }
                )

        logger.info(
            "Loaded %d samples from %s (%d skipped)",
            len(self.samples),
            jsonl_path,
            skipped,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Collator – builds model inputs and masks assistant loss only
# ---------------------------------------------------------------------------


class GroundingCollator:
    """
    Builds batched model inputs.

    The loss is masked so only the assistant response tokens contribute
    (the <point>…</point> output).  All prompt tokens are set to IGNORE_INDEX.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        from PIL import Image

        images_batch: list[Any] = []
        texts_batch: list[str] = []

        for sample in batch:
            img = Image.open(sample["img_path"]).convert("RGB")
            images_batch.append(img)

            # Build a chat-template formatted string
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["img_path"]},
                        {"type": "text", "text": sample["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": sample["target"],
                },
            ]
            # apply_chat_template returns a string with special tokens
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts_batch.append(text)

        # Tokenize (the processor handles image pixel extraction internally)
        encoding = self.processor(
            text=texts_batch,
            images=images_batch,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        labels = input_ids.clone()

        # Mask prompt tokens: find where assistant turn starts
        # by locating the assistant-role marker in the token sequence.
        # We do this per-sample in the batch.
        for i in range(len(batch)):
            target_text = batch[i]["target"]
            target_ids = self.processor.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            seq = input_ids[i]
            target_len = len(target_ids)
            # Find last occurrence of target_ids in seq
            found = -1
            for start in range(len(seq) - target_len, -1, -1):
                if torch.equal(seq[start : start + target_len], target_ids):
                    found = start
                    break
            if found >= 0:
                labels[i, :found] = IGNORE_INDEX
            else:
                # Fallback: mask everything (skip this sample's loss)
                labels[i, :] = IGNORE_INDEX

        encoding["labels"] = labels
        return encoding


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_processor(
    model_name: str,
) -> tuple[Any, Any]:
    """Load Qwen2.5-VL model in 4-bit QLoRA config + processor."""
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    logger.info("Loading processor from %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading model from %s (4-bit QLoRA)", model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    return model, processor


def apply_lora(
    model: Any,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Any:
    """Wrap model with LoRA adapters targeting LM linear layers only (not ViT)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
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

    model = get_peft_model(model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters: %s total, %s trainable (%.2f%%)",
        f"{total_params:,}",
        f"{trainable_params:,}",
        100.0 * trainable_params / total_params,
    )

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_training_args(args: argparse.Namespace, output_dir: Path) -> Any:
    """Build HuggingFace TrainingArguments from CLI args."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )


def run_training(args: argparse.Namespace) -> None:
    """Full training pipeline: load → LoRA → train → merge → save."""
    from transformers import Trainer

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + processor ──────────────────────────────────────────────
    model, processor = load_model_and_processor(args.model)
    model = apply_lora(model, args.lora_r, args.lora_alpha, lora_dropout=0.05)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = GroundingDataset(Path(args.train_data), processor)
    val_dataset = GroundingDataset(Path(args.val_data), processor)

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"Training dataset is empty after filtering. Check paths in {args.train_data}"
        )

    collator = GroundingCollator(processor)

    # ── Training arguments ───────────────────────────────────────────────────
    training_args = build_training_args(args, output_dir / "checkpoints")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    logger.info("Starting training (%d epochs, effective batch %d)", args.epochs, args.batch_size * args.grad_accum)
    t0 = time.time()

    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(str(output_dir / "checkpoints"))
    if last_checkpoint is not None:
        logger.info("Resuming from checkpoint: %s", last_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    elapsed = time.time() - t0
    logger.info(
        "Training complete in %.1f s (%.1f min). Final loss: %.4f",
        elapsed,
        elapsed / 60,
        train_result.training_loss,
    )

    # ── Save LoRA adapter, then re-merge against the full-precision base ─────
    # Merging into the 4-bit-quantised training base keeps the result 4-bit
    # and forces downstream serving stacks to carry a matching
    # `bitsandbytes` version. Re-merging into a bf16 copy of the base
    # produces a standard HuggingFace checkpoint servable by any vLLM /
    # SGLang / Transformers deployment without extra deps.
    logger.info("Saving trained LoRA adapter…")
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))

    # Free the training model + optimiser state before loading the
    # full-precision base — otherwise the 7B bf16 copy OOMs on top of
    # Trainer's GPU allocations.
    del trainer
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Reloading base model in bf16 for clean merge…")
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen25VL

    full_base = _Qwen25VL.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(full_base, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged (bf16) model to %s", merged_dir)
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))

    logger.info("Done. Merged model saved to: %s", merged_dir)
    print(f"\nTotal training time : {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"Final training loss : {train_result.training_loss:.4f}")
    print(f"Merged model saved  : {merged_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of UI-TARS-1.5-7B on grounding data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to vlm_train.jsonl",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to vlm_val.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ByteDance-Seed/UI-TARS-1.5-7B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/qontinui-grounding-v1",
        help="Output directory for checkpoints and merged model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main() -> None:
    # Suppress noisy deprecation warnings from deep dependencies
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

    args = parse_args()

    logger.info("=== Qontinui Grounding LoRA Fine-Tuning ===")
    logger.info("Model        : %s", args.model)
    logger.info("Train data   : %s", args.train_data)
    logger.info("Val data     : %s", args.val_data)
    logger.info("Output dir   : %s", args.output_dir)
    logger.info("Epochs       : %d", args.epochs)
    logger.info("LR           : %g", args.lr)
    logger.info("Batch size   : %d (x%d accum = %d effective)", args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    logger.info("LoRA r/alpha : %d / %d", args.lora_r, args.lora_alpha)
    logger.info("Seed         : %d", args.seed)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info("GPU %d: %s", i, torch.cuda.get_device_name(i))
    else:
        logger.warning("No CUDA device found – training will be extremely slow on CPU")

    run_training(args)


if __name__ == "__main__":
    main()
