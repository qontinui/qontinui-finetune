"""Smoke test: the RL fine-tune CLI parses all expected flags.

Does not train anything — only verifies argparse accepts the spec's arg set.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_FT_ROOT = Path(__file__).resolve().parents[2]  # qontinui-finetune/
_SCRIPT = _FT_ROOT / "scripts" / "finetune_grounding_rl.py"


def _load_rl_module():
    """Import scripts/finetune_grounding_rl.py as a module without a package."""
    if not _SCRIPT.exists():
        pytest.skip(f"{_SCRIPT} not present")
    spec = importlib.util.spec_from_file_location("finetune_grounding_rl", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["finetune_grounding_rl"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------


def test_parse_args_minimal_required() -> None:
    rl = _load_rl_module()
    args = rl.parse_args(["--train-data", "train.jsonl"])
    assert args.train_data == "train.jsonl"
    # Defaults
    assert args.num_rollouts == 8
    assert args.epochs == 1
    assert args.sft_adapter is None
    assert args.prm_checkpoint is None


def test_parse_args_missing_train_data_fails() -> None:
    rl = _load_rl_module()
    with pytest.raises(SystemExit):
        rl.parse_args([])


def test_parse_args_accepts_all_rl_flags() -> None:
    rl = _load_rl_module()
    args = rl.parse_args(
        [
            "--train-data",
            "train.jsonl",
            "--model",
            "ByteDance-Seed/UI-TARS-1.5-7B",
            "--sft-adapter",
            "/tmp/sft/merged",
            "--prm-checkpoint",
            "/tmp/prm/prm_checkpoint.pt",
            "--output-dir",
            "/tmp/rl-out",
            "--num-rollouts",
            "4",
            "--grpo-beta",
            "0.1",
            "--step-reward-weight",
            "0.25",
            "--tolerance",
            "0.03",
            "--epochs",
            "2",
            "--lr",
            "5e-6",
            "--batch-size",
            "1",
            "--grad-accum",
            "4",
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
            "--seed",
            "7",
        ]
    )
    assert args.sft_adapter == "/tmp/sft/merged"
    assert args.prm_checkpoint == "/tmp/prm/prm_checkpoint.pt"
    assert args.num_rollouts == 4
    assert args.grpo_beta == pytest.approx(0.1)
    assert args.step_reward_weight == pytest.approx(0.25)
    assert args.tolerance == pytest.approx(0.03)
    assert args.epochs == 2
    assert args.lr == pytest.approx(5e-6)
    assert args.grad_accum == 4
    assert args.lora_r == 8
    assert args.lora_alpha == 16
    assert args.seed == 7


def test_require_sft_adapter_flag_is_accepted() -> None:
    rl = _load_rl_module()
    args = rl.parse_args(["--train-data", "train.jsonl", "--require-sft-adapter"])
    assert args.require_sft_adapter is True


# ---------------------------------------------------------------------------
# Reward-fn plumbing — exercise the factory without trl or torch
# ---------------------------------------------------------------------------


def test_outcome_reward_in_tolerance() -> None:
    rl = _load_rl_module()
    assert rl.outcome_reward("<point>0.50 0.50</point>", (0.50, 0.50), 0.05) == 1.0
    assert rl.outcome_reward("<point>0.50 0.50</point>", (0.80, 0.80), 0.05) == 0.0
    # No GT → zero reward, no crash.
    assert rl.outcome_reward("<point>0.50 0.50</point>", None, 0.05) == 0.0
    # Unparseable completion → zero reward.
    assert rl.outcome_reward("garbage", (0.5, 0.5), 0.05) == 0.0


def test_parse_action_steps_single_point() -> None:
    rl = _load_rl_module()
    steps = rl.parse_action_steps("<point>0.4 0.6</point>")
    assert len(steps) == 1
    assert "<point>" in steps[0]


def test_parse_action_steps_multi_point() -> None:
    rl = _load_rl_module()
    steps = rl.parse_action_steps(
        "<point>0.1 0.2</point> some text <point>0.3 0.4</point>"
    )
    assert len(steps) == 2


def test_parse_action_steps_empty() -> None:
    rl = _load_rl_module()
    assert rl.parse_action_steps("") == []


def test_reward_fn_handles_missing_prm_gracefully() -> None:
    rl = _load_rl_module()
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "file:///no/such/img.png"},
                    {"type": "text", "text": "find the button"},
                ],
            },
            {"role": "assistant", "content": "<point>0.5 0.5</point>"},
        ]
    }
    fn = rl.make_reward_fn(
        sample_index={"find the button": sample},
        prm_inferencer=None,
        tolerance=0.05,
        step_reward_weight=0.5,
    )
    rewards = fn(
        prompts=["find the button"],
        completions=["<point>0.50 0.50</point>"],
    )
    assert rewards == [1.0]
