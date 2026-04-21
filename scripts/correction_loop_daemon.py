#!/usr/bin/env python3
"""Correction-loop daemon scaffold.

Plan §13 architectural call E: a long-running service, supervised by
the qontinui supervisor, that watches the VGA correction log + PG
metrics, decides when to retrain the grounding model, runs the shadow
evaluation, and gates the llama-swap config swap on the per-domain
regression check.

This file is intentionally a **scaffold** — the tick loop and gate
evaluation are real, but the retrain and llama-swap-swap actions are
stubbed with TODO markers. The next session wires them up.

Activation knobs (env vars)
---------------------------
``QONTINUI_VGA_CORRECTIONS_DIR``
    Source correction log directory. Default
    ``datasets/vga-corrections``.

``QONTINUI_VGA_RETRAIN_PER_DOMAIN_BUDGET`` (int, default 200)
    A single ``target_process`` reaching this correction count triggers
    a retrain attempt (plan §13 v6-retrain-trigger).

``QONTINUI_VGA_RETRAIN_AGGREGATE_BUDGET`` (int, default 500)
    Total correction count across all domains that triggers retrain.

``QONTINUI_VGA_AUTO_RETRAIN``
    When ``"true"``, the gate decision kicks off a retrain. When absent
    or ``"false"``, the daemon only *logs the intent*. Default
    ``"false"``.

``QONTINUI_VGA_AUTO_SHIP``
    When ``"true"`` AND the shadow eval gate passes, the daemon rewrites
    the llama-swap config to point the shipping alias at the new model.
    Still stubbed; prints what would be written. Default ``"false"``.

``QONTINUI_VGA_TICK_SECONDS``
    How often to re-evaluate gate conditions. Default 300s (5 min).

Run
---
``python -m qontinui_finetune.correction_loop_daemon`` (when the
``qontinui_finetune`` package is on the path) or ``python
scripts/correction_loop_daemon.py``.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("vga.correction_loop_daemon")

_DEFAULT_CORRECTIONS_DIR = Path("datasets/vga-corrections")
_ENV_CORRECTIONS_DIR = "QONTINUI_VGA_CORRECTIONS_DIR"
_ENV_PER_DOMAIN_BUDGET = "QONTINUI_VGA_RETRAIN_PER_DOMAIN_BUDGET"
_ENV_AGGREGATE_BUDGET = "QONTINUI_VGA_RETRAIN_AGGREGATE_BUDGET"
_ENV_AUTO_RETRAIN = "QONTINUI_VGA_AUTO_RETRAIN"
_ENV_AUTO_SHIP = "QONTINUI_VGA_AUTO_SHIP"
_ENV_TICK_SECONDS = "QONTINUI_VGA_TICK_SECONDS"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DaemonConfig:
    corrections_dir: Path
    per_domain_budget: int
    aggregate_budget: int
    auto_retrain: bool
    auto_ship: bool
    tick_seconds: int

    @classmethod
    def from_env(cls) -> DaemonConfig:
        corrections_dir = Path(
            os.environ.get(_ENV_CORRECTIONS_DIR, str(_DEFAULT_CORRECTIONS_DIR))
        )
        return cls(
            corrections_dir=corrections_dir,
            per_domain_budget=int(os.environ.get(_ENV_PER_DOMAIN_BUDGET, "200")),
            aggregate_budget=int(os.environ.get(_ENV_AGGREGATE_BUDGET, "500")),
            auto_retrain=_parse_bool_env(_ENV_AUTO_RETRAIN, default=False),
            auto_ship=_parse_bool_env(_ENV_AUTO_SHIP, default=False),
            tick_seconds=int(os.environ.get(_ENV_TICK_SECONDS, "300")),
        )


def _parse_bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "y")


# ---------------------------------------------------------------------------
# Correction stats
# ---------------------------------------------------------------------------


@dataclass
class CorrectionStats:
    total: int
    per_target_process: dict[str, int]

    @property
    def max_domain_count(self) -> int:
        if not self.per_target_process:
            return 0
        return max(self.per_target_process.values())


def _read_correction_stats(corrections_dir: Path) -> CorrectionStats:
    """Count entries without loading all of them into memory.

    Mirrors ``CorrectionLogger.stats()``. Reimplemented here rather than
    imported so the daemon can run without the ``qontinui`` package on
    sys.path (useful for a minimal supervisor-deployed venv).
    """
    jsonl = corrections_dir / "corrections.jsonl"
    if not jsonl.exists():
        return CorrectionStats(total=0, per_target_process={})

    total = 0
    per_target: dict[str, int] = {}
    with jsonl.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            tp = entry.get("target_process", "unknown")
            per_target[tp] = per_target.get(tp, 0) + 1
    return CorrectionStats(total=total, per_target_process=per_target)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    stats: CorrectionStats


def _evaluate_retrain_gate(
    stats: CorrectionStats, config: DaemonConfig
) -> RetrainDecision:
    """Apply the plan §13 trigger rule.

    Any one domain ≥ per_domain_budget OR aggregate ≥ aggregate_budget.
    """
    if stats.max_domain_count >= config.per_domain_budget:
        top_domain = max(
            stats.per_target_process.items(), key=lambda kv: kv[1]
        )
        return RetrainDecision(
            should_retrain=True,
            reason=(
                f"domain-budget: {top_domain[0]}={top_domain[1]} "
                f"≥ {config.per_domain_budget}"
            ),
            stats=stats,
        )
    if stats.total >= config.aggregate_budget:
        return RetrainDecision(
            should_retrain=True,
            reason=f"aggregate-budget: total={stats.total} ≥ {config.aggregate_budget}",
            stats=stats,
        )
    return RetrainDecision(
        should_retrain=False,
        reason=(
            f"below-thresholds: total={stats.total} "
            f"max_domain={stats.max_domain_count}"
        ),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Side-effect stubs — the real work lands in a future session
# ---------------------------------------------------------------------------


def _trigger_retrain(config: DaemonConfig, decision: RetrainDecision) -> None:
    """Kick off a v6+ training run.

    TODO(milestone c.3): real implementation:
        1. Run ``scripts/export_corrections_to_vlm_sft.py`` with
           ``--corrections-jsonl <config.corrections_dir>/corrections.jsonl``.
        2. Mix the resulting ``vlm_train.jsonl`` with the v5 balanced
           set per the 90/10 rule (plan §5 milestone c.3).
        3. Invoke ``scripts/finetune_grounding_lora.py`` with the
           combined dataset and a bumped output-dir.
        4. Apply ``scripts/patch_merged_for_vllm.py`` to the merged
           checkpoint.
        5. Kick off the shadow eval.

    For now: print the intent and move on.
    """
    if not config.auto_retrain:
        logger.info(
            "RETRAIN INTENT (QONTINUI_VGA_AUTO_RETRAIN=false): %s",
            decision.reason,
        )
        return

    logger.warning(
        "RETRAIN TRIGGER fired (reason=%s) — stub; not yet implemented",
        decision.reason,
    )
    # TODO: actually invoke the retrain pipeline.


def _swap_llama_swap_config(
    config: DaemonConfig, shadow_report: dict[str, Any]
) -> None:
    """Rewrite llama-swap config.yaml to ship the candidate model.

    TODO(milestone c.4): real implementation:
        1. Load ``qontinui/docker/llama-swap/config.yaml``.
        2. Point the shipping alias (``grounding-v5`` / ``grounding``)
           at the candidate-model entry.
        3. Fsync + atomic-replace (see cc-switch atomic-writes pattern,
           memory note ``proj_cc_switch_patterns.md``).
        4. Send SIGHUP to llama-swap.

    For now: print what would change.
    """
    if not config.auto_ship:
        logger.info(
            "SHIP INTENT (QONTINUI_VGA_AUTO_SHIP=false): "
            "gate passed, would swap alias to candidate model."
        )
        return

    logger.warning(
        "SHIP TRIGGER fired — stub; not yet implemented. shadow_report=%s",
        json.dumps(shadow_report, sort_keys=True),
    )


def _run_shadow_eval(candidate_model: str) -> dict[str, Any] | None:
    """Scaffold call into shadow_eval — returns None on any import error.

    Uses the public ``run_shadow_eval`` API from
    ``qontinui_train.evaluation.shadow_eval``.
    """
    try:
        from qontinui_train.evaluation.shadow_eval import run_shadow_eval
    except ImportError as exc:
        logger.warning("shadow_eval not importable: %s", exc)
        return None

    pg_url = os.environ.get("QONTINUI_PG_URL")
    if not pg_url:
        logger.info(
            "QONTINUI_PG_URL not set; skipping shadow eval step of the daemon"
        )
        return None

    report = run_shadow_eval(
        pg_url=pg_url,
        candidate_model=candidate_model,
        baseline_model=os.environ.get(
            "QONTINUI_VGA_BASELINE_MODEL", "qontinui-grounding-v5"
        ),
        api_base=os.environ.get(
            "QONTINUI_VGA_API_BASE", "http://localhost:5800/v1"
        ),
    )
    return report.to_dict()


# ---------------------------------------------------------------------------
# Tick loop
# ---------------------------------------------------------------------------


_shutdown_requested = False


def _handle_shutdown(signum: int, _frame: Any) -> None:  # noqa: D401
    global _shutdown_requested
    logger.info("correction_loop_daemon: received signal %d; shutting down", signum)
    _shutdown_requested = True


def tick(config: DaemonConfig) -> None:
    """One iteration of the daemon loop — isolated for unit tests."""
    stats = _read_correction_stats(config.corrections_dir)
    decision = _evaluate_retrain_gate(stats, config)

    logger.info(
        "tick @%s total=%d per_domain=%s decision=%s reason=%s",
        datetime.now(UTC).isoformat(),
        stats.total,
        dict(stats.per_target_process),
        decision.should_retrain,
        decision.reason,
    )

    if not decision.should_retrain:
        return

    _trigger_retrain(config, decision)

    # The shadow eval + ship step is only meaningful once a candidate
    # model exists. During milestone (c) phase 2 we'll advance to a
    # point where the pipeline writes a candidate tag we can read here.
    candidate_model = os.environ.get("QONTINUI_VGA_CANDIDATE_MODEL")
    if candidate_model:
        shadow_report = _run_shadow_eval(candidate_model)
        if shadow_report and shadow_report.get("overall_gate_pass"):
            _swap_llama_swap_config(config, shadow_report)
        else:
            logger.info(
                "Shadow eval did not pass the ship gate (report=%s)",
                json.dumps(shadow_report or {}, sort_keys=True),
            )


def run_forever(config: DaemonConfig) -> int:
    """Long-running entry point. Returns an exit code on shutdown."""
    logger.info(
        "correction_loop_daemon starting: corrections_dir=%s tick=%ss "
        "per_domain=%d aggregate=%d auto_retrain=%s auto_ship=%s",
        config.corrections_dir,
        config.tick_seconds,
        config.per_domain_budget,
        config.aggregate_budget,
        config.auto_retrain,
        config.auto_ship,
    )

    signal.signal(signal.SIGINT, _handle_shutdown)
    try:
        signal.signal(signal.SIGTERM, _handle_shutdown)
    except (AttributeError, ValueError):
        # SIGTERM unavailable on Windows in some contexts — ignore.
        pass

    while not _shutdown_requested:
        try:
            tick(config)
        except Exception:
            logger.exception("tick raised; continuing")
        # Chunked sleep so shutdown is responsive.
        for _ in range(max(1, config.tick_seconds)):
            if _shutdown_requested:
                break
            time.sleep(1)

    logger.info("correction_loop_daemon exited cleanly")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one tick and exit — useful for smoke tests and cron.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config = DaemonConfig.from_env()

    if args.once:
        tick(config)
        return 0

    return run_forever(config)


if __name__ == "__main__":
    sys.exit(main())
