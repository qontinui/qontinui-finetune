#!/usr/bin/env python3
"""Correction-loop daemon — the self-healing retrain + ship pipeline.

Plan §13 architectural call E: a long-running service, supervised by
the qontinui supervisor, that watches the VGA correction log + PG
metrics, decides when to retrain the grounding model, runs the shadow
evaluation, and gates the llama-swap config swap on the per-domain
regression check.

State machine
-------------
The daemon's tick loop is a simple three-state machine keyed off a
lockfile at ``<corrections_dir>/.retrain.lock``:

- ``idle``:      no lockfile. If the gate (plan §13) fires and
                 ``QONTINUI_VGA_AUTO_RETRAIN=1``, spawn a detached
                 retrain subprocess, write the lockfile, transition to
                 ``training``.
- ``training``:  lockfile present, PID still alive. Nothing to do —
                 training will take hours.
- ``ready``:     lockfile present, PID dead. Training finished. Apply
                 the post-merge vLLM compat patch. If the candidate
                 dir looks healthy AND ``QONTINUI_VGA_AUTO_SHIP=1``,
                 run shadow-eval, enforce the per-domain +5pp gate,
                 atomically swap the llama-swap config, and move the
                 candidate into the live model path. Clear the lock.

Activation knobs (env vars)
---------------------------
``QONTINUI_VGA_CORRECTIONS_DIR``
    Source correction log directory. Default ``datasets/vga-corrections``.

``QONTINUI_VGA_MODELS_DIR``
    Where merged models live (bind-mounted into llama-swap as ``/models``).
    Default ``D:/qontinui-root/models`` on Windows, ``/data/models``
    otherwise.

``QONTINUI_VGA_RETRAIN_PER_DOMAIN_BUDGET`` (int, default 200)
    A single ``target_process`` reaching this correction count triggers
    a retrain attempt (plan §13 v6-retrain-trigger).

``QONTINUI_VGA_RETRAIN_AGGREGATE_BUDGET`` (int, default 500)
    Total correction count across all domains that triggers retrain.

``QONTINUI_VGA_AUTO_RETRAIN``
    When ``"true"``/``"1"``, the gate decision kicks off a retrain.
    Default ``"false"`` (log intent only).

``QONTINUI_VGA_AUTO_SHIP``
    When ``"true"``/``"1"`` AND the shadow-eval gate passes, the daemon
    rewrites the llama-swap config to point the shipping alias at the
    new model. Default ``"false"`` (log intent only).

``QONTINUI_VGA_TICK_SECONDS``
    How often to re-evaluate gate conditions. Default 300s (5 min).

``QONTINUI_PG_URL``
    Postgres connection string for shadow-eval. Required when
    ``QONTINUI_VGA_AUTO_SHIP=1``.

``QONTINUI_VGA_BASELINE_MODEL`` (default ``qontinui-grounding-v5``).

``QONTINUI_VGA_CANDIDATE_VERSION`` (default ``v6``).
    Appended to ``qontinui-grounding-`` to form the candidate model
    served-model-name.

``QONTINUI_VGA_API_BASE`` (default ``http://localhost:8100/v1``).

``QONTINUI_VGA_LLAMA_SWAP_CONTAINER`` (default ``llama-swap-llama-swap-1``).

``QONTINUI_VGA_LLAMA_SWAP_CONFIG``
    Absolute path to the on-host llama-swap ``config.yaml``.
    Default ``D:/qontinui-root/qontinui/docker/llama-swap/config.yaml``
    on Windows, ``/opt/qontinui-root/qontinui/docker/llama-swap/config.yaml``
    otherwise.

Run
---
``python -m qontinui_finetune.correction_loop_daemon --once`` (one tick)
or ``python scripts/correction_loop_daemon.py --watch`` (long-running).

Operational notes
-----------------
- The training subprocess is detached: SIGINT-ing the daemon does NOT
  kill an in-flight training run. Use ``kill <pid-from-lockfile>`` if
  you need to abort training.
- The lockfile is a JSON blob keyed by PID + start time + output-dir.
  If the daemon crashes mid-retrain, the next daemon tick will see a
  live PID and keep waiting. If the PID is dead but the output dir is
  incomplete (no ``config.json``), the lock is treated as a failed
  retrain and cleared.
- ``.ship-history.jsonl`` at the top of ``corrections_dir`` is an
  append-only log of every ship decision (pass or block). Use it to
  audit automatic deployments.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("vga.correction_loop_daemon")

_DEFAULT_CORRECTIONS_DIR = Path("datasets/vga-corrections")
_ENV_CORRECTIONS_DIR = "QONTINUI_VGA_CORRECTIONS_DIR"
_ENV_MODELS_DIR = "QONTINUI_VGA_MODELS_DIR"
_ENV_PER_DOMAIN_BUDGET = "QONTINUI_VGA_RETRAIN_PER_DOMAIN_BUDGET"
_ENV_AGGREGATE_BUDGET = "QONTINUI_VGA_RETRAIN_AGGREGATE_BUDGET"
_ENV_AUTO_RETRAIN = "QONTINUI_VGA_AUTO_RETRAIN"
_ENV_AUTO_SHIP = "QONTINUI_VGA_AUTO_SHIP"
_ENV_TICK_SECONDS = "QONTINUI_VGA_TICK_SECONDS"
_ENV_BASELINE_MODEL = "QONTINUI_VGA_BASELINE_MODEL"
_ENV_CANDIDATE_VERSION = "QONTINUI_VGA_CANDIDATE_VERSION"
_ENV_API_BASE = "QONTINUI_VGA_API_BASE"
_ENV_LLAMA_SWAP_CONTAINER = "QONTINUI_VGA_LLAMA_SWAP_CONTAINER"
_ENV_LLAMA_SWAP_CONFIG = "QONTINUI_VGA_LLAMA_SWAP_CONFIG"
_ENV_PG_URL = "QONTINUI_PG_URL"

_LOCKFILE_NAME = ".retrain.lock"
_SHIP_HISTORY_NAME = ".ship-history.jsonl"

_SHIP_GATE_DELTA_PP = 5.0


# ---------------------------------------------------------------------------
# Path defaults — platform-aware
# ---------------------------------------------------------------------------


def _default_models_dir() -> Path:
    if sys.platform == "win32":
        return Path("D:/qontinui-root/models")
    return Path("/data/qontinui-root/models")


def _default_llama_swap_config() -> Path:
    if sys.platform == "win32":
        return Path("D:/qontinui-root/qontinui/docker/llama-swap/config.yaml")
    return Path("/opt/qontinui-root/qontinui/docker/llama-swap/config.yaml")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DaemonConfig:
    corrections_dir: Path
    models_dir: Path
    per_domain_budget: int
    aggregate_budget: int
    auto_retrain: bool
    auto_ship: bool
    tick_seconds: int
    baseline_model: str
    candidate_version: str
    api_base: str
    llama_swap_container: str
    llama_swap_config: Path
    pg_url: str | None

    @property
    def candidate_model_name(self) -> str:
        return f"qontinui-grounding-{self.candidate_version}"

    @property
    def candidate_output_dir(self) -> Path:
        # Candidate stays in a *-candidate suffix until the ship step
        # atomically renames it to the live path.
        return self.models_dir / f"{self.candidate_model_name}-candidate"

    @property
    def candidate_live_dir(self) -> Path:
        return self.models_dir / self.candidate_model_name

    @classmethod
    def from_env(cls) -> DaemonConfig:
        corrections_dir = Path(
            os.environ.get(_ENV_CORRECTIONS_DIR, str(_DEFAULT_CORRECTIONS_DIR))
        )
        models_dir = Path(os.environ.get(_ENV_MODELS_DIR, str(_default_models_dir())))
        llama_swap_config = Path(
            os.environ.get(_ENV_LLAMA_SWAP_CONFIG, str(_default_llama_swap_config()))
        )
        return cls(
            corrections_dir=corrections_dir,
            models_dir=models_dir,
            per_domain_budget=int(os.environ.get(_ENV_PER_DOMAIN_BUDGET, "200")),
            aggregate_budget=int(os.environ.get(_ENV_AGGREGATE_BUDGET, "500")),
            auto_retrain=_parse_bool_env(_ENV_AUTO_RETRAIN, default=False),
            auto_ship=_parse_bool_env(_ENV_AUTO_SHIP, default=False),
            tick_seconds=int(os.environ.get(_ENV_TICK_SECONDS, "300")),
            baseline_model=os.environ.get(_ENV_BASELINE_MODEL, "qontinui-grounding-v5"),
            candidate_version=os.environ.get(_ENV_CANDIDATE_VERSION, "v6"),
            api_base=os.environ.get(_ENV_API_BASE, "http://localhost:8100/v1"),
            llama_swap_container=os.environ.get(
                _ENV_LLAMA_SWAP_CONTAINER, "llama-swap-llama-swap-1"
            ),
            llama_swap_config=llama_swap_config,
            pg_url=os.environ.get(_ENV_PG_URL),
        )


def _parse_bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "y")


# ---------------------------------------------------------------------------
# Correction stats + retrain gate
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
    """Count entries without loading all of them into memory."""
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


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    stats: CorrectionStats


def _evaluate_retrain_gate(
    stats: CorrectionStats, config: DaemonConfig
) -> RetrainDecision:
    """Apply the plan §13 trigger rule."""
    if stats.max_domain_count >= config.per_domain_budget:
        top_domain = max(stats.per_target_process.items(), key=lambda kv: kv[1])
        return RetrainDecision(
            should_retrain=True,
            reason=(
                f"domain-budget: {top_domain[0]}={top_domain[1]} "
                f">= {config.per_domain_budget}"
            ),
            stats=stats,
        )
    if stats.total >= config.aggregate_budget:
        return RetrainDecision(
            should_retrain=True,
            reason=(
                f"aggregate-budget: total={stats.total} >= {config.aggregate_budget}"
            ),
            stats=stats,
        )
    return RetrainDecision(
        should_retrain=False,
        reason=(
            f"below-thresholds: total={stats.total} max_domain={stats.max_domain_count}"
        ),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Lockfile + PID liveness
# ---------------------------------------------------------------------------


def _lockfile_path(config: DaemonConfig) -> Path:
    return config.corrections_dir / _LOCKFILE_NAME


def _is_pid_alive(pid: int) -> bool:
    """Return True if ``pid`` is running. Cross-platform, dep-free."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Use the Win32 API: OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION.
        # If the handle is non-null we can GetExitCodeProcess; STILL_ACTIVE
        # (259) means the process is running.
        try:
            import ctypes
            from ctypes import wintypes
        except ImportError:  # pragma: no cover - ctypes always present on Windows
            return False

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't own it — still "alive".
            return True
        return True


def _read_lockfile(config: DaemonConfig) -> dict[str, Any] | None:
    path = _lockfile_path(config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Corrupt lockfile at %s: %s; treating as absent", path, exc)
        return None


def _write_lockfile(config: DaemonConfig, payload: dict[str, Any]) -> None:
    path = _lockfile_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _clear_lockfile(config: DaemonConfig) -> None:
    path = _lockfile_path(config)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _check_retrain_lock(config: DaemonConfig) -> dict[str, Any] | None:
    """Return the lockfile contents if training is (or was) in progress.

    Adds a computed ``pid_alive`` field so callers can distinguish
    "still training" from "process died, need to reconcile".
    """
    payload = _read_lockfile(config)
    if payload is None:
        return None
    pid = int(payload.get("pid", 0))
    payload["pid_alive"] = _is_pid_alive(pid)
    return payload


# ---------------------------------------------------------------------------
# Retrain action
# ---------------------------------------------------------------------------


@dataclass
class RetrainStatus:
    spawned: bool
    pid: int | None = None
    log_path: Path | None = None
    dataset_dir: Path | None = None
    output_dir: Path | None = None
    reason: str | None = None
    started_at: datetime | None = None


def _run_exporter_sync(config: DaemonConfig, dataset_dir: Path) -> None:
    """Run export_corrections_to_vlm_sft.py as a blocking subprocess.

    Exporter is fast (seconds). We run it inline so the lockfile + the
    detached trainer both see a complete dataset.
    """
    scripts_dir = Path(__file__).resolve().parent
    exporter = scripts_dir / "export_corrections_to_vlm_sft.py"
    if not exporter.exists():
        raise FileNotFoundError(f"Exporter not found at {exporter}")

    corrections_jsonl = config.corrections_dir / "corrections.jsonl"
    cmd = [
        sys.executable,
        str(exporter),
        "--corrections-jsonl",
        str(corrections_jsonl),
        "--output-dir",
        str(dataset_dir),
        "--include-private",
        "false",
        "--split",
        "0.8,0.1,0.1",
        "--seed",
        "42",
    ]
    logger.info("running exporter: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"exporter failed (rc={result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _spawn_trainer_detached(
    config: DaemonConfig,
    dataset_dir: Path,
    output_dir: Path,
    log_path: Path,
) -> int:
    """Spawn finetune_grounding_lora.py as a detached subprocess.

    Returns the spawned PID. Training runs independently of the daemon —
    killing the daemon does NOT kill the trainer.
    """
    scripts_dir = Path(__file__).resolve().parent
    trainer = scripts_dir / "finetune_grounding_lora.py"
    if not trainer.exists():
        raise FileNotFoundError(f"Trainer not found at {trainer}")

    cmd = [
        sys.executable,
        str(trainer),
        "--train-data",
        str(dataset_dir / "vlm_train.jsonl"),
        "--val-data",
        str(dataset_dir / "vlm_val.jsonl"),
        "--model",
        "ByteDance-Seed/UI-TARS-1.5-7B",
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--lr",
        "2e-4",
        "--batch-size",
        "1",
        "--grad-accum",
        "4",
        "--lora-r",
        "16",
        "--lora-alpha",
        "32",
        "--seed",
        "42",
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("a", encoding="utf-8")
    log_fp.write(
        f"\n=== retrain started at {datetime.now(UTC).isoformat()} ===\n"
        f"cmd: {' '.join(cmd)}\n\n"
    )
    log_fp.flush()

    popen_kwargs: dict[str, Any] = {
        "stdout": log_fp,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "cwd": str(scripts_dir.parent),  # repo root
    }
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — survives daemon exit.
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        popen_kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        popen_kwargs["close_fds"] = True
    else:
        # start_new_session puts the child in its own session, so SIGINT
        # to the daemon's terminal doesn't also SIGINT the trainer.
        popen_kwargs["start_new_session"] = True
        popen_kwargs["close_fds"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603
    logger.info("trainer spawned: pid=%d log=%s", proc.pid, log_path)
    return proc.pid


def _trigger_retrain(config: DaemonConfig, decision: RetrainDecision) -> RetrainStatus:
    """Orchestrate an async v6 retrain. Returns immediately.

    Steps:
        1. Refuse if another retrain is already in progress.
        2. Run exporter synchronously.
        3. Spawn trainer as DETACHED subprocess.
        4. Write lockfile with PID + output_dir + log_path.
        5. Return RetrainStatus.

    Subsequent daemon ticks inspect the lockfile + PID liveness and, on
    death, apply post-merge compat patches and (optionally) ship.
    """
    if not config.auto_retrain:
        logger.info(
            "RETRAIN INTENT (QONTINUI_VGA_AUTO_RETRAIN=false): %s",
            decision.reason,
        )
        return RetrainStatus(
            spawned=False,
            reason="auto_retrain disabled",
        )

    existing = _check_retrain_lock(config)
    if existing and existing.get("pid_alive"):
        logger.warning(
            "retrain already in progress: pid=%s started_at=%s; skipping",
            existing.get("pid"),
            existing.get("started_at"),
        )
        return RetrainStatus(
            spawned=False,
            reason="retrain already in progress",
            pid=int(existing.get("pid") or 0) or None,
        )

    ts = datetime.now(UTC)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
    dataset_dir = config.corrections_dir / f"vlm-sft-{ts_str}"
    output_dir = config.candidate_output_dir
    log_path = config.corrections_dir / "logs" / f"retrain-{ts_str}.log"

    # 2. Run exporter.
    try:
        _run_exporter_sync(config, dataset_dir)
    except Exception as exc:
        logger.exception("exporter failed; aborting retrain")
        return RetrainStatus(
            spawned=False,
            reason=f"exporter failed: {exc}",
            dataset_dir=dataset_dir,
        )

    # 3. Spawn trainer detached.
    try:
        pid = _spawn_trainer_detached(config, dataset_dir, output_dir, log_path)
    except Exception as exc:
        logger.exception("trainer spawn failed")
        return RetrainStatus(
            spawned=False,
            reason=f"trainer spawn failed: {exc}",
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            log_path=log_path,
        )

    # 4. Write lockfile.
    _write_lockfile(
        config,
        {
            "pid": pid,
            "started_at": ts.isoformat(),
            "reason": decision.reason,
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "candidate_model": config.candidate_model_name,
            "status": "training",
        },
    )

    return RetrainStatus(
        spawned=True,
        pid=pid,
        log_path=log_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        reason=decision.reason,
        started_at=ts,
    )


def _apply_post_merge_patches(merged_dir: Path) -> bool:
    """Apply the two transformers-5 → vLLM compat patches.

    Returns True on success. Logs + returns False on failure so the
    caller can record a ship-blocked reason.
    """
    scripts_dir = Path(__file__).resolve().parent
    patcher = scripts_dir / "patch_merged_for_vllm.py"
    if not patcher.exists():
        logger.error("patch_merged_for_vllm.py not found at %s", patcher)
        return False

    cmd = [sys.executable, str(patcher), str(merged_dir)]
    logger.info("applying vLLM compat patches: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error(
            "patch_merged_for_vllm failed (rc=%d)\nstdout: %s\nstderr: %s",
            result.returncode,
            result.stdout,
            result.stderr,
        )
        return False
    logger.info("patches applied: %s", result.stdout.strip())
    return True


# ---------------------------------------------------------------------------
# Ship action
# ---------------------------------------------------------------------------


@dataclass
class ShipStatus:
    shipped: bool
    timestamp: datetime | None = None
    swapped_from: str | None = None
    swapped_to: str | None = None
    blocking_domains: list[tuple[str, float]] = field(default_factory=list)
    reason: str | None = None
    eval_report_path: Path | None = None


def _evaluate_ship_gate(report: Any) -> tuple[bool, list[tuple[str, float]]]:
    """Enforce the plan §13 ship gate on a ShadowEvalReport.

    Returns (pass, blocking_domains). ``blocking_domains`` lists every
    (domain, delta_pp) that failed the +5pp floor OR regressed.

    ``report`` is duck-typed: either a ShadowEvalReport with
    ``per_domain_results`` mapping to objects with ``delta_pp`` and
    ``regression`` attrs, or a dict from ``to_dict()``.
    """
    per_domain: dict[str, Any]
    if hasattr(report, "per_domain_results"):
        per_domain = report.per_domain_results
    else:
        per_domain = (report or {}).get("per_domain_results", {})

    if not per_domain:
        return False, []

    blocking: list[tuple[str, float]] = []
    for domain, r in per_domain.items():
        if hasattr(r, "delta_pp"):
            delta = float(r.delta_pp)
            regression = bool(r.regression)
        else:
            delta = float(r.get("delta_pp", 0.0))
            regression = bool(r.get("regression", False))
        if regression or delta < _SHIP_GATE_DELTA_PP:
            blocking.append((domain, delta))

    return (len(blocking) == 0), blocking


def _atomic_swap_llama_swap_config(
    config: DaemonConfig,
) -> None:
    """Atomically rewrite llama-swap config.yaml to ship the candidate.

    Mutations:
        1. Uncomment the v6 block (if present as commented placeholder).
        2. Ensure the ``qontinui-grounding`` bare alias is attached to
           the candidate and removed from baseline.
        3. Point candidate block's ``--model`` at the live candidate
           path (``/models/qontinui-grounding-v6``).

    Write-then-rename for atomicity: any reader always sees either the
    old or the new full file, never a half-written YAML.
    """
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML required for config swap. Install with: pip install pyyaml"
        ) from exc

    cfg_path = config.llama_swap_config
    if not cfg_path.exists():
        raise FileNotFoundError(f"llama-swap config not found at {cfg_path}")

    raw = cfg_path.read_text(encoding="utf-8")

    candidate_block_name = config.candidate_model_name
    candidate_alias_versioned = f"grounding-{config.candidate_version}"

    # First pass: parse the current YAML so we know if the candidate
    # block already exists as real (not commented) content.
    parsed = yaml.safe_load(raw) or {}
    models = parsed.get("models", {})

    if candidate_block_name in models:
        # Candidate already in YAML; only need to swap the alias.
        data = parsed
        # Remove bare alias from baseline, add to candidate.
        baseline_entry = data["models"].get(config.baseline_model, {})
        baseline_aliases = list(baseline_entry.get("aliases", []))
        baseline_aliases = [a for a in baseline_aliases if a != "qontinui-grounding"]
        baseline_entry["aliases"] = baseline_aliases
        data["models"][config.baseline_model] = baseline_entry

        cand_entry = data["models"][candidate_block_name]
        cand_aliases = list(cand_entry.get("aliases", []))
        if "qontinui-grounding" not in cand_aliases:
            cand_aliases.append("qontinui-grounding")
        if candidate_alias_versioned not in cand_aliases:
            cand_aliases.append(candidate_alias_versioned)
        cand_entry["aliases"] = cand_aliases
        data["models"][candidate_block_name] = cand_entry

        new_text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    else:
        # Candidate block lives as a commented placeholder. Uncomment it,
        # then re-parse to swap aliases cleanly. We use a line-oriented
        # approach because PyYAML strips comments on round-trip.
        lines = raw.splitlines(keepends=True)
        out_lines: list[str] = []
        in_placeholder = False
        found_placeholder = False
        for line in lines:
            stripped_left = line.lstrip()
            # Recognise the placeholder start: a commented `<name>:` line.
            if not in_placeholder and stripped_left.startswith(
                f"# {candidate_block_name}:"
            ):
                in_placeholder = True
                found_placeholder = True
            if in_placeholder:
                # Continuation: any line that's indented-comment or blank.
                # The placeholder ends when we hit a line that is NOT a
                # comment AND is non-empty AND not inside the block.
                bare = line.rstrip("\r\n")
                if bare == "" or bare.lstrip().startswith("#"):
                    # Uncomment: strip the leading `# ` or `#`.
                    # Preserve indentation by keeping everything up to `#`.
                    idx = line.find("#")
                    if idx >= 0:
                        prefix = line[:idx]
                        rest = line[idx + 1 :]
                        if rest.startswith(" "):
                            rest = rest[1:]
                        out_lines.append(prefix + rest)
                    else:
                        out_lines.append(line)
                else:
                    # Non-comment, non-blank line ⇒ placeholder block
                    # finished. Emit this line verbatim.
                    in_placeholder = False
                    out_lines.append(line)
            else:
                out_lines.append(line)

        new_text = "".join(out_lines)

        if not found_placeholder:
            logger.warning(
                "no placeholder block for %s in config.yaml; appending a minimal entry",
                candidate_block_name,
            )
            # Append a minimal entry so YAML re-parse lands it.
            new_text = new_text.rstrip() + "\n\n"
            new_text += (
                f"  {candidate_block_name}:\n"
                f"    cmd: >\n"
                f"      python3 -m vllm.entrypoints.openai.api_server\n"
                f"      --model /models/{candidate_block_name}\n"
                f"      --served-model-name {candidate_block_name}\n"
                f"      --host 127.0.0.1\n"
                f"      --port ${{PORT}}\n"
                f"      --dtype bfloat16\n"
                f"      --max-model-len 8192\n"
                f"      --trust-remote-code\n"
                f'    proxy: "http://127.0.0.1:${{PORT}}"\n'
                f"    checkEndpoint: /health\n"
                f"    ttl: 300\n"
                f"    aliases:\n"
                f"      - {candidate_alias_versioned}\n"
            )

        # Re-parse and alias-swap on the uncommented text.
        data = yaml.safe_load(new_text) or {}
        if candidate_block_name not in data.get("models", {}):
            raise RuntimeError(
                f"after uncomment, {candidate_block_name} not parseable — "
                f"manual intervention needed on {cfg_path}"
            )

        # Remove bare alias from baseline.
        baseline_entry = data["models"].get(config.baseline_model, {})
        baseline_aliases = [
            a for a in baseline_entry.get("aliases", []) if a != "qontinui-grounding"
        ]
        baseline_entry["aliases"] = baseline_aliases
        data["models"][config.baseline_model] = baseline_entry

        # Add bare alias to candidate.
        cand_entry = data["models"][candidate_block_name]
        cand_aliases = list(cand_entry.get("aliases", []))
        if "qontinui-grounding" not in cand_aliases:
            cand_aliases.append("qontinui-grounding")
        if candidate_alias_versioned not in cand_aliases:
            cand_aliases.append(candidate_alias_versioned)
        cand_entry["aliases"] = cand_aliases
        data["models"][candidate_block_name] = cand_entry

        # Dump canonical YAML — comments will be lost, but the swap is
        # authoritative and the placeholder was already gone.
        new_text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)

    # Validate the YAML parses cleanly before committing.
    try:
        yaml.safe_load(new_text)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"new config.yaml failed to parse: {exc}") from exc

    # Atomic write-then-rename.
    new_path = cfg_path.with_name(cfg_path.name + ".new")
    new_path.write_text(new_text, encoding="utf-8")
    # fsync for durability on abrupt power loss; harmless otherwise.
    try:
        with new_path.open("rb") as fp:
            os.fsync(fp.fileno())
    except OSError:
        pass
    os.replace(new_path, cfg_path)


def _reload_llama_swap(config: DaemonConfig) -> str:
    """Force llama-swap to pick up the new config.

    llama-swap v201 (as shipped in the current docker-compose) exposes:
        - POST /unload  — unloads all running models
        - GET  /running — lists live models
        - (no /reload or /restart HTTP endpoint)

    The daemon is NOT started with ``-watch-config``, so file changes
    alone do not reload. A full container restart is the only reliable
    way to pick up a new ``config.yaml``. This is heavy (~seconds of
    downtime), but ships happen rarely and are expected to be gated by
    the shadow-eval passing.

    Returns a short description of the reload path taken, for logging.
    """
    # Try docker restart of the llama-swap container.
    cmd = ["docker", "restart", config.llama_swap_container]
    logger.info("reloading llama-swap via docker restart: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker restart failed (rc={result.returncode})\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return f"docker restart {config.llama_swap_container}"


def _append_ship_history(
    config: DaemonConfig, status: ShipStatus, gate_report: Any
) -> None:
    """Append a decision record to ``.ship-history.jsonl``."""
    history_path = config.corrections_dir / _SHIP_HISTORY_NAME
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ts": (status.timestamp or datetime.now(UTC)).isoformat(),
        "shipped": status.shipped,
        "swapped_from": status.swapped_from,
        "swapped_to": status.swapped_to,
        "blocking_domains": [
            {"target_process": tp, "delta_pp": dp} for tp, dp in status.blocking_domains
        ],
        "reason": status.reason,
        "eval_report_path": (
            str(status.eval_report_path) if status.eval_report_path else None
        ),
    }
    if gate_report is not None:
        if hasattr(gate_report, "to_dict"):
            record["report"] = gate_report.to_dict()
        elif isinstance(gate_report, dict):
            record["report"] = gate_report

    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")))
        f.write("\n")


def _trigger_ship(config: DaemonConfig, candidate_dir: Path) -> ShipStatus:
    """Run shadow-eval + enforce gate + atomic swap llama-swap config.

    Strict gate: every per-domain delta must be ≥ +5pp AND no regression.
    """
    if not config.auto_ship:
        logger.info(
            "SHIP INTENT (QONTINUI_VGA_AUTO_SHIP=false): candidate ready at %s",
            candidate_dir,
        )
        return ShipStatus(shipped=False, reason="auto_ship disabled")

    if not config.pg_url:
        logger.error("QONTINUI_PG_URL not set; cannot run shadow-eval")
        return ShipStatus(shipped=False, reason="QONTINUI_PG_URL missing")

    # 1. Run shadow eval.
    try:
        from qontinui_train.evaluation.shadow_eval import run_shadow_eval
    except ImportError as exc:
        logger.error("shadow_eval not importable: %s", exc)
        return ShipStatus(shipped=False, reason=f"shadow_eval import: {exc}")

    try:
        report = run_shadow_eval(
            pg_url=config.pg_url,
            candidate_model=config.candidate_model_name,
            baseline_model=config.baseline_model,
            api_base=config.api_base,
        )
    except Exception as exc:
        logger.exception("shadow_eval run failed")
        return ShipStatus(shipped=False, reason=f"shadow_eval failed: {exc}")

    # Persist the eval report for auditing.
    ts = datetime.now(UTC)
    ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
    eval_report_path = config.corrections_dir / "logs" / f"shadow-eval-{ts_str}.json"
    eval_report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        eval_report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 — best-effort persist
        logger.exception("failed to persist eval report")

    # 2. Enforce per-domain +5pp gate.
    passed, blocking = _evaluate_ship_gate(report)
    if not passed:
        logger.warning(
            "ship gate blocked: %d domain(s) failed +5pp floor or regressed: %s",
            len(blocking),
            blocking,
        )
        status = ShipStatus(
            shipped=False,
            timestamp=ts,
            swapped_from=config.baseline_model,
            swapped_to=config.candidate_model_name,
            blocking_domains=blocking,
            reason="gate-blocked",
            eval_report_path=eval_report_path,
        )
        _append_ship_history(config, status, report)
        return status

    # 3. Atomic rename candidate-dir → live-dir. Do this BEFORE the
    # config swap so llama-swap sees the model at /models/<name>.
    try:
        live_dir = config.candidate_live_dir
        if live_dir.exists():
            # A prior shipped copy exists. Move it aside for manual cleanup.
            backup = live_dir.with_name(live_dir.name + f".pre-{ts_str}")
            os.replace(live_dir, backup)
            logger.info("existing live dir moved to %s", backup)
        os.replace(candidate_dir, live_dir)
        logger.info("candidate promoted: %s -> %s", candidate_dir, live_dir)
    except OSError as exc:
        logger.exception("failed to promote candidate dir")
        status = ShipStatus(
            shipped=False,
            timestamp=ts,
            reason=f"promote failed: {exc}",
            eval_report_path=eval_report_path,
        )
        _append_ship_history(config, status, report)
        return status

    # 4. Atomic swap of llama-swap config.yaml.
    try:
        _atomic_swap_llama_swap_config(config)
    except Exception as exc:
        logger.exception("llama-swap config swap failed")
        status = ShipStatus(
            shipped=False,
            timestamp=ts,
            reason=f"config swap failed: {exc}",
            eval_report_path=eval_report_path,
        )
        _append_ship_history(config, status, report)
        return status

    # 5. Force llama-swap to reload.
    try:
        reload_method = _reload_llama_swap(config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("llama-swap reload failed")
        status = ShipStatus(
            shipped=True,  # config is already swapped
            timestamp=ts,
            swapped_from=config.baseline_model,
            swapped_to=config.candidate_model_name,
            reason=f"config swapped but reload failed: {exc}",
            eval_report_path=eval_report_path,
        )
        _append_ship_history(config, status, report)
        return status

    logger.info(
        "ship complete: %s → %s via %s",
        config.baseline_model,
        config.candidate_model_name,
        reload_method,
    )
    status = ShipStatus(
        shipped=True,
        timestamp=ts,
        swapped_from=config.baseline_model,
        swapped_to=config.candidate_model_name,
        reason=f"reload-via:{reload_method}",
        eval_report_path=eval_report_path,
    )
    _append_ship_history(config, status, report)
    return status


# ---------------------------------------------------------------------------
# Tick loop
# ---------------------------------------------------------------------------


_shutdown_requested = False


def _handle_shutdown(signum: int, _frame: Any) -> None:
    global _shutdown_requested
    logger.info("correction_loop_daemon: received signal %d; shutting down", signum)
    _shutdown_requested = True


def _log_retrain_status(status: RetrainStatus) -> None:
    if status.spawned:
        logger.info(
            "retrain spawned: pid=%s log=%s dataset=%s output=%s",
            status.pid,
            status.log_path,
            status.dataset_dir,
            status.output_dir,
        )
    else:
        logger.info("retrain not spawned: %s", status.reason)


def _log_ship_status(status: ShipStatus) -> None:
    if status.shipped:
        logger.info(
            "ship SUCCESS: %s → %s (report=%s)",
            status.swapped_from,
            status.swapped_to,
            status.eval_report_path,
        )
    else:
        logger.info(
            "ship blocked: reason=%s blocking=%s",
            status.reason,
            status.blocking_domains,
        )


def tick(config: DaemonConfig) -> None:
    """One iteration of the daemon loop — isolated for unit tests."""
    stats = _read_correction_stats(config.corrections_dir)
    decision = _evaluate_retrain_gate(stats, config)
    lock = _check_retrain_lock(config)

    logger.info(
        "tick @%s total=%d per_domain=%s decision=%s reason=%s lock=%s pid_alive=%s",
        datetime.now(UTC).isoformat(),
        stats.total,
        dict(stats.per_target_process),
        decision.should_retrain,
        decision.reason,
        bool(lock),
        lock.get("pid_alive") if lock else None,
    )

    if lock is None:
        # State = idle. Gate fires → start retrain.
        if decision.should_retrain:
            status = _trigger_retrain(config, decision)
            _log_retrain_status(status)
        return

    # Lockfile present.
    if lock.get("pid_alive"):
        # State = training. Nothing to do.
        return

    # State = ready (or crashed).
    output_dir_str = lock.get("output_dir")
    if not output_dir_str:
        logger.warning("lockfile has no output_dir; clearing")
        _clear_lockfile(config)
        return

    output_dir = Path(output_dir_str)
    merged_dir = output_dir / "merged"

    # Has training finished successfully? We expect merged/config.json.
    merged_ready = (merged_dir / "config.json").exists()
    if not merged_ready:
        # Trainer died without producing a merged checkpoint. Clear and
        # let the gate re-trigger on the next tick (with fresh data).
        logger.warning(
            "trainer exited without merged checkpoint at %s; clearing lock",
            merged_dir,
        )
        _clear_lockfile(config)
        return

    # Apply vLLM compat patches (idempotent).
    patched = _apply_post_merge_patches(merged_dir)
    if not patched:
        logger.error(
            "post-merge patches failed; leaving candidate at %s for manual fix",
            merged_dir,
        )
        _clear_lockfile(config)
        return

    # Ready to ship.
    ship_status = _trigger_ship(config, merged_dir)
    _log_ship_status(ship_status)
    # Whether or not the ship succeeded, the training cycle is done.
    # A blocked ship is a signal to gather more corrections, not to
    # keep the lockfile around.
    _clear_lockfile(config)


def run_forever(config: DaemonConfig) -> int:
    """Long-running entry point. Returns an exit code on shutdown."""
    logger.info(
        "correction_loop_daemon starting: corrections_dir=%s models_dir=%s "
        "tick=%ss per_domain=%d aggregate=%d auto_retrain=%s auto_ship=%s",
        config.corrections_dir,
        config.models_dir,
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
        "--watch",
        action="store_true",
        help="Long-running mode (default when --once not specified).",
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

    # --watch (or default).
    return run_forever(config)


# Re-export the status dataclasses for tests.
__all__ = [
    "CorrectionStats",
    "DaemonConfig",
    "RetrainDecision",
    "RetrainStatus",
    "ShipStatus",
    "main",
    "run_forever",
    "tick",
]


if __name__ == "__main__":
    sys.exit(main())
