"""Tests for scripts/correction_loop_daemon.py.

Covers the retrain + ship state machine end-to-end with heavy mocking:
- subprocess.Popen is patched so no real trainer ever spawns
- shadow_eval.run_shadow_eval is patched so no real PG hit
- _reload_llama_swap is patched so no real docker restart
- _apply_post_merge_patches is patched so no real JSON edits

Split into three groups:
1. Gate evaluation (no side effects)
2. Retrain state transitions (lockfile lifecycle)
3. Ship state transitions (gate block vs pass)
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from scripts import correction_loop_daemon as daemon_mod
from scripts.correction_loop_daemon import (
    CorrectionStats,
    DaemonConfig,
    _evaluate_retrain_gate,
    _evaluate_ship_gate,
    tick,
)


@contextlib.contextmanager
def _inject_shadow_eval(return_value: Any):
    """Inject a fake qontinui_train.evaluation.shadow_eval into sys.modules.

    The real package isn't a dependency of qontinui-finetune — it lives in
    the sibling qontinui-train repo. The daemon imports it lazily inside
    ``_trigger_ship`` so we can route around it at test time by stocking
    sys.modules with a shim that provides run_shadow_eval.
    """
    mock_mod = types.ModuleType("qontinui_train.evaluation.shadow_eval")
    mock_mod.run_shadow_eval = MagicMock(return_value=return_value)  # type: ignore[attr-defined]

    # Build the parent package chain so the import machinery resolves.
    parent = types.ModuleType("qontinui_train")
    parent_eval = types.ModuleType("qontinui_train.evaluation")
    parent_eval.shadow_eval = mock_mod  # type: ignore[attr-defined]
    parent.evaluation = parent_eval  # type: ignore[attr-defined]

    saved = {
        name: sys.modules.get(name)
        for name in (
            "qontinui_train",
            "qontinui_train.evaluation",
            "qontinui_train.evaluation.shadow_eval",
        )
    }
    sys.modules["qontinui_train"] = parent
    sys.modules["qontinui_train.evaluation"] = parent_eval
    sys.modules["qontinui_train.evaluation.shadow_eval"] = mock_mod
    try:
        yield mock_mod
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _write_corrections(
    corrections_dir: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Write a list of correction entries to corrections.jsonl."""
    corrections_dir.mkdir(parents=True, exist_ok=True)
    jsonl = corrections_dir / "corrections.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry))
            f.write("\n")


def _make_config(
    tmp_path: Path,
    *,
    auto_retrain: bool = False,
    auto_ship: bool = False,
    per_domain_budget: int = 200,
    aggregate_budget: int = 500,
    pg_url: str | None = None,
    candidate_version: str = "v6",
) -> DaemonConfig:
    return DaemonConfig(
        corrections_dir=tmp_path / "corrections",
        models_dir=tmp_path / "models",
        per_domain_budget=per_domain_budget,
        aggregate_budget=aggregate_budget,
        auto_retrain=auto_retrain,
        auto_ship=auto_ship,
        tick_seconds=300,
        baseline_model="qontinui-grounding-v5",
        candidate_version=candidate_version,
        api_base="http://localhost:8100/v1",
        llama_swap_container="llama-swap-test",
        llama_swap_config=tmp_path / "config.yaml",
        pg_url=pg_url,
    )


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


class TestRetrainGate:
    def test_below_thresholds_does_not_retrain(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        stats = CorrectionStats(total=10, per_target_process={"notepad.exe": 10})
        decision = _evaluate_retrain_gate(stats, config)
        assert decision.should_retrain is False
        assert "below-thresholds" in decision.reason

    def test_per_domain_budget_triggers(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, per_domain_budget=200)
        stats = CorrectionStats(
            total=250, per_target_process={"notepad++.exe": 200, "obs64.exe": 50}
        )
        decision = _evaluate_retrain_gate(stats, config)
        assert decision.should_retrain is True
        assert "notepad++.exe" in decision.reason

    def test_aggregate_budget_triggers(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, aggregate_budget=500)
        stats = CorrectionStats(
            total=500,
            per_target_process={"a": 100, "b": 100, "c": 100, "d": 100, "e": 100},
        )
        decision = _evaluate_retrain_gate(stats, config)
        assert decision.should_retrain is True
        assert "aggregate-budget" in decision.reason


class TestShipGate:
    def test_empty_report_blocks(self) -> None:
        passed, blocking = _evaluate_ship_gate({"per_domain_results": {}})
        assert passed is False
        assert blocking == []

    def test_one_regression_blocks(self) -> None:
        report = {
            "per_domain_results": {
                "notepad++.exe": {"delta_pp": 10.0, "regression": False},
                "obs64.exe": {"delta_pp": -2.0, "regression": True},
            }
        }
        passed, blocking = _evaluate_ship_gate(report)
        assert passed is False
        assert ("obs64.exe", -2.0) in blocking

    def test_below_5pp_blocks_even_without_regression(self) -> None:
        report = {
            "per_domain_results": {
                "notepad++.exe": {"delta_pp": 10.0, "regression": False},
                "obs64.exe": {"delta_pp": 3.0, "regression": False},
            }
        }
        passed, blocking = _evaluate_ship_gate(report)
        assert passed is False
        assert ("obs64.exe", 3.0) in blocking

    def test_all_above_5pp_passes(self) -> None:
        report = {
            "per_domain_results": {
                "notepad++.exe": {"delta_pp": 10.0, "regression": False},
                "obs64.exe": {"delta_pp": 5.5, "regression": False},
            }
        }
        passed, blocking = _evaluate_ship_gate(report)
        assert passed is True
        assert blocking == []


# ---------------------------------------------------------------------------
# Retrain lifecycle
# ---------------------------------------------------------------------------


def _seed_high_correction_count(tmp_path: Path) -> Path:
    corrections_dir = tmp_path / "corrections"
    entries = [
        {
            "ts": "2026-04-21T00:00:00Z",
            "state_machine_id": "abc",
            "image_sha": f"sha{i}",
            "image_path": "/non/existent.png",
            "prompt": "a",
            "corrected_bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
            "target_process": "notepad++.exe",
            "private": False,
        }
        for i in range(205)
    ]
    _write_corrections(corrections_dir, entries)
    return corrections_dir


class TestRetrainTick:
    def test_intent_only_when_flag_disabled(self, tmp_path: Path) -> None:
        _seed_high_correction_count(tmp_path)
        config = _make_config(tmp_path, auto_retrain=False, auto_ship=False)

        tick(config)

        # No lockfile should be written.
        assert not (config.corrections_dir / ".retrain.lock").exists()

    def test_spawns_trainer_when_gate_fires_and_flag_enabled(
        self, tmp_path: Path
    ) -> None:
        _seed_high_correction_count(tmp_path)
        config = _make_config(tmp_path, auto_retrain=True, auto_ship=False)

        popen_calls: list[list[str]] = []
        exporter_calls: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            exporter_calls.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        class FakePopen:
            _next_pid = 12345

            def __init__(self, cmd: list[str], **kwargs: Any) -> None:
                popen_calls.append(cmd)
                self.pid = FakePopen._next_pid
                FakePopen._next_pid += 1

        with (
            patch.object(daemon_mod.subprocess, "run", side_effect=fake_run),
            patch.object(daemon_mod.subprocess, "Popen", FakePopen),
            patch.object(daemon_mod, "_is_pid_alive", return_value=True),
        ):
            tick(config)

        # Exporter was invoked.
        assert len(exporter_calls) == 1
        assert any(
            "export_corrections_to_vlm_sft.py" in arg for arg in exporter_calls[0]
        )
        assert "--corrections-jsonl" in exporter_calls[0]
        assert "--include-private" in exporter_calls[0]

        # Trainer was spawned.
        assert len(popen_calls) == 1
        assert any("finetune_grounding_lora.py" in arg for arg in popen_calls[0])
        assert "--train-data" in popen_calls[0]
        assert "--val-data" in popen_calls[0]
        assert "--output-dir" in popen_calls[0]

        # Lockfile written with the fake pid.
        lock_path = config.corrections_dir / ".retrain.lock"
        assert lock_path.exists()
        lock = json.loads(lock_path.read_text())
        assert lock["pid"] == 12345
        assert lock["status"] == "training"
        assert lock["candidate_model"] == "qontinui-grounding-v6"

    def test_second_tick_while_training_is_a_noop(self, tmp_path: Path) -> None:
        _seed_high_correction_count(tmp_path)
        config = _make_config(tmp_path, auto_retrain=True)

        # Pre-write a lockfile to simulate "training in progress".
        config.corrections_dir.mkdir(parents=True, exist_ok=True)
        (config.corrections_dir / ".retrain.lock").write_text(
            json.dumps(
                {
                    "pid": 99999,
                    "started_at": "2026-04-21T00:00:00Z",
                    "reason": "domain-budget",
                    "output_dir": str(config.candidate_output_dir),
                    "log_path": "logs/retrain.log",
                    "dataset_dir": "datasets/corrections/vlm-sft",
                    "candidate_model": "qontinui-grounding-v6",
                    "status": "training",
                }
            )
        )

        popen_calls: list[Any] = []

        def fake_popen(*args: Any, **kwargs: Any) -> MagicMock:
            popen_calls.append(args)
            m = MagicMock()
            m.pid = 0
            return m

        with (
            patch.object(daemon_mod.subprocess, "Popen", side_effect=fake_popen),
            patch.object(daemon_mod, "_is_pid_alive", return_value=True),
        ):
            tick(config)

        # No new trainer spawned.
        assert popen_calls == []
        # Lockfile still present and unmodified pid.
        lock = json.loads((config.corrections_dir / ".retrain.lock").read_text())
        assert lock["pid"] == 99999


# ---------------------------------------------------------------------------
# Ship lifecycle
# ---------------------------------------------------------------------------


def _make_ready_candidate(tmp_path: Path, config: DaemonConfig) -> Path:
    """Simulate trainer-finished state: lockfile with dead PID + merged dir."""
    output_dir = config.candidate_output_dir
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    # Minimal config.json so the daemon considers the checkpoint ready.
    (merged_dir / "config.json").write_text('{"model_type": "qwen2_5_vl"}')
    (merged_dir / "tokenizer_config.json").write_text('{"extra_special_tokens": {}}')

    config.corrections_dir.mkdir(parents=True, exist_ok=True)
    (config.corrections_dir / ".retrain.lock").write_text(
        json.dumps(
            {
                "pid": 99999,  # dead pid (mocked via _is_pid_alive)
                "started_at": "2026-04-21T00:00:00Z",
                "reason": "domain-budget",
                "output_dir": str(output_dir),
                "log_path": "logs/retrain.log",
                "dataset_dir": "datasets/corrections/vlm-sft",
                "candidate_model": config.candidate_model_name,
                "status": "training",
            }
        )
    )
    return merged_dir


def _make_fake_shadow_report(
    per_domain: dict[str, dict[str, float | bool | int]],
) -> MagicMock:
    """Build a ShadowEvalReport-shaped mock."""
    domain_objs = {}
    for name, v in per_domain.items():
        dm = MagicMock()
        dm.delta_pp = v["delta_pp"]
        dm.regression = v.get("regression", False)
        dm.baseline_acc = v.get("baseline_acc", 0.5)
        dm.candidate_acc = v.get("candidate_acc", 0.55)
        dm.samples = v.get("samples", 100)
        domain_objs[name] = dm

    report = MagicMock()
    report.per_domain_results = domain_objs
    report.to_dict.return_value = {
        "candidate_model": "qontinui-grounding-v6",
        "baseline_model": "qontinui-grounding-v5",
        "per_domain_results": {
            k: {"delta_pp": v["delta_pp"], "regression": v.get("regression", False)}
            for k, v in per_domain.items()
        },
    }
    return report


def _make_config_yaml(path: Path) -> None:
    """Write a config.yaml that matches the real one's shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """healthCheckTimeout: 900
logLevel: info
startPort: 5800

models:
  qontinui-grounding-v5:
    cmd: python3 -m vllm.entrypoints.openai.api_server --model /models/qontinui-grounding-v5
    proxy: "http://127.0.0.1:${PORT}"
    checkEndpoint: /health
    ttl: 300
    aliases:
      - qontinui-grounding
      - grounding-v5

  # qontinui-grounding-v6:
  #   cmd: python3 -m vllm.entrypoints.openai.api_server --model /models/qontinui-grounding-v6 --served-model-name qontinui-grounding-v6
  #   proxy: "http://127.0.0.1:${PORT}"
  #   checkEndpoint: /health
  #   ttl: 300
  #   aliases:
  #     - grounding-v6
""",
        encoding="utf-8",
    )


class TestShipTick:
    def test_ship_blocks_when_gate_fails(self, tmp_path: Path) -> None:
        _seed_high_correction_count(tmp_path)
        config = _make_config(
            tmp_path,
            auto_retrain=True,
            auto_ship=True,
            pg_url="postgresql://user:pw@localhost/x",
        )
        _make_config_yaml(config.llama_swap_config)
        merged_dir = _make_ready_candidate(tmp_path, config)

        # Capture the original config text for the post-assertion.
        original_yaml = config.llama_swap_config.read_text()

        failing_report = _make_fake_shadow_report(
            {
                "notepad++.exe": {"delta_pp": 7.0, "regression": False},
                "obs64.exe": {"delta_pp": 2.0, "regression": False},  # blocks
            }
        )

        reload_calls: list[Any] = []
        with (
            patch.object(daemon_mod, "_is_pid_alive", return_value=False),
            patch.object(daemon_mod, "_apply_post_merge_patches", return_value=True),
            _inject_shadow_eval(
                return_value=failing_report,
            ),
            patch.object(
                daemon_mod,
                "_reload_llama_swap",
                side_effect=lambda c: reload_calls.append(c),
            ),
        ):
            tick(config)

        # Config YAML must NOT be modified.
        assert config.llama_swap_config.read_text() == original_yaml
        # Reload must NOT be triggered.
        assert reload_calls == []
        # Lockfile cleared (training cycle done, blocked ship or not).
        assert not (config.corrections_dir / ".retrain.lock").exists()
        # Candidate dir stays in *-candidate form (not promoted).
        assert not config.candidate_live_dir.exists()
        assert merged_dir.exists()

        # Ship history records the block.
        history = (config.corrections_dir / ".ship-history.jsonl").read_text()
        records = [json.loads(line) for line in history.splitlines() if line]
        assert len(records) == 1
        assert records[0]["shipped"] is False
        assert any(
            d["target_process"] == "obs64.exe" for d in records[0]["blocking_domains"]
        )

    def test_ship_succeeds_when_all_domains_lift(self, tmp_path: Path) -> None:
        _seed_high_correction_count(tmp_path)
        config = _make_config(
            tmp_path,
            auto_retrain=True,
            auto_ship=True,
            pg_url="postgresql://user:pw@localhost/x",
        )
        _make_config_yaml(config.llama_swap_config)
        merged_dir = _make_ready_candidate(tmp_path, config)

        passing_report = _make_fake_shadow_report(
            {
                "notepad++.exe": {"delta_pp": 8.0, "regression": False},
                "obs64.exe": {"delta_pp": 6.0, "regression": False},
            }
        )

        reload_calls: list[Any] = []

        def _fake_reload(c: Any) -> str:
            reload_calls.append(c)
            return "fake-reload"

        with (
            patch.object(daemon_mod, "_is_pid_alive", return_value=False),
            patch.object(daemon_mod, "_apply_post_merge_patches", return_value=True),
            _inject_shadow_eval(
                return_value=passing_report,
            ),
            patch.object(
                daemon_mod,
                "_reload_llama_swap",
                side_effect=_fake_reload,
            ),
        ):
            tick(config)

        # Candidate promoted: output_dir/merged → live_dir.
        # (The daemon renames the merged dir, not the candidate output_dir.)
        assert config.candidate_live_dir.exists()
        assert not merged_dir.exists()

        # Config YAML rewritten. The v6 block must be present and have
        # the bare `qontinui-grounding` alias.
        new_text = config.llama_swap_config.read_text()
        # atomic rename leaves no .new file behind
        assert not config.llama_swap_config.with_name(
            config.llama_swap_config.name + ".new"
        ).exists()

        import yaml  # type: ignore[import-not-found]

        parsed = yaml.safe_load(new_text)
        assert "qontinui-grounding-v6" in parsed["models"]
        v6_aliases = parsed["models"]["qontinui-grounding-v6"].get("aliases", [])
        assert "qontinui-grounding" in v6_aliases
        v5_aliases = parsed["models"]["qontinui-grounding-v5"].get("aliases", [])
        assert "qontinui-grounding" not in v5_aliases
        # v5 keeps its versioned alias.
        assert "grounding-v5" in v5_aliases

        # Reload was triggered.
        assert len(reload_calls) == 1
        # Lockfile cleared.
        assert not (config.corrections_dir / ".retrain.lock").exists()
        # History recorded the ship.
        history = (config.corrections_dir / ".ship-history.jsonl").read_text()
        records = [json.loads(line) for line in history.splitlines() if line]
        assert len(records) == 1
        assert records[0]["shipped"] is True
        assert records[0]["swapped_from"] == "qontinui-grounding-v5"
        assert records[0]["swapped_to"] == "qontinui-grounding-v6"

    def test_ship_intent_only_when_flag_disabled(self, tmp_path: Path) -> None:
        _seed_high_correction_count(tmp_path)
        config = _make_config(
            tmp_path,
            auto_retrain=True,
            auto_ship=False,
            pg_url="postgresql://user:pw@localhost/x",
        )
        _make_config_yaml(config.llama_swap_config)
        merged_dir = _make_ready_candidate(tmp_path, config)

        original_yaml = config.llama_swap_config.read_text()

        with (
            patch.object(daemon_mod, "_is_pid_alive", return_value=False),
            patch.object(daemon_mod, "_apply_post_merge_patches", return_value=True),
        ):
            tick(config)

        # Config untouched; candidate still at *-candidate.
        assert config.llama_swap_config.read_text() == original_yaml
        assert not config.candidate_live_dir.exists()
        assert merged_dir.exists()

    def test_missing_merged_config_clears_lock(self, tmp_path: Path) -> None:
        """Trainer died without producing a merged checkpoint."""
        _seed_high_correction_count(tmp_path)
        config = _make_config(
            tmp_path, auto_retrain=True, auto_ship=True, pg_url="postgresql://u:p@h/x"
        )

        # Lockfile with dead PID but NO merged/config.json.
        config.candidate_output_dir.mkdir(parents=True, exist_ok=True)
        config.corrections_dir.mkdir(parents=True, exist_ok=True)
        (config.corrections_dir / ".retrain.lock").write_text(
            json.dumps(
                {
                    "pid": 99999,
                    "started_at": "2026-04-21T00:00:00Z",
                    "output_dir": str(config.candidate_output_dir),
                    "status": "training",
                }
            )
        )

        with patch.object(daemon_mod, "_is_pid_alive", return_value=False):
            tick(config)

        assert not (config.corrections_dir / ".retrain.lock").exists()
