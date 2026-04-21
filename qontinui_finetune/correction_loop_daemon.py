"""Package-path shim for ``scripts/correction_loop_daemon.py``.

Lets the supervisor spawn the daemon via
``python -m qontinui_finetune.correction_loop_daemon`` — the canonical
way to invoke a managed service — without needing the ``scripts/``
directory on sys.path.

The real implementation lives in ``scripts/correction_loop_daemon.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prepend the repo-level scripts/ dir so we can re-export from it.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from correction_loop_daemon import (  # type: ignore[import-not-found]  # noqa: E402
    CorrectionStats,
    DaemonConfig,
    RetrainDecision,
    RetrainStatus,
    ShipStatus,
    main,
    run_forever,
    tick,
)

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
