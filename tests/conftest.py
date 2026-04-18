"""pytest configuration — put qontinui-finetune/ on sys.path.

This lets tests do ``from prm.dataset import ...`` / ``from scripts... import
...`` without requiring an editable install of the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# qontinui-finetune/ (parent of tests/) — add so that ``prm`` and ``scripts``
# resolve as top-level packages.
_FT_ROOT = Path(__file__).resolve().parent.parent
if str(_FT_ROOT) not in sys.path:
    sys.path.insert(0, str(_FT_ROOT))
