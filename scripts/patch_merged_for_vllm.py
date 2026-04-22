#!/usr/bin/env python3
"""Patch a merged Qwen2.5-VL checkpoint so older vLLM (in llama-swap) can load it.

Two known drifts between transformers 5 save format and vLLM's embedded
transformers version:

1. `tokenizer_config.json`
   - `extra_special_tokens`: list → {}

2. `config.json`
   - Flatten `text_config.*` fields up to top level
   - Drop `text_config.layer_types` (new in transformers 5)
   - Convert `text_config.rope_parameters` → top-level `rope_theta` + `rope_scaling`
   - Keep `vision_config` + top-level vision/image token IDs untouched

Usage:
    python scripts/patch_merged_for_vllm.py <model_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def patch_tokenizer_config(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    if isinstance(data.get("extra_special_tokens"), list):
        data["extra_special_tokens"] = {}
        changed = True
    if changed:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return changed


def patch_config(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    tc = data.get("text_config")
    if isinstance(tc, dict):
        # Drop layer_types (not in old config format)
        tc.pop("layer_types", None)

        # Convert rope_parameters → rope_theta + rope_scaling (old format)
        rp = tc.pop("rope_parameters", None)
        if isinstance(rp, dict):
            rope_theta = rp.get("rope_theta")
            mrope_section = rp.get("mrope_section")
            rope_type = rp.get("rope_type", "default")
            if rope_theta is not None:
                data["rope_theta"] = rope_theta
            if mrope_section is not None:
                data["rope_scaling"] = {
                    "type": "mrope",
                    "mrope_section": mrope_section,
                    "rope_type": rope_type,
                }

        # Fix nested text model_type
        if tc.get("model_type") == "qwen2_5_vl_text":
            tc.pop("model_type")

        # Flatten remaining text_config fields to top level
        for k, v in tc.items():
            if k not in data:
                data[k] = v

        # Keep a minimal text_config for anything that still looks it up,
        # OR drop it entirely. vLLM's old format doesn't expect it — drop.
        data.pop("text_config", None)
        changed = True

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    model_dir: Path = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"Not a directory: {model_dir}", file=sys.stderr)
        return 1

    tok_cfg = model_dir / "tokenizer_config.json"
    cfg = model_dir / "config.json"

    if tok_cfg.exists():
        changed = patch_tokenizer_config(tok_cfg)
        print(f"tokenizer_config.json: {'patched' if changed else 'unchanged'}")
    else:
        print("tokenizer_config.json: missing", file=sys.stderr)

    if cfg.exists():
        changed = patch_config(cfg)
        print(f"config.json: {'patched' if changed else 'unchanged'}")
    else:
        print("config.json: missing", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
