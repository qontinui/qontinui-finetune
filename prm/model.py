"""``GroundingPRM`` — small process-reward model for grounding rollouts.

Design choice
-------------
The PRM needs to score (image, instruction, predicted_grounding) tuples cheaply
because the RL rollout scorer calls it K times per prompt (default K=8) per
GRPO step. The SFT VLM backbone (UI-TARS-1.5-7B) would dominate step cost if
reused here, so we pick a **separate, small encoder** and freeze it. Only the
final linear head trains.

The default backbone is ``openai/clip-vit-base-patch32`` (~150 M params) which:
- Accepts image + text jointly via HuggingFace ``CLIPModel`` API.
- Exposes a pooled joint embedding (``logits_per_image``/``logits_per_text``
  inputs: ``pixel_values`` and ``input_ids``).
- Fits easily alongside the RL-trained 7 B model on a single A100 / 3090.

Alternative backbones (any HuggingFace ``AutoModel`` producing a pooled hidden
state) can be passed via ``backbone_name``. The head projects the pooled
embedding to a scalar.

Inputs
------
The ``forward`` signature mirrors the tuple described in the top-level
docstring: an image tensor, tokenized instruction + predicted-grounding text.
Callers are expected to build those inputs with a ``AutoProcessor`` that
matches ``backbone_name``; this keeps ``GroundingPRM`` independent of the
backbone's exact preprocessing semantics.

Output
------
A scalar (shape ``(batch,)``) — the raw process reward. The training loss is
``mse_loss(reward, label)``; at inference time the raw scalar is the reward.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# The default backbone is a small CLIP model: public, ~150 M params, and
# exposes image+text encoders we can pool. Swap via ``--backbone`` on the
# train.py CLI or by passing ``backbone_name`` to ``GroundingPRM``.
DEFAULT_BACKBONE = "openai/clip-vit-base-patch32"


class GroundingPRM:
    """Tiny reward head on top of a frozen image+text encoder.

    Parameters
    ----------
    backbone_name:
        HuggingFace model ID or local path. Must be loadable via
        ``AutoModel.from_pretrained``. Defaults to CLIP ViT-B/32.
    freeze_backbone:
        When ``True`` (default), the backbone's parameters are frozen and only
        the linear head trains. Set ``False`` for full fine-tuning if you have
        enough labels and VRAM.
    hidden_dim:
        Explicit override for the pooled embedding dimension. When ``None``,
        we infer from the loaded backbone's ``config.projection_dim`` (CLIP) or
        ``config.hidden_size`` (generic encoders).

    Notes
    -----
    Instantiation lazily imports torch + transformers so tests that mock the
    class or only touch the dataset loader don't require the heavy ML stack.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        freeze_backbone: bool = True,
        hidden_dim: int | None = None,
    ) -> None:
        import torch
        from torch import nn
        from transformers import AutoModel

        self.backbone_name = backbone_name
        self.backbone: Any = AutoModel.from_pretrained(backbone_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Resolve pooled embedding dim.
        dim = hidden_dim
        if dim is None:
            cfg = self.backbone.config
            # CLIP exposes projection_dim; generic models expose hidden_size.
            dim = (
                getattr(cfg, "projection_dim", None)
                or getattr(cfg, "hidden_size", None)
            )
            if dim is None:
                raise ValueError(
                    f"Cannot infer embedding dim for backbone {backbone_name!r}; "
                    "pass hidden_dim explicitly."
                )

        self.hidden_dim: int = int(dim)
        self.head: Any = nn.Linear(self.hidden_dim, 1)
        self._torch = torch
        self._nn = nn

        logger.info(
            "GroundingPRM initialised: backbone=%s, dim=%d, freeze=%s",
            backbone_name,
            self.hidden_dim,
            freeze_backbone,
        )

    # ------------------------------------------------------------------
    # torch.nn.Module-like surface
    # ------------------------------------------------------------------

    def parameters(self) -> Any:
        """Yield trainable parameters (head + any unfrozen backbone weights)."""
        for p in self.backbone.parameters():
            if p.requires_grad:
                yield p
        yield from self.head.parameters()

    def to(self, device: Any) -> GroundingPRM:
        """Move backbone + head to *device*. Returns self for chaining."""
        self.backbone = self.backbone.to(device)
        self.head = self.head.to(device)
        return self

    def train(self, mode: bool = True) -> GroundingPRM:
        self.head.train(mode)
        # Keep frozen backbone in eval mode so e.g. dropout doesn't activate.
        self.backbone.train(False)
        return self

    def eval(self) -> GroundingPRM:
        return self.train(False)

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint dict with head weights + backbone identifier.

        We deliberately exclude the (frozen, huge) backbone state: it can be
        restored by name at load time. The linear head is the only trainable
        component.
        """
        return {
            "backbone_name": self.backbone_name,
            "hidden_dim": self.hidden_dim,
            "head_state": self.head.state_dict(),
        }

    def load_head_state(self, head_state: dict[str, Any]) -> None:
        self.head.load_state_dict(head_state)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _pool(self, outputs: Any) -> Any:
        """Extract a pooled (batch, hidden_dim) tensor from backbone outputs.

        Handles CLIPModel (joint image+text embed) and generic encoder outputs
        (``last_hidden_state`` → mean pool).
        """
        torch = self._torch
        # CLIP path
        image_embeds = getattr(outputs, "image_embeds", None)
        text_embeds = getattr(outputs, "text_embeds", None)
        if image_embeds is not None and text_embeds is not None:
            # Fuse by element-wise multiplication then L2-normalise.
            fused = image_embeds * text_embeds
            return torch.nn.functional.normalize(fused, dim=-1)

        # Generic encoder path: mean-pool last_hidden_state over sequence.
        last = getattr(outputs, "last_hidden_state", None)
        if last is not None:
            return last.mean(dim=1)

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled

        raise RuntimeError(
            "GroundingPRM: backbone outputs lack a poolable field "
            "(expected image_embeds+text_embeds, last_hidden_state, or pooler_output)."
        )

    def forward(self, **inputs: Any) -> Any:
        """Score a batch of (image, text) pairs.

        Accepts the kwargs produced by the matching ``AutoProcessor``:
        typically ``pixel_values``, ``input_ids``, ``attention_mask`` (and
        optionally ``pixel_mask``, ``token_type_ids``, ...). All are forwarded
        to the backbone unchanged.

        Returns
        -------
        torch.Tensor of shape ``(batch,)`` — raw process reward scalars.
        """
        outputs = self.backbone(**inputs)
        pooled = self._pool(outputs)
        reward = self.head(pooled).squeeze(-1)
        return reward

    __call__ = forward
