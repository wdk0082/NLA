"""The target model (Qwen2.5-7B-Instruct): activation extraction and patched output.

`layer` follows the NLA convention: the residual stream AFTER block `layer`
(HF hidden_states[layer+1]); the AR reads/writes that stream and the "downstream part" F
is blocks layer+1..N-1 + final norm + lm_head.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nla.qwen2 import Qwen2Weights, embed_tokens, final_logits, load_qwen2, run_blocks


class Target:
    def __init__(
        self, ckpt_dir: str | Path, device: torch.device, dtype: torch.dtype, layer: int = 20
    ):
        self.dir = Path(ckpt_dir)
        self.device = device
        self.layer = layer
        self.w: Qwen2Weights = load_qwen2(self.dir, device, dtype)
        assert 0 <= layer < self.w.cfg.num_hidden_layers - 1

    @torch.no_grad()
    def prefix(self, ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """ids/valid [B,T] (right-padded) -> residual after block `layer`, [B,T,d] fp32."""
        return run_blocks(self.w, embed_tokens(self.w, ids), 0, self.layer + 1, valid)

    @torch.no_grad()
    def tail_logits(
        self,
        h_layer: torch.Tensor,  # [B,T,d] fp32 residual after block `layer`
        valid: torch.Tensor,  # [B,T]
        lengths: torch.Tensor,  # [B] — read logits at lengths-1
        replace: torch.Tensor | None = None,  # [B,d] fp32: patch position lengths-1 with this
    ) -> torch.Tensor:
        """Downstream blocks + head -> next-token logits [B,V] fp32 at each row's last token,
        optionally with the layer-`layer` residual at that token replaced."""
        b, t, _ = h_layer.shape
        if replace is not None:
            pos = torch.arange(t, device=h_layer.device)
            at = (pos[None, :] == (lengths - 1)[:, None])[:, :, None]  # [B,T,1]
            h_layer = torch.where(at, replace.to(h_layer.dtype)[:, None, :], h_layer)
        x = run_blocks(self.w, h_layer, self.layer + 1, self.w.cfg.num_hidden_layers, valid)
        last = x[torch.arange(b, device=x.device), lengths - 1]  # [B,d]
        return final_logits(self.w, last[:, None, :])[:, 0]

    @torch.no_grad()
    def extract(
        self, ids: torch.Tensor, valid: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (h [B,d] fp32 at lengths-1 after block `layer`, logits [B,V] fp32 at lengths-1)."""
        hl = self.prefix(ids, valid)
        b = ids.shape[0]
        h = hl[torch.arange(b, device=ids.device), lengths - 1]
        return h, self.tail_logits(hl, valid, lengths)
