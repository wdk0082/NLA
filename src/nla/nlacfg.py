"""The NLA sidecar contract (`nla_meta.yaml`) for the released kitft checkpoints.

Mirrors nla-inference's `load_nla_config`: never hardcode token IDs, prompt templates or
scale factors — load them from the sidecar and assert against the live tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)


def extract_explanation(text: str) -> str | None:
    """Payload between <explanation> tags (stripped); None if the close tag is missing."""
    m = EXPLANATION_RE.search(text)
    return m.group(1).strip() if m else None


def chat_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """`apply_chat_template(tokenize=True, add_generation_prompt=True)` as a plain id list
    (transformers 4.x returns a list, 5.x may return a BatchEncoding)."""
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    ids = list(out)
    if ids and isinstance(ids[0], list):  # batched form
        ids = ids[0]
    return [int(i) for i in ids]


@dataclass(frozen=True)
class NLAMeta:
    role: str
    d_model: int
    layer_index: int
    injection_scale: float | None
    mse_scale: float
    injection_char: str
    injection_token_id: int
    left_neighbor_id: int
    right_neighbor_id: int
    critic_suffix_ids: tuple[int, ...] | None
    av_template: str
    ar_template: str

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> NLAMeta:
        m = yaml.safe_load((Path(checkpoint_dir) / "nla_meta.yaml").read_text())
        assert m.get("kind") == "nla_model", m.get("kind")
        tok = m["tokens"]
        ext = m.get("extraction", {})
        suffix = tok.get("critic_suffix_ids")
        return cls(
            role=m.get("role", "?"),
            d_model=int(m["d_model"]),
            layer_index=int(
                m.get("extraction_layer_index", m.get("critic", {}).get("extraction_layer_index"))
            ),
            injection_scale=None
            if ext.get("injection_scale") is None
            else float(ext["injection_scale"]),
            mse_scale=float(ext["mse_scale"]),
            injection_char=tok["injection_char"],
            injection_token_id=int(tok["injection_token_id"]),
            left_neighbor_id=int(tok["injection_left_neighbor_id"]),
            right_neighbor_id=int(tok["injection_right_neighbor_id"]),
            critic_suffix_ids=None if suffix is None else tuple(int(i) for i in suffix),
            av_template=m["prompt_templates"]["av"],
            ar_template=m["prompt_templates"]["ar"],
        )

    # --- AV side -------------------------------------------------------------------
    def av_prompt_ids(self, tokenizer: Any) -> tuple[list[int], int]:
        """Canonical AV prompt ids and the injection position, verified against the
        live tokenizer (injection char is a single token, neighbours match the sidecar)."""
        live = tokenizer.encode(self.injection_char, add_special_tokens=False)
        assert live == [self.injection_token_id], (
            f"tokenizer drift: {self.injection_char!r} -> {live}, sidecar says [{self.injection_token_id}]"
        )
        content = self.av_template.format(injection_char=self.injection_char)
        ids = chat_ids(tokenizer, [{"role": "user", "content": content}])
        hits = [i for i, t in enumerate(ids) if t == self.injection_token_id]
        assert len(hits) == 1, f"injection token appears {len(hits)}x in the canonical prompt"
        p = hits[0]
        assert ids[p - 1] == self.left_neighbor_id and ids[p + 1] == self.right_neighbor_id, (
            f"neighbour drift at injection position {p}: {ids[p - 1]}, {ids[p + 1]} vs sidecar "
            f"{self.left_neighbor_id}, {self.right_neighbor_id}"
        )
        return ids, p

    # --- AR side -------------------------------------------------------------------
    def ar_prompt(self, explanation: str) -> str:
        return self.ar_template.format(explanation=explanation)

    def ar_prompt_ids(self, tokenizer: Any, explanation: str) -> list[int]:
        """Critic prompt ids; the AR reads its value at the LAST token. add_special_tokens=True
        matches training (no-op for Qwen, which has no BOS)."""
        ids = [
            int(i)
            for i in tokenizer(self.ar_prompt(explanation), add_special_tokens=True)["input_ids"]
        ]
        if self.critic_suffix_ids is not None:
            n = len(self.critic_suffix_ids)
            assert tuple(ids[-n:]) == self.critic_suffix_ids, (
                f"AR prompt does not end with the sidecar suffix: {ids[-n:]} vs {self.critic_suffix_ids}"
            )
        return ids
