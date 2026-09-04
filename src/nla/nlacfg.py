"""The NLA sidecar contract (`nla_meta.yaml`) for released checkpoints.

Mirrors nla-inference's / EasyNLA's `load_nla_config`: never hardcode token IDs, prompt
templates or scale factors — load them from the sidecar and assert against the live tokenizer.
Two sidecar kinds exist: `nla_model` (kitft checkpoints: `d_model`, `prompt_templates.av/ar`)
and `nla_dataset` (EasyNLA training-data sidecars shipped with the ceselder checkpoints:
`extraction.{d_model,layer_index}`, `prompt_templates.actor/critic`). Scale fields may be a
number, the sentinel `sqrt_d_model`, or absent (mse_scale then defaults to sqrt(d_model), the
EasyNLA default; injection_scale defaults to None = raw vectors).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)
SCALE_SQRT_D = "sqrt_d_model"


def extract_explanation(text: str) -> str | None:
    """Payload between <explanation> tags (stripped); None if the close tag is missing."""
    m = EXPLANATION_RE.search(text)
    return m.group(1).strip() if m else None


def chat_ids(tokenizer: Any, messages: list[dict[str, str]], **chat_kwargs: Any) -> list[int]:
    """`apply_chat_template(tokenize=True, add_generation_prompt=True)` as a plain id list
    (transformers 4.x returns a list, 5.x a BatchEncoding). Extra kwargs (e.g.
    `enable_thinking=False`) go to the template."""
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **chat_kwargs
    )
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    ids = list(out)
    if ids and isinstance(ids[0], list):  # batched form
        ids = ids[0]
    return [int(i) for i in ids]


def resolve_scale(raw: Any, d_model: int, default: Any = None) -> float | None:
    """number | 'sqrt_d_model' | None/'raw'/'none' -> float | None (absent -> `default`)."""
    if raw is None:
        raw = default
    if raw is None or raw in ("raw", "none"):
        return None
    if raw == SCALE_SQRT_D:
        return math.sqrt(d_model)
    return float(raw)


@dataclass(frozen=True)
class NLAMeta:
    kind: str  # nla_model | nla_dataset
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
    chat_kwargs: dict[str, Any] = field(default_factory=dict)  # e.g. {"enable_thinking": False}

    @classmethod
    def load(cls, path: str | Path, chat_kwargs: dict[str, Any] | None = None) -> NLAMeta:
        """`path` is a checkpoint dir (reads `<dir>/nla_meta.yaml`) or the yaml file itself."""
        p = Path(path)
        p = p / "nla_meta.yaml" if p.is_dir() else p
        m = yaml.safe_load(p.read_text())
        kind = m.get("kind")
        assert kind in ("nla_model", "nla_dataset"), kind
        tok = m["tokens"]
        templates = m.get("prompt_templates", {})
        ext = m.get("extraction", {}) or {}
        if kind == "nla_model":
            d_model = int(m["d_model"])
            layer = m.get(
                "extraction_layer_index", m.get("critic", {}).get("extraction_layer_index")
            )
            role = m.get("role", "?")
        else:
            d_model = int(ext["d_model"])
            layer = ext["layer_index"]
            role = m.get("stage", "?")
        suffix = tok.get("critic_suffix_ids")
        return cls(
            kind=kind,
            role=role,
            d_model=d_model,
            layer_index=int(layer),
            injection_scale=resolve_scale(ext.get("injection_scale"), d_model),
            mse_scale=resolve_scale(ext.get("mse_scale"), d_model, default=SCALE_SQRT_D),
            injection_char=tok["injection_char"],
            injection_token_id=int(tok["injection_token_id"]),
            left_neighbor_id=int(tok["injection_left_neighbor_id"]),
            right_neighbor_id=int(tok["injection_right_neighbor_id"]),
            critic_suffix_ids=None if suffix is None else tuple(int(i) for i in suffix),
            av_template=templates.get("av") or templates["actor"],
            ar_template=templates.get("ar") or templates["critic"],
            chat_kwargs=dict(chat_kwargs or {}),
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
        ids = chat_ids(tokenizer, [{"role": "user", "content": content}], **self.chat_kwargs)
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
        """Critic prompt ids; the AR reads its value at the LAST token. No special tokens
        (EasyNLA scores with add_special_tokens=False; the Qwen tokenizers add no BOS, so
        this also matches the kitft training path)."""
        ids = [
            int(i)
            for i in tokenizer(self.ar_prompt(explanation), add_special_tokens=False)["input_ids"]
        ]
        if self.critic_suffix_ids is not None:
            n = len(self.critic_suffix_ids)
            assert tuple(ids[-n:]) == self.critic_suffix_ids, (
                f"AR prompt does not end with the sidecar suffix: {ids[-n:]} vs {self.critic_suffix_ids}"
            )
        return ids
