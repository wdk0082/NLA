"""Claim decomposition and minimal edits of explanations.

Backends:
  * `LocalEditor` — Qwen2.5-7B-Instruct on the TPU through `nla.generate` (the automated
    pipeline path; the weights are the target model's, so the edit stage reuses them).
  * `FileEditor` — pre-authored edits from a JSONL file (one object per explanation:
    {"idx", "claims": [...], "edits": [{"claim_id", "contradicted", "deleted"}],
    "paraphrase", "translation"}) — the manual / gold path.

Output formats are TAG-based (not JSON): explanations are full of double quotes, which
7B models fail to escape reliably.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from nla.device import manual_seed
from nla.generate import GenConfig, cut_at_eos, generate, left_pad
from nla.nlacfg import chat_ids
from nla.qwen2 import Qwen2Weights, embed_tokens

SYSTEM = (
    "You are a precise text editor working inside an interpretability research pipeline. "
    "You follow the requested output format exactly and add no commentary."
)

DECOMPOSE = """<explanation>
{z}
</explanation>

The text above is an explanation produced by an interpretability tool; it describes what a language model's internal state encodes about a text.

Task: list the distinct atomic claims the explanation makes. Each claim must be a short, self-contained declarative sentence that can be checked on its own, reusing the explanation's own words where possible. Give at most {k} claims, most central first. Do not add claims the explanation does not make.

Output ONLY a numbered list, one claim per line:
1. ...
2. ..."""

EDIT = """<explanation>
{z}
</explanation>

<claim>
{c}
</claim>

Task: produce two edited versions of the explanation above.
(A) CONTRADICTED: the same explanation, except that the claim is replaced by a claim that contradicts it (asserts the opposite, or an incompatible alternative). Change as little else as possible: keep every other claim, the structure, the wording and the length.
(B) DELETED: the same explanation with the claim removed and not replaced. Change as little else as possible: keep every other claim and the wording; only fix grammar where the removal requires it.

Output exactly this format and nothing else:
<contradicted>
...
</contradicted>
<deleted>
...
</deleted>"""

PARAPHRASE = """<explanation>
{z}
</explanation>

Task: rewrite the explanation so that it has exactly the same meaning but different wording: change sentence structure and vocabulary, keep every claim, add nothing, remove nothing, keep quoted words quoted.

Output exactly this format and nothing else:
<paraphrase>
...
</paraphrase>"""

TRANSLATE = """<explanation>
{z}
</explanation>

Task: translate the explanation into French, preserving every claim exactly (keep quoted English words in English, inside their quotes).

Output exactly this format and nothing else:
<translation>
...
</translation>"""

_NUM_LINE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")


def _tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(rf"<{tag}>\s*(.*)$", text, re.DOTALL)  # truncated close tag
    return m.group(1).strip() if m and m.group(1).strip() else None


def parse_claims(text: str, k: int) -> list[str]:
    claims: list[str] = []
    for line in text.splitlines():
        m = _NUM_LINE.match(line)
        if m:
            c = m.group(2).strip().strip('"').strip()
            if len(c) > 3:
                claims.append(c)
    return claims[:k]


@dataclass
class Edits:
    idx: int
    claims: list[str] = field(default_factory=list)
    contradicted: list[str | None] = field(default_factory=list)  # per claim
    deleted: list[str | None] = field(default_factory=list)
    paraphrase: str | None = None
    translation: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ programmatic variants


def shuffle_snippets(z: str, seed: int) -> str | None:
    """Reorder the explanation's snippets (blank-line separated; else sentences)."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", z) if p.strip()]
    if len(parts) < 2:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", z) if p.strip()]
    if len(parts) < 2:
        return None
    rng = random.Random(seed)
    order = list(range(len(parts)))
    while order == list(range(len(parts))):
        rng.shuffle(order)
    sep = "\n\n" if "\n\n" in z else " "
    return sep.join(parts[i] for i in order)


def derangement(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


# ------------------------------------------------------------------ backends


class LocalEditor:
    """Qwen2.5-7B-Instruct as editor (greedy decoding, chat template, left-padded batches)."""

    def __init__(
        self,
        w: Qwen2Weights,
        tokenizer: Any,
        device: torch.device,
        batch_size: int = 16,
        prompt_len: int = 768,
    ):
        self.w = w
        self.tok = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        self.eos_ids = tuple(int(i) for i in eos if i is not None)

    def _ids(self, user: str) -> list[int]:
        ids = chat_ids(
            self.tok, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        )
        if len(ids) > self.prompt_len:  # keep the tail (the instructions) — rare
            ids = ids[-self.prompt_len :]
        return ids

    @torch.no_grad()
    def complete(self, prompts: list[str], max_new: int, log_tag: str = "") -> list[str]:
        """Greedy completions for a list of user prompts (batched, static shapes)."""
        manual_seed(0, self.device)
        outs: list[str] = []
        gen = GenConfig(max_new=max_new, temperature=0.0, eos_ids=self.eos_ids, check_every=16)
        bs = self.batch_size
        for start in range(0, len(prompts), bs):
            chunk = [self._ids(p) for p in prompts[start : start + bs]]
            b_real = len(chunk)
            chunk += [chunk[-1]] * (bs - b_real)
            ids, valid = left_pad(chunk, self.pad_id, self.prompt_len)
            emb = embed_tokens(self.w, ids.to(self.device))
            toks, _ = generate(self.w, emb, valid.to(self.device), gen)
            for i in range(b_real):
                outs.append(
                    self.tok.decode(cut_at_eos(toks[i], self.eos_ids), skip_special_tokens=True)
                )
            print(f"[editor{log_tag}] {len(outs)}/{len(prompts)}", flush=True)
        return outs

    def edit_all(self, explanations: dict[int, str], max_claims: int = 4) -> list[Edits]:
        idxs = sorted(explanations)
        # 1) claims
        dec = self.complete(
            [DECOMPOSE.format(z=explanations[i], k=max_claims) for i in idxs],
            max_new=192,
            log_tag="/claims",
        )
        edits = {
            i: Edits(idx=i, claims=parse_claims(t, max_claims), raw={"decompose": t})
            for i, t in zip(idxs, dec, strict=True)
        }
        # 2) per-claim contradiction + deletion
        jobs = [(i, j) for i in idxs for j in range(len(edits[i].claims))]
        res = self.complete(
            [EDIT.format(z=explanations[i], c=edits[i].claims[j]) for i, j in jobs],
            max_new=512,
            log_tag="/edits",
        )
        for (i, j), t in zip(jobs, res, strict=True):
            e = edits[i]
            while len(e.contradicted) <= j:
                e.contradicted.append(None)
                e.deleted.append(None)
            e.contradicted[j] = _tag(t, "contradicted")
            e.deleted[j] = _tag(t, "deleted")
            e.raw.setdefault("edit", {})[j] = t
        # 3) paraphrase, 4) translation
        par = self.complete(
            [PARAPHRASE.format(z=explanations[i]) for i in idxs], max_new=384, log_tag="/paraphrase"
        )
        tra = self.complete(
            [TRANSLATE.format(z=explanations[i]) for i in idxs], max_new=448, log_tag="/translate"
        )
        for i, p, t in zip(idxs, par, tra, strict=True):
            edits[i].paraphrase = _tag(p, "paraphrase")
            edits[i].translation = _tag(t, "translation")
            edits[i].raw["paraphrase"], edits[i].raw["translation"] = p, t
        return [edits[i] for i in idxs]


class FileEditor:
    """Edits pre-authored in a JSONL file (manual / gold path)."""

    def __init__(self, path: str | Path):
        self.rows = {
            int(r["idx"]): r for r in map(json.loads, Path(path).read_text().splitlines()) if r
        }

    def edit_all(self, explanations: dict[int, str], max_claims: int = 4) -> list[Edits]:
        out: list[Edits] = []
        for i in sorted(explanations):
            r = self.rows.get(i)
            if r is None:
                continue
            claims = list(r.get("claims", []))[:max_claims]
            by_claim = {int(e["claim_id"]): e for e in r.get("edits", [])}
            out.append(
                Edits(
                    idx=i,
                    claims=claims,
                    contradicted=[
                        by_claim.get(j, {}).get("contradicted") for j in range(len(claims))
                    ],
                    deleted=[by_claim.get(j, {}).get("deleted") for j in range(len(claims))],
                    paraphrase=r.get("paraphrase"),
                    translation=r.get("translation"),
                    raw={"source": "file"},
                )
            )
        return out
