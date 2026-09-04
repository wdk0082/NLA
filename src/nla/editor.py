"""Claim decomposition and minimal edits of explanations (span-anchored).

A 7B editor asked to "rewrite the whole explanation with one claim changed" appends a
negation or returns the text unchanged (EXP001 smoke round). So edits are anchored to a
verbatim EXCERPT of the explanation that the editor names for each claim:

  * deletion      z^{-c}  = z with the excerpt removed (programmatic, always minimal);
  * contradiction z^{¬c}  = z with the excerpt replaced by an editor-written contradicting
                            rewrite of that excerpt only (short output, verified != excerpt).

Backends: `LocalEditor` (Qwen2.5-7B-Instruct on the TPU via `nla.generate`; the weights are
the target model's) and `FileEditor` (pre-authored JSONL: manual / gold path).
Output formats are TAG/line based (explanations are full of double quotes, which small
models fail to escape in JSON).
"""

from __future__ import annotations

import difflib
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

Task: list the distinct atomic claims the explanation makes, at most {k}, most central first. For each claim give (a) the claim as a short self-contained declarative sentence and (b) the EXCERPT: the exact contiguous excerpt of the explanation that expresses this claim, copied verbatim character for character (a phrase, clause or sentence of at most 30 words; it must occur in the explanation exactly as written). Different claims must use different excerpts.

Output ONLY lines of this form, nothing else:
1. CLAIM: <sentence> | EXCERPT: <verbatim excerpt>
2. CLAIM: <sentence> | EXCERPT: <verbatim excerpt>"""

CONTRADICT = """<explanation>
{z}
</explanation>

The explanation above contains this excerpt:
<excerpt>{span}</excerpt>
which makes the claim: {claim}

Task: rewrite ONLY that excerpt so that it asserts the opposite of the claim (or an incompatible alternative), keeping the excerpt's grammatical role, style and length, so that the rewrite can replace the original excerpt in place. Keep every other part of the explanation as it is (do not output it). Do not mention that anything was changed.

Output exactly this format and nothing else:
<replacement>
...
</replacement>"""

PARAPHRASE = """<explanation>
{z}
</explanation>

Task: rewrite the explanation in different words, sentence by sentence, using synonyms and different sentence structures throughout, so that no sentence is copied verbatim. Keep every claim, keep the paragraph structure, keep any quoted text unchanged inside its quotes, add nothing and remove nothing.

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

_CLAIM_LINE = re.compile(
    r"^\s*(\d+)[.)]\s*CLAIM:\s*(.+?)\s*\|\s*EXCERPT:\s*(.+?)\s*$", re.IGNORECASE
)


def _tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(rf"<{tag}>\s*(.*)$", text, re.DOTALL)  # truncated close tag
    return m.group(1).strip() if m and m.group(1).strip() else None


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'“‘" and s[-1] in "\"'”’":
        return s[1:-1].strip()
    return s


def locate_span(z: str, excerpt: str, min_ratio: float = 0.85) -> tuple[int, int] | None:
    """Find `excerpt` in `z`: exact, then case-insensitive, then the longest fuzzy match
    (difflib) if it covers >= min_ratio of the excerpt. Returns (start, end) in z."""
    for cand in (excerpt, _unquote(excerpt)):
        if not cand:
            continue
        i = z.find(cand)
        if i >= 0:
            return i, i + len(cand)
        i = z.lower().find(cand.lower())
        if i >= 0:
            return i, i + len(cand)
    cand = _unquote(excerpt)
    if len(cand) < 8:
        return None
    sm = difflib.SequenceMatcher(None, z, cand, autojunk=False)
    m = sm.find_longest_match(0, len(z), 0, len(cand))
    if m.size >= min_ratio * len(cand):
        return m.a, m.a + m.size
    return None


def parse_claims(text: str, z: str, k: int) -> list[tuple[str, tuple[int, int] | None]]:
    """-> [(claim, span or None)], at most k, spans located in z (None if not found)."""
    out: list[tuple[str, tuple[int, int] | None]] = []
    seen: set[tuple[int, int]] = set()
    for line in text.splitlines():
        m = _CLAIM_LINE.match(line)
        if not m:
            continue
        claim = _unquote(m.group(2))
        if len(claim) < 4:
            continue
        span = locate_span(z, m.group(3))
        if span is not None and span in seen:
            span = None  # duplicate excerpt -> keep the claim, drop the anchor
        if span is not None:
            seen.add(span)
        out.append((claim, span))
        if len(out) >= k:
            break
    return out


def delete_span(z: str, span: tuple[int, int]) -> str | None:
    """Remove z[start:end] and tidy punctuation/whitespace; None if nothing changes."""
    s, e = span
    out = z[:s] + z[e:]
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"[ \t]+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,;:])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"^[ \t]*[,;:]\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    lines = [ln.rstrip() for ln in out.split("\n")]
    out = "\n".join(lines).strip()
    return out if out and out != z else None


def replace_span(z: str, span: tuple[int, int], new: str) -> str | None:
    s, e = span
    new = new.strip()
    if not new or new == z[s:e].strip():
        return None
    return z[:s] + new + z[e:]


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


WHOLE_KINDS = ("polarity", "vocab", "cat")  # EXP002 hand-made / mechanical whole-explanation edits


@dataclass
class Edits:
    idx: int
    claims: list[str] = field(default_factory=list)
    spans: list[str | None] = field(default_factory=list)  # excerpt text per claim
    replacements: list[str | None] = field(default_factory=list)  # contradicting excerpt per claim
    contradicted: list[str | None] = field(default_factory=list)  # per claim (full text)
    deleted: list[str | None] = field(default_factory=list)
    paraphrase: str | None = None
    translation: str | None = None
    whole: dict[str, str | None] = field(default_factory=dict)  # kind -> full text (WHOLE_KINDS)
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


def cat_edit(z: str, final_token: str, replacement: str = "cat") -> tuple[str | None, int]:
    """Replace every whole-word mention of the context's final token in `z` (quoted or bare,
    case-sensitive) by `replacement`. Returns (edited text or None, number of replacements)."""
    word = final_token.strip()
    if not word or not word.isalpha():
        return None, 0
    out, n = re.subn(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", replacement, z)
    return (out, n) if n else (None, 0)


def write_hand_template(
    path: str | Path, rows: list[dict[str, Any]], with_claims: bool = False
) -> None:
    """One JSON line per explanation for hand authoring: the fields to fill are `polarity`,
    `vocab`, `paraphrase`, `translation`, optionally `cat` (overrides the mechanical edit) and,
    with `with_claims`, `claims` (list of {claim, excerpt, contradiction}; EXP001-style, dormant
    since EXP002 round 3). Values already present in `rows` (e.g. a translation reused from an
    earlier round) are written as given. Everything else is context for the author and is
    ignored on load."""
    fields = ["polarity", "vocab", "paraphrase", "translation", "cat"]
    with open(path, "w") as f:
        for r in rows:
            item: dict[str, Any] = {
                "idx": int(r["idx"]),
                "final_token": r.get("final_token"),
                "x_tail": r.get("x_tail"),
                "explanation": r["explanation"],
            }
            if with_claims:
                item["claims"] = r.get("claims") or []
            for k in fields:
                item[k] = r.get(k)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def swap_token(text: str, token: str, placeholder: str = "cat") -> str | None:
    """`text` is a "cat" variant (every final-token mention replaced by `placeholder`); put
    `token` there instead. Used for `unrelated_token`: a donor explanation with the donor's
    final token replaced by this activation's. None if the placeholder does not occur."""
    word = token.strip()
    if not word:
        return None
    out, n = re.subn(rf"(?<![A-Za-z]){re.escape(placeholder)}(?![A-Za-z])", word, text)
    return out if n else None


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
        if not prompts:
            return []
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
        # 1) claims + anchoring excerpts
        dec = self.complete(
            [DECOMPOSE.format(z=explanations[i], k=max_claims) for i in idxs],
            max_new=320,
            log_tag="/claims",
        )
        edits: dict[int, Edits] = {}
        spans: dict[int, list[tuple[int, int] | None]] = {}
        for i, t in zip(idxs, dec, strict=True):
            parsed = parse_claims(t, explanations[i], max_claims)
            e = Edits(idx=i, claims=[c for c, _ in parsed], raw={"decompose": t})
            spans[i] = [s for _, s in parsed]
            e.spans = [explanations[i][s[0] : s[1]] if s else None for s in spans[i]]
            e.contradicted = [None] * len(parsed)
            e.replacements = [None] * len(parsed)
            e.deleted = [delete_span(explanations[i], s) if s else None for s in spans[i]]
            edits[i] = e
        # 2) contradiction of the anchored excerpt only
        jobs = [(i, j) for i in idxs for j, s in enumerate(spans[i]) if s is not None]
        res = self.complete(
            [
                CONTRADICT.format(
                    z=explanations[i], span=edits[i].spans[j], claim=edits[i].claims[j]
                )
                for i, j in jobs
            ],
            max_new=96,
            log_tag="/contradict",
        )
        for (i, j), t in zip(jobs, res, strict=True):
            rep = _tag(t, "replacement")
            edits[i].contradicted[j] = (
                replace_span(explanations[i], spans[i][j], rep) if rep else None
            )
            edits[i].replacements[j] = rep if edits[i].contradicted[j] else None
            edits[i].raw.setdefault("contradict", {})[j] = t
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
    """Edits pre-authored in a JSONL file (hand / gold path). One object per explanation:
    {"idx", "claims": [{"claim", "excerpt", "contradiction"}...], "paraphrase", "translation",
    and optionally the EXP002 whole-explanation kinds "polarity", "vocab", "cat"} where
    `excerpt` is a verbatim span of the explanation and `contradiction` is the replacement for
    that span (deletion is derived programmatically). Whole-explanation texts equal to the
    original are dropped."""

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
            z = explanations[i]
            e = Edits(
                idx=i,
                paraphrase=r.get("paraphrase"),
                translation=r.get("translation"),
                whole={k: (r.get(k) or None) for k in WHOLE_KINDS if (r.get(k) or None) != z},
                raw={"source": "file"},
            )
            for c in list(r.get("claims", []))[:max_claims]:
                span = locate_span(z, c.get("excerpt", "")) if c.get("excerpt") else None
                e.claims.append(c["claim"])
                e.spans.append(z[span[0] : span[1]] if span else None)
                e.deleted.append(delete_span(z, span) if span else None)
                con = (
                    replace_span(z, span, c["contradiction"])
                    if span and c.get("contradiction")
                    else None
                )
                e.contradicted.append(con)
                e.replacements.append(c["contradiction"].strip() if con else None)
            out.append(e)
        return out
