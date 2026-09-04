"""Corpus loading and context sampling.

EXP001: Ultra-FineWeb (the 7B NLA's training distribution) streamed from the Hub, ONE
extraction position per document, uniformly >= `min_pos` tokens in (the release's datagen uses
min position 50), context = the raw page from token 0.

EXP002: FineFineWeb (the 27B NLA's training corpus, domain-labelled jsonl shards), fixed clean
start (the first prose paragraph) and a random whole-word end position — see
`sample_contexts_prose`.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

CORPORA: dict[str, dict[str, Any]] = {
    "ultrafineweb": {
        "path": "openbmb/Ultra-FineWeb",
        "data_files": "data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet",
        "text_col": "content",
    },
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "data_files": "sample/10BT/000_00000.parquet",
        "text_col": "text",
    },
}


def iter_docs(corpus: str, skip: int = 0):
    spec = CORPORA[corpus]
    ds = load_dataset(
        spec["path"], data_files={"train": spec["data_files"]}, split="train", streaming=True
    )
    for i, row in enumerate(ds):
        if i < skip:
            continue
        yield f"{spec['path']}:{spec['data_files']}:{i}", row[spec["text_col"]]


def _doc_rng(seed: int, doc_id: str) -> random.Random:
    return random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())


def sample_contexts(
    corpus: str,
    tokenizer: Any,
    n: int,
    max_ctx: int = 512,
    min_pos: int = 50,
    seed: int = 0,
    skip: int = 0,
) -> pd.DataFrame:
    """Sample `n` (doc, position) pairs. Returns columns:
    idx, doc_id, pos (0-based token index of the extraction position), n_ctx (= pos+1),
    ids (list[int] of the context tokens), x_text (decoded context)."""
    rows: list[dict[str, Any]] = []
    for doc_id, text in iter_docs(corpus, skip=skip):
        if len(rows) >= n:
            break
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_ctx)[
            "input_ids"
        ]
        if len(ids) <= min_pos + 1:
            continue
        pos = _doc_rng(seed, doc_id).randint(min_pos, len(ids) - 1)
        ctx = [int(t) for t in ids[: pos + 1]]
        rows.append(
            {
                "idx": len(rows),
                "doc_id": doc_id,
                "pos": pos,
                "n_ctx": pos + 1,
                "ids": ctx,
                "x_text": tokenizer.decode(ctx),
            }
        )
    assert len(rows) == n, f"only {len(rows)} usable documents (wanted {n})"
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- FineFineWeb (EXP002)

FFW_REPO = "m-a-p/FineFineWeb"  # domain-labelled FineWeb; the 27B NLA's training corpus
_LIST_RE = re.compile(r"^(\d+[.)]|[-*•–]|[A-Za-z][.)]\s|#{1,6}\s|\S+:$|\S+:\s)")
_WORD_RE = re.compile(r"^ [A-Za-z]+$")
_ALPHA_START_RE = re.compile(r"^[A-Za-z]")


def ffw_shard_path(domain: str, shard: int = 0) -> Path:
    """Local path of one FineFineWeb jsonl shard (downloaded into the HF cache if missing)."""
    return Path(
        hf_hub_download(FFW_REPO, f"{domain}/{domain}_{shard:06d}.jsonl", repo_type="dataset")
    )


def iter_ffw_docs(domain: str, shard: int = 0, skip: int = 0, min_lang_score: float = 0.0):
    """Yield (doc_id, text) from one shard, in file order."""
    with open(ffw_shard_path(domain, shard)) as fh:
        for i, line in enumerate(fh):
            if i < skip:
                continue
            row = json.loads(line)
            if float(row.get("language_score", 1.0)) < min_lang_score:
                continue
            yield f"ffw:{domain}:{shard}:{i}", row["text"]


def prose_paragraph_start(text: str, min_words: int = 40) -> int | None:
    """Character offset of the first paragraph of running prose: starts with a capital letter,
    at least `min_words` words, no list / numbering / heading pattern, no URL, ends in
    sentence punctuation. None if the document has no such paragraph."""
    pos = 0
    for raw in text.split("\n"):
        start = pos + (len(raw) - len(raw.lstrip()))
        p = raw.strip()
        pos += len(raw) + 1
        if len(p.split()) < min_words or not p[0].isupper() or _LIST_RE.match(p):
            continue
        if p[-1] not in ".!?\"'”’)" or "http" in p or "www." in p:
            continue
        return start
    return None


def word_token_positions(tokenizer: Any, ids: list[int]) -> list[int]:
    """Positions t whose token is a whole alphabetic word: a leading-space letters-only token
    that the next token does not continue (the next token starts with a space, punctuation,
    a newline or a digit, or there is no next token)."""
    pieces = [tokenizer.decode([i]) for i in ids]
    out = []
    for t, s in enumerate(pieces):
        if not _WORD_RE.match(s):
            continue
        if t + 1 < len(pieces) and _ALPHA_START_RE.match(pieces[t + 1]):
            continue
        out.append(t)
    return out


def sample_contexts_prose(
    tokenizer: Any,
    domains: list[str],
    n: int,
    max_ctx: int = 256,
    min_pos: int = 50,
    seed: int = 0,
    shard: int = 0,
    skip: int = 0,
    min_lang_score: float = 0.9,
    min_words: int = 40,
) -> pd.DataFrame:
    """`n` contexts spread evenly over `domains`, one per document. Each context starts at the
    first prose paragraph of its document (fixed clean start) and ends at a whole-word token
    drawn uniformly from positions >= `min_pos` within the first `max_ctx` tokens after that
    start (random end, never a sentence-final punctuation token). Columns: idx, doc_id,
    domain, pos, n_ctx, ids, x_text, final_token, para_start_char."""
    per = [n // len(domains) + (1 if k < n % len(domains) else 0) for k in range(len(domains))]
    rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}
    for domain, want in zip(domains, per, strict=True):
        st = stats[domain] = {"seen": 0, "no_prose": 0, "short": 0, "no_word_end": 0, "kept": 0}
        for doc_id, text in iter_ffw_docs(domain, shard, skip, min_lang_score):
            if st["kept"] >= want:
                break
            st["seen"] += 1
            start = prose_paragraph_start(text, min_words)
            if start is None:
                st["no_prose"] += 1
                continue
            ids = tokenizer(
                text[start:], add_special_tokens=False, truncation=True, max_length=max_ctx + 1
            )["input_ids"]
            if len(ids) <= min_pos + 1:
                st["short"] += 1
                continue
            cands = [t for t in word_token_positions(tokenizer, ids) if min_pos <= t < max_ctx]
            if not cands:
                st["no_word_end"] += 1
                continue
            pos = _doc_rng(seed, doc_id).choice(cands)
            ctx = [int(t) for t in ids[: pos + 1]]
            rows.append(
                {
                    "idx": len(rows),
                    "doc_id": doc_id,
                    "domain": domain,
                    "pos": pos,
                    "n_ctx": pos + 1,
                    "ids": ctx,
                    "x_text": tokenizer.decode(ctx),
                    "final_token": tokenizer.decode([ctx[-1]]),
                    "para_start_char": start,
                }
            )
            st["kept"] += 1
        assert st["kept"] == want, f"{domain}: only {st['kept']} usable documents (wanted {want})"
    df = pd.DataFrame(rows)
    df.attrs["sampling_stats"] = stats
    return df
