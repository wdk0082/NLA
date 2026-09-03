"""Corpus loading and context sampling.

Ultra-FineWeb is the NLA's training distribution (50/50 with WildChat). We stream one
parquet shard and sample ONE extraction position per document, at least `min_pos` tokens
in (the release's datagen uses min position 50), so the activation has real context.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import pandas as pd
from datasets import load_dataset

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
