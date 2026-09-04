"""Hugging Face Hub helpers (uses the HF_HOME cache; all repos used here are ungated).

Weights are fetched from the Hub on demand (~1 GB/s on the studio) and are NOT kept in the
studio's persistent disk between sessions if that disk gets too large. `NLA_MODEL_STORE`
(e.g. the Teamspace drive, `/teamspace/uploads/nla-metrics/hf`) may hold flat copies named
`<org>--<name>/...`; they are used only as a FALLBACK when the Hub download fails (repo gone,
offline), because reading through the drive's FUSE mount is slower than a fresh download.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

TOKENIZER_PATTERNS = ["*.json", "*.txt", "*.yaml", "*.jinja", "*.model", "LICENSE", "README.md"]


def snapshot(
    repo_id: str,
    tokenizer_only: bool = False,
    allow_patterns: list[str] | None = None,
    repo_type: str | None = None,
) -> Path:
    """Local snapshot dir for `repo_id` (downloads what is missing). `allow_patterns` limits
    the download to a subset of the repo (e.g. one adapter of a multi-checkpoint repo)."""
    patterns = TOKENIZER_PATTERNS if tokenizer_only else allow_patterns
    try:
        return Path(snapshot_download(repo_id, allow_patterns=patterns, repo_type=repo_type))
    except Exception as e:  # any hub failure: try the local/drive store
        store = os.environ.get("NLA_MODEL_STORE", "")
        local = Path(store) / repo_id.replace("/", "--") if store else None
        if local is not None and local.is_dir():
            print(f"[hub] {repo_id}: hub download failed ({e!r}); using {local}", flush=True)
            return local
        raise
