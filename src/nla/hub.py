"""Hugging Face Hub helpers (uses the HF_HOME cache; all repos used here are ungated)."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

TOKENIZER_PATTERNS = ["*.json", "*.txt", "*.yaml", "*.jinja", "*.model", "LICENSE", "README.md"]


def snapshot(repo_id: str, tokenizer_only: bool = False) -> Path:
    """Local snapshot dir for `repo_id` (downloads what is missing)."""
    patterns = TOKENIZER_PATTERNS if tokenizer_only else None
    return Path(snapshot_download(repo_id, allow_patterns=patterns))
