"""Hugging Face Hub helpers (uses the HF_HOME cache; all repos used here are ungated)."""

from __future__ import annotations

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
    return Path(snapshot_download(repo_id, allow_patterns=patterns, repo_type=repo_type))
