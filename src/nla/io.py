"""Artifact I/O: experiment dirs, parquet/JSON helpers, GCS sync (gsutil on the VM)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def artifact_root() -> Path:
    return Path(os.environ.get("ARTIFACT_DIR") or "./artifacts").expanduser()


def exp_dir(name: str) -> Path:
    d = artifact_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_npy(arr: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def sync_to_gcs(local_dir: Path, subdir: str | None = None) -> None:
    """Mirror `local_dir` to $GCS_ARTIFACTS[/subdir] with gsutil (VM). No-op when unset
    or when gsutil is missing (laptop analysis runs)."""
    dest = os.environ.get("GCS_ARTIFACTS", "")
    if not dest or shutil.which("gsutil") is None or os.environ.get("DEVICE", "").lower() != "tpu":
        return  # only the VM pushes; laptop runs work on pulled copies
    dest = dest.rstrip("/") + "/" + (subdir or local_dir.name)
    cmd = ["gsutil", "-m", "-q", "rsync", "-r", str(local_dir), dest]
    print(f"[gcs] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[gcs] WARNING rsync failed (rc={r.returncode}): {r.stderr[-800:]}", flush=True)
