"""Device selection and accelerator helpers.

`DEVICE=cuda` (the H200 studio, EXP002) selects CUDA; `DEVICE=tpu` (the retired EXP001 VM)
selects the torch_xla device (imported lazily); anything else is CPU (blank = CUDA if
available, else CPU).
"""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    dev = os.environ.get("DEVICE", "").lower()
    if dev == "tpu":
        import torch_xla  # Linux/TPU only

        return torch_xla.device()
    if dev == "cuda" or (not dev and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def is_xla(device: torch.device) -> bool:
    return device.type == "xla"


def sync(device: torch.device) -> None:
    """Materialise pending lazy-tensor work (XLA) / wait for queued kernels (CUDA)."""
    if is_xla(device):
        import torch_xla

        torch_xla.sync()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def matmul_dtype(device: torch.device) -> torch.dtype:
    """bf16 on accelerators; fp32 on CPU (bf16 is emulated and ~50x slower there)."""
    return torch.bfloat16 if device.type in ("xla", "cuda") else torch.float32


def manual_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if is_xla(device):
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def memory_info(device: torch.device) -> dict[str, float]:
    """Accelerator memory usage in GB (empty on CPU)."""
    if is_xla(device):
        import torch_xla.core.xla_model as xm

        info = xm.get_memory_info(device)
        return {k: round(v / 1e9, 2) for k, v in info.items() if isinstance(v, int | float)}
    if device.type == "cuda":
        return {
            "allocated_gb": round(torch.cuda.memory_allocated(device) / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(device) / 1e9, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        }
    return {}


def free_accelerator() -> None:
    """Release cached blocks after a model is deleted (stages swap 27B-class models)."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
