"""Device selection and XLA helpers.

`DEVICE=tpu` (set by gcp/launch*.sh on the VM) selects the torch_xla device; anything
else is CPU. torch_xla is imported lazily so the package imports on the laptop.
"""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    if os.environ.get("DEVICE", "").lower() == "tpu":
        import torch_xla  # Linux/TPU only

        return torch_xla.device()
    return torch.device("cpu")


def is_xla(device: torch.device) -> bool:
    return device.type == "xla"


def sync(device: torch.device) -> None:
    """Materialise pending lazy-tensor work (no-op on CPU)."""
    if is_xla(device):
        import torch_xla

        torch_xla.sync()


def matmul_dtype(device: torch.device) -> torch.dtype:
    """bf16 on the TPU; fp32 on CPU (bf16 is emulated and ~50x slower there)."""
    return torch.bfloat16 if is_xla(device) else torch.float32


def manual_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if is_xla(device):
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(seed)


def memory_info(device: torch.device) -> dict[str, float]:
    """HBM usage in GB (TPU only)."""
    if not is_xla(device):
        return {}
    import torch_xla.core.xla_model as xm

    info = xm.get_memory_info(device)
    return {k: round(v / 1e9, 2) for k, v in info.items() if isinstance(v, int | float)}
