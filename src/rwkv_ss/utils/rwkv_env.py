"""RWKV environment bootstrap utilities.

Set RWKV_* environment variables before importing RWKV modules.
Provide a fail-fast checker for CUDA kernel availability.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

_DEFAULTS: Dict[str, str] = {
    "RWKV_JIT_ON": "1",
    "RWKV_CUDA_ON": "1",
    "RWKV_MY_TESTING": "x070",
    "RWKV_FLOAT_MODE": "bf16",
    "RWKV_HEAD_SIZE_A": "64",
}


def apply_default_rwkv_env(overrides: Optional[Dict[str, str]] = None) -> None:
    """Apply default RWKV_* env vars without overwriting existing values.

    Call this before importing any RWKV modules.
    """
    env = dict(_DEFAULTS)
    if overrides:
        env.update(overrides)

    for k, v in env.items():
        os.environ.setdefault(k, v)


def apply_default_rwkv_env_and_check(overrides: Optional[Dict[str, str]] = None, require_cuda: bool = True) -> bool:
    """Apply defaults and optionally ensure CUDA RWKV kernels are available.

    Returns True if CUDA kernels are present (or not required), False otherwise.
    If `require_cuda` is True and CUDA kernels are missing, raises RuntimeError.
    """
    apply_default_rwkv_env(overrides)
    return ensure_cuda_available(require_cuda=require_cuda)


def ensure_cuda_available(require_cuda: bool = True) -> bool:
    """Attempt to import RWKV CUDA symbol to verify availability.

    If ``require_cuda`` is True and the import fails, raises RuntimeError.
    Returns True if CUDA symbol present, False otherwise.
    """
    try:
        # Module path mirrors where older code imported RUN_CUDA_RWKV7g
        from RWKV.RWKV_v7.train_temp.src.model import RUN_CUDA_RWKV7g  # type: ignore
        return True
    except Exception as exc:
        if require_cuda:
            raise RuntimeError(
                "RWKV CUDA kernels not available. Install the RWKV CUDA package or set RWKV_CUDA_ON=0"
            ) from exc
        return False
