"""Model registry for rwkv_ss.

Provides a simple `build_model(cfg)` factory that dispatches to
registered model builders. For now, supports RWKV v7 separator.
"""
from __future__ import annotations

from typing import Any

try:
    from rwkv_ss.models.rwkv_v7 import build_rwkv7_separator
    from rwkv_ss.models.rwkv_v7.config import SeparatorV7Config
except Exception:
    build_rwkv7_separator = None  # type: ignore
    SeparatorV7Config = None  # type: ignore


def build_model(cfg: Any):
    """Build a model from a config object or dict.

    If `cfg` is a `SeparatorV7Config` dataclass, dispatch to RWKVv7 factory.
    If `cfg` is a dict containing keys for RWKV, dispatch accordingly.
    """
    if build_rwkv7_separator is None:
        raise RuntimeError("RWKV v7 builder not available in registry")

    # If it's the SeparatorV7Config dataclass
    if type(cfg).__name__ == "SeparatorV7Config":
        # mypy: dynamic attributes
        return build_rwkv7_separator(
            n_embd=cfg.n_embd,
            codec_dim=cfg.codec_dim,
            n_layer=cfg.n_layer,
            num_sources=cfg.num_sources,
            head_size_a=cfg.head_size_a,
            head_hidden=cfg.head_hidden,
            head_mode=cfg.head_mode,
            enforce_bf16=cfg.enforce_bf16,
            n_groups=cfg.n_groups,
        )

    # If cfg is a plain dict
    if isinstance(cfg, dict):
        return build_rwkv7_separator(
            n_embd=cfg["n_embd"],
            codec_dim=cfg["codec_dim"],
            n_layer=cfg.get("n_layer", 8),
            num_sources=cfg.get("num_sources", 2),
            head_size_a=cfg.get("head_size_a", 64),
            head_hidden=cfg["head_hidden"],
            head_mode=cfg.get("head_mode", "residual"),
            enforce_bf16=cfg.get("enforce_bf16", True),
            n_groups=cfg.get("n_groups", 2),
        )

    raise TypeError("Unsupported config type for build_model")
