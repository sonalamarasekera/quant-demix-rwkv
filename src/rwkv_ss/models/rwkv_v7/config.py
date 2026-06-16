"""Configuration dataclasses for RWKV v7 separator."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SeparatorV7Config:
    n_embd: int = 512
    codec_dim: int = 1024
    n_layer: int = 8
    head_size_a: int = 64
    enforce_bf16: bool = True
    num_sources: int = 2
    head_hidden: int = 256
    head_mode: str = "residual"
    dropout: float = 0.1
    n_groups: int = 2
