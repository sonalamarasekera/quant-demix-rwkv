"""Simple checkpoint callback to persist training state."""
from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import torch


class CheckpointCallback:
    def __init__(self, directory: str, prefix: str = "ckpt"):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.prefix = prefix

    def _path_for_epoch(self, epoch: int) -> str:
        return os.path.join(self.directory, f"{self.prefix}-epoch{epoch:03d}.pt")

    def save(self, epoch: int, state: Dict[str, Any]):
        path = self._path_for_epoch(epoch)
        torch.save(state, path)
        return path
