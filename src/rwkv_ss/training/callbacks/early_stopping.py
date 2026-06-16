"""Minimal early stopping callback."""
from __future__ import annotations

from typing import Optional


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, metric: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric < self.best - self.min_delta if self.mode == "min" else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
