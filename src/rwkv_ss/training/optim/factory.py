"""Optimizer and scheduler factory helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    eps = float(cfg.get("eps", 1e-8))

    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = cfg.get("scheduler")
    if not sched_cfg:
        return None

    name = sched_cfg.get("name", "ReduceLROnPlateau")
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "min"),
            factor=float(sched_cfg.get("factor", 0.1)),
            patience=int(sched_cfg.get("patience", 10)),
            verbose=bool(sched_cfg.get("verbose", False)),
        )
    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )

    raise ValueError(f"Unsupported scheduler name: {name}")
