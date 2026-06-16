"""Lightweight training entrypoint. Tries to use Hydra if available, otherwise falls back to a YAML config loader."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

try:
    import hydra
    from omegaconf import DictConfig
    _HAS_HYDRA = True
except Exception:
    _HAS_HYDRA = False

import yaml

from rwkv_ss.models.registry import build_model
from rwkv_ss.transforms.stft import STFTProcessor
from rwkv_ss.data.datamodule import Libri2MixDataModule
from rwkv_ss.training.trainer import Trainer


def _run_with_cfg(cfg: Dict[str, Any]):
    # Build datamodule
    data_cfg = cfg.get("data", {})
    dm = Libri2MixDataModule(data_cfg)

    # STFT
    stft_cfg = cfg.get("stft", {})
    stft = STFTProcessor(**stft_cfg)

    # Model
    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg)

    trainer = Trainer(model=model, datamodule=dm, stft_processor=stft, cfg=cfg)
    trainer.fit(max_epochs=int(cfg.get("max_epochs", 100)))


if _HAS_HYDRA:
    @hydra.main(config_path=None)
    def main(cfg: DictConfig):
        _run_with_cfg(cfg)


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf8") as f:
        cfg = yaml.safe_load(f)

    _run_with_cfg(cfg)


if __name__ == "__main__":
    if _HAS_HYDRA and len(sys.argv) == 1:
        # hydra will handle commandline if used via `python -m scripts.train`
        main()
    else:
        main_cli()
