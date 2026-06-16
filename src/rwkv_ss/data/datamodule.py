"""DataLoader factory for Libri2Mix CSV datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from rwkv_ss.data.datasets.libri2mix import (
    Libri2MixDataset,
    Libri2MixEvalDataset,
    collate_eval_fn,
    collate_train_fn,
)


@dataclass
class DataConfig:
    train_csv: str
    valid_csv: str
    sample_rate: int = 16000
    segment_seconds: float = 3.0
    batch_size: int = 8
    subset_frac: float | None = None
    subset_seed: int = 42
    num_workers_train: int = 4
    num_workers_valid: int = 2
    drop_last: bool = True
    pin_memory: bool = True
    shuffle_train: bool = True


class Libri2MixDataModule:
    """Builds train/val/test dataloaders without training-loop dependencies."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.train_ds: Libri2MixDataset | None = None
        self.valid_ds: Libri2MixDataset | None = None
        self.test_ds: Libri2MixEvalDataset | None = None

    def setup(self, test_csv: str | None = None) -> None:
        cfg = self.config

        for path, label in (
            (cfg.train_csv, "train_csv"),
            (cfg.valid_csv, "valid_csv"),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        self.train_ds = Libri2MixDataset(
            cfg.train_csv,
            sample_rate=cfg.sample_rate,
            segment_seconds=cfg.segment_seconds,
            subset_frac=cfg.subset_frac,
            subset_seed=cfg.subset_seed,
        )
        self.valid_ds = Libri2MixDataset(
            cfg.valid_csv,
            sample_rate=cfg.sample_rate,
            segment_seconds=cfg.segment_seconds,
            subset_frac=cfg.subset_frac,
            subset_seed=cfg.subset_seed,
        )

        if test_csv is not None:
            if not Path(test_csv).is_file():
                raise FileNotFoundError(f"test_csv not found: {test_csv}")
            self.test_ds = Libri2MixEvalDataset(
                test_csv,
                sample_rate=cfg.sample_rate,
                subset_frac=cfg.subset_frac,
                subset_seed=cfg.subset_seed,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("Call setup() before requesting train_dataloader().")
        cfg = self.config
        return DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_train,
            num_workers=cfg.num_workers_train,
            collate_fn=collate_train_fn,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.valid_ds is None:
            raise RuntimeError("Call setup() before requesting val_dataloader().")
        cfg = self.config
        return DataLoader(
            self.valid_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers_valid,
            collate_fn=collate_train_fn,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory,
        )

    def test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("Call setup(test_csv=...) before requesting test_dataloader().")
        cfg = self.config
        return DataLoader(
            self.test_ds,
            batch_size=batch_size or cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers_valid,
            collate_fn=collate_eval_fn,
            drop_last=False,
            pin_memory=cfg.pin_memory,
        )
