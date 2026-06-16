"""Shared pytest fixtures."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rwkv_ss.data.datamodule import DataConfig, Libri2MixDataModule


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def synthetic_audio_dir(tmp_path: Path, sample_rate: int) -> Path:
    audio_dir = tmp_path / "audio"
    mix_dir = audio_dir / "mix_clean"
    s1_dir = audio_dir / "s1"
    s2_dir = audio_dir / "s2"
    for d in (mix_dir, s1_dir, s2_dir):
        d.mkdir(parents=True)

    duration_sec = 0.5
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    for i in range(3):
        stem = f"utt_{i}"
        mix = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        s1 = np.sin(2 * np.pi * 440 * t)
        s2 = np.sin(2 * np.pi * 880 * t)
        sf.write(mix_dir / f"{stem}.wav", mix.astype(np.float32), sample_rate)
        sf.write(s1_dir / f"{stem}.wav", s1.astype(np.float32), sample_rate)
        sf.write(s2_dir / f"{stem}.wav", s2.astype(np.float32), sample_rate)

    return audio_dir


@pytest.fixture
def synthetic_csv(synthetic_audio_dir: Path, tmp_path: Path) -> Path:
    mix_dir = synthetic_audio_dir / "mix_clean"
    s1_dir = synthetic_audio_dir / "s1"
    s2_dir = synthetic_audio_dir / "s2"

    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mix_path", "s1_path", "s2_path"])
        for mix_path in sorted(mix_dir.glob("*.wav")):
            stem = mix_path.stem
            writer.writerow(
                [
                    str(mix_path),
                    str(s1_dir / f"{stem}.wav"),
                    str(s2_dir / f"{stem}.wav"),
                ]
            )
    return csv_path


@pytest.fixture
def datamodule_config(synthetic_csv: Path) -> DataConfig:
    return DataConfig(
        train_csv=str(synthetic_csv),
        valid_csv=str(synthetic_csv),
        sample_rate=16000,
        segment_seconds=0.25,
        batch_size=2,
        num_workers_train=0,
        num_workers_valid=0,
        drop_last=False,
        pin_memory=False,
    )
