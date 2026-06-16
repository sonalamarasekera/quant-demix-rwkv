"""Libri2Mix CSV-backed datasets for speech separation."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


def load_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("mix_path") or not row.get("s1_path") or not row.get("s2_path"):
                continue
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid rows found in CSV: {csv_path}")
    return rows


def subset_rows(
    rows: list[dict[str, str]],
    *,
    max_samples: int | None = None,
    subset_frac: float | None = None,
    subset_seed: int = 42,
) -> list[dict[str, str]]:
    n = len(rows)

    if max_samples is not None:
        n_keep = min(max_samples, n)
    elif subset_frac is not None:
        n_keep = max(1, int(round(n * subset_frac)))
    else:
        return rows

    if n_keep >= n:
        return rows

    rng = random.Random(subset_seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return [rows[i] for i in indices[:n_keep]]


def load_mono(path: str, sample_rate: int) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data.T)
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


class Libri2MixDataset(Dataset):
    """Training dataset with random fixed-length segment crop/pad."""

    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        segment_seconds: float = 3.0,
        max_samples: int | None = None,
        subset_frac: float | None = None,
        subset_seed: int = 42,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.rows = subset_rows(
            load_csv_rows(csv_path),
            max_samples=max_samples,
            subset_frac=subset_frac,
            subset_seed=subset_seed,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        mix = load_mono(row["mix_path"], self.sample_rate)
        s1 = load_mono(row["s1_path"], self.sample_rate)
        s2 = load_mono(row["s2_path"], self.sample_rate)

        t = min(mix.size(-1), s1.size(-1), s2.size(-1))
        mix = mix[..., :t]
        s1 = s1[..., :t]
        s2 = s2[..., :t]

        seg = self.segment_samples
        if t > seg:
            start = torch.randint(0, t - seg + 1, (1,)).item()
            end = start + seg
            mix = mix[..., start:end]
            s1 = s1[..., start:end]
            s2 = s2[..., start:end]
        elif t < seg:
            pad = seg - t
            mix = F.pad(mix, (0, pad))
            s1 = F.pad(s1, (0, pad))
            s2 = F.pad(s2, (0, pad))

        sources = torch.stack([s1, s2], dim=0)
        return {"mix": mix, "sources": sources}


class Libri2MixEvalDataset(Dataset):
    """Evaluation dataset returning full utterances and CSV row metadata."""

    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 16000,
        max_samples: int | None = None,
        subset_frac: float | None = None,
        subset_seed: int = 42,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.rows = subset_rows(
            load_csv_rows(csv_path),
            max_samples=max_samples,
            subset_frac=subset_frac,
            subset_seed=subset_seed,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        mix = load_mono(row["mix_path"], self.sample_rate)
        s1 = load_mono(row["s1_path"], self.sample_rate)
        s2 = load_mono(row["s2_path"], self.sample_rate)

        t = min(mix.size(-1), s1.size(-1), s2.size(-1))
        mix = mix[..., :t]
        s1 = s1[..., :t]
        s2 = s2[..., :t]

        sources = torch.stack([s1, s2], dim=0)
        return {"mix": mix, "sources": sources, "row": row}


def collate_train_fn(
    batch: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [b["mix"].shape[-1] for b in batch]
    t_max = max(lengths)

    mix_list: list[torch.Tensor] = []
    sources_list: list[torch.Tensor] = []

    for b in batch:
        mix = b["mix"]
        sources = b["sources"]
        pad_t = t_max - mix.shape[-1]

        if pad_t > 0:
            mix = F.pad(mix, (0, pad_t))
            sources = F.pad(sources, (0, pad_t))

        mix_list.append(mix)
        sources_list.append(sources)

    return torch.stack(mix_list, dim=0), torch.stack(sources_list, dim=0)


def collate_eval_fn(
    batch: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    lengths = [b["mix"].shape[-1] for b in batch]
    t_max = max(lengths)

    mix_list: list[torch.Tensor] = []
    sources_list: list[torch.Tensor] = []
    rows: list[dict[str, str]] = []

    for b in batch:
        mix = b["mix"]
        sources = b["sources"]
        pad_t = t_max - mix.shape[-1]

        if pad_t > 0:
            mix = F.pad(mix, (0, pad_t))
            sources = F.pad(sources, (0, pad_t))

        mix_list.append(mix)
        sources_list.append(sources)
        rows.append(b["row"])

    return torch.stack(mix_list, 0), torch.stack(sources_list, 0), rows
