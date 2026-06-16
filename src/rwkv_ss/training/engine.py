"""Training engine: train/validate epoch loops using the shared pipeline."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from rwkv_ss.transforms.stft import STFTProcessor
from rwkv_ss.training.losses.pit_si_sdr import pit_si_sdr_loss
from rwkv_ss.training.pipeline import run_separation


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    stft_processor: STFTProcessor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
    writer: Optional[SummaryWriter] = None,
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch:03d} TRAIN]", ncols=120)
    for batch_idx, (mix_wav, src_wav) in enumerate(pbar):
        mix_wav = mix_wav.to(device)
        src_wav = src_wav.to(device)

        optimizer.zero_grad()

        sep_wav = run_separation(mix_wav, model, stft_processor)

        B, S, _, T_wav = src_wav.shape
        src_wav = src_wav.squeeze(2)
        T_min = min(sep_wav.size(-1), src_wav.size(-1))
        loss = pit_si_sdr_loss(sep_wav[..., :T_min], src_wav[..., :T_min])

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        num_batches += 1
        pbar.set_postfix(train_loss=total_loss / num_batches)

        if writer is not None:
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar("batch/train_loss", batch_loss, global_step)

    return total_loss / max(1, num_batches)


def validate(
    epoch: int,
    model: nn.Module,
    stft_processor: STFTProcessor,
    valid_loader: DataLoader,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(valid_loader, desc=f"[Epoch {epoch:03d} VALID]", ncols=120)
    with torch.no_grad():
        for mix_wav, src_wav in pbar:
            mix_wav = mix_wav.to(device)
            src_wav = src_wav.to(device)

            sep_wav = run_separation(mix_wav, model, stft_processor)

            B, S, _, T_wav = src_wav.shape
            src_wav = src_wav.squeeze(2)
            T_min = min(sep_wav.size(-1), src_wav.size(-1))
            loss = pit_si_sdr_loss(sep_wav[..., :T_min], src_wav[..., :T_min])

            batch_loss = float(loss.item())
            total_loss += batch_loss
            num_batches += 1
            pbar.set_postfix(val_loss=total_loss / num_batches)

    return total_loss / max(1, num_batches)
