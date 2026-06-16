"""Shared separation pipeline: STFT -> model -> iSTFT reconstruction."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from rwkv_ss.transforms.stft import STFTProcessor


def run_separation(mix_wav: torch.Tensor, model: torch.nn.Module, stft_processor: STFTProcessor) -> torch.Tensor:
    """Run STFT -> model -> iSTFT and return separated waveforms.

    Args:
        mix_wav: [B, 1, T_wav]
        model: model expecting [B, T_tf, F] input and returning [B, T_tf, S, F]
        stft_processor: STFTProcessor instance

    Returns:
        sep_wav: [B, S, T_wav]
    """
    B, _, T_wav = mix_wav.shape

    mix_mag, mix_phase = stft_processor.stft(mix_wav)  # [B, F, T_tf]
    mix_mag_t = mix_mag.transpose(1, 2)  # [B, T_tf, F]

    model_output = model(mix_mag_t)
    if isinstance(model_output, tuple):
        sep_mag_t = model_output[0]
    else:
        sep_mag_t = model_output

    # Handle shapes: [B, T_tf, S, F] or [B, T_tf, S*F]
    if sep_mag_t.dim() == 4:
        # [B, T_tf, S, F] -> [B, S, F, T_tf]
        sep_mag = sep_mag_t.permute(0, 2, 3, 1)
    elif sep_mag_t.dim() == 3:
        B_out, T_tf, sep_dim = sep_mag_t.shape
        n_freq = mix_mag.shape[1]
        S = sep_dim // n_freq
        sep_mag = sep_mag_t.view(B_out, T_tf, S, n_freq).permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unexpected model output shape: {sep_mag_t.shape}")

    sep_mag = F.relu(sep_mag)

    sep_wav = stft_processor.istft(sep_mag, mix_phase, length=T_wav)  # [B, S, T_wav]

    # Align lengths to minimum
    return sep_wav
