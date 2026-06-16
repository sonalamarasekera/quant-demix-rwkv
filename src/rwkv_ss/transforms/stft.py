"""Fixed STFT/iSTFT transform for TF-domain speech separation."""

from __future__ import annotations

import torch
import torch.nn as nn


class STFTProcessor(nn.Module):
    """
    Fixed STFT/iSTFT transform for TF-domain processing.
    No learning needed - pure mathematical transform.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int | None = None,
        window: str = "hann",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window_type = window

        if window == "hann":
            self.register_buffer("window", torch.hann_window(self.win_length))
        elif window == "hamming":
            self.register_buffer("window", torch.hamming_window(self.win_length))
        else:
            self.register_buffer("window", torch.ones(self.win_length))

    def stft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute STFT.

        Args:
            x: [B, 1, T_wav] or [B, T_wav] waveform

        Returns:
            magnitude: [B, F, T_tf]
            phase: [B, F, T_tf]
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            normalized=False,
        )

        return spec.abs(), spec.angle()

    def istft(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
        length: int | None = None,
    ) -> torch.Tensor:
        """
        Compute inverse STFT.

        Args:
            magnitude: [B, F, T_tf] or [B, S, F, T_tf]
            phase: [B, F, T_tf]
            length: original waveform length

        Returns:
            wav: [B, T_wav] or [B, S, T_wav]
        """
        if magnitude.dim() == 4:
            b, s, f, t_tf = magnitude.shape
            phase = phase.unsqueeze(1).expand(b, s, f, t_tf)

            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            complex_spec = torch.complex(real, imag)
            complex_spec = complex_spec.reshape(b * s, f, t_tf)

            wav = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                length=length,
                center=True,
                normalized=False,
            )
            return wav.view(b, s, -1)

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        complex_spec = torch.complex(real, imag)

        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
            center=True,
            normalized=False,
        )
