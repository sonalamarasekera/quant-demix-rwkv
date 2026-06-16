"""PIT SI-SDR loss utilities for training and evaluation."""
from __future__ import annotations

from typing import Tuple

import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB. est, ref: [B, T] -> returns [B]"""
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)

    dot = (est_zm * ref_zm).sum(dim=-1, keepdim=True)
    ref_energy = (ref_zm ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = dot / ref_energy * ref_zm

    e_noise = est_zm - s_target
    s_target_energy = (s_target ** 2).sum(dim=-1) + eps
    e_noise_energy = (e_noise ** 2).sum(dim=-1) + eps

    return 10 * torch.log10(s_target_energy / e_noise_energy)


def pit_si_sdr_loss(est_sources: torch.Tensor, ref_sources: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Permutation-invariant SI-SDR loss. Returns negative average SI-SDR (to minimize)."""
    B, S, T = est_sources.shape
    assert S == ref_sources.shape[1]

    # Compute pairwise SI-SDRs
    perms = []
    for p in range(S):
        perm_ref = ref_sources.flip(dims=[1]) if p == 1 and S == 2 else ref_sources
        perms.append(si_sdr(est_sources.reshape(B * S, T), perm_ref.reshape(B * S, T), eps).view(B, S))

    if len(perms) == 1:
        best = perms[0].sum(dim=1)
    else:
        stacked = torch.stack(perms, dim=0)  # [P, B, S]
        summed = stacked.sum(dim=-1)  # [P, B]
        best, _ = torch.max(summed, dim=0)

    return -best.mean()


def pit_si_sdr_with_perm(est_sources: torch.Tensor, true_sources: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return average SI-SDR and best permutation index for S=2 case.

    est_sources, true_sources: [B, S, T]
    Returns: (avg_sdr: [B], best_perm: [B])
    """
    B, S, T = est_sources.shape
    assert S == 2

    est1, est2 = est_sources[:, 0], est_sources[:, 1]
    s1, s2 = true_sources[:, 0], true_sources[:, 1]

    sdr11 = si_sdr(est1, s1)
    sdr22 = si_sdr(est2, s2)
    sum_perm1 = sdr11 + sdr22

    sdr12 = si_sdr(est1, s2)
    sdr21 = si_sdr(est2, s1)
    sum_perm2 = sdr12 + sdr21

    best_perm = (sum_perm2 > sum_perm1).long()

    avg_sdr = torch.where(
        best_perm == 0,
        (sdr11 + sdr22) / 2.0,
        (sdr12 + sdr21) / 2.0,
    )

    return avg_sdr, best_perm


def reorder_sources(est_sources: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    B, S, T = est_sources.shape
    assert S == 2
    reordered = torch.zeros_like(est_sources)
    for i in range(B):
        if perm[i] == 0:
            reordered[i, 0] = est_sources[i, 0]
            reordered[i, 1] = est_sources[i, 1]
        else:
            reordered[i, 0] = est_sources[i, 1]
            reordered[i, 1] = est_sources[i, 0]
    return reordered
