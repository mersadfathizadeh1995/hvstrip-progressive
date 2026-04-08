"""Batch statistics for dual-resonance analyses."""

from typing import List

import numpy as np

from .data_structures import BatchDualResonanceStats, DualResonanceResult


def compute_batch_statistics(
    results: List[DualResonanceResult],
) -> BatchDualResonanceStats:
    """Compute aggregate statistics from a list of profile results."""
    successful = [r for r in results if r.success]
    n_ok = len(successful)

    if n_ok == 0:
        return BatchDualResonanceStats(
            n_profiles=len(results),
            n_successful=0,
            success_rate=0.0,
        )

    f0s = np.array([r.f0 for r in successful])
    f1s = np.array([r.f1 for r in successful])
    f0t = np.array([r.f0_theoretical for r in successful])
    f1t = np.array([r.f1_theoretical for r in successful])
    ratios = np.array([r.freq_ratio for r in successful])

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    # Per-step shift distributions
    max_steps = max(len(r.freq_per_step) for r in successful)
    shift_lists: List[List[float]] = [[] for _ in range(max(max_steps - 1, 0))]
    for r in successful:
        for i in range(1, len(r.freq_per_step)):
            if i - 1 < len(shift_lists):
                shift_lists[i - 1].append(
                    r.freq_per_step[i] - r.freq_per_step[i - 1]
                )

    mean_shifts = [float(np.mean(s)) if s else 0.0 for s in shift_lists]
    std_shifts = [float(np.std(s)) if s else 0.0 for s in shift_lists]
    sep_count = sum(1 for r in successful if r.separation_success)

    return BatchDualResonanceStats(
        n_profiles=len(results),
        n_successful=n_ok,
        success_rate=n_ok / len(results) * 100 if results else 0.0,
        f0_mean=float(np.mean(f0s)),
        f0_std=float(np.std(f0s)),
        f0_min=float(np.min(f0s)),
        f0_max=float(np.max(f0s)),
        f1_mean=float(np.mean(f1s)),
        f1_std=float(np.std(f1s)),
        f1_min=float(np.min(f1s)),
        f1_max=float(np.max(f1s)),
        freq_ratio_mean=float(np.mean(ratios)),
        freq_ratio_std=float(np.std(ratios)),
        f0_theoretical_correlation=_safe_corr(f0s, f0t),
        f1_theoretical_correlation=_safe_corr(f1s, f1t),
        separation_success_rate=sep_count / n_ok * 100 if n_ok else 0.0,
        mean_freq_shift_per_step=mean_shifts,
        std_freq_shift_per_step=std_shifts,
    )
