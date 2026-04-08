"""I/O helpers for dual-resonance results."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .data_structures import BatchDualResonanceStats, DualResonanceResult


def save_results_csv(
    results: List[DualResonanceResult],
    output_path: str,
) -> None:
    """Save batch results to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "profile_name", "success", "n_layers", "total_depth_m",
        "f0_Hz", "a0", "f0_theoretical_Hz",
        "f1_Hz", "a1", "f1_theoretical_Hz",
        "freq_ratio", "max_freq_shift_Hz",
        "controlling_step", "separation_success", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "profile_name": r.profile_name,
                "success": r.success,
                "n_layers": r.n_layers,
                "total_depth_m": r.total_depth,
                "f0_Hz": r.f0,
                "a0": r.a0,
                "f0_theoretical_Hz": r.f0_theoretical,
                "f1_Hz": r.f1,
                "a1": r.a1,
                "f1_theoretical_Hz": r.f1_theoretical,
                "freq_ratio": r.freq_ratio,
                "max_freq_shift_Hz": r.max_freq_shift,
                "controlling_step": r.controlling_step,
                "separation_success": r.separation_success,
                "error": r.error_message,
            })


def save_statistics_json(
    stats: BatchDualResonanceStats,
    output_path: str,
) -> None:
    """Serialise batch statistics to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(stats), fh, indent=2)
