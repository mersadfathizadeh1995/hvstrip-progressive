"""
Metrics — quantitative comparison measures between engine outputs.

Computes peak frequency agreement, curve shape similarity,
amplitude ratios, and statistical summaries.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .forward_comparison import ComparisonDataset, EngineResult, ProfileComparison
from .config import MetricsConfig

logger = logging.getLogger(__name__)


@dataclass
class PeakAgreement:
    """Agreement metrics for peak frequencies between two engines."""

    engine_a: str
    engine_b: str
    n_profiles: int = 0
    n_both_have_peaks: int = 0
    mean_freq_difference: float = 0.0
    mean_freq_ratio: float = 0.0
    std_freq_difference: float = 0.0
    correlation: float = 0.0
    agreement_rate: float = 0.0  # fraction within tolerance


@dataclass
class CurveAgreement:
    """Curve-shape agreement metrics between two engines."""

    engine_a: str
    engine_b: str
    n_profiles: int = 0
    mean_rmse: float = 0.0
    mean_correlation: float = 0.0
    mean_gof: float = 0.0  # goodness-of-fit


@dataclass
class EngineStatistics:
    """Per-engine aggregate statistics."""

    engine_name: str
    n_successful: int = 0
    n_failed: int = 0
    success_rate: float = 0.0
    mean_time: float = 0.0
    mean_n_peaks: float = 0.0
    f0_mean: float = 0.0
    f0_std: float = 0.0


@dataclass
class ComparisonMetrics:
    """Complete metrics for the comparison study."""

    peak_agreements: List[PeakAgreement] = field(default_factory=list)
    curve_agreements: List[CurveAgreement] = field(default_factory=list)
    engine_stats: List[EngineStatistics] = field(default_factory=list)
    per_category: Dict[str, "ComparisonMetrics"] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_agreements": [pa.__dict__ for pa in self.peak_agreements],
            "curve_agreements": [ca.__dict__ for ca in self.curve_agreements],
            "engine_stats": [es.__dict__ for es in self.engine_stats],
            "per_category": {
                cat: m.to_dict() for cat, m in self.per_category.items()
            },
        }


def compute_metrics(
    dataset: ComparisonDataset,
    config: MetricsConfig,
) -> ComparisonMetrics:
    """Compute all comparison metrics for the dataset.

    Parameters
    ----------
    dataset : ComparisonDataset
    config : MetricsConfig

    Returns
    -------
    ComparisonMetrics
    """
    metrics = ComparisonMetrics()

    # Engine statistics
    for engine_name in dataset.engine_names:
        stats = _compute_engine_stats(dataset, engine_name)
        metrics.engine_stats.append(stats)

    # Pairwise peak agreement
    for i, eng_a in enumerate(dataset.engine_names):
        for eng_b in dataset.engine_names[i + 1:]:
            pa = _compute_peak_agreement(dataset, eng_a, eng_b, config)
            metrics.peak_agreements.append(pa)

    # Pairwise curve agreement
    for i, eng_a in enumerate(dataset.engine_names):
        for eng_b in dataset.engine_names[i + 1:]:
            ca = _compute_curve_agreement(dataset, eng_a, eng_b, config)
            metrics.curve_agreements.append(ca)

    # Per-category breakdown
    categories = set(c.category for c in dataset.comparisons)
    for category in categories:
        subset = ComparisonDataset(
            comparisons=[c for c in dataset.comparisons if c.category == category],
            engine_names=dataset.engine_names,
        )
        metrics.per_category[category] = compute_metrics(subset, config)

    return metrics


def _compute_engine_stats(
    dataset: ComparisonDataset,
    engine_name: str,
) -> EngineStatistics:
    """Compute aggregate statistics for one engine."""
    results = dataset.get_engine_results(engine_name)
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    f0_values = []
    for r in successful:
        if r.peaks:
            f0_values.append(r.peaks[0].get("frequency", 0))

    return EngineStatistics(
        engine_name=engine_name,
        n_successful=len(successful),
        n_failed=len(failed),
        success_rate=len(successful) / max(len(results), 1),
        mean_time=np.mean([r.elapsed_seconds for r in successful]) if successful else 0,
        mean_n_peaks=np.mean([len(r.peaks) for r in successful]) if successful else 0,
        f0_mean=float(np.mean(f0_values)) if f0_values else 0,
        f0_std=float(np.std(f0_values)) if f0_values else 0,
    )


def _compute_peak_agreement(
    dataset: ComparisonDataset,
    eng_a: str,
    eng_b: str,
    config: MetricsConfig,
) -> PeakAgreement:
    """Compute peak frequency agreement between two engines."""
    pa = PeakAgreement(engine_a=eng_a, engine_b=eng_b)

    freq_diffs = []
    freq_ratios = []
    fa_list = []
    fb_list = []

    for comp in dataset.comparisons:
        ra = comp.engine_results.get(eng_a)
        rb = comp.engine_results.get(eng_b)
        if not ra or not rb or not ra.success or not rb.success:
            continue

        pa.n_profiles += 1

        if ra.peaks and rb.peaks:
            pa.n_both_have_peaks += 1
            fa = ra.peaks[0].get("frequency", 0)
            fb = rb.peaks[0].get("frequency", 0)
            if fa > 0 and fb > 0:
                freq_diffs.append(abs(fa - fb))
                freq_ratios.append(fa / fb)
                fa_list.append(fa)
                fb_list.append(fb)

    if freq_diffs:
        pa.mean_freq_difference = float(np.mean(freq_diffs))
        pa.std_freq_difference = float(np.std(freq_diffs))
        pa.mean_freq_ratio = float(np.mean(freq_ratios))

        # Agreement rate: fraction within tolerance
        within = sum(
            1 for d, r in zip(freq_diffs, freq_ratios)
            if d <= config.freq_tolerance_hz
            or abs(r - 1.0) <= config.freq_tolerance_ratio
        )
        pa.agreement_rate = within / len(freq_diffs)

        # Correlation
        if len(fa_list) >= 3:
            pa.correlation = float(np.corrcoef(fa_list, fb_list)[0, 1])

    return pa


def _compute_curve_agreement(
    dataset: ComparisonDataset,
    eng_a: str,
    eng_b: str,
    config: MetricsConfig,
) -> CurveAgreement:
    """Compute curve-shape agreement between two engines."""
    ca = CurveAgreement(engine_a=eng_a, engine_b=eng_b)

    rmse_values = []
    corr_values = []
    gof_values = []

    fmin, fmax = config.freq_range_for_rmse

    for comp in dataset.comparisons:
        ra = comp.engine_results.get(eng_a)
        rb = comp.engine_results.get(eng_b)
        if not ra or not rb or not ra.success or not rb.success:
            continue
        if not ra.frequencies or not rb.frequencies:
            continue

        ca.n_profiles += 1

        # Interpolate to common frequency grid
        try:
            freq_a = np.array(ra.frequencies)
            amp_a = np.array(ra.amplitudes)
            freq_b = np.array(rb.frequencies)
            amp_b = np.array(rb.amplitudes)

            # Common grid
            common_freq = np.logspace(
                np.log10(max(fmin, freq_a.min(), freq_b.min())),
                np.log10(min(fmax, freq_a.max(), freq_b.max())),
                200,
            )

            interp_a = np.interp(common_freq, freq_a, amp_a)
            interp_b = np.interp(common_freq, freq_b, amp_b)

            if config.normalize_curves:
                max_a = interp_a.max() or 1.0
                max_b = interp_b.max() or 1.0
                interp_a = interp_a / max_a
                interp_b = interp_b / max_b

            # RMSE
            rmse = float(np.sqrt(np.mean((interp_a - interp_b) ** 2)))
            rmse_values.append(rmse)

            # Correlation
            if np.std(interp_a) > 0 and np.std(interp_b) > 0:
                corr = float(np.corrcoef(interp_a, interp_b)[0, 1])
                corr_values.append(corr)

            # Goodness-of-fit (1 - NRMSE)
            range_val = max(interp_a.max() - interp_a.min(), 1e-10)
            gof = 1.0 - rmse / range_val
            gof_values.append(max(0.0, gof))

        except Exception:
            pass

    if rmse_values:
        ca.mean_rmse = float(np.mean(rmse_values))
    if corr_values:
        ca.mean_correlation = float(np.mean(corr_values))
    if gof_values:
        ca.mean_gof = float(np.mean(gof_values))

    return ca


# --- Convenience functions ---

def compute_f0_comparison_table(
    dataset: ComparisonDataset,
) -> List[Dict[str, Any]]:
    """Build a table of f0 values per profile × engine."""
    rows = []
    for comp in dataset.comparisons:
        row: Dict[str, Any] = {
            "profile": comp.profile_name,
            "category": comp.category,
        }
        for eng_name, result in comp.engine_results.items():
            if result.success and result.peaks:
                row[f"{eng_name}_f0"] = result.peaks[0].get("frequency")
                row[f"{eng_name}_amp"] = result.peaks[0].get("amplitude")
            else:
                row[f"{eng_name}_f0"] = None
                row[f"{eng_name}_amp"] = None
        rows.append(row)
    return rows
