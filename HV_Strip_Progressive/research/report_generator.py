"""
Report Generator — automated output generation for the comparison study.

Produces CSV summary tables, JSON data files, and LaTeX-compatible
text output for direct inclusion in the paper.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .config import ComparisonStudyConfig
from .forward_comparison import ComparisonDataset
from .metrics import ComparisonMetrics, compute_f0_comparison_table
from .field_data import FieldValidation

logger = logging.getLogger(__name__)


def generate_report(
    dataset: ComparisonDataset,
    metrics: ComparisonMetrics,
    config: ComparisonStudyConfig,
    field_validations: Optional[List[FieldValidation]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate the full report package.

    Returns
    -------
    dict
        Mapping of report component names to file paths.
    """
    if output_dir is None:
        output_dir = os.path.join(config.output.output_dir, "report")
    os.makedirs(output_dir, exist_ok=True)

    files: Dict[str, str] = {}

    # CSV: f0 comparison table
    if config.output.save_csv:
        p = _write_f0_csv(dataset, output_dir)
        files["f0_comparison_csv"] = p

        p = _write_engine_stats_csv(metrics, output_dir)
        files["engine_stats_csv"] = p

        p = _write_pairwise_csv(metrics, output_dir)
        files["pairwise_agreement_csv"] = p

    # JSON: full results
    if config.output.save_json:
        p = os.path.join(output_dir, "comparison_dataset.json")
        with open(p, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2, default=str)
        files["dataset_json"] = p

        p = os.path.join(output_dir, "metrics.json")
        with open(p, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2, default=str)
        files["metrics_json"] = p

    # LaTeX tables
    if config.output.save_latex:
        p = _write_latex_f0_table(dataset, metrics, config, output_dir)
        files["latex_f0_table"] = p

        p = _write_latex_agreement_table(metrics, config, output_dir)
        files["latex_agreement_table"] = p

        p = _write_latex_engine_stats(metrics, config, output_dir)
        files["latex_engine_stats"] = p

    # Field validation summary
    if field_validations:
        p = _write_field_validation_csv(field_validations, output_dir)
        files["field_validation_csv"] = p

    # Figures
    if config.output.save_figures:
        from .visualization import generate_figure, FIGURE_TYPES

        figs_dir = os.path.join(output_dir, "figures")
        os.makedirs(figs_dir, exist_ok=True)
        fmt = config.visualization.figure_format

        for fig_type in FIGURE_TYPES:
            try:
                path = os.path.join(figs_dir, f"{fig_type}.{fmt}")
                generate_figure(
                    fig_type,
                    path,
                    dataset=dataset,
                    metrics=metrics,
                    config=config.visualization,
                )
                files[f"figure_{fig_type}"] = path
            except Exception as exc:
                logger.warning("Failed to generate %s: %s", fig_type, exc)

    logger.info("Report generated: %d files in %s", len(files), output_dir)
    return files


# --- CSV writers ---

def _write_f0_csv(dataset: ComparisonDataset, output_dir: str) -> str:
    """Write f0 comparison table as CSV."""
    path = os.path.join(output_dir, "f0_comparison.csv")
    rows = compute_f0_comparison_table(dataset)
    if not rows:
        return path

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_engine_stats_csv(metrics: ComparisonMetrics, output_dir: str) -> str:
    """Write per-engine stats as CSV."""
    path = os.path.join(output_dir, "engine_statistics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "engine", "n_successful", "n_failed", "success_rate",
            "mean_time_s", "mean_n_peaks", "f0_mean", "f0_std",
        ])
        for es in metrics.engine_stats:
            writer.writerow([
                es.engine_name, es.n_successful, es.n_failed,
                f"{es.success_rate:.3f}", f"{es.mean_time:.3f}",
                f"{es.mean_n_peaks:.1f}", f"{es.f0_mean:.3f}", f"{es.f0_std:.3f}",
            ])
    return path


def _write_pairwise_csv(metrics: ComparisonMetrics, output_dir: str) -> str:
    """Write pairwise agreement as CSV."""
    path = os.path.join(output_dir, "pairwise_agreement.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "engine_a", "engine_b", "n_profiles", "n_both_peaks",
            "mean_freq_diff_hz", "std_freq_diff_hz", "mean_freq_ratio",
            "correlation", "agreement_rate",
        ])
        for pa in metrics.peak_agreements:
            writer.writerow([
                pa.engine_a, pa.engine_b, pa.n_profiles, pa.n_both_have_peaks,
                f"{pa.mean_freq_difference:.4f}", f"{pa.std_freq_difference:.4f}",
                f"{pa.mean_freq_ratio:.4f}", f"{pa.correlation:.4f}",
                f"{pa.agreement_rate:.4f}",
            ])
    return path


def _write_field_validation_csv(
    validations: List[FieldValidation],
    output_dir: str,
) -> str:
    """Write field validation results as CSV."""
    path = os.path.join(output_dir, "field_validation.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "site", "measured_f0", "best_engine", "best_rmse",
        ])
        for v in validations:
            writer.writerow([
                v.site_name, v.measured_f0, v.best_engine,
                f"{v.best_rmse:.4f}" if v.best_rmse < float("inf") else "N/A",
            ])
    return path


# --- LaTeX writers ---

def _write_latex_f0_table(
    dataset: ComparisonDataset,
    metrics: ComparisonMetrics,
    config: ComparisonStudyConfig,
    output_dir: str,
) -> str:
    """Generate a LaTeX table of f0 comparisons."""
    path = os.path.join(output_dir, "table_f0_comparison.tex")
    labels = config.visualization.engine_labels

    rows = compute_f0_comparison_table(dataset)

    with open(path, "w") as f:
        n_eng = len(dataset.engine_names)
        cols = "l" * (2 + n_eng)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write("\\hline\n")
        header = "Profile & Category"
        for eng in dataset.engine_names:
            header += f" & {labels.get(eng, eng)}"
        f.write(header + " \\\\\n")
        f.write("\\hline\n")

        for row in rows[:30]:  # Limit to 30 rows
            line = f"{row['profile']} & {row['category']}"
            for eng in dataset.engine_names:
                val = row.get(f"{eng}_f0")
                line += f" & {val:.2f}" if val else " & --"
            f.write(line + " \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return path


def _write_latex_agreement_table(
    metrics: ComparisonMetrics,
    config: ComparisonStudyConfig,
    output_dir: str,
) -> str:
    """Generate a LaTeX pairwise agreement table."""
    path = os.path.join(output_dir, "table_agreement.tex")
    labels = config.visualization.engine_labels

    with open(path, "w") as f:
        f.write("\\begin{tabular}{llccccc}\n")
        f.write("\\hline\n")
        f.write("Engine A & Engine B & N & $\\Delta f_0$ (Hz) & Ratio & $r$ & Agreement \\\\\n")
        f.write("\\hline\n")
        for pa in metrics.peak_agreements:
            la = labels.get(pa.engine_a, pa.engine_a)
            lb = labels.get(pa.engine_b, pa.engine_b)
            f.write(
                f"{la} & {lb} & {pa.n_both_have_peaks} & "
                f"{pa.mean_freq_difference:.3f} & {pa.mean_freq_ratio:.3f} & "
                f"{pa.correlation:.3f} & {pa.agreement_rate:.1%} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return path


def _write_latex_engine_stats(
    metrics: ComparisonMetrics,
    config: ComparisonStudyConfig,
    output_dir: str,
) -> str:
    """Generate a LaTeX engine statistics table."""
    path = os.path.join(output_dir, "table_engine_stats.tex")
    labels = config.visualization.engine_labels

    with open(path, "w") as f:
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\hline\n")
        f.write("Engine & Success & $\\bar{f}_0$ (Hz) & $\\sigma_{f_0}$ (Hz) & $\\bar{N}_{peaks}$ & Time (s) \\\\\n")
        f.write("\\hline\n")
        for es in metrics.engine_stats:
            label = labels.get(es.engine_name, es.engine_name)
            f.write(
                f"{label} & {es.success_rate:.0%} & "
                f"{es.f0_mean:.2f} & {es.f0_std:.2f} & "
                f"{es.mean_n_peaks:.1f} & {es.mean_time:.2f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return path
