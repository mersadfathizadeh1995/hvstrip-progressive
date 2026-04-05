"""
Visualization — publication-quality figures for the comparison study.

10+ figure types covering single-profile comparisons, aggregate
statistics, correlation plots, and multi-panel dashboards.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .config import ComparisonStudyConfig, VisualizationConfig
from .forward_comparison import ComparisonDataset, ProfileComparison
from .metrics import ComparisonMetrics

logger = logging.getLogger(__name__)

# All available figure types
FIGURE_TYPES = {
    "single_comparison": "HV curves from all engines for one profile",
    "f0_scatter": "f0 scatter plot: engine A vs engine B",
    "f0_boxplot": "f0 distribution boxplots by engine",
    "agreement_heatmap": "Pairwise agreement heatmap",
    "curve_overlay_grid": "Grid of HV curve comparisons (N profiles)",
    "residual_map": "Curve residuals across frequency",
    "category_comparison": "Performance breakdown by geological category",
    "runtime_comparison": "Computation time comparison",
    "peak_correlation_matrix": "Multi-engine peak correlation matrix",
    "field_validation": "Measured vs modeled for field sites",
    "comprehensive_dashboard": "Multi-panel summary dashboard",
}


def generate_figure(
    figure_type: str,
    output_path: str,
    dataset: Optional[ComparisonDataset] = None,
    metrics: Optional[ComparisonMetrics] = None,
    config: Optional[VisualizationConfig] = None,
    **kwargs: Any,
) -> str:
    """Generate a publication figure.

    Parameters
    ----------
    figure_type : str
        One of FIGURE_TYPES keys.
    output_path : str
        Path for the output image.
    dataset : ComparisonDataset, optional
    metrics : ComparisonMetrics, optional
    config : VisualizationConfig, optional

    Returns
    -------
    str
        Path to the generated figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if config is None:
        config = VisualizationConfig()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dispatch = {
        "single_comparison": _plot_single_comparison,
        "f0_scatter": _plot_f0_scatter,
        "f0_boxplot": _plot_f0_boxplot,
        "agreement_heatmap": _plot_agreement_heatmap,
        "curve_overlay_grid": _plot_curve_overlay_grid,
        "residual_map": _plot_residual_map,
        "category_comparison": _plot_category_comparison,
        "runtime_comparison": _plot_runtime_comparison,
        "peak_correlation_matrix": _plot_peak_correlation_matrix,
        "field_validation": _plot_field_validation,
        "comprehensive_dashboard": _plot_comprehensive_dashboard,
    }

    func = dispatch.get(figure_type)
    if func is None:
        raise ValueError(
            f"Unknown figure type: '{figure_type}'. "
            f"Available: {list(FIGURE_TYPES.keys())}"
        )

    fig = func(
        dataset=dataset,
        metrics=metrics,
        config=config,
        **kwargs,
    )

    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)
    return output_path


def list_figure_types() -> List[Dict[str, str]]:
    """Return available figure types."""
    return [
        {"type": k, "description": v}
        for k, v in FIGURE_TYPES.items()
    ]


# --- Individual figure generators ---

def _plot_single_comparison(
    dataset: ComparisonDataset,
    config: VisualizationConfig,
    profile_name: str = "",
    **kwargs: Any,
):
    """HV curves from all engines for a single profile."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize_single)

    comp = None
    if profile_name and dataset:
        comp = dataset.get_comparison(profile_name)
    elif dataset and dataset.comparisons:
        comp = dataset.comparisons[0]

    if comp is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    for eng_name, result in comp.engine_results.items():
        if not result.success or not result.frequencies:
            continue
        color = config.engine_colors.get(eng_name, "#666")
        label = config.engine_labels.get(eng_name, eng_name)
        ax.semilogx(result.frequencies, result.amplitudes, color=color, label=label, lw=1.5)

        for peak in result.peaks:
            ax.axvline(peak.get("frequency", 0), color=color, ls="--", alpha=0.4, lw=0.8)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H/V Amplitude")
    ax.set_title(f"HVSR Comparison — {comp.profile_name}")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    return fig


def _plot_f0_scatter(
    dataset: ComparisonDataset,
    config: VisualizationConfig,
    engine_a: str = "",
    engine_b: str = "",
    **kwargs: Any,
):
    """f0 scatter: engine A vs engine B."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize_single)

    engines = dataset.engine_names if dataset else []
    if not engine_a and len(engines) >= 2:
        engine_a, engine_b = engines[0], engines[1]

    fa_vals, fb_vals, cats = [], [], []
    if dataset:
        for comp in dataset.comparisons:
            ra = comp.engine_results.get(engine_a)
            rb = comp.engine_results.get(engine_b)
            if ra and rb and ra.success and rb.success and ra.peaks and rb.peaks:
                fa_vals.append(ra.peaks[0].get("frequency", 0))
                fb_vals.append(rb.peaks[0].get("frequency", 0))
                cats.append(comp.category)

    if fa_vals:
        ax.scatter(fa_vals, fb_vals, c=range(len(fa_vals)), cmap="viridis", alpha=0.7, s=30)
        lim = [min(min(fa_vals), min(fb_vals)) * 0.8, max(max(fa_vals), max(fb_vals)) * 1.2]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="1:1 line")
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    label_a = config.engine_labels.get(engine_a, engine_a)
    label_b = config.engine_labels.get(engine_b, engine_b)
    ax.set_xlabel(f"f₀ — {label_a} (Hz)")
    ax.set_ylabel(f"f₀ — {label_b} (Hz)")
    ax.set_title("Fundamental Frequency Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    return fig


def _plot_f0_boxplot(
    dataset: ComparisonDataset,
    config: VisualizationConfig,
    **kwargs: Any,
):
    """f0 distribution boxplots by engine."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize_single)

    data_per_engine = {}
    if dataset:
        for eng in dataset.engine_names:
            f0s = []
            for comp in dataset.comparisons:
                r = comp.engine_results.get(eng)
                if r and r.success and r.peaks:
                    f0s.append(r.peaks[0].get("frequency", 0))
            if f0s:
                data_per_engine[eng] = f0s

    if data_per_engine:
        labels = [config.engine_labels.get(e, e) for e in data_per_engine]
        colors = [config.engine_colors.get(e, "#666") for e in data_per_engine]
        bp = ax.boxplot(
            list(data_per_engine.values()),
            labels=labels,
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("f₀ (Hz)")
    ax.set_title("f₀ Distribution by Engine")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def _plot_agreement_heatmap(
    metrics: ComparisonMetrics,
    config: VisualizationConfig,
    **kwargs: Any,
):
    """Pairwise agreement heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize_single)

    if metrics and metrics.peak_agreements:
        engines = sorted(set(
            [pa.engine_a for pa in metrics.peak_agreements]
            + [pa.engine_b for pa in metrics.peak_agreements]
        ))
        n = len(engines)
        matrix = np.ones((n, n))
        for pa in metrics.peak_agreements:
            i = engines.index(pa.engine_a)
            j = engines.index(pa.engine_b)
            matrix[i, j] = pa.agreement_rate
            matrix[j, i] = pa.agreement_rate

        labels = [config.engine_labels.get(e, e) for e in engines]
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, label="Agreement Rate")

    ax.set_title("Pairwise f₀ Agreement")
    return fig


def _plot_curve_overlay_grid(
    dataset: ComparisonDataset,
    config: VisualizationConfig,
    n_profiles: int = 9,
    **kwargs: Any,
):
    """Grid of HV curve comparisons."""
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = (n_profiles + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=config.figsize_panel)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    comps = dataset.comparisons[:n_profiles] if dataset else []
    for i, ax in enumerate(axes_flat):
        if i < len(comps):
            comp = comps[i]
            for eng_name, result in comp.engine_results.items():
                if result.success and result.frequencies:
                    color = config.engine_colors.get(eng_name, "#666")
                    ax.semilogx(result.frequencies, result.amplitudes, color=color, lw=1)
            ax.set_title(comp.profile_name, fontsize=8)
            ax.grid(True, alpha=0.2)
        else:
            ax.set_visible(False)

    fig.suptitle("HV Curve Comparison Grid", fontsize=14)
    fig.tight_layout()
    return fig


def _plot_residual_map(dataset, config, **kwargs):
    """Curve residuals placeholder."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=config.figsize_single)
    ax.text(0.5, 0.5, "Residual Map\n(requires computed curves)", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Frequency-Dependent Residuals")
    return fig


def _plot_category_comparison(metrics, config, **kwargs):
    """Performance by geological category."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize_comparison)

    if metrics and metrics.per_category:
        categories = list(metrics.per_category.keys())
        x = np.arange(len(categories))
        width = 0.25

        for i, es in enumerate(metrics.engine_stats):
            rates = []
            for cat in categories:
                cat_m = metrics.per_category.get(cat)
                if cat_m:
                    cat_es = [s for s in cat_m.engine_stats if s.engine_name == es.engine_name]
                    rates.append(cat_es[0].success_rate if cat_es else 0)
                else:
                    rates.append(0)
            color = config.engine_colors.get(es.engine_name, "#666")
            label = config.engine_labels.get(es.engine_name, es.engine_name)
            ax.bar(x + i * width, rates, width, color=color, label=label, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend()

    ax.set_ylabel("Success Rate")
    ax.set_title("Performance by Geological Category")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def _plot_runtime_comparison(dataset, config, **kwargs):
    """Runtime bar chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize_single)

    if dataset:
        times = {}
        for eng in dataset.engine_names:
            results = dataset.get_engine_results(eng)
            ok = [r.elapsed_seconds for r in results if r.success]
            times[eng] = np.mean(ok) if ok else 0

        names = [config.engine_labels.get(e, e) for e in times]
        colors = [config.engine_colors.get(e, "#666") for e in times]
        ax.bar(names, list(times.values()), color=colors, alpha=0.8)

    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("Engine Computation Time")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def _plot_peak_correlation_matrix(metrics, config, **kwargs):
    """Peak correlation matrix."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.figsize_single)

    if metrics and metrics.peak_agreements:
        engines = sorted(set(
            [pa.engine_a for pa in metrics.peak_agreements]
            + [pa.engine_b for pa in metrics.peak_agreements]
        ))
        n = len(engines)
        matrix = np.eye(n)
        for pa in metrics.peak_agreements:
            i = engines.index(pa.engine_a)
            j = engines.index(pa.engine_b)
            matrix[i, j] = pa.correlation
            matrix[j, i] = pa.correlation

        labels = [config.engine_labels.get(e, e) for e in engines]
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_title("Peak Frequency Correlation")
    return fig


def _plot_field_validation(config, **kwargs):
    """Field validation — measured vs modeled."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=config.figsize_single)
    ax.text(0.5, 0.5, "Field Validation\n(pass field_validations kwarg)", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Measured vs Modeled HVSR")
    return fig


def _plot_comprehensive_dashboard(
    dataset: ComparisonDataset,
    metrics: ComparisonMetrics,
    config: VisualizationConfig,
    **kwargs: Any,
):
    """Multi-panel summary dashboard."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # f0 boxplot
    ax1 = fig.add_subplot(gs[0, 0])
    _inline_f0_boxplot(ax1, dataset, config)

    # Runtime
    ax2 = fig.add_subplot(gs[0, 1])
    _inline_runtime(ax2, dataset, config)

    # Agreement heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    _inline_agreement(ax3, metrics, config)

    # Grid sample (4 profiles)
    for idx in range(4):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        if dataset and idx < len(dataset.comparisons):
            comp = dataset.comparisons[idx]
            for eng, res in comp.engine_results.items():
                if res.success and res.frequencies:
                    c = config.engine_colors.get(eng, "#666")
                    ax.semilogx(res.frequencies, res.amplitudes, color=c, lw=1)
            ax.set_title(comp.profile_name, fontsize=9)
            ax.grid(True, alpha=0.2)

    # Stats table
    ax_tab = fig.add_subplot(gs[1:, 2])
    _inline_stats_table(ax_tab, metrics, config)

    fig.suptitle("HVSR Forward Modeling Comparison — Summary", fontsize=16, y=0.98)
    return fig


# --- Inline helpers for dashboard ---

def _inline_f0_boxplot(ax, dataset, config):
    if not dataset:
        return
    data = {}
    for eng in dataset.engine_names:
        f0s = [r.peaks[0].get("frequency", 0) for r in dataset.get_engine_results(eng) if r.success and r.peaks]
        if f0s:
            data[eng] = f0s
    if data:
        labels = [config.engine_labels.get(e, e)[:12] for e in data]
        colors = [config.engine_colors.get(e, "#666") for e in data]
        bp = ax.boxplot(list(data.values()), labels=labels, patch_artist=True)
        for p, c in zip(bp["boxes"], colors):
            p.set_facecolor(c)
            p.set_alpha(0.6)
    ax.set_title("f₀ Distribution", fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")


def _inline_runtime(ax, dataset, config):
    if not dataset:
        return
    times = {}
    for eng in dataset.engine_names:
        ok = [r.elapsed_seconds for r in dataset.get_engine_results(eng) if r.success]
        times[eng] = np.mean(ok) if ok else 0
    names = [config.engine_labels.get(e, e)[:12] for e in times]
    colors = [config.engine_colors.get(e, "#666") for e in times]
    ax.bar(names, list(times.values()), color=colors, alpha=0.8)
    ax.set_title("Mean Runtime (s)", fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")


def _inline_agreement(ax, metrics, config):
    if not metrics or not metrics.peak_agreements:
        return
    engines = sorted(set([pa.engine_a for pa in metrics.peak_agreements] + [pa.engine_b for pa in metrics.peak_agreements]))
    n = len(engines)
    m = np.ones((n, n))
    for pa in metrics.peak_agreements:
        i, j = engines.index(pa.engine_a), engines.index(pa.engine_b)
        m[i, j] = m[j, i] = pa.agreement_rate
    labels = [config.engine_labels.get(e, e)[:8] for e in engines]
    ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{m[i,j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Agreement", fontsize=10)


def _inline_stats_table(ax, metrics, config):
    ax.axis("off")
    if not metrics or not metrics.engine_stats:
        return
    rows = []
    for es in metrics.engine_stats:
        label = config.engine_labels.get(es.engine_name, es.engine_name)[:15]
        rows.append([
            label,
            f"{es.success_rate:.0%}",
            f"{es.f0_mean:.2f}",
            f"{es.f0_std:.2f}",
            f"{es.mean_time:.2f}s",
        ])
    ax.table(
        cellText=rows,
        colLabels=["Engine", "Success", "f₀ Mean", "f₀ Std", "Time"],
        loc="center",
        cellLoc="center",
    )
    ax.set_title("Engine Statistics", fontsize=10)
