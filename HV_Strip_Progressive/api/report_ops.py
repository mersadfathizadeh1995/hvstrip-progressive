"""
Report Operations — figure and report generation.

Wraps :class:`core.report_generator.ProgressiveStrippingReporter`
with a clean API exposing individual figure types and batch reports.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ReportConfig, HVStripConfig

logger = logging.getLogger(__name__)

# Available figure types with descriptions
FIGURE_TYPES = {
    "hv_overlay": "All stripping-step HV curves overlaid on one figure",
    "peak_evolution": "Peak frequency tracking across stripping steps",
    "interface_analysis": "Velocity-interface depth identification",
    "waterfall": "3-D waterfall of step HV curves",
    "comprehensive": "Multi-panel comprehensive dashboard",
    "publication": "Clean, publication-ready figure",
    "text_report": "ASCII text summary report",
    "metadata": "JSON metadata file",
    "pdf_report": "Multi-page PDF report",
    "analysis_csv": "Analysis summary CSV",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_strip_report(
    strip_dir: str,
    output_dir: Optional[str] = None,
    config: Optional[ReportConfig] = None,
    vs_data: Optional[Dict] = None,
) -> Dict[str, str]:
    """Generate a comprehensive report for a stripping result.

    Parameters
    ----------
    strip_dir : str
        Path to the stripping output directory (containing step folders).
    output_dir : str, optional
        Report output directory.  Defaults to *strip_dir*/report.
    config : ReportConfig, optional
    vs_data : dict, optional
        Pre-loaded velocity data (passed to the reporter).

    Returns
    -------
    dict
        Mapping of report component name → file path.
    """
    if config is None:
        config = ReportConfig()

    if output_dir is None:
        output_dir = os.path.join(strip_dir, "report")
    os.makedirs(output_dir, exist_ok=True)

    from ..core.report_generator import ProgressiveStrippingReporter

    reporter = ProgressiveStrippingReporter(
        strip_directory=strip_dir,
        output_dir=output_dir,
        vs_data=vs_data,
    )

    try:
        result_paths = reporter.generate_comprehensive_report()
        # Convert Path objects to strings
        return {k: str(v) for k, v in result_paths.items()}
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        return {"error": str(exc)}


def generate_figure(
    strip_dir: str,
    figure_type: str,
    output_path: str,
    config: Optional[ReportConfig] = None,
    vs_data: Optional[Dict] = None,
) -> str:
    """Generate a single figure type.

    Parameters
    ----------
    strip_dir : str
    figure_type : str
        One of the keys in :data:`FIGURE_TYPES`.
    output_path : str
        Output file path (e.g. ``"output/overlay.png"``).
    config : ReportConfig, optional
    vs_data : dict, optional

    Returns
    -------
    str
        The written file path.
    """
    if config is None:
        config = ReportConfig()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    from ..core.report_generator import ProgressiveStrippingReporter

    reporter = ProgressiveStrippingReporter(
        strip_directory=strip_dir,
        output_dir=os.path.dirname(output_path),
        vs_data=vs_data,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8), dpi=config.dpi)

    method_map = {
        "hv_overlay": reporter.draw_hv_overlay_on_figure,
        "peak_evolution": reporter.draw_peak_evolution_on_figure,
        "interface_analysis": reporter.draw_interface_analysis_on_figure,
        "waterfall": reporter.draw_waterfall_on_figure,
        "publication": reporter.draw_publication_on_figure,
    }

    draw_fn = method_map.get(figure_type)
    if draw_fn is None:
        plt.close(fig)
        raise ValueError(
            f"Unknown figure type '{figure_type}'. "
            f"Available: {list(method_map.keys())}"
        )

    success = draw_fn(fig)
    if success:
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
        logger.info("Saved %s figure → %s", figure_type, output_path)
    plt.close(fig)

    return output_path if success else ""


def generate_text_report(
    strip_dir: str,
    output_path: str,
    vs_data: Optional[Dict] = None,
) -> str:
    """Generate a text summary report.

    Returns the written file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    from ..core.report_generator import ProgressiveStrippingReporter

    reporter = ProgressiveStrippingReporter(
        strip_directory=strip_dir,
        output_dir=os.path.dirname(output_path),
        vs_data=vs_data,
    )

    try:
        path = reporter._create_text_report()
        return str(path)
    except Exception as exc:
        logger.error("Text report failed: %s", exc)
        return ""


def generate_pdf_report(
    strip_dir: str,
    output_path: str,
    vs_data: Optional[Dict] = None,
) -> str:
    """Generate a multi-page PDF report.

    Returns the written file path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    from ..core.report_generator import ProgressiveStrippingReporter

    reporter = ProgressiveStrippingReporter(
        strip_directory=strip_dir,
        output_dir=os.path.dirname(output_path),
        vs_data=vs_data,
    )

    try:
        path = reporter._create_pdf_report()
        return str(path)
    except Exception as exc:
        logger.error("PDF report failed: %s", exc)
        return ""


def list_figure_types() -> List[Dict[str, str]]:
    """Return metadata for all available figure types."""
    return [
        {"name": name, "description": desc}
        for name, desc in FIGURE_TYPES.items()
    ]
