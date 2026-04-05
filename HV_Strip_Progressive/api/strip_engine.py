"""
Strip Engine — progressive layer stripping workflow.

Wraps :func:`core.batch_workflow.run_complete_workflow` and
:func:`core.stripper.write_peel_sequence` with typed result dataclasses.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.stripper import write_peel_sequence
from ..core.batch_workflow import (
    run_complete_workflow,
    find_step_folders,
    save_hv_csv,
    DEFAULT_WORKFLOW_CONFIG,
)
from ..core.hv_forward import compute_hv_curve
from ..core.hv_postprocess import process as hv_postprocess
from ..core.peak_detection import detect_peak as core_detect_peak

from .config import HVStripConfig, StripConfig
from .profile_io import load_profile, get_profile_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result for a single stripping step."""

    step_number: int = 0
    n_layers: int = 0
    model_path: str = ""
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_frequency: float = 0.0
    peak_amplitude: float = 0.0
    profile_summary: Dict[str, Any] = field(default_factory=dict)
    figure_paths: Dict[str, str] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "n_layers": self.n_layers,
            "model_path": self.model_path,
            "peak_frequency": self.peak_frequency,
            "peak_amplitude": self.peak_amplitude,
            "profile_summary": self.profile_summary,
            "figure_paths": self.figure_paths,
            "success": self.success,
            "error": self.error,
            "n_frequencies": len(self.frequencies),
        }


@dataclass
class StripResult:
    """Result of a full progressive-stripping workflow."""

    initial_profile: str = ""
    output_directory: str = ""
    strip_directory: str = ""
    steps: List[StepResult] = field(default_factory=list)
    dual_resonance: Optional[Dict[str, Any]] = None
    report_files: Dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def peak_evolution(self) -> List[Dict[str, float]]:
        """Return peak frequency/amplitude per step."""
        return [
            {
                "step": s.step_number,
                "n_layers": s.n_layers,
                "frequency": s.peak_frequency,
                "amplitude": s.peak_amplitude,
            }
            for s in self.steps
            if s.success
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_profile": self.initial_profile,
            "output_directory": self.output_directory,
            "strip_directory": self.strip_directory,
            "n_steps": self.n_steps,
            "steps": [s.to_dict() for s in self.steps],
            "peak_evolution": self.peak_evolution,
            "dual_resonance": self.dual_resonance,
            "report_files": self.report_files,
            "elapsed_seconds": self.elapsed_seconds,
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_stripping(
    profile_or_path: Any,
    output_dir: str,
    config: Optional[HVStripConfig] = None,
    generate_report: bool = True,
) -> StripResult:
    """Run the full progressive-stripping workflow.

    1. Generate peeled model sequence
    2. Compute HV curve for each step (via selected engine)
    3. Post-process (peak detection, figures)
    4. Optionally generate comprehensive report

    Parameters
    ----------
    profile_or_path
        Path to a soil-profile file, or a :class:`SoilProfile` object.
    output_dir : str
        Base output directory.
    config : HVStripConfig, optional
    generate_report : bool
        Generate the full ProgressiveStrippingReporter report.

    Returns
    -------
    StripResult
    """
    if config is None:
        config = HVStripConfig()

    t0 = time.perf_counter()

    # Resolve model path
    from ..core.soil_profile import SoilProfile

    if isinstance(profile_or_path, SoilProfile):
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w"
        )
        tmp.write(profile_or_path.to_hvf_format())
        tmp.close()
        model_path = tmp.name
        profile_name = profile_or_path.name or "unnamed"
    else:
        model_path = str(profile_or_path)
        profile_name = Path(model_path).stem

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Build workflow config dict
        workflow_cfg = config.build_workflow_config()
        workflow_cfg["generate_report"] = generate_report

        # Run the core workflow
        core_results = run_complete_workflow(
            initial_model_path=model_path,
            output_base_dir=output_dir,
            workflow_config=workflow_cfg,
            engine_name=config.engine.name,
        )

        elapsed = time.perf_counter() - t0

        # Parse core results into StepResult objects
        steps = _parse_core_results(core_results, config)

        # Extract strip directory
        strip_dir = str(core_results.get("strip_directory", ""))

        # Report files
        report_files = {}
        for key, val in core_results.get("report_files", {}).items():
            report_files[key] = str(val)

        # Dual resonance
        dual_res = core_results.get("dual_resonance")
        if dual_res is not None and hasattr(dual_res, "__dict__"):
            from dataclasses import asdict

            try:
                dual_res = asdict(dual_res)
            except Exception:
                dual_res = str(dual_res)

        return StripResult(
            initial_profile=model_path,
            output_directory=output_dir,
            strip_directory=strip_dir,
            steps=steps,
            dual_resonance=dual_res,
            report_files=report_files,
            elapsed_seconds=round(elapsed, 3),
            success=True,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("Stripping failed for %s: %s", profile_name, exc)
        return StripResult(
            initial_profile=model_path,
            output_directory=output_dir,
            elapsed_seconds=round(elapsed, 3),
            success=False,
            error=str(exc),
        )


def generate_peel_sequence(
    profile_path: str,
    output_dir: str,
) -> List[str]:
    """Generate the peeled model files without running HV computation.

    Returns a list of model file paths (one per step).
    """
    strip_dir = write_peel_sequence(profile_path, output_dir)
    step_folders = find_step_folders(strip_dir)

    model_paths = []
    for folder in step_folders:
        folder_p = Path(folder)
        models = list(folder_p.glob("model_*.txt"))
        if models:
            model_paths.append(str(models[0]))

    return model_paths


def compute_step(
    model_path: str,
    config: Optional[HVStripConfig] = None,
) -> StepResult:
    """Compute the HV curve for a single stripping step.

    Parameters
    ----------
    model_path : str
        Path to the model file for this step.
    config : HVStripConfig, optional

    Returns
    -------
    StepResult
    """
    if config is None:
        config = HVStripConfig()

    engine_cfg = config.build_engine_config()

    try:
        freqs_list, amps_list = compute_hv_curve(
            model_path,
            config=engine_cfg,
            engine_name=config.engine.name,
        )
        freqs = np.array(freqs_list, dtype=float)
        amps = np.array(amps_list, dtype=float)

        # Detect peak
        peak_cfg = config.peak_detection.to_core_config()
        try:
            f0, a0, idx = core_detect_peak(freqs, amps, peak_cfg)
        except Exception:
            f0, a0 = 0.0, 0.0

        # Count layers from the model
        n_layers = _count_layers(model_path)

        return StepResult(
            n_layers=n_layers,
            model_path=model_path,
            frequencies=freqs,
            amplitudes=amps,
            peak_frequency=float(f0),
            peak_amplitude=float(a0),
            success=True,
        )

    except Exception as exc:
        return StepResult(
            model_path=model_path,
            success=False,
            error=str(exc),
        )


def get_step_comparison(strip_result: StripResult) -> Dict[str, Any]:
    """Extract peak evolution data for comparison.

    Returns
    -------
    dict
        Keys: ``step_numbers``, ``n_layers``, ``frequencies``,
        ``amplitudes``, ``max_shift``.
    """
    steps = [s for s in strip_result.steps if s.success]
    if not steps:
        return {}

    freqs = [s.peak_frequency for s in steps]
    amps = [s.peak_amplitude for s in steps]

    shifts = [abs(freqs[i] - freqs[i - 1]) for i in range(1, len(freqs))]
    max_shift = max(shifts) if shifts else 0.0

    return {
        "step_numbers": [s.step_number for s in steps],
        "n_layers": [s.n_layers for s in steps],
        "frequencies": freqs,
        "amplitudes": amps,
        "max_shift": max_shift,
        "n_steps": len(steps),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_core_results(
    core_results: Dict[str, Any],
    config: HVStripConfig,
) -> List[StepResult]:
    """Convert the core workflow dict into :class:`StepResult` objects."""
    steps: List[StepResult] = []
    step_results = core_results.get("step_results", {})

    for i, (step_name, data) in enumerate(sorted(step_results.items())):
        try:
            # Read HV data if available
            hv_csv = data.get("hv_csv")
            freqs = np.array([])
            amps = np.array([])
            if hv_csv and Path(str(hv_csv)).exists():
                from ..core.hv_postprocess import read_hv_csv

                freqs, amps = read_hv_csv(str(hv_csv))

            step = StepResult(
                step_number=i,
                n_layers=data.get("n_layers", 0),
                model_path=str(data.get("model_file", "")),
                frequencies=freqs,
                amplitudes=amps,
                peak_frequency=float(data.get("peak_frequency", 0.0)),
                peak_amplitude=float(data.get("peak_amplitude", 0.0)),
                figure_paths={
                    k: str(v)
                    for k, v in data.items()
                    if k.endswith("_png") or k.endswith("_figure")
                },
                success=True,
            )
            steps.append(step)
        except Exception as exc:
            steps.append(StepResult(
                step_number=i,
                success=False,
                error=str(exc),
            ))

    return steps


def _count_layers(model_path: str) -> int:
    """Count layers in an HVf-format model file."""
    try:
        with open(model_path, "r") as f:
            first_line = f.readline().strip()
            return int(first_line)
    except Exception:
        return 0
