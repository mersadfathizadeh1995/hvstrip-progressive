"""
Runner — end-to-end orchestrator for the comparison study.

Ties together profile generation, forward comparison, metrics,
field validation, visualization, and report generation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from .config import ComparisonStudyConfig
from .profile_suite import (
    generate_profile_suite,
    load_profile_suite,
    get_suite_summary,
    ProfileEntry,
)
from .forward_comparison import (
    run_forward_comparison,
    ComparisonDataset,
)
from .metrics import compute_metrics, ComparisonMetrics
from .field_data import run_field_validation, FieldValidation
from .report_generator import generate_report

logger = logging.getLogger(__name__)


class ComparisonStudyRunner:
    """Orchestrator for the full comparison study pipeline.

    Usage::

        runner = ComparisonStudyRunner()
        runner.configure(engines={"engines": ["sh_wave", "diffuse_field"]})
        runner.generate_profiles()
        runner.run_comparison()
        runner.compute_metrics()
        runner.generate_report()
    """

    def __init__(self, config: Optional[ComparisonStudyConfig] = None):
        self._config = config or ComparisonStudyConfig()
        self._profiles: List[ProfileEntry] = []
        self._dataset: Optional[ComparisonDataset] = None
        self._metrics: Optional[ComparisonMetrics] = None
        self._field_validations: Optional[List[FieldValidation]] = None
        self._progress_callback: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> Dict[str, Any]:
        """Update study configuration.

        Accepts any top-level ComparisonStudyConfig field as keyword args.
        Nested configs can be passed as dicts.
        """
        for key, val in kwargs.items():
            if key == "profiles" and isinstance(val, dict):
                for k, v in val.items():
                    if hasattr(self._config.profiles, k):
                        setattr(self._config.profiles, k, v)
            elif key == "engines" and isinstance(val, dict):
                for k, v in val.items():
                    if hasattr(self._config.engines, k):
                        setattr(self._config.engines, k, v)
            elif key == "metrics" and isinstance(val, dict):
                for k, v in val.items():
                    if hasattr(self._config.metrics, k):
                        setattr(self._config.metrics, k, v)
            elif key == "visualization" and isinstance(val, dict):
                for k, v in val.items():
                    if hasattr(self._config.visualization, k):
                        setattr(self._config.visualization, k, v)
            elif key == "output" and isinstance(val, dict):
                for k, v in val.items():
                    if hasattr(self._config.output, k):
                        setattr(self._config.output, k, v)
            elif hasattr(self._config, key):
                setattr(self._config, key, val)

        return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        return self._config.to_dict()

    def set_progress_callback(self, callback: Callable) -> None:
        """Set a callback for progress updates: callback(current, total, message)."""
        self._progress_callback = callback

    # ------------------------------------------------------------------
    # Phase 1: Profile generation
    # ------------------------------------------------------------------

    def generate_profiles(
        self,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate synthetic profile suite."""
        self._profiles = generate_profile_suite(
            self._config,
            output_dir=output_dir,
        )
        summary = get_suite_summary(self._profiles)
        return {
            "phase": "profile_generation",
            "n_profiles": len(self._profiles),
            "summary": summary,
        }

    def load_profiles(self, profiles_dir: str) -> Dict[str, Any]:
        """Load a previously generated profile suite."""
        self._profiles = load_profile_suite(profiles_dir)
        summary = get_suite_summary(self._profiles)
        return {
            "phase": "profile_loading",
            "n_profiles": len(self._profiles),
            "summary": summary,
        }

    def add_profile(self, entry: ProfileEntry) -> None:
        """Add a single profile to the suite."""
        self._profiles.append(entry)

    # ------------------------------------------------------------------
    # Phase 2: Forward comparison
    # ------------------------------------------------------------------

    def run_comparison(self) -> Dict[str, Any]:
        """Run all engines on all profiles."""
        if not self._profiles:
            return {"error": "No profiles loaded. Call generate_profiles() first."}

        self._dataset = run_forward_comparison(
            self._profiles,
            self._config,
            progress_callback=self._progress_callback,
        )
        return {
            "phase": "forward_comparison",
            "total_runs": self._dataset.total_runs,
            "successful_runs": self._dataset.successful_runs,
            "elapsed_seconds": self._dataset.elapsed_seconds,
        }

    # ------------------------------------------------------------------
    # Phase 3: Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute comparison metrics."""
        if not self._dataset:
            return {"error": "No comparison data. Call run_comparison() first."}

        self._metrics = compute_metrics(
            self._dataset,
            self._config.metrics,
        )
        return {
            "phase": "metrics",
            "n_peak_agreements": len(self._metrics.peak_agreements),
            "n_curve_agreements": len(self._metrics.curve_agreements),
            "n_categories": len(self._metrics.per_category),
            "engine_stats": [es.__dict__ for es in self._metrics.engine_stats],
        }

    # ------------------------------------------------------------------
    # Phase 4: Field validation
    # ------------------------------------------------------------------

    def run_field_validation(self) -> Dict[str, Any]:
        """Run field validation if sites are configured."""
        if not self._config.field_sites:
            return {"phase": "field_validation", "n_sites": 0, "message": "No field sites configured"}

        self._field_validations = run_field_validation(
            self._config,
            progress_callback=self._progress_callback,
        )
        return {
            "phase": "field_validation",
            "n_sites": len(self._field_validations),
            "results": [v.to_dict() for v in self._field_validations],
        }

    # ------------------------------------------------------------------
    # Phase 5: Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate the full report."""
        if not self._dataset or not self._metrics:
            return {"error": "Run comparison and metrics first."}

        files = generate_report(
            self._dataset,
            self._metrics,
            self._config,
            field_validations=self._field_validations,
            output_dir=output_dir,
        )
        return {
            "phase": "report",
            "n_files": len(files),
            "files": files,
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_study(
        self,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the complete study pipeline end-to-end.

        Returns
        -------
        dict
            Summary of all phases.
        """
        t0 = time.time()
        results: Dict[str, Any] = {}

        # Phase 1
        if not self._profiles:
            results["profiles"] = self.generate_profiles()
        else:
            results["profiles"] = {
                "n_profiles": len(self._profiles),
                "message": "Using pre-loaded profiles",
            }

        # Phase 2
        results["comparison"] = self.run_comparison()

        # Phase 3
        results["metrics"] = self.compute_metrics()

        # Phase 4
        if self._config.field_sites:
            results["field_validation"] = self.run_field_validation()

        # Phase 5
        results["report"] = self.generate_report(output_dir=output_dir)

        results["total_elapsed_seconds"] = time.time() - t0
        return results

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save study state to JSON."""
        state = {
            "config": self._config.to_dict(),
            "n_profiles": len(self._profiles),
            "has_dataset": self._dataset is not None,
            "has_metrics": self._metrics is not None,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> Dict[str, Any]:
        """Load study configuration from JSON."""
        with open(path) as f:
            state = json.load(f)
        self._config = ComparisonStudyConfig.from_dict(state["config"])
        return state
