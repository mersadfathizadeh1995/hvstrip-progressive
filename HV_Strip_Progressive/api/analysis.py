"""
HVStripAnalysis — main orchestrator for HV Strip Progressive API.

Stateful class that holds profiles, results, and configuration.
Single entry point for all operations, consumed by both GUI and MCP.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .config import (
    HVStripConfig,
    EngineConfig,
    FrequencyConfig,
    PeakDetectionConfig,
    AutoPeakConfig,
    StripConfig,
    DualResonanceConfig,
    ReportConfig,
    OutputConfig,
    BatchConfig,
    PostProcessConfig,
    AdaptiveConfig,
    HVPlotConfig,
    VsPlotConfig,
)
from .profile_io import (
    load_profile,
    load_profile_dinver,
    load_profiles_from_directory,
    create_profile,
    save_profile,
    profile_to_dict,
    profile_from_dict,
    get_profile_summary,
    suggest_layer_fill,
    validate_profile,
)
from .forward_engine import (
    compute_forward,
    compute_forward_batch,
    detect_peaks_on_curve,
    set_manual_peaks,
    list_engines,
    ForwardResult,
    MultiForwardResult,
    PeakInfo,
)
from .strip_engine import (
    run_stripping,
    compute_step,
    get_step_comparison,
    StripResult,
)
from .batch_engine import (
    run_batch_stripping,
    get_batch_statistics,
    BatchStripResult,
)
from .peak_ops import (
    detect_peak,
    detect_all_peaks,
    detect_peaks_with_ranges,
    set_manual_peak,
    list_presets,
    get_preset,
)
from .dual_resonance_ops import (
    extract_dual_resonance,
    compute_theoretical_frequencies,
)
from .report_ops import (
    generate_strip_report,
    generate_figure,
    list_figure_types,
)
from .export import (
    export_forward_result,
    export_strip_result,
    export_batch_result,
    export_profile_csv,
    export_hv_curve_csv,
    export_peak_summary,
)

logger = logging.getLogger(__name__)


class HVStripAnalysis:
    """Stateful orchestrator for HV Strip Progressive workflows.

    Holds profiles, forward results, stripping results, and configuration.
    All methods return JSON-serialisable dicts.

    Usage::

        analysis = HVStripAnalysis()
        analysis.set_engine(name="sh_wave")
        analysis.load_profile("path/to/model.txt", name="site_A")
        result = analysis.compute_forward("site_A")
        analysis.run_stripping("site_A", output_dir="output/site_A")
    """

    def __init__(self, session_id: str = "default"):
        self._session_id = session_id
        self._config = HVStripConfig()
        self._profiles: Dict[str, Any] = {}  # name → SoilProfile
        self._forward_results: Dict[str, ForwardResult] = {}
        self._strip_results: Dict[str, StripResult] = {}
        self._batch_result: Optional[BatchStripResult] = None
        self._research_runner: Optional[Any] = None  # ComparisonStudyRunner

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> Dict[str, Any]:
        """Apply top-level config changes.

        Accepts any field of :class:`HVStripConfig` as a keyword arg.
        Nested configs can be passed as dicts.
        """
        from .config import _apply_dict

        _apply_dict(self._config, kwargs)
        return self.get_config()

    def set_engine(self, **kwargs: Any) -> Dict[str, Any]:
        """Update engine configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.engine, kwargs)
        return {"engine": self._config.engine.__dict__}

    def set_frequency(self, **kwargs: Any) -> Dict[str, Any]:
        """Update frequency configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.frequency, kwargs)
        return {"frequency": self._config.frequency.__dict__}

    def set_peak_detection(self, **kwargs: Any) -> Dict[str, Any]:
        """Update peak detection configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.peak_detection, kwargs)
        return {"peak_detection": self._config.peak_detection.__dict__}

    def set_auto_peak(self, **kwargs: Any) -> Dict[str, Any]:
        """Update auto-peak configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.auto_peak, kwargs)
        return {"auto_peak": self._config.auto_peak.__dict__}

    def set_strip(self, **kwargs: Any) -> Dict[str, Any]:
        """Update stripping configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.strip, kwargs)
        return {"strip": self._config.strip.__dict__}

    def set_dual_resonance(self, **kwargs: Any) -> Dict[str, Any]:
        """Update dual-resonance configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.dual_resonance, kwargs)
        return {"dual_resonance": self._config.dual_resonance.__dict__}

    def set_report(self, **kwargs: Any) -> Dict[str, Any]:
        """Update report configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.report, kwargs)
        return {"report": self._config.report.__dict__}

    def set_output(self, **kwargs: Any) -> Dict[str, Any]:
        """Update output configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.output, kwargs)
        return {"output": self._config.output.__dict__}

    def set_postprocess(self, **kwargs: Any) -> Dict[str, Any]:
        """Update post-processing configuration."""
        from .config import _apply_dict

        _apply_dict(self._config.postprocess, kwargs)
        return {"postprocess": self._config.postprocess.__dict__}

    def get_config(self) -> Dict[str, Any]:
        """Return the full current configuration."""
        return self._config.to_dict()

    def get_defaults(self) -> Dict[str, Any]:
        """Return the default configuration."""
        return HVStripConfig().to_dict()

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def load_profile_from_file(
        self,
        path: str,
        name: Optional[str] = None,
        fmt: str = "auto",
    ) -> Dict[str, Any]:
        """Load a soil profile from file and add to the session.

        Returns the profile summary.
        """
        profile = load_profile(path, fmt=fmt, name=name)
        pname = profile.name or os.path.basename(path)
        self._profiles[pname] = profile
        summary = get_profile_summary(profile)
        logger.info("Added profile '%s' to session", pname)
        return {
            "name": pname,
            "path": path,
            "summary": summary.__dict__,
        }

    def load_profile_dinver(
        self,
        vs_file: str,
        vp_file: Optional[str] = None,
        rho_file: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a profile from SEPARATE Dinver files (Vs + optional Vp/ρ)
        and add it to the session — the legacy "Dinver Files" input mode."""
        profile = load_profile_dinver(
            vs_file, vp_file=vp_file, rho_file=rho_file, name=name)
        pname = profile.name
        self._profiles[pname] = profile
        summary = get_profile_summary(profile)
        return {"name": pname, "path": str(vs_file),
                "summary": summary.__dict__}

    def load_profiles_from_directory(
        self,
        directory: str,
        pattern: str = "*.txt",
    ) -> Dict[str, Any]:
        """Load every matching profile in *directory* into the session.

        Per-file failures are collected under ``errors``, never fatal.
        """
        profiles, errors = load_profiles_from_directory(directory, pattern)
        loaded = []
        for profile in profiles:
            pname = profile.name
            self._profiles[pname] = profile
            loaded.append(pname)
        return {
            "success": True,
            "directory": str(directory),
            "loaded": loaded,
            "errors": [{"path": p, "error": e} for p, e in errors],
        }

    def create_profile_from_layers(
        self,
        layers: List[Dict[str, Any]],
        name: str = "custom",
    ) -> Dict[str, Any]:
        """Create a profile from layer dicts and add to session."""
        profile = create_profile(layers, name=name)
        self._profiles[name] = profile
        summary = get_profile_summary(profile)
        return {
            "name": name,
            "summary": summary.__dict__,
        }

    def update_profile(
        self,
        name: str,
        layers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Replace a profile's layers (the Data-stage table's Apply).

        INVALIDATES that profile's forward/strip results — they no longer
        describe the edited model.
        """
        if name not in self._profiles:
            return {"success": False, "error": f"Profile '{name}' not found"}
        profile = create_profile(layers, name=name)
        self._profiles[name] = profile
        self._forward_results.pop(name, None)
        self._strip_results.pop(name, None)
        summary = get_profile_summary(profile)
        return {"success": True, "name": name, "summary": summary.__dict__}

    @staticmethod
    def suggest_layer_fill(
        vs: float, nu: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Auto-fill suggestions (ν, Vp, density, soil type) for one Vs;
        pass *nu* to derive Vp from a user-typed Poisson's ratio."""
        return suggest_layer_fill(vs, nu=nu)

    def add_profile(self, name: str, profile: Any) -> Dict[str, Any]:
        """Add an existing SoilProfile object to the session."""
        self._profiles[name] = profile
        summary = get_profile_summary(profile)
        return {"name": name, "summary": summary.__dict__}

    def get_profiles(self) -> List[Dict[str, Any]]:
        """Return summaries for all loaded profiles."""
        result = []
        for name, profile in self._profiles.items():
            summary = get_profile_summary(profile)
            result.append({
                "name": name,
                "n_layers": summary.n_layers,
                "total_depth": summary.total_depth,
                "vs30": summary.vs30,
                "f0_estimate": summary.f0_estimate,
            })
        return result

    def get_profile(self, name: str) -> Dict[str, Any]:
        """Return full profile dict including layers."""
        if name not in self._profiles:
            raise KeyError(f"Profile '{name}' not found")
        return profile_to_dict(self._profiles[name])

    def remove_profile(self, name: str) -> Dict[str, Any]:
        """Remove a profile from the session."""
        if name in self._profiles:
            del self._profiles[name]
        self._forward_results.pop(name, None)
        self._strip_results.pop(name, None)
        return {"removed": name}

    # ------------------------------------------------------------------
    # Forward computation
    # ------------------------------------------------------------------

    def compute_forward_single(
        self,
        profile_name: Optional[str] = None,
        engine_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute forward HV curve for one profile.

        Parameters
        ----------
        profile_name : str, optional
            If None, uses the first (or only) loaded profile.
        engine_name : str, optional
            Override engine.
        """
        profile = self._resolve_profile(profile_name)
        result = compute_forward(
            profile,
            config=self._config,
            engine_name=engine_name,
        )
        self._forward_results[profile.name] = result
        return result.to_dict()

    def compute_forward_all(
        self,
        engine_name: Optional[str] = None,
        progress_cb=None,
    ) -> Dict[str, Any]:
        """Compute forward HV curves for all loaded profiles.

        ``progress_cb(frame: dict)`` receives one ``profile`` frame per
        profile; ``None`` (default) = unchanged behaviour.
        """
        profiles = list(self._profiles.values())
        if not profiles:
            return {"error": "No profiles loaded"}

        config = self._config.copy()
        if engine_name:
            config.engine.name = engine_name

        multi_result = compute_forward_batch(
            profiles, config=config, detect_peaks=True,
            progress_cb=progress_cb,
        )

        # Store individual results
        for res in multi_result.results:
            if res.profile_name:
                self._forward_results[res.profile_name] = res

        return multi_result.to_dict()

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def detect_peaks_for_profile(
        self,
        profile_name: str,
    ) -> Dict[str, Any]:
        """Re-detect peaks on an existing forward result."""
        if profile_name not in self._forward_results:
            raise KeyError(
                f"No forward result for '{profile_name}'. "
                "Run compute_forward first."
            )
        result = self._forward_results[profile_name]
        peaks = detect_peaks_on_curve(
            result.frequencies, result.amplitudes, self._config.peak_detection
        )
        result.peaks = peaks
        return {"peaks": [p.__dict__ for p in peaks]}

    def set_manual_peaks_for_profile(
        self,
        profile_name: str,
        peaks: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Override peaks with manual selections."""
        if profile_name not in self._forward_results:
            raise KeyError(f"No forward result for '{profile_name}'")
        result = set_manual_peaks(self._forward_results[profile_name], peaks)
        self._forward_results[profile_name] = result
        return {"peaks": [p.__dict__ for p in result.peaks]}

    def get_peaks(self, profile_name: str) -> Dict[str, Any]:
        """Return peaks for a profile."""
        if profile_name not in self._forward_results:
            return {"peaks": []}
        return {
            "peaks": [
                p.__dict__ for p in self._forward_results[profile_name].peaks
            ]
        }

    # ------------------------------------------------------------------
    # Stripping
    # ------------------------------------------------------------------

    def run_stripping_for_profile(
        self,
        profile_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        progress_cb=None,
    ) -> Dict[str, Any]:
        """Run progressive stripping on one profile.

        ``progress_cb(frame: dict)`` streams phase/log frames parsed from
        the (frozen) core workflow narration; ``None`` = unchanged.
        """
        profile = self._resolve_profile(profile_name)
        if output_dir is None:
            output_dir = os.path.join(
                self._config.output.output_dir or ".",
                profile.name or "strip_output",
            )

        result = run_stripping(
            profile,
            output_dir=output_dir,
            config=self._config,
            generate_report=self._config.strip.generate_report,
            progress_cb=progress_cb,
        )
        self._strip_results[profile.name] = result
        return result.to_dict()

    def run_batch_stripping_all(
        self,
        output_dir: Optional[str] = None,
        progress_cb=None,
    ) -> Dict[str, Any]:
        """Run stripping on all loaded profiles.

        ``progress_cb(frame: dict)`` streams one ``profile`` frame per
        profile plus the per-profile workflow narration; ``None`` =
        unchanged behaviour.
        """
        if not self._profiles:
            return {"error": "No profiles loaded"}

        if output_dir is None:
            output_dir = self._config.output.output_dir or "batch_output"

        # Write profiles to temp files
        import tempfile

        profile_paths: List[str] = []
        for name, profile in self._profiles.items():
            tmp = tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w",
                prefix=f"{name}_",
            )
            tmp.write(profile.to_hvf_format())
            tmp.close()
            profile_paths.append(tmp.name)

        result = run_batch_stripping(
            profile_paths,
            output_dir=output_dir,
            config=self._config,
            progress_cb=progress_cb,
        )
        self._batch_result = result

        # Store individual results
        for pr in result.results:
            if pr.success and pr.strip_result:
                self._strip_results[pr.profile_name] = pr.strip_result

        return result.to_dict()

    # ------------------------------------------------------------------
    # Dual resonance
    # ------------------------------------------------------------------

    def extract_dual_resonance_for_profile(
        self,
        profile_name: str,
    ) -> Dict[str, Any]:
        """Extract f0/f1 dual resonance from stripping result."""
        if profile_name not in self._strip_results:
            raise KeyError(
                f"No strip result for '{profile_name}'. "
                "Run run_stripping first."
            )
        strip_res = self._strip_results[profile_name]
        return extract_dual_resonance(
            strip_dir=strip_res.strip_directory,
            config=self._config.dual_resonance,
            peak_config=self._config.peak_detection,
            profile_name=profile_name,
        )

    # ------------------------------------------------------------------
    # Reports & export
    # ------------------------------------------------------------------

    def generate_report_for_profile(
        self,
        profile_name: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate report for a single profile's stripping result."""
        if profile_name not in self._strip_results:
            raise KeyError(f"No strip result for '{profile_name}'")
        strip_res = self._strip_results[profile_name]
        if output_dir is None:
            output_dir = os.path.join(strip_res.output_directory, "report")
        return generate_strip_report(
            strip_dir=strip_res.strip_directory,
            output_dir=output_dir,
            config=self._config.report,
        )

    def generate_figure_for_profile(
        self,
        profile_name: str,
        figure_type: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate a specific figure for a profile."""
        if profile_name not in self._strip_results:
            raise KeyError(f"No strip result for '{profile_name}'")
        strip_res = self._strip_results[profile_name]
        path = generate_figure(
            strip_dir=strip_res.strip_directory,
            figure_type=figure_type,
            output_path=output_path,
            config=self._config.report,
        )
        return {"path": path}

    def export_results(
        self,
        output_dir: str,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Export all results to the specified directory."""
        if formats is None:
            formats = ["csv", "json"]

        os.makedirs(output_dir, exist_ok=True)
        all_paths: Dict[str, str] = {}

        # Export forward results
        for name, result in self._forward_results.items():
            paths = export_forward_result(
                result.to_dict(),
                os.path.join(output_dir, name),
                formats=formats,
                base_name=name,
            )
            for k, v in paths.items():
                all_paths[f"{name}_{k}"] = v

        # Export strip results
        for name, result in self._strip_results.items():
            paths = export_strip_result(
                result.to_dict(),
                os.path.join(output_dir, name),
                formats=formats,
                base_name=name,
            )
            for k, v in paths.items():
                all_paths[f"{name}_{k}"] = v

        # Export batch result
        if self._batch_result:
            paths = export_batch_result(
                self._batch_result.to_dict(),
                output_dir,
                formats=formats,
            )
            all_paths.update(paths)

        return {"files": all_paths, "n_files": len(all_paths)}

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def save_session(self, session_dir: str) -> Dict[str, Any]:
        """Save session state to directory."""
        from .session_io import save_session

        return save_session(self, session_dir)

    def load_session(self, session_dir: str) -> Dict[str, Any]:
        """Load session state from directory."""
        from .session_io import load_session

        return load_session(self, session_dir)

    # ------------------------------------------------------------------
    # Validation & discovery
    # ------------------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """Validate current configuration and loaded profiles."""
        from .validation import validate_config, validate_engine_availability

        errors: List[str] = []
        warnings: List[str] = []

        cfg_result = validate_config(self._config)
        errors.extend(cfg_result.get("errors", []))
        warnings.extend(cfg_result.get("warnings", []))

        eng_result = validate_engine_availability(self._config.engine.name)
        if not eng_result.get("available", False):
            warnings.append(eng_result.get("message", "Engine not available"))

        for name, profile in self._profiles.items():
            vr = validate_profile(profile)
            if not vr.valid:
                errors.extend([f"{name}: {e}" for e in vr.errors])
            warnings.extend([f"{name}: {w}" for w in vr.warnings])

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def list_engines() -> List[Dict[str, Any]]:
        """List available forward engines."""
        return list_engines()

    def check_engines(self) -> Dict[str, Any]:
        """Cheap per-engine availability probe (existence checks ONLY —
        never spawns a subprocess), so a GUI can surface per-tool status
        instead of crashing on a missing binary.

        Returns ``{engine: {"available": bool, "reason": str}}``.
        """
        import os
        from pathlib import Path

        report: Dict[str, Any] = {}

        # sh_wave — pure Python, always available.
        report["sh_wave"] = {"available": True, "reason": "pure Python"}

        # diffuse_field — the vendored HVf executable (or a config override).
        exe = self._config.engine.exe_path
        if not exe:
            base = Path(__file__).resolve().parent.parent / "core" / "engines" \
                / "diffuse_wave_field"
            for cand in (base / "exe_Win" / "HVf.exe",
                         base / "exe_Linux" / "HVf",
                         base / "exe_Linux" / "HVf_Serial"):
                if cand.is_file():
                    exe = str(cand)
                    break
        if exe and os.path.isfile(exe):
            report["diffuse_field"] = {"available": True, "reason": exe}
        else:
            report["diffuse_field"] = {
                "available": False,
                "reason": "HVf executable not found "
                          "(set EngineConfig.exe_path)",
            }

        # ellipticity — Geopsy gpell via Git Bash (config or local_config).
        gpell = self._config.engine.gpell_path
        bash = self._config.engine.git_bash_path
        if not gpell or not bash:
            try:
                from .. import local_config as _lc

                gpell = gpell or getattr(_lc, "GPELL_PATH", "")
                bash = bash or getattr(_lc, "GIT_BASH_PATH", "")
            except ImportError:
                pass
        missing = [label for label, path in
                   (("gpell", gpell), ("git-bash", bash))
                   if not (path and os.path.isfile(path))]
        if not missing:
            report["ellipticity"] = {"available": True, "reason": gpell}
        else:
            report["ellipticity"] = {
                "available": False,
                "reason": f"missing: {', '.join(missing)} "
                          "(configure local_config.py)",
            }
        return report

    def run_research_study(
        self,
        study_config: Optional[Dict[str, Any]] = None,
        phase: str = "full",
        progress_cb=None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """Run the research comparison-study pipeline through the api.

        Wraps :class:`research.runner.ComparisonStudyRunner` (what the GUI's
        Research tab drives) behind the ONE facade.  ``phase`` is ``"full"``
        or one of ``profiles | comparison | metrics | field_validation |
        report``.  ``progress_cb`` receives the runner's ``(current, total,
        message)`` progress as ``{"type": "study", ...}`` frames plus a
        ``study_phase`` frame per completed phase.

        The facade holds ONE runner across calls — the study phases are
        stateful (profiles → dataset → metrics), so sequencing them as
        separate calls (the GUI's cooperative-cancel loop) must land on the
        same instance.  ``reset=True`` (or a new ``study_config``) starts a
        fresh study.

        ``study_config`` may carry a top-level ``profiles_dir`` (not a
        ComparisonStudyConfig field): the profiles phase then LOADS that
        existing suite (a folder of ``.txt`` models) instead of generating
        one — the clean degrade path when SoilGen is not installed.
        """
        from ._progress import emit
        from ..research.runner import ComparisonStudyRunner

        if reset or self._research_runner is None:
            self._research_runner = ComparisonStudyRunner()
        runner = self._research_runner
        profiles_dir = None
        if study_config:
            study_config = dict(study_config)
            profiles_dir = study_config.pop("profiles_dir", None)
            runner.configure(**study_config)
        if progress_cb is not None:
            runner.set_progress_callback(
                lambda cur, total, msg: emit(
                    progress_cb, type="study", index=cur, total=total,
                    label=str(msg),
                )
            )

        phases = {
            "profiles": (
                (lambda: runner.load_profiles(profiles_dir))
                if profiles_dir else runner.generate_profiles),
            "comparison": runner.run_comparison,
            "metrics": runner.compute_metrics,
            "field_validation": runner.run_field_validation,
            "report": runner.generate_report,
        }
        try:
            if phase == "full":
                results: Dict[str, Any] = {}
                for name, fn in phases.items():
                    emit(progress_cb, type="study_phase", phase=name,
                         state="started")
                    results[name] = fn()
                    if isinstance(results[name], dict) and \
                            results[name].get("error"):
                        return {"success": False, "phase": name,
                                "error": str(results[name]["error"]),
                                "results": results}
                    emit(progress_cb, type="study_phase", phase=name,
                         state="finished")
                return {"success": True, "phase": "full", "results": results}
            if phase not in phases:
                return {"success": False,
                        "error": f"Unknown study phase: {phase!r}"}
            emit(progress_cb, type="study_phase", phase=phase, state="started")
            result = phases[phase]()
            if isinstance(result, dict) and result.get("error"):
                return {"success": False, "phase": phase,
                        "error": str(result["error"]), "results": result}
            emit(progress_cb, type="study_phase", phase=phase, state="finished")
            return {"success": True, "phase": phase, "results": result}
        except Exception as exc:                              # noqa: BLE001
            return {"success": False, "phase": phase, "error": str(exc)}

    @staticmethod
    def list_peak_presets() -> List[Dict[str, Any]]:
        """List available peak detection presets."""
        return list_presets()

    @staticmethod
    def list_figure_types() -> List[Dict[str, str]]:
        """List available figure types."""
        return list_figure_types()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_profile(self, name: Optional[str] = None) -> Any:
        """Resolve a profile by name, defaulting to the first loaded."""
        if name and name in self._profiles:
            return self._profiles[name]
        if not name and len(self._profiles) == 1:
            return next(iter(self._profiles.values()))
        if not name and not self._profiles:
            raise ValueError("No profiles loaded")
        raise KeyError(
            f"Profile '{name}' not found. "
            f"Available: {list(self._profiles.keys())}"
        )
