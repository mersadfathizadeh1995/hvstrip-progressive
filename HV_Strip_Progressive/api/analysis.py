"""
HVStripAnalysis — main orchestrator for HV Strip Progressive API.

Stateful class that holds profiles, results, and configuration.
Single entry point for all operations, consumed by both GUI and MCP.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
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
    set_exact_peaks,
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
from .persist_ops import (
    peak_to_dict,
    persist_peaks,
    rehydrate_results_folder,
    resolve_step_folder,
    _normalize_step_picks,
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
        #: profile → step-folder → canonical picks (spec 002; the session
        #: side of the picked_peaks.json sidecar).
        self._picked_peaks: Dict[str, Dict[str, Dict[str, Any]]] = {}

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
    # Public state accessors + the config funnel (spec 002 DC-5 — the
    # GUI reads THESE; the private backing fields are not its business)
    # ------------------------------------------------------------------

    @property
    def config(self) -> HVStripConfig:
        """The live config object (READ it freely; mutate only through
        ``set_*`` / :meth:`update_section` / :meth:`apply_config_payload`)."""
        return self._config

    def forward_results(self) -> Dict[str, ForwardResult]:
        """Live forward results by profile name (read-only snapshot dict)."""
        return dict(self._forward_results)

    def strip_results(self) -> Dict[str, StripResult]:
        """Live strip results by profile name (read-only snapshot dict)."""
        return dict(self._strip_results)

    def batch_result(self) -> Optional[BatchStripResult]:
        """The last batch-stripping result, if any."""
        return self._batch_result

    def update_section(self, section: str, **fields: Any) -> Dict[str, Any]:
        """Set fields on ONE config section — the sanctioned GUI funnel.

        Unknown sections fail; unknown field names are applied best-effort
        and reported under ``unmapped`` (never silently dropped).
        """
        from .config import _apply_dict

        target = getattr(self._config, section, None)
        if target is None or not hasattr(target, "__dataclass_fields__"):
            return {"success": False,
                    "error": f"Unknown config section: {section!r}"}
        unmapped: List[str] = []
        _apply_dict(target, dict(fields), unmapped=unmapped)
        return {"success": True, "section": section, "unmapped": unmapped}

    def apply_config_payload(self, payload: Any) -> Dict[str, Any]:
        """Replace the whole config from a persisted payload (v2 or
        legacy — routed through the ONE load funnel)."""
        from .session_io import load_config_payload

        self._config = load_config_payload(payload)
        return {"success": True}

    def config_payload(self) -> Dict[str, Any]:
        """The persistable v2 payload (``config_version`` included)."""
        return self._config.to_dict()

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

    def profile_names(self) -> List[str]:
        """Loaded profile names only — CHEAP (no summary/Vs30 computation).
        For status probes and refresh paths (spec 002 FR-11); use
        :meth:`get_profiles` when the summaries are actually needed."""
        return list(self._profiles)

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

    def set_profile_peaks(
        self,
        profile_name: str,
        peaks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Store user-picked forward peaks VERBATIM (spec 002 FR-1).

        The legacy click semantics: the EXACT picked frequency is kept and
        a missing amplitude interpolated — no snapping (contrast
        :meth:`set_manual_peaks_for_profile`).  ``label_pos`` entries (a
        dragged annotation position) round-trip.
        """
        if profile_name not in self._forward_results:
            return {"success": False,
                    "error": f"No forward result for '{profile_name}'"}
        result = set_exact_peaks(self._forward_results[profile_name], peaks)
        self._forward_results[profile_name] = result
        return {"success": True,
                "peaks": [peak_to_dict(p) for p in result.peaks]}

    # ------------------------------------------------------------------
    # Picked peaks per strip step (spec 002 FR-4/FR-5) — the session
    # store behind the interactive figure + the write-back chain
    # ------------------------------------------------------------------

    _EMPTY_STEP_PICKS: Dict[str, Any] = {
        "f0": None, "secondary": [], "vs30": None, "vsavg": None,
        "bedrock_depth": None,
    }

    def _resolve_step_key(self, profile_name: str, step: str) -> str:
        """Canonicalise *step* to the on-disk step-folder name when the
        profile's strip directory is known."""
        strip_res = self._strip_results.get(profile_name)
        if strip_res is not None and strip_res.strip_directory:
            folder = resolve_step_folder(
                Path(strip_res.strip_directory), str(step))
            if folder is not None:
                return folder.name
        return str(step)

    def set_step_peaks(
        self,
        profile_name: str,
        step: str,
        f0: Any = None,
        secondary: Optional[List[Any]] = None,
        vs30: Optional[float] = None,
        vsavg: Optional[float] = None,
        bedrock_depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Merge picks for ONE strip step into the session store.

        ``None`` arguments leave the stored value unchanged (pass
        ``secondary=[]`` to clear the secondaries; use
        :meth:`clear_step_peaks` to wipe a step).  Peaks may be dicts,
        :class:`PeakInfo`, or legacy ``(freq, amp[, idx])`` tuples.
        """
        key = self._resolve_step_key(profile_name, step)
        entry = self._picked_peaks.setdefault(profile_name, {}).setdefault(
            key, dict(self._EMPTY_STEP_PICKS, secondary=[]))
        updates: Dict[str, Any] = {}
        if f0 is not None:
            updates["f0"] = f0
        if secondary is not None:
            updates["secondary"] = secondary
        norm = _normalize_step_picks(updates)
        if f0 is not None:
            entry["f0"] = norm["f0"]
        if secondary is not None:
            entry["secondary"] = norm["secondary"]
        for k, v in (("vs30", vs30), ("vsavg", vsavg),
                     ("bedrock_depth", bedrock_depth)):
            if v is not None:
                entry[k] = v
        return {"success": True, "step": key, **self._copy_entry(entry)}

    @staticmethod
    def _copy_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Detach an envelope copy from the live store."""
        return {k: (list(v) if isinstance(v, list) else
                    (dict(v) if isinstance(v, dict) else v))
                for k, v in entry.items()}

    def get_step_peaks(self, profile_name: str, step: str) -> Dict[str, Any]:
        """Picks for one step (an empty entry when nothing is stored)."""
        key = self._resolve_step_key(profile_name, step)
        entry = self._picked_peaks.get(profile_name, {}).get(key)
        if entry is None:
            entry = dict(self._EMPTY_STEP_PICKS, secondary=[])
        return {"success": True, "step": key, **self._copy_entry(entry)}

    def picked_peaks(self, profile_name: str) -> Dict[str, Any]:
        """The whole picks store for a profile (deep copy)."""
        import copy as _copy

        return {"success": True,
                "steps": _copy.deepcopy(
                    self._picked_peaks.get(profile_name, {}))}

    def clear_step_peaks(
        self,
        profile_name: str,
        step: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Clear one step's picks, or the profile's whole store."""
        if step is None:
            self._picked_peaks.pop(profile_name, None)
            return {"success": True, "cleared": "all"}
        key = self._resolve_step_key(profile_name, step)
        self._picked_peaks.get(profile_name, {}).pop(key, None)
        return {"success": True, "cleared": key}

    def persist_picked_peaks(
        self,
        profile_name: str,
        regenerate_report: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """The legacy Finish write-back chain (spec 002 FR-4), Qt-free.

        Writes the sidecar + per-step summary CSVs + ``vs_results.json``
        (via :func:`persist_ops.persist_peaks`), mirrors the f0 picks onto
        the in-memory :class:`StripResult`, and — when
        *regenerate_report* is true (default: the ``strip.generate_report``
        config flag) — regenerates the comprehensive report from the
        rewritten files, exactly as the legacy app did.
        """
        store = self._picked_peaks.get(profile_name)
        if not store:
            return {"success": False,
                    "error": f"No picked peaks for '{profile_name}'"}
        strip_res = self._strip_results.get(profile_name)
        if strip_res is None or not strip_res.strip_directory:
            return {"success": False,
                    "error": f"No strip result with a strip directory "
                             f"for '{profile_name}'"}
        env = persist_peaks(strip_res.strip_directory, store)
        if not env.get("success"):
            return env

        # Mirror the legacy in-memory update: f0 picks overwrite step peaks.
        by_num: Dict[int, Dict[str, Any]] = {}
        for folder_name, pdata in store.items():
            f0 = pdata.get("f0")
            if not f0:
                continue
            try:
                num = int(folder_name.split("_")[0].replace("Step", ""))
            except ValueError:
                continue
            by_num[num] = f0
        for step_res in strip_res.steps:
            f0 = by_num.get(step_res.step_number)
            if f0:
                step_res.peak_frequency = float(f0["frequency"])
                step_res.peak_amplitude = float(f0["amplitude"])

        if regenerate_report is None:
            regenerate_report = bool(self._config.strip.generate_report)
        if regenerate_report:
            report = generate_strip_report(
                strip_dir=strip_res.strip_directory,
                output_dir=str(Path(strip_res.strip_directory).parent),
                config=self._config.report,
            )
            env["report"] = report
            if "error" not in report:
                strip_res.report_files = dict(report)
        return env

    @staticmethod
    def vs_context_for_layers(
        layers: List[Dict[str, Any]],
        bedrock_depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Vs30 + VsAvg-to-bedrock + interface depths for a layer stack —
        the legacy wizard's Vs mini-panel numbers (spec 002 FR-9):
        Vs30 = 30 m WITH half-space extrapolation; VsAvg = to the bedrock
        interface (default: the bottom of the finite layers) WITHOUT it.
        """
        from ..core.vs_average import compute_vs_average

        pairs = [(float(ly.get("thickness", ly.get("h", 0.0))),
                  float(ly["vs"])) for ly in layers]
        finite = [h for h, _ in pairs if h > 0]
        interfaces: List[float] = []
        z = 0.0
        for h in finite:
            z += h
            interfaces.append(round(z, 6))
        total = z
        target = float(bedrock_depth) if bedrock_depth else total
        out: Dict[str, Any] = {"success": True, "interfaces": interfaces,
                               "bedrock_depth": target if target > 0 else None,
                               "vs30": None, "vs30_extrapolated": False,
                               "vsavg": None}
        if not pairs:
            return out
        try:
            vs30 = compute_vs_average(pairs, target_depth=30.0,
                                      use_halfspace=True)
            out["vs30"] = float(vs30.vs_avg)
            out["vs30_extrapolated"] = bool(vs30.extrapolated)
        except Exception as exc:  # noqa: BLE001 — a status, never a crash
            out["vs30_error"] = str(exc)
        if target > 0:
            try:
                vsavg = compute_vs_average(pairs, target_depth=target,
                                           use_halfspace=False)
                out["vsavg"] = float(vsavg.vs_avg)
            except Exception as exc:  # noqa: BLE001
                out["vsavg_error"] = str(exc)
        return out

    def dual_resonance_overrides(self, profile_name: str) -> Dict[str, Any]:
        """Picked f0 per step as ``{step_folder: (freq, amp)}`` — the
        ``peak_overrides`` the dual-resonance figure substitutes for its
        auto-detected f0/f1 (the legacy figure-studio hand-off)."""
        store = self._picked_peaks.get(profile_name, {})
        overrides: Dict[str, Any] = {}
        for step, pdata in store.items():
            f0 = pdata.get("f0")
            if f0:
                overrides[step] = (float(f0["frequency"]),
                                   float(f0["amplitude"]))
        return {"success": True, "overrides": overrides}

    def load_results_folder(
        self,
        path: str,
        profile_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Re-open an existing strip results tree WITHOUT recompute
        (spec 002 FR-5) — v2 trees (sidecar) and legacy trees alike.

        The rehydrated :class:`StripResult` + picks register under
        *profile_name* (default: the folder's name), ready for further
        picking and :meth:`persist_picked_peaks`.
        """
        env = rehydrate_results_folder(path)
        if not env.get("success"):
            return env
        result: StripResult = env["result"]
        name = profile_name or Path(path).name
        result.initial_profile = name
        self._strip_results[name] = result

        def _entry(pdata: Dict[str, Any]) -> Dict[str, Any]:
            e = dict(self._EMPTY_STEP_PICKS, secondary=[])  # fresh list
            for k, v in (pdata or {}).items():
                if k in e:
                    e[k] = list(v) if isinstance(v, list) else v
            return e

        self._picked_peaks[name] = {
            step: _entry(pdata)
            for step, pdata in (env["picks"] or {}).items()
        }
        return {"success": True, "name": name,
                "n_steps": len(result.steps),
                "strip_directory": result.strip_directory,
                "picks_source": env["picks_source"]}

    # ------------------------------------------------------------------
    # Checked-set dispatch (spec 002 FR-7) — run EXACTLY the given
    # profiles, with per-profile setting overrides + cooperative cancel
    # ------------------------------------------------------------------

    def compute_forward_for(
        self,
        names: List[str],
        settings_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
        progress_cb=None,
        cancel=None,
    ) -> Dict[str, Any]:
        """Forward-model the given profiles (the checked run set).

        *settings_by_name* maps a profile to config-section overrides
        (``{"frequency": {...}, ...}``) applied to a COPY of the session
        config — the session config itself is never mutated.  *cancel* is
        any object with ``is_set()`` (e.g. ``threading.Event``), checked
        BEFORE each item.
        """
        from ._progress import emit
        from .config import _apply_dict

        settings_by_name = settings_by_name or {}
        names = list(names)
        total = len(names)
        completed: List[str] = []
        failed: List[Dict[str, str]] = []
        cancelled = False
        for i, name in enumerate(names, 1):
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            emit(progress_cb, type="profile", index=i, total=total,
                 name=name, state="started")
            if name not in self._profiles:
                failed.append({"name": name, "error": "profile not loaded"})
                emit(progress_cb, type="profile", index=i, total=total,
                     name=name, state="failed")
                continue
            cfg = self._config.copy()
            overrides = settings_by_name.get(name)
            if overrides:
                _apply_dict(cfg, dict(overrides))
            result = compute_forward(self._profiles[name], config=cfg)
            self._forward_results[name] = result
            if result.success:
                completed.append(name)
                emit(progress_cb, type="profile", index=i, total=total,
                     name=name, state="finished")
            else:
                failed.append({"name": name,
                               "error": result.error or "failed"})
                emit(progress_cb, type="profile", index=i, total=total,
                     name=name, state="failed")
        return {"success": not failed and not cancelled,
                "cancelled": cancelled, "n_total": total,
                "completed": completed, "failed": failed}

    def run_stripping_for(
        self,
        names: List[str],
        settings_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
        output_dir: Optional[str] = None,
        progress_cb=None,
        cancel=None,
    ) -> Dict[str, Any]:
        """Strip the given profiles, each into ``<output>/<name>/``.

        Same override/cancel contract as :meth:`compute_forward_for`.
        Cancellation is cooperative BETWEEN profiles — one profile's
        workflow is atomic (the frozen core loop has no cancel hook).
        """
        from ._progress import emit
        from .config import _apply_dict

        settings_by_name = settings_by_name or {}
        names = list(names)
        total = len(names)
        base = output_dir or self._config.output.output_dir or "batch_output"
        completed: List[str] = []
        failed: List[Dict[str, str]] = []
        cancelled = False
        for i, name in enumerate(names, 1):
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            emit(progress_cb, type="profile", index=i, total=total,
                 name=name, state="started")
            if name not in self._profiles:
                failed.append({"name": name, "error": "profile not loaded"})
                emit(progress_cb, type="profile", index=i, total=total,
                     name=name, state="failed")
                continue
            cfg = self._config.copy()
            overrides = settings_by_name.get(name)
            if overrides:
                _apply_dict(cfg, dict(overrides))
            result = run_stripping(
                self._profiles[name],
                output_dir=os.path.join(base, name),
                config=cfg,
                generate_report=cfg.strip.generate_report,
                progress_cb=progress_cb,
            )
            self._strip_results[name] = result
            if result.success:
                completed.append(name)
                emit(progress_cb, type="profile", index=i, total=total,
                     name=name, state="finished")
            else:
                failed.append({"name": name,
                               "error": result.error or "failed"})
                emit(progress_cb, type="profile", index=i, total=total,
                     name=name, state="failed")
        return {"success": not failed and not cancelled,
                "cancelled": cancelled, "n_total": total,
                "completed": completed, "failed": failed,
                "output_dir": base}

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
