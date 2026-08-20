"""``AppState`` — the house spine over ONE :class:`HVStripAnalysis`.

Panels talk ONLY to this object (never ``analysis.*`` directly, never disk,
never ``core.*``).  Long ops run on TWO :class:`OpQueue`s — the MAIN queue
(Forward + Strip) and a dedicated RESEARCH queue so a long comparison study
never blocks the interactive tools (the agreed archetype: per-tool status,
peer tools).  Progress streams through :class:`ProgressBridge` (latest-wins
per frame type) from the api's ``progress_cb`` frames.

Config is :class:`HVStripConfig` ONLY; persisted payloads go through the
facade's ``apply_config_payload``/``config_payload`` (the ONE funnel — v2
payloads; legacy dicts migrate on read), and section edits through
``update_section`` (spec 002 DC-5: no private reach-ins).
``status_for(tool)`` is the single source of tool-switcher truth (pull
model).  Engine availability comes from ``analysis.check_engines()`` —
an unavailable engine is a STATUS, never a crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Signal

from HV_Strip_Progressive.api import HVStripAnalysis
from HV_Strip_Progressive.gui.v2.state.profile_status import ProcessingStatus
from HV_Strip_Progressive.gui.v2.state.tool import StripTool, ToolStatus
from HV_Strip_Progressive.gui.v2.workers.op_worker import OpQueue, ProgressBridge

#: op name → the tool whose status it drives.
_OP_TOOL: Dict[str, StripTool] = {
    "run_forward": StripTool.FORWARD,
    "run_forward_all": StripTool.FORWARD,
    "run_strip": StripTool.STRIP,
    "run_batch_strip": StripTool.STRIP,
    "run_research": StripTool.RESEARCH,
}

#: The standalone settings file (v2 payload; legacy YAML migrates on read).
_SETTINGS_PATH = Path.home() / ".hvstrip" / "settings.yaml"


def _ok(env: Any) -> bool:
    """The api returns plain dicts with mixed conventions — normalise."""
    if not isinstance(env, dict):
        return False
    if env.get("error"):
        return False
    return bool(env.get("success", True))


class AppState(QObject):
    """Signals + ops + status over one :class:`HVStripAnalysis`."""

    session_opened = Signal()
    profiles_changed = Signal()
    forward_changed = Signal()
    strip_changed = Signal()
    research_changed = Signal()
    config_changed = Signal(str)          # section name
    active_tool_changed = Signal(object)  # StripTool

    op_started = Signal(str)
    op_progress = Signal(dict)            # coalesced frames (both queues)
    op_finished = Signal(str, dict)
    error = Signal(list)
    dirty_changed = Signal(bool)
    #: Transient UI-thread busy note (spec 002 FR-12): emitted with a
    #: message before a deliberate main-thread block (the one-time heavy
    #: preload), and with "" when it ends.  The window shows a busy
    #: cursor + status text so the first Run never looks frozen.
    busy_hint = Signal(str)

    # The profile-centric state model (the ProfilesPanel contract).
    checked_changed = Signal(list)                 # checked profile names
    focus_changed = Signal(object)                 # focused name | None
    profile_status_changed = Signal(object, str)   # (StripTool, name)
    profile_settings_changed = Signal(str)         # profile name

    def __init__(
        self,
        analysis: Optional[HVStripAnalysis] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._analysis: Optional[HVStripAnalysis] = None
        self._dirty = False
        self._active_tool = StripTool.DATA
        self._error_tools: set = set()
        self._research_cancelled = False
        self._engines_report: Optional[Dict[str, Any]] = None
        self._research_results: Dict[str, Any] = {}
        self._profile_checked: set = set()
        self._profile_focus: Optional[str] = None
        self._profile_settings: Dict[StripTool, Dict[str, dict]] = {}
        self._profile_status: Dict[StripTool, Dict[str, Any]] = {}
        self._op_profiles: Dict[str, List[List[str]]] = {}

        # Two queues: main (Forward + Strip) and research (long studies).
        self._queue = OpQueue(self)
        self._research_queue = OpQueue(self)
        for q in (self._queue, self._research_queue):
            q.op_started.connect(self._on_op_started)
            q.op_finished.connect(self._on_op_finished)

        # One coalescing bridge feeds op_progress from either queue's op.
        self._progress = ProgressBridge(self)
        self._progress.frame.connect(self.op_progress)
        self._progress.frame.connect(self._on_progress_frame)

        self.open_session(analysis or HVStripAnalysis())

    # ==================================================================
    #  Session
    # ==================================================================
    def open_session(self, analysis: HVStripAnalysis) -> None:
        self._analysis = analysis
        self._error_tools.clear()
        self._research_results = {}
        self._engines_report = None
        self._profile_checked.clear()
        self._profile_focus = None
        self._profile_settings.clear()
        self._profile_status.clear()
        self._op_profiles.clear()
        self._set_dirty(False)
        self.session_opened.emit()

    @property
    def analysis(self) -> Optional[HVStripAnalysis]:
        return self._analysis

    @property
    def config(self):
        return self._analysis.config if self._analysis else None

    @property
    def is_busy(self) -> bool:
        return self._queue.is_busy or self._research_queue.is_busy

    @property
    def current_op(self) -> Optional[str]:
        return self._queue.current_op or self._research_queue.current_op

    @property
    def dirty(self) -> bool:
        return self._dirty

    # ==================================================================
    #  Tool activity + status (the switcher's single source of truth)
    # ==================================================================
    @property
    def active_tool(self) -> StripTool:
        return self._active_tool

    def set_active_tool(self, tool: StripTool) -> None:
        if tool is self._active_tool:
            return
        self._active_tool = tool
        self.active_tool_changed.emit(tool)

    def status_for(self, tool: StripTool) -> ToolStatus:
        a = self._analysis
        if a is None:
            return ToolStatus.IDLE
        for q in (self._queue, self._research_queue):
            if q.current_op and _OP_TOOL.get(q.current_op) is tool:
                return ToolStatus.RUNNING
        if tool in self._error_tools:
            return ToolStatus.ERROR
        if tool is StripTool.DATA and a.profile_names():
            return ToolStatus.DONE
        if tool is StripTool.FORWARD and a.forward_results():
            return ToolStatus.DONE
        if tool is StripTool.STRIP and (
                a.strip_results() or a.batch_result()):
            return ToolStatus.DONE
        if tool is StripTool.RESEARCH and self._research_results:
            return ToolStatus.DONE
        return ToolStatus.IDLE

    def engines_report(self, refresh: bool = False) -> Dict[str, Any]:
        """Cached ``check_engines()`` (existence probes only)."""
        if self._analysis is None:
            return {}
        if self._engines_report is None or refresh:
            self._engines_report = self._analysis.check_engines()
        return self._engines_report

    # ==================================================================
    #  Profiles (fast, synchronous)
    # ==================================================================
    def load_profile(
        self, path: str, name: Optional[str] = None, fmt: str = "auto",
    ) -> Dict[str, Any]:
        return self._sync(
            lambda: self._analysis.load_profile_from_file(
                path, name=name, fmt=fmt),
            on_success=(self.profiles_changed,), tool=None, mutate=True,
        )

    def load_profile_dinver(
        self,
        vs_file: str,
        vp_file: Optional[str] = None,
        rho_file: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._sync(
            lambda: self._analysis.load_profile_dinver(
                vs_file, vp_file=vp_file, rho_file=rho_file, name=name),
            on_success=(self.profiles_changed,), tool=None, mutate=True,
        )

    def load_profiles_from_directory(
        self, directory: str, pattern: str = "*.txt",
    ) -> Dict[str, Any]:
        env = self._sync(
            lambda: self._analysis.load_profiles_from_directory(
                directory, pattern),
            on_success=(self.profiles_changed,), tool=None, mutate=True,
        )
        for entry in env.get("errors") or []:
            self.error.emit([f"{entry['path']}: {entry['error']}"])
        return env

    def add_profile_from_layers(
        self, layers: List[Dict[str, Any]], name: str,
    ) -> Dict[str, Any]:
        return self._sync(
            lambda: self._analysis.create_profile_from_layers(layers, name=name),
            on_success=(self.profiles_changed,), tool=None, mutate=True,
        )

    def profiles(self) -> Dict[str, Any]:
        if self._analysis is None:
            return {}
        return self._analysis.get_profiles()

    def profile_names(self) -> List[str]:
        # CHEAP — names only, no per-profile Vs30/summary computation
        # (spec 002 FR-11); use profiles() when summaries are needed.
        return self._analysis.profile_names() if self._analysis else []

    def profile_dict(self, name: str) -> Optional[Dict[str, Any]]:
        """Full profile dict (name + layers) or None."""
        if self._analysis is None or not name:
            return None
        try:
            return self._analysis.get_profile(name)
        except KeyError:
            return None

    def profile_info(self, name: str) -> Optional[Dict[str, Any]]:
        """The summary row (n_layers / total_depth / vs30 / f0) or None."""
        for row in self.profiles() or []:
            if row["name"] == name:
                return row
        return None

    def update_profile(
        self, name: str, layers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Replace a profile's layers (the Data-stage table's Apply).

        Downstream results are invalidated by the api; all of this
        profile's run badges reset."""
        env = self._sync(
            lambda: self._analysis.update_profile(name, layers),
            on_success=(self.profiles_changed, self.forward_changed,
                        self.strip_changed),
            tool=None, mutate=True,
        )
        if _ok(env):
            self._reset_profile_statuses(name)
        return env

    def remove_profile(self, name: str) -> Dict[str, Any]:
        env = self._sync(
            lambda: self._analysis.remove_profile(name),
            on_success=(self.profiles_changed, self.forward_changed,
                        self.strip_changed),
            tool=None, mutate=True,
        )
        self._profile_checked.discard(name)
        if self._profile_focus == name:
            self.set_focus(None)
        self._reset_profile_statuses(name)
        self.checked_changed.emit(sorted(self._profile_checked))
        return env

    def suggest_layer_fill(
        self, vs: float, nu: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Auto-fill suggestions (ν, Vp, density, soil type) for one Vs."""
        return HVStripAnalysis.suggest_layer_fill(vs, nu=nu)

    # ==================================================================
    #  The run set + focus + per-tool settings/status (ProfilesPanel)
    # ==================================================================
    def checked_profiles(self) -> List[str]:
        """The RUN SET — names checked in the ProfilesPanel."""
        existing = set(self.profile_names())
        return sorted(self._profile_checked & existing)

    def set_checked(self, names: List[str]) -> None:
        new = set(names)
        if new == self._profile_checked:
            return
        self._profile_checked = new
        self.checked_changed.emit(sorted(new))

    def set_profile_checked(self, name: str, checked: bool) -> None:
        if checked == (name in self._profile_checked):
            return
        (self._profile_checked.add if checked
         else self._profile_checked.discard)(name)
        self.checked_changed.emit(sorted(self._profile_checked))

    @property
    def focus(self) -> Optional[str]:
        return self._profile_focus

    def set_focus(self, name: Optional[str]) -> None:
        if name == self._profile_focus:
            return
        self._profile_focus = name
        self.focus_changed.emit(name)

    def assign_settings(
        self, tool: StripTool, names: List[str], cfg: Dict[str, Any],
    ) -> None:
        """Attach a per-profile settings clone for one tool (HV Pro's
        assign-settings-to-checked pattern)."""
        import copy

        bucket = self._profile_settings.setdefault(tool, {})
        for name in names:
            bucket[name] = copy.deepcopy(cfg)
            self.profile_settings_changed.emit(name)
        self._set_dirty(True)

    def profile_settings(
        self, tool: StripTool, name: str,
    ) -> Optional[Dict[str, Any]]:
        return self._profile_settings.get(tool, {}).get(name)

    def has_settings(self, tool: StripTool, name: str) -> bool:
        return name in self._profile_settings.get(tool, {})

    def profile_status(self, tool: StripTool, name: str) -> ProcessingStatus:
        return self._profile_status.get(tool, {}).get(
            name, ProcessingStatus.NOT_STARTED)

    def _set_profile_statuses(
        self, tool: StripTool, names: List[str], status: ProcessingStatus,
    ) -> None:
        bucket = self._profile_status.setdefault(tool, {})
        for name in names:
            if bucket.get(name) is status:
                continue
            bucket[name] = status
            self.profile_status_changed.emit(tool, name)

    def _reset_profile_statuses(self, name: str) -> None:
        for tool, bucket in self._profile_status.items():
            if bucket.pop(name, None) is not None:
                self.profile_status_changed.emit(tool, name)

    def _resolve_op_profile(self, requested: Optional[str]) -> Optional[str]:
        """Resolve the profile a single-profile op targets, BEFORE submit.

        The api's ``_resolve_profile(None)`` RAISES whenever the count isn't
        exactly one, so ``None`` must never reach the worker — default to the
        first loaded profile here, or surface a clean error."""
        if requested is not None:
            return requested
        names = self.profile_names()
        if not names:
            self.error.emit(["No profiles loaded."])
            return None
        return names[0]

    def _op_profile_names(self, op: str, requested: Optional[str]) -> List[str]:
        """Which profiles an op involves (for badge transitions)."""
        names = self.profile_names()
        if requested is not None:
            return [requested]
        if op in ("run_forward", "run_strip"):
            return names[:1]   # unreachable — run_* resolve None pre-submit
        if op == "run_research":
            checked = self.checked_profiles()
            return checked or names
        return names          # run_forward_all / run_batch_strip

    # ==================================================================
    #  Config (dataclass sections; panels push scalars here)
    # ==================================================================
    def update_config(self, section: str, **fields) -> None:
        """Set fields on one ``HVStripConfig`` section; emits
        ``config_changed(section)``.  No compute.  Routes through the
        facade's sanctioned funnel (spec 002 DC-5)."""
        if self._analysis is None:
            self.error.emit(["No session open."])
            return
        env = self._analysis.update_section(section, **fields)
        if not env.get("success"):
            self.error.emit([env.get("error", "config update failed")])
            return
        if env.get("unmapped"):
            self.error.emit(
                [f"Ignored unknown config field(s): "
                 f"{', '.join(env['unmapped'])}"])
        self._engines_report = None if section == "engine" else self._engines_report
        self._set_dirty(True)
        self.config_changed.emit(section)

    def set_engine(self, name: str) -> None:
        self.update_config("engine", name=name)

    # ==================================================================
    #  Long ops — the MAIN queue (Forward + Strip)
    # ==================================================================
    _compute_imported = False

    def _ensure_compute_imports(self) -> None:
        """First-import the heavy compute/report stack (matplotlib + the
        core postprocess/report modules) on the GUI thread, ONCE, before
        any main-queue op runs.  Same doctrine as
        :meth:`_ensure_research_imports`: a worker thread FIRST-importing
        heavy native extensions after another worker has touched the
        scipy/sklearn stack hard-aborts the process on Windows.  The
        deliberate block is wrapped in :attr:`busy_hint` so the UI can
        show it (spec 002 FR-12)."""
        if AppState._compute_imported:
            return
        self.busy_hint.emit(
            "Preparing compute libraries (one-time)…")
        try:
            self._do_compute_imports()
        finally:
            self.busy_hint.emit("")

    @staticmethod
    def _do_compute_imports() -> None:
        if AppState._compute_imported:
            return
        from HV_Strip_Progressive.api import preload_heavy_modules

        preload_heavy_modules()
        AppState._compute_imported = True

    def _track_op(self, op: str, requested: Optional[str] = None) -> None:
        """Record the op's profiles (FIFO per op name — the queues run
        same-name submits in order) + flip them to QUEUED."""
        names = self._op_profile_names(op, requested)
        self._op_profiles.setdefault(op, []).append(names)
        tool = _OP_TOOL.get(op)
        if tool is not None:
            self._set_profile_statuses(tool, names, ProcessingStatus.QUEUED)

    def run_forward(self, profile_name: Optional[str] = None) -> None:
        a = self._require()
        if a is None:
            return
        profile_name = self._resolve_op_profile(profile_name)
        if profile_name is None:
            return
        self._ensure_compute_imports()
        self._track_op("run_forward", profile_name)
        self._queue.submit(
            "run_forward",
            lambda: a.compute_forward_single(profile_name),
        )

    def run_forward_all(self) -> None:
        a = self._require()
        if a is None:
            return
        self._ensure_compute_imports()
        self._track_op("run_forward_all")
        self._queue.submit(
            "run_forward_all",
            lambda: a.compute_forward_all(progress_cb=self._progress),
        )

    def run_strip(
        self,
        profile_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        a = self._require()
        if a is None:
            return
        profile_name = self._resolve_op_profile(profile_name)
        if profile_name is None:
            return
        self._ensure_compute_imports()
        self._track_op("run_strip", profile_name)
        self._queue.submit(
            "run_strip",
            lambda: a.run_stripping_for_profile(
                profile_name, output_dir=output_dir,
                progress_cb=self._progress,
            ),
        )

    def run_batch_strip(self, output_dir: Optional[str] = None) -> None:
        a = self._require()
        if a is None:
            return
        self._ensure_compute_imports()
        self._track_op("run_batch_strip")
        self._queue.submit(
            "run_batch_strip",
            lambda: a.run_batch_stripping_all(
                output_dir=output_dir, progress_cb=self._progress,
            ),
        )

    # ==================================================================
    #  Long ops — the RESEARCH queue (own lane + cooperative cancel)
    # ==================================================================
    RESEARCH_PHASES = (
        "profiles", "comparison", "metrics", "field_validation", "report",
    )

    _research_imported = False

    def _ensure_research_imports(self) -> None:
        """First-import the research stack (sklearn/matplotlib-heavy) on the
        GUI thread, ONCE, before any research op runs.  Two worker threads
        first-importing heavy native extensions concurrently (research on its
        queue + a forward op on the main queue) hard-aborts the process on
        Windows — serialising the import here removes the race."""
        if AppState._research_imported:
            return
        self.busy_hint.emit(
            "Preparing research libraries (one-time)…")
        try:
            self._do_compute_imports()   # the study's report phase plots too
            import HV_Strip_Progressive.research.runner  # noqa: F401
            import HV_Strip_Progressive.research.metrics  # noqa: F401

            AppState._research_imported = True
        finally:
            self.busy_hint.emit("")

    def run_research_phase(
        self, phase: str, study_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        a = self._require()
        if a is None:
            return
        self._ensure_research_imports()
        self._research_cancelled = False
        self._track_op("run_research")
        self._research_queue.submit(
            "run_research",
            lambda: a.run_research_study(
                study_config=study_config, phase=phase,
                progress_cb=self._progress,
                reset=(phase == "profiles"),
            ),
        )

    def run_full_study(
        self, study_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """The full pipeline with COOPERATIVE cancel between phases (the
        api's "full" mode can't be interrupted; AppState sequences the
        phases itself and checks the flag between them)."""
        a = self._require()
        if a is None:
            return
        self._ensure_research_imports()
        self._research_cancelled = False

        def _run() -> Dict[str, Any]:
            results: Dict[str, Any] = {}
            for phase in self.RESEARCH_PHASES:
                if self._research_cancelled:
                    return {"success": False, "cancelled": True,
                            "results": results}
                env = a.run_research_study(
                    study_config=study_config if phase == "profiles" else None,
                    phase=phase, progress_cb=self._progress,
                    reset=(phase == "profiles"),
                )
                results[phase] = env
                if not _ok(env):
                    return {"success": False, "phase": phase,
                            "error": env.get("error", "study phase failed"),
                            "results": results}
            return {"success": True, "results": results}

        self._track_op("run_research")
        self._research_queue.submit("run_research", _run)

    def cancel_research(self) -> None:
        """Cooperative: takes effect between study phases."""
        self._research_cancelled = True

    def research_results(self) -> Dict[str, Any]:
        return dict(self._research_results)

    def research_phase_result(self, phase: str) -> Dict[str, Any]:
        """The RAW result dict for one study phase, whatever the run shape
        (single-phase runs store raw results; the full-study loop stores
        per-phase envelopes carrying ``results``)."""
        entry = self._research_results.get(phase)
        if not isinstance(entry, dict):
            return {}
        if "results" in entry and isinstance(entry["results"], dict) \
                and ("success" in entry or "phase" in entry):
            return entry["results"]
        return entry

    def research_figures(self) -> List[tuple]:
        """``(name, path)`` pairs for the study's generated figure files."""
        files = self.research_phase_result("report").get("files", {})
        if not isinstance(files, dict):
            return []
        out = []
        for name, path in sorted(files.items()):
            if str(path).lower().endswith((".png", ".jpg", ".svg")):
                out.append((str(name), str(path)))
        return out

    # ==================================================================
    #  Results accessors (read-only)
    # ==================================================================
    def forward_results(self) -> Dict[str, Any]:
        a = self._analysis
        return a.forward_results() if a else {}

    def strip_results(self) -> Dict[str, Any]:
        a = self._analysis
        return a.strip_results() if a else {}

    def batch_result(self):
        a = self._analysis
        return a.batch_result() if a else None

    def vs_context(
        self,
        layers: List[Dict[str, Any]],
        bedrock_depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Vs30/VsAvg/interfaces for a layer stack (the Vs mini-panel)."""
        if self._analysis is None:
            return {"success": False, "error": "No session open."}
        return self._analysis.vs_context_for_layers(
            layers, bedrock_depth=bedrock_depth)

    def profile_layers_from_file(self, path: str) -> List[Dict[str, Any]]:
        """Parse a model file into plot-ready ``[{thickness, vs}, …]`` (for
        the per-step Vs view) — through the api, never core directly."""
        try:
            from HV_Strip_Progressive.api.profile_io import load_profile

            profile = load_profile(path)
            return [{"thickness": float(ly.thickness), "vs": float(ly.vs)}
                    for ly in profile.layers]
        except Exception:                                 # noqa: BLE001
            return []

    def report_overlay_on_figure(self, fig) -> bool:
        """Draw the strip-report HV overlay onto a matplotlib figure (the
        Figure Studio view).  Uses the ACTIVE strip result's directory via
        the report analysis module (importable per the layering audit)."""
        strips = self.strip_results()
        for result in strips.values():
            strip_dir = getattr(result, "strip_directory", "")
            if not strip_dir:
                continue
            try:
                from HV_Strip_Progressive.core.report_generator import (
                    ProgressiveStrippingReporter,
                )

                reporter = ProgressiveStrippingReporter(strip_dir)
                return bool(reporter.draw_hv_overlay_on_figure(fig))
            except Exception as exc:                      # noqa: BLE001
                self.error.emit([f"Figure Studio: {exc}"])
                return False
        return False

    # ==================================================================
    #  Persistence — v2 payloads through the ONE funnel
    # ==================================================================
    def load_settings(self, path: Optional[Path] = None) -> bool:
        """Best-effort read of the standalone settings (v2 or legacy)."""
        if self._analysis is None:
            return False
        target = Path(path) if path else _SETTINGS_PATH
        if not target.is_file():
            return False
        try:
            import yaml

            payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        except Exception:                                # noqa: BLE001
            return False
        self._analysis.apply_config_payload(payload)
        self._engines_report = None
        self.config_changed.emit("*")
        return True

    def save_settings(self, path: Optional[Path] = None) -> bool:
        if self._analysis is None:
            return False
        target = Path(path) if path else _SETTINGS_PATH
        try:
            import yaml

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                yaml.safe_dump(self._analysis.config_payload(),
                               sort_keys=False),
                encoding="utf-8",
            )
        except Exception as exc:                          # noqa: BLE001
            self.error.emit([f"Could not save settings: {exc}"])
            return False
        self._set_dirty(False)
        return True

    def config_payload(self) -> Dict[str, Any]:
        """The v2 payload for project persistence (``hvstrip_state_io``)."""
        return self._analysis.config_payload() if self._analysis else {}

    def apply_config_payload(self, payload: Any) -> None:
        """Project persistence load — v2 or legacy, via the funnel."""
        if self._analysis is None:
            return
        self._analysis.apply_config_payload(payload)
        self._engines_report = None
        self.config_changed.emit("*")

    # ==================================================================
    #  Internals
    # ==================================================================
    def _require(self) -> Optional[HVStripAnalysis]:
        if self._analysis is None:
            self.error.emit(["No session open."])
        return self._analysis

    def _sync(
        self,
        fn: Callable[[], Dict[str, Any]],
        *,
        on_success: tuple,
        tool: Optional[StripTool],
        mutate: bool,
    ) -> Dict[str, Any]:
        if self._analysis is None:
            self.error.emit(["No session open."])
            return {"success": False, "error": "No session open."}
        try:
            env = fn()
        except Exception as exc:                          # noqa: BLE001
            # A bad user file (unparsable profile, missing path) is a
            # STATUS, never a crash.
            env = {"success": False, "error": str(exc)}
        if _ok(env):
            if tool is not None:
                self._error_tools.discard(tool)
            for signal in on_success:
                signal.emit()
            if mutate:
                self._set_dirty(True)
        else:
            if tool is not None:
                self._error_tools.add(tool)
            self.error.emit([str(env.get("error", "operation failed"))])
        return env

    def _on_op_started(self, name: str) -> None:
        tool = _OP_TOOL.get(name)
        if tool is not None:
            pending = self._op_profiles.get(name) or [[]]
            self._set_profile_statuses(tool, pending[0],
                                       ProcessingStatus.RUNNING)
        self.op_started.emit(name)

    def _on_progress_frame(self, frame: dict) -> None:
        """Per-profile batch frames narrow RUNNING to the active profile."""
        if frame.get("type") != "profile":
            return
        op = self.current_op
        tool = _OP_TOOL.get(op or "")
        profile = frame.get("profile")
        if tool is None or not profile:
            return
        self._set_profile_statuses(tool, [str(profile)],
                                   ProcessingStatus.RUNNING)

    def _on_op_finished(self, name: str, env: dict) -> None:
        tool = _OP_TOOL.get(name)
        pending = self._op_profiles.get(name) or []
        op_names = pending.pop(0) if pending else []
        if tool is not None and op_names:
            if env.get("cancelled"):
                final = ProcessingStatus.NOT_STARTED
            elif _ok(env):
                final = ProcessingStatus.DONE
            else:
                final = ProcessingStatus.FAILED
            self._set_profile_statuses(tool, op_names, final)
        if _ok(env):
            if tool is not None:
                self._error_tools.discard(tool)
            if name in ("run_forward", "run_forward_all"):
                self.forward_changed.emit()
            elif name in ("run_strip", "run_batch_strip"):
                self.strip_changed.emit()
            elif name == "run_research":
                phase = env.get("phase")
                results = env.get("results", {}) or {}
                if phase and phase != "full":
                    self._research_results[phase] = results
                else:
                    self._research_results = (
                        dict(results) if isinstance(results, dict) else {})
                self.research_changed.emit()
            self._set_dirty(True)
        elif env.get("cancelled"):
            # A cancelled study is a benign finish, not an error.
            results = env.get("results", {}) or {}
            if isinstance(results, dict):
                self._research_results = dict(results)
            self.research_changed.emit()
        else:
            if tool is not None:
                self._error_tools.add(tool)
            self.error.emit([str(env.get("error", f"{name} failed"))])
        self.op_finished.emit(name, env)

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)

    def shutdown(self) -> None:
        self._research_cancelled = True
        self._queue.shutdown()
        self._research_queue.shutdown()


__all__ = ["AppState"]
