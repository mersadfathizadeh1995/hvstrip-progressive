"""``StripMainWindow`` — the HV Strip house-style workbench (archetype C).

The agreed tool-collection shell (``HV_Pro_docs/hvstrip_gui/SKETCH.txt``):
a :class:`ToolSwitcher` of **3 peer tools** (Forward Model · HV Strip ·
Research; per-tool status pulled from :meth:`AppState.status_for`), over a
3-pane ``QSplitter``:

* **left**  — a :class:`CollapsibleSideRail` wrapping a ``QStackedWidget``
  of per-tool panels (``gui/v2/tools/*``);
* **centre**— a ``QStackedWidget`` of per-tool canvas view-sets;
* **right** — the Layers | Properties rail (bedrock nesting) over one
  :class:`LayerModel` — a strip run's STEPS are the layers.

Plus the tabbed **Log | Problems | Progress** dock (Progress = the live
batch :class:`RunTable`) and a status bar with the ENGINE BADGES from
``check_engines()`` (an unavailable engine is a status, never a crash).
Follows ``theme_core.theme_authority`` per-window (amber accent).
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hvsr_pro.packages.theme_core import theme_authority

from HV_Strip_Progressive.api import HVStripAnalysis
from HV_Strip_Progressive.gui.v2.state import AppState, StripTool
from HV_Strip_Progressive.gui.v2.state.layer_model import LayerModel
from HV_Strip_Progressive.gui.v2.theme import apply_theme, resolve_palette
from HV_Strip_Progressive.gui.v2.widgets.house.layer_tree import LayerTree
from HV_Strip_Progressive.gui.v2.widgets.house.profiles_panel import (
    ProfilesPanel,
)
from HV_Strip_Progressive.gui.v2.widgets.house.properties_panel import (
    PropertiesPanel,
)
from HV_Strip_Progressive.gui.v2.widgets.house.run_table import RunTable
from HV_Strip_Progressive.gui.v2.widgets.house.tool_switcher import ToolSwitcher
from HV_Strip_Progressive.gui.v2.workbench.side_rail import CollapsibleSideRail

WINDOW_TITLE = "HV Pro — HV Strip"

_FILES_W = 235
_LEFT_W = 340
_LAYERS_W = 190
_PROPS_W = 210
_RIGHT_W = _LAYERS_W + _PROPS_W


def _placeholder(text: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.addStretch(1)
    lbl = QLabel(text)
    lbl.setProperty("role", "muted")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    lay.addWidget(lbl)
    lay.addStretch(1)
    return w


class StripMainWindow(QMainWindow):
    """The house-style tool-collection shell (see module docstring)."""

    def __init__(
        self,
        analysis: Optional[HVStripAnalysis] = None,
        *,
        project_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        state_payload: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project_name = project_name
        self._init_output_dir = output_dir
        self.resize(1320, 840)

        self.app_state = AppState(analysis or HVStripAnalysis(), parent=self)
        self.layer_model = LayerModel(self.app_state, self)

        self.app_state.dirty_changed.connect(lambda _d: self._refresh_title())
        self.app_state.op_started.connect(self._on_op_started)
        self.app_state.op_progress.connect(self._on_op_progress)
        self.app_state.op_finished.connect(self._on_op_finished)
        self.app_state.error.connect(self._on_error)
        self.app_state.busy_hint.connect(self._on_busy_hint)
        # Engine badges follow engine-config changes live (spec 002 FR-12).
        self.app_state.config_changed.connect(
            lambda s: self._refresh_engine_badges()
            if s in ("engine", "*") else None)

        self._panels: Dict[StripTool, QWidget] = {}
        self._canvases: Dict[StripTool, QWidget] = {}

        self._build_menus()
        self._build_body()
        self._build_dock()
        self._build_status_bar()
        self._install_tools()

        # Theme authority (ADR-0014): self-theme now, follow mode_changed.
        self._retheme(theme_authority.current_mode)
        theme_authority.mode_changed.connect(self._retheme)

        if state_payload:
            self.app_state.apply_config_payload(state_payload)
        elif self.app_state.load_settings():
            pass  # standalone settings restored

        if self._init_output_dir:
            self.app_state.update_config(
                "output", output_dir=str(self._init_output_dir))

        self._activate_tool(StripTool.DATA)
        self._restore_window_state()
        self._refresh_title()
        self._refresh_engine_badges()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_body(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.switcher = ToolSwitcher(self.app_state, central)
        self.switcher.tool_activated.connect(self._activate_tool)
        self.switcher.set_project_label(self._project_name or "")
        layout.addWidget(self.switcher)

        self._splitter = QSplitter(Qt.Horizontal, central)
        self._splitter.setChildrenCollapsible(False)

        # Left: per-tool panels (populated by the tool ports; placeholders
        # until each tool lands).
        self._panel_stack = QStackedWidget()
        self._canvas_stack = QStackedWidget()
        for tool in StripTool:
            panel = _placeholder(
                f"{tool.label} controls arrive with its tool port.")
            canvas = _placeholder(f"{tool.label} views arrive with its port.")
            self._panels[tool] = panel
            self._canvases[tool] = canvas
            self._panel_stack.addWidget(panel)
            self._canvas_stack.addWidget(canvas)

        # The GLOBAL files rail — leftmost, in every stage (Round 2).
        self.profiles_panel = ProfilesPanel(self.app_state)
        self.profiles_panel.load_requested.connect(
            lambda: self._activate_tool(StripTool.DATA))
        self.profiles_panel.assign_settings_requested.connect(
            self._on_assign_settings)
        self._files_rail = CollapsibleSideRail(
            self.profiles_panel, side="left", expanded_width=_FILES_W)

        self._left_rail = CollapsibleSideRail(
            self._panel_stack, side="left", expanded_width=_LEFT_W)
        self._splitter.addWidget(self._files_rail)
        self._splitter.addWidget(self._left_rail)
        self._splitter.addWidget(self._canvas_stack)
        self._splitter.addWidget(self._build_right_dock())
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setStretchFactor(2, 1)
        self._splitter.setStretchFactor(3, 0)
        for rail in (self._files_rail, self._left_rail, self._right_rail):
            rail.collapsed_changed.connect(
                lambda _c: self._rebalance_splitter())
        layout.addWidget(self._splitter, 1)

        self.setCentralWidget(central)
        self._rebalance_splitter()

    def _build_right_dock(self) -> CollapsibleSideRail:
        self._layer_tree = LayerTree(self.layer_model)
        self._properties = PropertiesPanel(self.layer_model, self.app_state)
        self._layer_tree.layer_selected.connect(self._properties.set_layer)

        # Bedrock nesting: collapse the OUTER rail → both vanish; collapse
        # the INNER props rail → only Properties vanishes.
        self._layers_box = self._titled("Layers", self._layer_tree)
        self._props_rail = CollapsibleSideRail(
            self._titled("Properties", self._properties),
            side="right", expanded_width=_PROPS_W)
        region = QWidget()
        rl = QHBoxLayout(region)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._layers_box, 1)
        rl.addWidget(self._props_rail, 0)
        self._right_rail = CollapsibleSideRail(
            region, side="right", expanded_width=_RIGHT_W)
        self._props_rail.collapsed_changed.connect(
            lambda _c: self._on_props_toggled())
        return self._right_rail

    def _on_props_toggled(self) -> None:
        self._right_rail.set_expanded_width(
            _LAYERS_W + self._props_rail.current_width())
        self._rebalance_splitter()

    @staticmethod
    def _titled(title: str, body: QWidget) -> QWidget:
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(6, 6, 6, 0)
        lay.setSpacing(2)
        head = QLabel(title)
        head.setProperty("role", "h2")
        lay.addWidget(head)
        lay.addWidget(body, 1)
        return holder

    def _rebalance_splitter(self) -> None:
        total = max(self._splitter.width(), 1100)
        files = self._files_rail.current_width()
        left = self._left_rail.current_width()
        right = self._right_rail.current_width()
        self._splitter.setSizes(
            [files, left, max(360, total - files - left - right), right])

    def _build_dock(self) -> None:
        self._dock = QDockWidget("Console", self)
        self._dock.setAllowedAreas(
            Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self._dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        tabs = QTabWidget(self._dock)
        tabs.setObjectName("WorkbenchDock")
        tabs.setDocumentMode(True)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        tabs.addTab(self._log, "Log")

        self._problems = QListWidget()
        tabs.addTab(self._problems, "Problems")

        progress_tab = QWidget()
        pl = QVBoxLayout(progress_tab)
        pl.setContentsMargins(6, 6, 6, 6)
        pl.setSpacing(4)
        self.run_table = RunTable()
        pl.addWidget(self.run_table, 1)
        self._cancel_btn = QPushButton("Cancel research")
        self._cancel_btn.clicked.connect(self.app_state.cancel_research)
        self._cancel_btn.setEnabled(False)
        pl.addWidget(self._cancel_btn, 0, Qt.AlignRight)
        tabs.addTab(progress_tab, "Progress")

        self._dock_tabs = tabs
        self._dock.setWidget(tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._dock)

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        self._status_msg = QLabel("Ready")
        status.addWidget(self._status_msg, 1)
        self._engine_badges = QLabel("")
        self._engine_badges.setProperty("role", "caption")
        status.addPermanentWidget(self._engine_badges)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.hide()
        status.addPermanentWidget(self._progress)

    def _build_menus(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        settings_act = QAction("&Settings…", self)
        settings_act.setShortcut("Ctrl+Shift+,")
        settings_act.triggered.connect(self._on_settings)
        file_menu.addAction(settings_act)
        save_act = QAction("Save se&ttings", self)
        save_act.setShortcut(QKeySequence.Save)
        # Qt passes `checked: bool` into a triggered slot — never bind a
        # method with meaningful defaults directly (spec 002 audit O19).
        save_act.triggered.connect(lambda: self.app_state.save_settings())
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = bar.addMenu("&View")
        for i, tool in enumerate(StripTool, start=1):
            act = QAction(f"&{i}. {tool.label}", self)
            act.setShortcut(f"Ctrl+{i}")
            act.triggered.connect(
                lambda _c=False, t=tool: self._activate_tool(t))
            view_menu.addAction(act)
        view_menu.addSeparator()
        for label, shortcut, slot in (
            ("Toggle &Files", "Ctrl+Shift+[",
             lambda: self._files_rail.toggle()),
            ("Toggle &Controls", "Ctrl+[", lambda: self._left_rail.toggle()),
            ("Toggle &Layers", "Ctrl+]", lambda: self._right_rail.toggle()),
            ("Toggle &Properties", "Ctrl+\\",
             lambda: self._props_rail.toggle()),
            ("Toggle Cons&ole", "Ctrl+`",
             lambda: self._dock.setVisible(not self._dock.isVisible())),
        ):
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            view_menu.addAction(act)

        help_menu = bar.addMenu("&Help")
        about = QAction("&About HV Strip", self)
        about.triggered.connect(self._on_about)
        help_menu.addAction(about)

    # ------------------------------------------------------------------
    # Tool registration (the tool ports call these)
    # ------------------------------------------------------------------
    def _install_tools(self) -> None:
        """Replace placeholders with the ported tools (S3–S5)."""
        from HV_Strip_Progressive.gui.v2.tools.data.panel import (
            DataCanvas,
            DataToolPanel,
        )
        from HV_Strip_Progressive.gui.v2.tools.forward.panel import (
            ForwardCanvas,
            ForwardToolPanel,
        )
        from HV_Strip_Progressive.gui.v2.tools.research.panel import (
            ResearchCanvas,
            ResearchToolPanel,
        )
        from HV_Strip_Progressive.gui.v2.tools.strip.panel import (
            StripCanvas,
            StripToolPanel,
        )

        self.set_tool_widgets(
            StripTool.DATA,
            DataToolPanel(self.app_state),
            DataCanvas(self.app_state, self.layer_model),
        )
        self.set_tool_widgets(
            StripTool.FORWARD,
            ForwardToolPanel(self.app_state),
            ForwardCanvas(self.app_state, self.layer_model),
        )
        self.set_tool_widgets(
            StripTool.STRIP,
            StripToolPanel(self.app_state),
            StripCanvas(self.app_state, self.layer_model),
        )
        self.set_tool_widgets(
            StripTool.RESEARCH,
            ResearchToolPanel(self.app_state),
            ResearchCanvas(self.app_state, self.layer_model),
        )

    def set_tool_widgets(
        self, tool: StripTool, panel: QWidget, canvas: QWidget,
    ) -> None:
        """Replace a tool's placeholder panel + canvas (S3–S5 ports)."""
        old_panel = self._panels[tool]
        idx = self._panel_stack.indexOf(old_panel)
        self._panel_stack.removeWidget(old_panel)
        old_panel.deleteLater()
        self._panel_stack.insertWidget(idx, panel)
        self._panels[tool] = panel

        old_canvas = self._canvases[tool]
        cidx = self._canvas_stack.indexOf(old_canvas)
        self._canvas_stack.removeWidget(old_canvas)
        old_canvas.deleteLater()
        self._canvas_stack.insertWidget(cidx, canvas)
        self._canvases[tool] = canvas
        self._activate_tool(self.switcher.active_tool)

    def panel(self, tool: StripTool) -> QWidget:
        return self._panels[tool]

    def canvas(self, tool: StripTool) -> QWidget:
        return self._canvases[tool]

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def show_tool(self, tool: StripTool) -> None:
        self._activate_tool(tool)

    def _activate_tool(self, tool: StripTool) -> None:
        self.app_state.set_active_tool(tool)
        self.switcher.set_active(tool)
        self._panel_stack.setCurrentWidget(self._panels[tool])
        self._canvas_stack.setCurrentWidget(self._canvases[tool])
        self.layer_model.set_tool(tool)
        self._cancel_btn.setEnabled(tool is StripTool.RESEARCH)

    @property
    def active_tool(self) -> StripTool:
        return self.switcher.active_tool

    # ------------------------------------------------------------------
    # Console / status wiring
    # ------------------------------------------------------------------
    def _on_op_started(self, name: str) -> None:
        self._status_msg.setText(f"Running {name}…")
        self._progress.setRange(0, 0)
        self._progress.show()
        self._log.appendPlainText(f"▶ {name} started")
        if name in ("run_batch_strip", "run_forward_all"):
            self.run_table.begin_run()
            self._dock_tabs.setCurrentIndex(2)
        if name == "run_research":
            self._cancel_btn.setEnabled(True)

    def _on_op_progress(self, frame: dict) -> None:
        kind = frame.get("type")
        if kind == "log":
            self._log.appendPlainText(str(frame.get("text", "")))
        elif kind == "phase":
            self._status_msg.setText(
                f"[{frame.get('index')}/{frame.get('total')}] "
                f"{frame.get('label', '')}")
        elif kind == "study":
            self._status_msg.setText(
                f"Study {frame.get('index')}/{frame.get('total')}: "
                f"{frame.get('label', '')}")
        self.run_table.on_frame(frame)

    def _on_op_finished(self, name: str, env: dict) -> None:
        ok = bool(env.get("success", True)) and not env.get("error")
        cancelled = bool(env.get("cancelled"))
        self._progress.hide()
        self._status_msg.setText(
            "Ready" if ok else
            f"{name} cancelled" if cancelled else f"{name} failed")
        self._log.appendPlainText(
            f"{'✓' if ok else '⏹' if cancelled else '✗'} {name} finished")
        if name in ("run_batch_strip", "run_forward_all"):
            self.run_table.finalize(env)
        self.switcher.refresh_statuses()
        self._refresh_title()

    def _on_error(self, errors: list) -> None:
        for err in errors or []:
            self._problems.addItem(str(err))
            self._log.appendPlainText(f"✗ {err}")

    def _on_busy_hint(self, text: str) -> None:
        """Busy feedback around a deliberate main-thread block (the
        one-time heavy preload) — spec 002 FR-12."""
        from PySide6.QtWidgets import QApplication

        if text:
            QApplication.setOverrideCursor(Qt.BusyCursor)
            self._status_msg.setText(text)
            QApplication.processEvents()   # paint the note BEFORE the block
        else:
            QApplication.restoreOverrideCursor()

    #: The config sections whose current values an "Assign settings" snapshot
    #: captures, per tool (Track 2 dispatches runs with these per-profile).
    _ASSIGN_SECTIONS = {
        StripTool.FORWARD: ("engine", "frequency", "peak_detection"),
        StripTool.STRIP: ("engine", "frequency", "strip", "adaptive",
                          "dual_resonance"),
    }

    def _on_assign_settings(self, names: list) -> None:
        tool = self.app_state.active_tool
        sections = self._ASSIGN_SECTIONS.get(tool)
        if not sections:
            self._status_msg.setText(
                f"{tool.label} has no per-profile settings to assign.")
            return
        cfg_dict = self.app_state.config_payload()
        snapshot = {s: cfg_dict[s] for s in sections if s in cfg_dict}
        self.app_state.assign_settings(tool, list(names), snapshot)
        self._status_msg.setText(
            f"{tool.label} settings assigned to {len(names)} profile(s).")

    def _refresh_title(self) -> None:
        star = " •" if self.app_state.dirty else ""
        name = f" — {self._project_name}" if self._project_name else ""
        self.setWindowTitle(f"{WINDOW_TITLE}{name}{star}")

    def _refresh_engine_badges(self) -> None:
        report = self.app_state.engines_report()
        parts = []
        for eng, label in (("diffuse_field", "HVf"), ("sh_wave", "SH"),
                           ("ellipticity", "ellipticity")):
            entry = report.get(eng, {})
            mark = "✓" if entry.get("available") else "✗"
            parts.append(f"{label} {mark}")
        self._engine_badges.setText("engines:  " + "   ".join(parts))

    # ------------------------------------------------------------------
    def _on_settings(self) -> None:
        from HV_Strip_Progressive.gui.v2.dialogs.settings_dialog import (
            SettingsDialog,
        )

        dlg = SettingsDialog(self.app_state, self)
        dlg.exec()
        self._refresh_engine_badges()

    def _on_about(self) -> None:
        QMessageBox.about(
            self, "About HV Strip",
            "HV Strip — progressive HVSR layer stripping.\n"
            "Part of the HV Pro family (house-style workbench).")

    # ------------------------------------------------------------------
    def _retheme(self, mode: str) -> None:
        apply_theme(self, mode)
        self._palette = resolve_palette(mode)
        for canvas in self._canvases.values():
            if hasattr(canvas, "set_palette"):
                canvas.set_palette(self._palette)

    # ------------------------------------------------------------------
    # Window-state persistence (spec 002 FR-14 — geometry, docks,
    # splitter sizes, rail collapse survive restarts)
    # ------------------------------------------------------------------
    _QS_ORG = "HV_Pro"
    _QS_APP = "hv_strip_v2"

    @staticmethod
    def _persist_enabled() -> bool:
        # Offscreen (test) sessions must never touch the user's real
        # window state; a persistence test opts in explicitly.
        import os

        return (os.environ.get("QT_QPA_PLATFORM") != "offscreen"
                or bool(os.environ.get("HVSTRIP_TEST_PERSIST")))

    def _rails(self) -> Dict[str, QWidget]:
        return {"files": self._files_rail, "left": self._left_rail,
                "right": self._right_rail, "props": self._props_rail}

    def _restore_window_state(self) -> None:
        from PySide6.QtCore import QSettings

        if not self._persist_enabled():
            return
        s = QSettings(self._QS_ORG, self._QS_APP)
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        st = s.value("windowState")
        if st is not None:
            self.restoreState(st)
        for key, rail in self._rails().items():
            val = s.value(f"rail/{key}")
            if val is not None:
                rail.set_collapsed(val in (True, "true", "1", 1))
        sp = s.value("splitter")
        if sp is not None:
            self._splitter.restoreState(sp)

    def _save_window_state(self) -> None:
        from PySide6.QtCore import QSettings

        if not self._persist_enabled():
            return
        s = QSettings(self._QS_ORG, self._QS_APP)
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())
        s.setValue("splitter", self._splitter.saveState())
        for key, rail in self._rails().items():
            s.setValue(f"rail/{key}", rail.is_collapsed())

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_window_state()
        self.app_state.shutdown()
        super().closeEvent(event)


__all__ = ["StripMainWindow", "WINDOW_TITLE"]
