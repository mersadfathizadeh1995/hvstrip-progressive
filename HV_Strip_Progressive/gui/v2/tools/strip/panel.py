"""The HV Strip tool — the 3-sub-stage wizard + Batch, and the canvas.

The agreed design (SKETCH.txt + DECISIONS #5/#6):

* **Single** = a gated wizard: **1. Model** (profile + output folder) →
  **2. Configure & Run** (strip options · adaptive · dual resonance ·
  engine/frequency · ▶ Run with live narration) → **3. Review** (per-step
  readout · dual resonance · report shortcuts).
* **Batch** = a profiles list + shared options + ▶ Run batch (streams into
  the Progress dock's RunTable).
* Canvas views: **Waterfall Overlay** (per-STEP keyed curves — the layers
  the right rail toggles/styles) · **Step HV ∥ Vs** · **Summary Table** ·
  **Figure Studio** (the ONE matplotlib view — publication export).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.canvas.hv_curve_canvas import HVCurveCanvas
from HV_Strip_Progressive.gui.v2.canvas.vs_profile_canvas import VsProfileCanvas
from HV_Strip_Progressive.gui.v2.stages.base import PhasePanel, card, status_line
from HV_Strip_Progressive.gui.v2.state import AppState, StripTool
from HV_Strip_Progressive.gui.v2.state.layer_model import LayerModel
from HV_Strip_Progressive.gui.v2.tools.forward.panel import (
    _ConfigCards,
    _pen_from_display,
)
from HV_Strip_Progressive.gui.v2.widgets.house.sub_breadcrumb import (
    SubBreadcrumb,
)


# ======================================================================
#  Sub-stage 1 — Model
# ======================================================================
class StripModelPanel(PhasePanel):
    """Profile in, output folder out."""

    def build(self) -> None:
        self._loading = False
        body = self.scroll_host()
        head = QLabel("1 · Model")
        head.setProperty("role", "h2")
        body.addWidget(head)
        body.addWidget(status_line(
            "Pick the layered model to strip and where results go.",
            role="caption"))

        grp, gl = card("Profile")
        frow = QHBoxLayout()
        self._file_lbl = status_line("No profile loaded.", role="muted")
        browse = QPushButton("Load profile…")
        browse.setProperty("primary", "true")
        browse.clicked.connect(self._on_browse)
        frow.addWidget(self._file_lbl, 1)
        frow.addWidget(browse)
        gl.addLayout(frow)
        body.addWidget(grp)

        grp, gl = card("Output")
        orow = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output folder…")
        self._out_edit.editingFinished.connect(self._on_out)
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._on_browse_out)
        orow.addWidget(self._out_edit, 1)
        orow.addWidget(out_btn)
        gl.addLayout(orow)
        body.addWidget(grp)
        body.addStretch(1)
        self.profile_name: Optional[str] = None

    def refresh(self) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            profiles = self.app_state.profiles()
            if self.profile_name is None and profiles:
                self.profile_name = profiles[-1]["name"]
            info = next((p for p in profiles
                         if p["name"] == self.profile_name), None)
            if info:
                self._file_lbl.setText(
                    f"{info['name']} · {info['n_layers']} layers")
            cfg = self.app_state.config
            if cfg and not self._out_edit.hasFocus():
                self._out_edit.setText(cfg.output.output_dir or "")
        finally:
            self._loading = False

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load soil profile", "",
            "Profiles (*.txt *.csv);;All files (*)")
        if not path:
            return
        env = self.app_state.load_profile(path)
        if env.get("name"):
            self.profile_name = env["name"]

    def _on_browse_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder")
        if folder:
            self._out_edit.setText(folder)
            self._on_out()

    def _on_out(self) -> None:
        self.app_state.update_config(
            "output", output_dir=self._out_edit.text().strip())


# ======================================================================
#  Sub-stage 2 — Configure & Run
# ======================================================================
class StripRunPanel(PhasePanel):
    """Strip options + the live run."""

    def __init__(self, app_state, model_panel: StripModelPanel, parent=None):
        self._model_panel = model_panel
        super().__init__(app_state, parent)

    def build(self) -> None:
        self._loading = False
        body = self.scroll_host()
        head = QLabel("2 · Configure && Run")
        head.setProperty("role", "h2")
        body.addWidget(head)

        grp, gl = card("Strip options")
        self._report_chk = QCheckBox("Generate the full report")
        self._report_chk.toggled.connect(
            lambda on: self.app_state.update_config(
                "strip", generate_report=bool(on))
            if not self._loading else None)
        gl.addWidget(self._report_chk)
        body.addWidget(grp)

        grp, gl = card("Adaptive frequency scan", collapsed=True)
        self._adaptive_chk = QCheckBox("Enable (re-scan when a peak hugs "
                                       "the band edge)")
        self._adaptive_chk.toggled.connect(
            lambda on: self.app_state.update_config(
                "adaptive", enable=bool(on))
            if not self._loading else None)
        gl.addWidget(self._adaptive_chk)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Max passes:"))
        self._passes = QSpinBox()
        self._passes.setRange(1, 5)
        self._passes.valueChanged.connect(
            lambda v: self.app_state.update_config(
                "adaptive", max_passes=int(v))
            if not self._loading else None)
        prow.addWidget(self._passes)
        prow.addStretch(1)
        gl.addLayout(prow)
        body.addWidget(grp)

        grp, gl = card("Dual resonance", collapsed=True)
        self._dual_chk = QCheckBox("Extract f0/f1 dual resonance")
        self._dual_chk.toggled.connect(
            lambda on: self.app_state.update_config(
                "dual_resonance", enabled=bool(on))
            if not self._loading else None)
        gl.addWidget(self._dual_chk)
        body.addWidget(grp)

        self.cards = _ConfigCards()
        self.cards.build(self, body)

        rrow = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run stripping")
        self._run_btn.setProperty("primary", "true")
        self._run_btn.clicked.connect(self._on_run)
        rrow.addStretch(1)
        rrow.addWidget(self._run_btn)
        body.addLayout(rrow)
        self._note = status_line("", role="muted")
        body.addWidget(self._note)
        body.addStretch(1)

    def refresh(self) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            cfg = self.app_state.config
            if cfg:
                self._report_chk.setChecked(bool(cfg.strip.generate_report))
                self._adaptive_chk.setChecked(bool(cfg.adaptive.enable))
                self._passes.setValue(int(cfg.adaptive.max_passes))
                self._dual_chk.setChecked(bool(cfg.dual_resonance.enabled))
            self.cards.refresh(self)
            ready = bool(self._model_panel.profile_name
                         and (cfg and cfg.output.output_dir))
            self._run_btn.setEnabled(ready and not self.app_state.is_busy)
            self._note.setText(
                "" if ready else
                "Load a profile and set the output folder on sub-stage 1.")
        finally:
            self._loading = False

    def _on_run(self) -> None:
        cfg = self.app_state.config
        out = Path(cfg.output.output_dir or ".") / (
            self._model_panel.profile_name or "strip_output")
        self.app_state.run_strip(
            self._model_panel.profile_name, output_dir=str(out))


# ======================================================================
#  Sub-stage 3 — Review
# ======================================================================
class StripReviewPanel(PhasePanel):
    """Per-step readout + dual resonance + report shortcuts."""

    def build(self) -> None:
        body = self.scroll_host()
        head = QLabel("3 · Review")
        head.setProperty("role", "h2")
        body.addWidget(head)

        grp, gl = card("Steps")
        self._steps = QTableWidget(0, 4)
        self._steps.setHorizontalHeaderLabels(
            ["Step", "Layers", "f₀ (Hz)", "A₀"])
        self._steps.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._steps.verticalHeader().setVisible(False)
        self._steps.setEditTriggers(QTableWidget.NoEditTriggers)
        self._steps.setMinimumHeight(180)
        gl.addWidget(self._steps)
        body.addWidget(grp)

        grp, gl = card("Dual resonance", collapsed=True)
        self._dual_lbl = status_line("Not extracted.", role="muted")
        gl.addWidget(self._dual_lbl)
        body.addWidget(grp)

        grp, gl = card("Report")
        rrow = QHBoxLayout()
        open_btn = QPushButton("Open output folder")
        open_btn.clicked.connect(self._on_open_folder)
        rrow.addWidget(open_btn)
        rrow.addStretch(1)
        gl.addLayout(rrow)
        self._report_lbl = status_line("", role="caption")
        gl.addWidget(self._report_lbl)
        body.addWidget(grp)
        body.addStretch(1)

    def refresh(self) -> None:
        strips = self.app_state.strip_results()
        result = next(iter(strips.values()), None)
        steps = getattr(result, "steps", None) or []
        self._steps.setRowCount(len(steps))
        for r, step in enumerate(steps):
            cells = (
                f"Step{step.step_number}",
                str(step.n_layers),
                f"{step.peak_frequency:.3f}" if step.peak_frequency else "—",
                f"{step.peak_amplitude:.2f}" if step.peak_amplitude else "—",
            )
            for c, text in enumerate(cells):
                self._steps.setItem(r, c, QTableWidgetItem(text))
        dual = getattr(result, "dual_resonance", None)
        self._dual_lbl.setText(str(dual) if dual else "Not extracted.")
        report = getattr(result, "report_files", None) or {}
        self._report_lbl.setText(
            f"{len(report)} report file(s) generated." if report else
            "No report generated (enable it on sub-stage 2).")

    def _on_open_folder(self) -> None:
        import os

        result = next(iter(self.app_state.strip_results().values()), None)
        out = getattr(result, "output_directory", "")
        if out and os.path.isdir(out):
            os.startfile(out)  # noqa: S606 — user-requested open


# ======================================================================
#  Batch
# ======================================================================
class StripBatchPanel(PhasePanel):
    """Many profiles → the Progress-dock table."""

    def build(self) -> None:
        self._loading = False
        body = self.scroll_host()
        head = QLabel("Batch stripping")
        head.setProperty("role", "h2")
        body.addWidget(head)
        body.addWidget(status_line(
            "Every loaded profile strips into its own sub-folder; watch the "
            "Progress dock's table.", role="caption"))

        grp, gl = card("Profiles")
        self._list = QListWidget()
        self._list.setMaximumHeight(140)
        gl.addWidget(self._list)
        prow = QHBoxLayout()
        add_btn = QPushButton("Add files…")
        add_btn.clicked.connect(self._on_add)
        prow.addWidget(add_btn)
        prow.addStretch(1)
        gl.addLayout(prow)
        body.addWidget(grp)

        grp, gl = card("Output")
        orow = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Batch output folder…")
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._on_browse_out)
        orow.addWidget(self._out_edit, 1)
        orow.addWidget(out_btn)
        gl.addLayout(orow)
        body.addWidget(grp)

        rrow = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run batch")
        self._run_btn.setProperty("primary", "true")
        self._run_btn.clicked.connect(self._on_run)
        rrow.addStretch(1)
        rrow.addWidget(self._run_btn)
        body.addLayout(rrow)
        body.addStretch(1)

    def refresh(self) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            self._list.clear()
            profiles = self.app_state.profiles()
            for p in profiles:
                self._list.addItem(f"{p['name']}  ·  {p['n_layers']} layers")
            self._run_btn.setEnabled(
                bool(profiles) and not self.app_state.is_busy)
        finally:
            self._loading = False

    def _on_add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add soil profiles", "",
            "Profiles (*.txt *.csv);;All files (*)")
        for path in paths:
            self.app_state.load_profile(path)

    def _on_browse_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Batch output folder")
        if folder:
            self._out_edit.setText(folder)

    def _on_run(self) -> None:
        out = self._out_edit.text().strip() or None
        self.app_state.run_batch_strip(output_dir=out)


# ======================================================================
#  The tool container
# ======================================================================
class StripToolPanel(QWidget):
    """Single (the 3-sub-stage wizard) | Batch."""

    tool = StripTool.STRIP

    def __init__(self, app_state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._app = app_state
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._sub_bar = SubBreadcrumb(
            ["Model", "Configure & Run", "Review", "Batch"], self,
            numbered=False)
        outer.addWidget(self._sub_bar)

        self._stack = QStackedWidget(self)
        self.model_panel = StripModelPanel(app_state)
        self.run_panel = StripRunPanel(app_state, self.model_panel)
        self.review_panel = StripReviewPanel(app_state)
        self.batch_panel = StripBatchPanel(app_state)
        for p in (self.model_panel, self.run_panel, self.review_panel,
                  self.batch_panel):
            self._stack.addWidget(p)
        outer.addWidget(self._stack, 1)

        nav = QWidget(self)
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(10, 6, 10, 8)
        self._back_btn = QPushButton("◀ Back")
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setProperty("primary", "true")
        self._back_btn.clicked.connect(
            lambda: self._set_index(self._sub_bar.active_index - 1))
        self._next_btn.clicked.connect(
            lambda: self._set_index(self._sub_bar.active_index + 1))
        nl.addWidget(self._back_btn)
        nl.addStretch(1)
        nl.addWidget(self._next_btn)
        outer.addWidget(nav)

        self._sub_bar.tab_clicked.connect(self._set_index)
        app_state.profiles_changed.connect(self._update_gating)
        app_state.strip_changed.connect(self._update_gating)
        app_state.config_changed.connect(lambda _s: self._update_gating())
        self._set_index(0)
        self._update_gating()

    # ------------------------------------------------------------------
    def _has_profile(self) -> bool:
        return bool(self.model_panel.profile_name
                    or self._app.profiles())

    def _has_result(self) -> bool:
        return bool(self._app.strip_results())

    def _set_index(self, index: int) -> None:
        index = max(0, min(index, self._stack.count() - 1))
        if index in (1,) and not self._has_profile():
            return
        if index == 2 and not self._has_result():
            return
        self._stack.setCurrentIndex(index)
        self._sub_bar.set_active(index)
        self._update_nav()

    def _update_gating(self) -> None:
        self._sub_bar.set_tab_enabled(1, self._has_profile())
        self._sub_bar.set_tab_enabled(2, self._has_result())
        self._update_nav()

    def _update_nav(self) -> None:
        cur = self._sub_bar.active_index
        self._back_btn.setEnabled(cur > 0)
        wizard_last = cur >= 2         # Review; Batch is a peer, not a step
        self._next_btn.setVisible(not wizard_last)
        if cur == 0:
            self._next_btn.setEnabled(self._has_profile())
        elif cur == 1:
            self._next_btn.setEnabled(self._has_result())

    @property
    def active_index(self) -> int:
        return self._sub_bar.active_index


# ======================================================================
#  The Strip canvas
# ======================================================================
class StripCanvas(QWidget):
    """Waterfall Overlay · Step HV ∥ Vs · Summary · Figure Studio."""

    def __init__(
        self,
        app_state: AppState,
        layer_model: LayerModel,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._app = app_state
        self._layers = layer_model

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs)

        self.overlay = HVCurveCanvas()
        self.tabs.addTab(self.overlay, "Waterfall Overlay")

        step_view = QWidget()
        sv = QVBoxLayout(step_view)
        sv.setContentsMargins(4, 4, 4, 4)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Step:"))
        self.step_combo = QComboBox()
        self.step_combo.currentIndexChanged.connect(self._on_step_selected)
        srow.addWidget(self.step_combo, 1)
        sv.addLayout(srow)
        split = QSplitter(Qt.Horizontal)
        self.step_hv = HVCurveCanvas()
        self.step_vs = VsProfileCanvas()
        split.addWidget(self.step_hv)
        split.addWidget(self.step_vs)
        sv.addWidget(split, 1)
        self.tabs.addTab(step_view, "Step HV ∥ Vs")

        self.summary = QTableWidget(0, 4)
        self.summary.setHorizontalHeaderLabels(
            ["Step", "Layers", "f₀ (Hz)", "A₀"])
        self.summary.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.summary.verticalHeader().setVisible(False)
        self.summary.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.summary, "Summary Table")

        self.figure_studio = _FigureStudio(app_state)
        self.tabs.addTab(self.figure_studio, "Figure Studio")

        app_state.strip_changed.connect(self.refresh)
        layer_model.visibility_changed.connect(self._on_visibility)
        layer_model.display_changed.connect(self._on_display)
        layer_model.layers_rebuilt.connect(self._apply_all_visibility)
        self.refresh()

    # ------------------------------------------------------------------
    def set_palette(self, palette) -> None:
        for canvas in (self.overlay, self.step_hv, self.step_vs):
            canvas.apply_theme(palette)

    def _result(self):
        return next(iter(self._app.strip_results().values()), None)

    def refresh(self) -> None:
        result = self._result()
        steps = getattr(result, "steps", None) or []

        self.overlay.clear_items()
        for step in steps:
            if not len(step.frequencies):
                continue
            key = f"step::{step.step_number}"
            self.overlay.set_item_hv(
                key, step.frequencies, step.amplitudes,
                _pen_from_display(self._layers.display(key)))
        self._apply_all_visibility()

        self.step_combo.blockSignals(True)
        self.step_combo.clear()
        for step in steps:
            self.step_combo.addItem(
                f"Step{step.step_number} · {step.n_layers}-layer",
                step.step_number)
        self.step_combo.blockSignals(False)
        if steps:
            self._on_step_selected(0)

        self.summary.setRowCount(len(steps))
        for r, step in enumerate(steps):
            cells = (
                f"Step{step.step_number}",
                str(step.n_layers),
                f"{step.peak_frequency:.3f}" if step.peak_frequency else "—",
                f"{step.peak_amplitude:.2f}" if step.peak_amplitude else "—",
            )
            for c, text in enumerate(cells):
                self.summary.setItem(r, c, QTableWidgetItem(text))

    def _on_step_selected(self, index: int) -> None:
        result = self._result()
        steps = getattr(result, "steps", None) or []
        if not (0 <= index < len(steps)):
            return
        step = steps[index]
        if len(step.frequencies):
            self.step_hv.update_synthetic(step.frequencies, step.amplitudes)
            if step.peak_frequency:
                self.step_hv.set_peak_freqs([step.peak_frequency])
        layers = self._app.profile_layers_from_file(step.model_path)
        if layers:
            self.step_vs.update_profile(layers)

    # ------------------------------------------------------------------
    def _on_visibility(self, key: str) -> None:
        if key.startswith("step::"):
            self.overlay.set_item_visible(key, self._layers.is_visible(key))

    def _apply_all_visibility(self) -> None:
        for key in self.overlay.item_keys():
            self.overlay.set_item_visible(key, self._layers.is_visible(key))

    def _on_display(self, key: str) -> None:
        if key.startswith("step::") and key in self.overlay.item_keys():
            self.overlay.set_item_pen(
                key, _pen_from_display(self._layers.display(key)))


class _FigureStudio(QWidget):
    """The ONE matplotlib view — publication overlay + export."""

    def __init__(self, app_state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._app = app_state
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        draw_btn = QPushButton("Draw HV overlay")
        draw_btn.setProperty("primary", "true")
        draw_btn.clicked.connect(self._on_draw)
        save_btn = QPushButton("Export…")
        save_btn.clicked.connect(self._on_save)
        row.addWidget(draw_btn)
        row.addWidget(save_btn)
        row.addStretch(1)
        outer.addLayout(row)
        self._figure = Figure(figsize=(10, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        outer.addWidget(self._canvas, 1)
        self._note = status_line("Run a strip first, then draw.", role="muted")
        outer.addWidget(self._note)

    def _on_draw(self) -> None:
        self._figure.clear()
        ok = self._app.report_overlay_on_figure(self._figure)
        self._canvas.draw_idle()
        self._note.setText("" if ok else "No strip result to draw yet.")

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "hv_overlay.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self._figure.savefig(path, dpi=200, bbox_inches="tight")
            self._note.setText(f"Exported {path}")


__all__ = ["StripToolPanel", "StripCanvas"]
