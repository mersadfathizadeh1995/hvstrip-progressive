"""The Forward Model tool — panel (Single | Multiple sub-stages) + canvas.

The agreed layout (SKETCH.txt): the left panel is a sub-stage container —
**Single** (one profile: file OR an editable layer table, engine/frequency/
peaks cards, ▶ Run) and **Multiple** (a profiles list + ▶ Run all, streamed
per-profile to the Progress dock).  The canvas is a view-tab set: Single =
``HV Curve ∥ Vs Profile``; Multiple = ``Overlay`` (keyed per-profile curves
driven by the right-rail layers) + ``Summary`` table.  Panels talk ONLY to
AppState.
"""

from __future__ import annotations

from typing import List, Optional

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
from HV_Strip_Progressive.gui.v2.widgets.house.sub_breadcrumb import (
    SubBreadcrumb,
)

_ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]


def _pen_from_display(disp: dict) -> "pg.mkPen":
    color = QColor(disp.get("color") or "#C87A20")
    color.setAlphaF(max(0.0, min(1.0, disp.get("opacity", 1.0))))
    style = {"solid": Qt.SolidLine, "dash": Qt.DashLine,
             "dot": Qt.DotLine}.get(disp.get("line_style", "solid"),
                                    Qt.SolidLine)
    return pg.mkPen(color, width=float(disp.get("line_width", 2)),
                    style=style)


# ======================================================================
#  Shared config cards (engine / frequency / peaks)
# ======================================================================
class _ConfigCards:
    """Builds the shared Engine + Frequency + Peaks cards onto a panel."""

    def build(self, panel: PhasePanel, body) -> None:
        app = panel.app_state

        grp, gl = card("Engine")
        row = QHBoxLayout()
        row.addWidget(QLabel("Engine:"))
        self.engine = QComboBox()
        self.engine.addItems(_ENGINES)
        self.engine.currentTextChanged.connect(
            lambda t: app.set_engine(t) if not self._loading(panel) else None)
        row.addWidget(self.engine, 1)
        gl.addLayout(row)
        self.engine_note = status_line("", role="caption")
        gl.addWidget(self.engine_note)
        body.addWidget(grp)

        grp, gl = card("Frequency")
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(6)
        self.fmin = QDoubleSpinBox()
        self.fmin.setRange(0.001, 100.0)
        self.fmin.setDecimals(3)
        self.fmin.valueChanged.connect(
            lambda v: app.update_config("frequency", fmin=float(v))
            if not self._loading(panel) else None)
        self.fmax = QDoubleSpinBox()
        self.fmax.setRange(0.01, 200.0)
        self.fmax.setDecimals(2)
        self.fmax.valueChanged.connect(
            lambda v: app.update_config("frequency", fmax=float(v))
            if not self._loading(panel) else None)
        self.nf = QSpinBox()
        self.nf.setRange(8, 4096)
        self.nf.valueChanged.connect(
            lambda v: app.update_config("frequency", nf=int(v))
            if not self._loading(panel) else None)
        self.n_samples = QSpinBox()
        self.n_samples.setRange(8, 8192)
        self.n_samples.valueChanged.connect(
            lambda v: app.update_config("frequency", n_samples=int(v))
            if not self._loading(panel) else None)
        form.addWidget(QLabel("fmin (Hz):"), 0, 0)
        form.addWidget(self.fmin, 0, 1)
        form.addWidget(QLabel("fmax (Hz):"), 0, 2)
        form.addWidget(self.fmax, 0, 3)
        form.addWidget(QLabel("nf (HVf):"), 1, 0)
        form.addWidget(self.nf, 1, 1)
        form.addWidget(QLabel("samples:"), 1, 2)
        form.addWidget(self.n_samples, 1, 3)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        gl.addLayout(form)
        body.addWidget(grp)

        grp, gl = card("Peak detection", collapsed=True)
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(6)
        self.preset = QComboBox()
        self.preset.addItems(["default", "forward_modeling", "conservative",
                              "forward_modeling_sharp", "custom"])
        self.preset.currentTextChanged.connect(
            lambda t: app.update_config("peak_detection", preset=t)
            if not self._loading(panel) else None)
        self.select = QComboBox()
        self.select.addItems(["leftmost", "sharpest", "leftmost_sharpest",
                              "max"])
        self.select.currentTextChanged.connect(
            lambda t: app.update_config("peak_detection", select=t)
            if not self._loading(panel) else None)
        form.addWidget(QLabel("Preset:"), 0, 0)
        form.addWidget(self.preset, 0, 1)
        form.addWidget(QLabel("Select:"), 0, 2)
        form.addWidget(self.select, 0, 3)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        gl.addLayout(form)
        body.addWidget(grp)

    @staticmethod
    def _loading(panel) -> bool:
        return getattr(panel, "_loading", False)

    def refresh(self, panel: PhasePanel) -> None:
        cfg = panel.app_state.config
        if cfg is None:
            return
        self.engine.setCurrentText(cfg.engine.name)
        report = panel.app_state.engines_report()
        entry = report.get(cfg.engine.name, {})
        self.engine_note.setText(
            "available" if entry.get("available")
            else f"UNAVAILABLE — {entry.get('reason', '')}")
        self.fmin.setValue(float(cfg.frequency.fmin))
        self.fmax.setValue(float(cfg.frequency.fmax))
        self.nf.setValue(int(cfg.frequency.nf))
        self.n_samples.setValue(int(cfg.frequency.n_samples))
        self.preset.setCurrentText(cfg.peak_detection.preset)
        self.select.setCurrentText(cfg.peak_detection.select)


# ======================================================================
#  Sub-stage 1 — Single
# ======================================================================
class ForwardSinglePanel(PhasePanel):
    """One profile (file or table) → one forward curve."""

    def build(self) -> None:
        self._loading = False
        body = self.scroll_host()
        head = QLabel("Forward · Single")
        head.setProperty("role", "h2")
        body.addWidget(head)
        body.addWidget(status_line(
            "One profile → its forward H/V curve.", role="caption"))

        grp, gl = card("Profile")
        frow = QHBoxLayout()
        self._file_lbl = status_line("No profile loaded.", role="muted")
        browse = QPushButton("Load file…")
        browse.setProperty("primary", "true")
        browse.clicked.connect(self._on_browse)
        frow.addWidget(self._file_lbl, 1)
        frow.addWidget(browse)
        gl.addLayout(frow)
        gl.addWidget(status_line("…or edit layers (vp/ρ auto-derive):",
                                 role="caption"))
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Thickness (m)", "Vs (m/s)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setMaximumHeight(150)
        gl.addWidget(self._table)
        trow = QHBoxLayout()
        add_btn = QPushButton("+ Layer")
        add_btn.clicked.connect(self._on_add_row)
        rem_btn = QPushButton("− Layer")
        rem_btn.clicked.connect(self._on_remove_row)
        use_btn = QPushButton("Use table")
        use_btn.clicked.connect(self._on_use_table)
        trow.addWidget(add_btn)
        trow.addWidget(rem_btn)
        trow.addWidget(use_btn)
        trow.addStretch(1)
        gl.addLayout(trow)
        body.addWidget(grp)

        self.cards = _ConfigCards()
        self.cards.build(self, body)

        rrow = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run forward model")
        self._run_btn.setProperty("primary", "true")
        self._run_btn.clicked.connect(self._on_run)
        rrow.addStretch(1)
        rrow.addWidget(self._run_btn)
        body.addLayout(rrow)
        body.addStretch(1)
        self._profile_name: Optional[str] = None

    def refresh(self) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            self.cards.refresh(self)
            profiles = self.app_state.profiles()
            if self._profile_name is None and profiles:
                self._profile_name = profiles[-1]["name"]
            if self._profile_name:
                info = next((p for p in profiles
                             if p["name"] == self._profile_name), None)
                if info:
                    self._file_lbl.setText(
                        f"{info['name']} · {info['n_layers']} layers · "
                        f"Vs30 {info['vs30']:.0f}" if info.get("vs30")
                        else info["name"])
            self._run_btn.setEnabled(
                bool(self._profile_name) and not self.app_state.is_busy)
        finally:
            self._loading = False

    # -- handlers -----------------------------------------------------
    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load soil profile", "",
            "Profiles (*.txt *.csv);;All files (*)")
        if not path:
            return
        env = self.app_state.load_profile(path)
        name = env.get("name") or env.get("profile_name")
        if name:
            self._profile_name = name

    def _on_add_row(self) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QTableWidgetItem("10.0"))
        self._table.setItem(r, 1, QTableWidgetItem("300.0"))

    def _on_remove_row(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()},
                      reverse=True) or (
            [self._table.rowCount() - 1] if self._table.rowCount() else [])
        for r in rows:
            self._table.removeRow(r)

    def _on_use_table(self) -> None:
        layers: List[dict] = []
        for r in range(self._table.rowCount()):
            try:
                layers.append({
                    "thickness": float(self._table.item(r, 0).text()),
                    "vs": float(self._table.item(r, 1).text()),
                })
            except (AttributeError, TypeError, ValueError):
                continue
        if not layers:
            self.app_state.error.emit(["Add at least one layer row."])
            return
        layers[-1]["thickness"] = 0.0     # the last row is the half-space
        env = self.app_state.add_profile_from_layers(layers, name="table_profile")
        if env.get("name"):
            self._profile_name = env["name"]

    def _on_run(self) -> None:
        self.app_state.run_forward(self._profile_name)


# ======================================================================
#  Sub-stage 2 — Multiple
# ======================================================================
class ForwardMultiPanel(PhasePanel):
    """Many profiles → overlaid curves + a summary table."""

    def build(self) -> None:
        self._loading = False
        body = self.scroll_host()
        head = QLabel("Forward · Multiple")
        head.setProperty("role", "h2")
        body.addWidget(head)
        body.addWidget(status_line(
            "Load several profiles, run them all; each becomes a toggleable "
            "layer on the Overlay view.", role="caption"))

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

        self.cards = _ConfigCards()
        self.cards.build(self, body)

        rrow = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run all")
        self._run_btn.setProperty("primary", "true")
        self._run_btn.clicked.connect(
            lambda: self.app_state.run_forward_all())
        rrow.addStretch(1)
        rrow.addWidget(self._run_btn)
        body.addLayout(rrow)
        body.addStretch(1)

    def refresh(self) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            self.cards.refresh(self)
            self._list.clear()
            profiles = self.app_state.profiles()
            for p in profiles:
                self._list.addItem(
                    f"{p['name']}  ·  {p['n_layers']} layers")
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


# ======================================================================
#  The tool container + canvas
# ======================================================================
class ForwardToolPanel(QWidget):
    """Single | Multiple sub-stage container (the house pattern)."""

    tool = StripTool.FORWARD
    advance_requested = Signal()

    def __init__(self, app_state: AppState, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._sub_bar = SubBreadcrumb(["Single", "Multiple"], self)
        outer.addWidget(self._sub_bar)
        self._stack = QStackedWidget(self)
        self.single_panel = ForwardSinglePanel(app_state)
        self.multi_panel = ForwardMultiPanel(app_state)
        self._stack.addWidget(self.single_panel)
        self._stack.addWidget(self.multi_panel)
        outer.addWidget(self._stack, 1)
        self._sub_bar.tab_clicked.connect(self._set_index)
        self._set_index(0)

    def _set_index(self, index: int) -> None:
        index = max(0, min(index, self._stack.count() - 1))
        self._stack.setCurrentIndex(index)
        self._sub_bar.set_active(index)

    @property
    def active_index(self) -> int:
        return self._sub_bar.active_index


class ForwardCanvas(QWidget):
    """View tabs: HV∥Vs (single) · Overlay (keyed per-profile) · Summary."""

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

        self.hv = HVCurveCanvas()
        self.vs = VsProfileCanvas()
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.hv)
        split.addWidget(self.vs)
        self.tabs.addTab(split, "HV Curve ∥ Vs Profile")

        self.overlay = HVCurveCanvas()
        self.tabs.addTab(self.overlay, "Overlay")

        self.summary = QTableWidget(0, 4)
        self.summary.setHorizontalHeaderLabels(
            ["Profile", "f₀ (Hz)", "A₀", "Engine"])
        self.summary.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.summary.verticalHeader().setVisible(False)
        self.summary.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.summary, "Summary")

        app_state.forward_changed.connect(self.refresh)
        app_state.profiles_changed.connect(self.refresh)
        layer_model.visibility_changed.connect(self._on_visibility)
        layer_model.display_changed.connect(self._on_display)
        layer_model.layers_rebuilt.connect(self._apply_all_visibility)
        self.refresh()

    # ------------------------------------------------------------------
    def set_palette(self, palette) -> None:
        for canvas in (self.hv, self.vs, self.overlay):
            canvas.apply_theme(palette)

    def refresh(self) -> None:
        results = self._app.forward_results()
        # Single view: the most recent result.
        if results:
            last = list(results.values())[-1]
            if len(last.frequencies):
                self.hv.update_synthetic(last.frequencies, last.amplitudes)
                self.hv.set_peak_freqs(
                    [p.frequency for p in (last.peaks or [])])
        # Overlay: one keyed item per profile.
        self.overlay.clear_items()
        for name, res in results.items():
            if not len(res.frequencies):
                continue
            key = f"prof::{name}"
            self.overlay.set_item_hv(
                key, res.frequencies, res.amplitudes,
                _pen_from_display(self._layers.display(key)))
        self._apply_all_visibility()
        # Summary.
        self.summary.setRowCount(len(results))
        for r, (name, res) in enumerate(results.items()):
            peaks = res.peaks or []
            cells = (
                name,
                f"{peaks[0].frequency:.3f}" if peaks else "—",
                f"{peaks[0].amplitude:.2f}" if peaks else "—",
                res.engine_name,
            )
            for c, text in enumerate(cells):
                self.summary.setItem(r, c, QTableWidgetItem(text))

    def show_profile(self, layers: list) -> None:
        if layers:
            self.vs.update_profile(layers)

    # ------------------------------------------------------------------
    def _on_visibility(self, key: str) -> None:
        if key.startswith("prof::"):
            self.overlay.set_item_visible(key, self._layers.is_visible(key))
        elif key == "peaks":
            pass  # peak markers toggle arrives with the strip port

    def _apply_all_visibility(self) -> None:
        for key in self.overlay.item_keys():
            self.overlay.set_item_visible(key, self._layers.is_visible(key))

    def _on_display(self, key: str) -> None:
        if key.startswith("prof::") and key in self.overlay.item_keys():
            self.overlay.set_item_pen(
                key, _pen_from_display(self._layers.display(key)))


__all__ = ["ForwardToolPanel", "ForwardCanvas",
           "ForwardSinglePanel", "ForwardMultiPanel"]
