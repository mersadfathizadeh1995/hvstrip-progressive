"""The Research tool — the comparison-study pipeline as 5 phase sub-stages.

The agreed design (SKETCH.txt + DECISIONS #7): **Profiles → Comparison →
Metrics → Field → Report**, each phase runnable ALONE ([▶ Run phase] with
per-phase config cards); the Report page adds [▶ Run full study] and
[⏹ Cancel] (cooperative — takes effect between phases).  Everything runs on
the dedicated RESEARCH OpQueue so a long study never blocks Forward/Strip.

Canvas views: **Comparison Figures** (the study's generated figures as a
keyed ``fig::`` gallery driven by the right rail) · **Metrics Tables** ·
**Report** (the file manifest).
"""

from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.stages.base import PhasePanel, card, status_line
from HV_Strip_Progressive.gui.v2.state import AppState, StripTool
from HV_Strip_Progressive.gui.v2.state.layer_model import LayerModel
from HV_Strip_Progressive.gui.v2.widgets.house.sub_breadcrumb import (
    SubBreadcrumb,
)

_PHASES = ("profiles", "comparison", "metrics", "field_validation", "report")
_ENGINES = ("diffuse_field", "sh_wave", "ellipticity")


# ======================================================================
#  Phase pages
# ======================================================================
class _ResearchPage(PhasePanel):
    """Shared shape: header + cards + [▶ Run phase]."""

    phase = ""
    title = ""
    blurb = ""

    def __init__(self, app_state, owner: "ResearchToolPanel", parent=None):
        self._owner = owner
        super().__init__(app_state, parent)

    def build(self) -> None:
        self._loading = False
        body = self.scroll_host()
        head = QLabel(self.title)
        head.setProperty("role", "h2")
        body.addWidget(head)
        body.addWidget(status_line(self.blurb, role="caption"))
        self.build_cards(body)
        rrow = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run phase")
        self._run_btn.setProperty("primary", "true")
        self._run_btn.clicked.connect(self._on_run)
        rrow.addStretch(1)
        rrow.addWidget(self._run_btn)
        body.addLayout(rrow)
        self.build_footer(body)
        self._status = status_line("", role="muted")
        body.addWidget(self._status)
        body.addStretch(1)

    def build_cards(self, body) -> None:  # pragma: no cover - subclass
        pass

    def build_footer(self, body) -> None:
        pass

    def refresh(self) -> None:
        result = self.app_state.research_phase_result(self.phase)
        self._status.setText(self.summarize(result) if result else "")

    def summarize(self, result: Dict[str, Any]) -> str:
        return "Phase completed."

    def _on_run(self) -> None:
        self.app_state.run_research_phase(
            self.phase, study_config=self._owner.study_payload())


class ProfilesPage(_ResearchPage):
    phase = "profiles"
    title = "1 · Profiles"
    blurb = ("Generate the synthetic profile suite (SoilGen scenarios + "
             "random profiles), or load an existing folder of models.")

    def build_cards(self, body) -> None:
        grp, gl = card("Source")
        srow = QHBoxLayout()
        srow.addWidget(QLabel("SoilGen path:"))
        self.soilgen_edit = QLineEdit()
        self.soilgen_edit.setPlaceholderText(
            "Path to the SoilGen package (for generation)…")
        sg_btn = QPushButton("Browse…")
        sg_btn.clicked.connect(self._on_browse_soilgen)
        srow.addWidget(self.soilgen_edit, 1)
        srow.addWidget(sg_btn)
        gl.addLayout(srow)
        drow = QHBoxLayout()
        drow.addWidget(QLabel("…or load suite:"))
        self.suite_edit = QLineEdit()
        self.suite_edit.setPlaceholderText(
            "Existing profiles folder (.txt models) — skips generation")
        su_btn = QPushButton("Browse…")
        su_btn.clicked.connect(self._on_browse_suite)
        drow.addWidget(self.suite_edit, 1)
        drow.addWidget(su_btn)
        gl.addLayout(drow)
        body.addWidget(grp)

        grp, gl = card("Suite")
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(6)
        self.n_random = QSpinBox()
        self.n_random.setRange(0, 500)
        self.n_random.setValue(20)
        self.n_per_scenario = QSpinBox()
        self.n_per_scenario.setRange(0, 200)
        self.n_per_scenario.setValue(15)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999_999)
        self.seed.setValue(42)
        form.addWidget(QLabel("Random profiles:"), 0, 0)
        form.addWidget(self.n_random, 0, 1)
        form.addWidget(QLabel("Per scenario:"), 0, 2)
        form.addWidget(self.n_per_scenario, 0, 3)
        form.addWidget(QLabel("Seed:"), 1, 0)
        form.addWidget(self.seed, 1, 1)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        gl.addLayout(form)
        body.addWidget(grp)

    def _on_browse_soilgen(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "SoilGen package directory")
        if folder:
            self.soilgen_edit.setText(folder)

    def _on_browse_suite(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Existing profiles folder")
        if folder:
            self.suite_edit.setText(folder)

    def summarize(self, result: Dict[str, Any]) -> str:
        n = result.get("n_profiles")
        verb = ("loaded" if result.get("phase") == "profile_loading"
                else "generated")
        return f"{n} profiles {verb}." if n else "Phase completed."


class ComparisonPage(_ResearchPage):
    phase = "comparison"
    title = "2 · Comparison"
    blurb = "Run every enabled engine on every generated profile."

    def build_cards(self, body) -> None:
        grp, gl = card("Engines")
        self.engine_checks: Dict[str, QCheckBox] = {}
        for eng in _ENGINES:
            chk = QCheckBox(eng)
            chk.setChecked(True)
            self.engine_checks[eng] = chk
            gl.addWidget(chk)
        body.addWidget(grp)

        grp, gl = card("Frequency")
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(6)
        self.fmin = QLineEdit("0.1")
        self.fmax = QLineEdit("30.0")
        self.nf = QSpinBox()
        self.nf.setRange(16, 4096)
        self.nf.setValue(500)
        form.addWidget(QLabel("fmin (Hz):"), 0, 0)
        form.addWidget(self.fmin, 0, 1)
        form.addWidget(QLabel("fmax (Hz):"), 0, 2)
        form.addWidget(self.fmax, 0, 3)
        form.addWidget(QLabel("frequencies:"), 1, 0)
        form.addWidget(self.nf, 1, 1)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        gl.addLayout(form)
        body.addWidget(grp)

    def refresh(self) -> None:
        super().refresh()
        report = self.app_state.engines_report()
        for eng, chk in self.engine_checks.items():
            available = bool(report.get(eng, {}).get("available"))
            chk.setEnabled(available)
            if not available:
                chk.setChecked(False)
                chk.setText(f"{eng}  (unavailable)")

    def summarize(self, result: Dict[str, Any]) -> str:
        total = result.get("total_runs")
        good = result.get("successful_runs")
        if total is None:
            return "Phase completed."
        return f"{good}/{total} runs succeeded " \
               f"({result.get('elapsed_seconds', 0):.0f} s)."


class MetricsPage(_ResearchPage):
    phase = "metrics"
    title = "3 · Metrics"
    blurb = ("Peak/curve agreement + per-engine statistics over the "
             "comparison dataset (see the Metrics Tables view).")

    def summarize(self, result: Dict[str, Any]) -> str:
        n = result.get("n_peak_agreements")
        if n is None:
            return "Phase completed."
        return (f"{n} peak agreements · "
                f"{result.get('n_curve_agreements', 0)} curve agreements · "
                f"{result.get('n_categories', 0)} categories.")


class FieldPage(_ResearchPage):
    phase = "field_validation"
    title = "4 · Field validation"
    blurb = ("Validate engines against configured field sites (skipped "
             "cleanly when no sites are configured).")

    def summarize(self, result: Dict[str, Any]) -> str:
        n = result.get("n_sites", 0)
        return result.get("message") or f"{n} site(s) validated."


class ReportPage(_ResearchPage):
    phase = "report"
    title = "5 · Report"
    blurb = ("Generate the study report (figures + tables + LaTeX); also "
             "the place to run the FULL study end-to-end.")

    def build_cards(self, body) -> None:
        grp, gl = card("Output")
        orow = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Study output folder…")
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._on_browse_out)
        orow.addWidget(self.out_edit, 1)
        orow.addWidget(out_btn)
        gl.addLayout(orow)
        body.addWidget(grp)

    def build_footer(self, body) -> None:
        grp, gl = card("Full study")
        frow = QHBoxLayout()
        self.full_btn = QPushButton("▶  Run full study")
        self.full_btn.setProperty("primary", "true")
        self.full_btn.clicked.connect(self._on_full)
        self.cancel_btn = QPushButton("⏹  Cancel")
        self.cancel_btn.clicked.connect(self.app_state.cancel_research)
        frow.addWidget(self.full_btn)
        frow.addWidget(self.cancel_btn)
        frow.addStretch(1)
        gl.addLayout(frow)
        gl.addWidget(status_line(
            "Cancel is cooperative — it takes effect between phases.",
            role="caption"))
        body.addWidget(grp)

    def _on_browse_out(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Study output folder")
        if folder:
            self.out_edit.setText(folder)

    def _on_full(self) -> None:
        self.app_state.run_full_study(self._owner.study_payload())

    def summarize(self, result: Dict[str, Any]) -> str:
        n = result.get("n_files")
        return f"{n} report file(s) generated." if n else "Phase completed."


# ======================================================================
#  The tool container
# ======================================================================
class ResearchToolPanel(QWidget):
    """5 phase sub-stages, freely navigable (each runnable alone)."""

    tool = StripTool.RESEARCH

    def __init__(self, app_state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._app = app_state
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._sub_bar = SubBreadcrumb(
            ["Profiles", "Comparison", "Metrics", "Field", "Report"], self,
            numbered=False)
        outer.addWidget(self._sub_bar)

        self._stack = QStackedWidget(self)
        self.profiles_page = ProfilesPage(app_state, self)
        self.comparison_page = ComparisonPage(app_state, self)
        self.metrics_page = MetricsPage(app_state, self)
        self.field_page = FieldPage(app_state, self)
        self.report_page = ReportPage(app_state, self)
        self.pages = (self.profiles_page, self.comparison_page,
                      self.metrics_page, self.field_page, self.report_page)
        for p in self.pages:
            self._stack.addWidget(p)
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

    # ------------------------------------------------------------------
    def study_payload(self) -> Dict[str, Any]:
        """The ComparisonStudyConfig overrides from the phase pages."""
        engines = [e for e, c in self.comparison_page.engine_checks.items()
                   if c.isChecked()]
        profiles: Dict[str, Any] = {
            "n_random": int(self.profiles_page.n_random.value()),
            "n_per_scenario": int(
                self.profiles_page.n_per_scenario.value()),
            "seed": int(self.profiles_page.seed.value()),
        }
        soilgen = self.profiles_page.soilgen_edit.text().strip()
        if soilgen:
            profiles["soilgen_path"] = soilgen
        payload: Dict[str, Any] = {
            "profiles": profiles,
            "engines": {
                "engines": engines,
                "fmin": _to_float(self.comparison_page.fmin.text(), 0.1),
                "fmax": _to_float(self.comparison_page.fmax.text(), 30.0),
                "n_frequencies": int(self.comparison_page.nf.value()),
            },
        }
        suite = self.profiles_page.suite_edit.text().strip()
        if suite:
            payload["profiles_dir"] = suite   # load, don't generate
        out = self.report_page.out_edit.text().strip()
        if out:
            payload["output"] = {"output_dir": out}
        return payload


def _to_float(text: str, default: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


# ======================================================================
#  The Research canvas
# ======================================================================
class ResearchCanvas(QWidget):
    """Comparison Figures (fig:: gallery) · Metrics Tables · Report."""

    def __init__(
        self,
        app_state: AppState,
        layer_model: LayerModel,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._app = app_state
        self._layers = layer_model
        self._figure_labels: Dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs)

        # Figures gallery (scrollable column of study figures).
        self._gallery_scroll = QScrollArea()
        self._gallery_scroll.setWidgetResizable(True)
        gal_host = QWidget()
        self._gallery = QVBoxLayout(gal_host)
        self._gallery.setContentsMargins(8, 8, 8, 8)
        self._gallery.setSpacing(10)
        self._gallery.addStretch(1)
        self._gallery_scroll.setWidget(gal_host)
        self.tabs.addTab(self._gallery_scroll, "Comparison Figures")

        # Metrics tables.
        metrics_view = QWidget()
        ml = QVBoxLayout(metrics_view)
        ml.setContentsMargins(4, 4, 4, 4)
        self._metrics_note = status_line(
            "Run the Metrics phase to populate.", role="muted")
        ml.addWidget(self._metrics_note)
        self.engine_stats = QTableWidget(0, 0)
        self.engine_stats.verticalHeader().setVisible(False)
        self.engine_stats.setEditTriggers(QTableWidget.NoEditTriggers)
        ml.addWidget(self.engine_stats, 1)
        self.tabs.addTab(metrics_view, "Metrics Tables")

        # Report manifest.
        report_view = QWidget()
        rl = QVBoxLayout(report_view)
        rl.setContentsMargins(4, 4, 4, 4)
        self.report_files = QTableWidget(0, 2)
        self.report_files.setHorizontalHeaderLabels(["File", "Path"])
        self.report_files.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.report_files.verticalHeader().setVisible(False)
        self.report_files.setEditTriggers(QTableWidget.NoEditTriggers)
        rl.addWidget(self.report_files, 1)
        self.tabs.addTab(report_view, "Report")

        app_state.research_changed.connect(self.refresh)
        layer_model.visibility_changed.connect(self._on_visibility)
        layer_model.layers_rebuilt.connect(self._apply_all_visibility)
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        # Figures gallery.
        while self._gallery.count() > 1:      # keep the trailing stretch
            item = self._gallery.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._figure_labels.clear()
        for name, path in self._app.research_figures():
            box = QWidget()
            bl = QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(2)
            title = status_line(name, role="caption")
            img = QLabel()
            pix = QPixmap(path)
            if not pix.isNull():
                img.setPixmap(pix.scaledToWidth(
                    720, Qt.SmoothTransformation))
            else:
                img.setText(path)
            bl.addWidget(title)
            bl.addWidget(img)
            self._gallery.insertWidget(self._gallery.count() - 1, box)
            self._figure_labels[f"fig::{name}"] = box
        self._apply_all_visibility()

        # Metrics tables (columns derived from the api's engine_stats dicts).
        stats = self._app.research_phase_result("metrics") \
            .get("engine_stats") or []
        if stats:
            cols = list(stats[0].keys())
            self.engine_stats.setColumnCount(len(cols))
            self.engine_stats.setHorizontalHeaderLabels(cols)
            self.engine_stats.horizontalHeader().setSectionResizeMode(
                QHeaderView.Stretch)
            self.engine_stats.setRowCount(len(stats))
            for r, entry in enumerate(stats):
                for c, col in enumerate(cols):
                    val = entry.get(col)
                    text = f"{val:.4g}" if isinstance(val, float) else str(val)
                    self.engine_stats.setItem(r, c, QTableWidgetItem(text))
            self._metrics_note.setText("")
        else:
            self.engine_stats.setRowCount(0)
            self._metrics_note.setText("Run the Metrics phase to populate.")

        # Report manifest.
        files = self._app.research_phase_result("report").get("files", {})
        files = files if isinstance(files, dict) else {}
        self.report_files.setRowCount(len(files))
        for r, (name, path) in enumerate(sorted(files.items())):
            self.report_files.setItem(r, 0, QTableWidgetItem(str(name)))
            self.report_files.setItem(r, 1, QTableWidgetItem(str(path)))

    # ------------------------------------------------------------------
    def _on_visibility(self, key: str) -> None:
        if key.startswith("fig::") and key in self._figure_labels:
            self._figure_labels[key].setVisible(self._layers.is_visible(key))

    def _apply_all_visibility(self) -> None:
        for key, w in self._figure_labels.items():
            w.setVisible(self._layers.is_visible(key))


__all__ = ["ResearchToolPanel", "ResearchCanvas"]
