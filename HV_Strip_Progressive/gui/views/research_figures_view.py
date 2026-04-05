"""ResearchFiguresView — gallery of publication figure types with preview."""

import os
import tempfile

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QProgressBar, QCheckBox, QScrollArea,
    QGridLayout, QFrame, QMessageBox,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Available figure types and descriptions
FIGURE_TYPES = {
    "single_comparison": "HV curves from all engines for one profile",
    "f0_scatter": "f₀ scatter plot: engine A vs engine B",
    "f0_boxplot": "f₀ distribution boxplots by engine",
    "agreement_heatmap": "Pairwise agreement heatmap",
    "curve_overlay_grid": "Grid of HV curve comparisons",
    "residual_map": "Curve residuals across frequency",
    "category_comparison": "Performance by geological category",
    "runtime_comparison": "Computation time comparison",
    "peak_correlation_matrix": "Multi-engine peak correlation matrix",
    "field_validation": "Measured vs modeled for field sites",
    "comprehensive_dashboard": "Multi-panel summary dashboard",
}


class ResearchFiguresView(QWidget):
    """Canvas view for generating and previewing comparison figures.

    Shows a list of figure types, a preview area, and batch generation controls.
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._last_output_dir = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>📈 Publication Figures</b>"))
        header.addStretch()

        self._btn_gen_all = QPushButton("Generate All")
        self._btn_gen_all.clicked.connect(self._generate_all)
        header.addWidget(self._btn_gen_all)

        self._btn_save_dir = QPushButton("📁 Set Output Dir...")
        self._btn_save_dir.clicked.connect(self._set_output_dir)
        header.addWidget(self._btn_save_dir)
        lay.addLayout(header)

        # Figure type selector + preview
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Figure Type:"))
        self._type_combo = QComboBox()
        for ft, desc in FIGURE_TYPES.items():
            self._type_combo.addItem(f"{ft} — {desc}", ft)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        selector_row.addWidget(self._type_combo, 1)

        self._btn_preview = QPushButton("👁 Preview")
        self._btn_preview.clicked.connect(self._preview_figure)
        selector_row.addWidget(self._btn_preview)

        self._btn_save = QPushButton("💾 Save")
        self._btn_save.clicked.connect(self._save_figure)
        selector_row.addWidget(self._btn_save)
        lay.addLayout(selector_row)

        # Preview area
        if HAS_MPL:
            self._figure = Figure(figsize=(10, 6), dpi=100)
            self._canvas = FigureCanvasQTAgg(self._figure)
            lay.addWidget(self._canvas, 1)
        else:
            self._canvas = None
            lbl = QLabel("matplotlib not available — install for previews")
            lbl.setAlignment(Qt.AlignCenter)
            lay.addWidget(lbl, 1)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._status = QLabel("Select a figure type and click Preview.")
        self._status.setStyleSheet("color: #888;")
        lay.addWidget(self._status)

    def _on_type_changed(self, idx):
        ft = self._type_combo.itemData(idx)
        desc = FIGURE_TYPES.get(ft, "")
        self._status.setText(f"Selected: {ft} — {desc}")

    def _get_runner(self):
        """Get the ComparisonStudyRunner from the panel."""
        if not self._mw:
            return None
        panel = self._mw.get_panel()
        if panel and hasattr(panel, '_get_runner'):
            return panel._get_runner()
        return None

    def _preview_figure(self):
        """Generate the selected figure type as a preview."""
        ft = self._type_combo.currentData()
        if not ft:
            return

        runner = self._get_runner()
        if not runner or not runner._dataset:
            QMessageBox.information(
                self, "No Data",
                "Run a comparison study first to generate figures.")
            return

        try:
            from ...research.visualization import generate_figure

            # Save to temp file and display
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False)
            tmp.close()

            generate_figure(
                ft, tmp.name,
                dataset=runner._dataset,
                metrics=runner._metrics,
                config=runner._config.visualization,
            )

            # Display in canvas
            if self._canvas:
                self._figure.clear()
                import matplotlib.image as mpimg
                img = mpimg.imread(tmp.name)
                ax = self._figure.add_subplot(111)
                ax.imshow(img)
                ax.axis("off")
                self._figure.tight_layout(pad=0)
                self._canvas.draw()

            self._status.setText(f"Preview: {ft}")
            os.unlink(tmp.name)

        except Exception as e:
            self._status.setText(f"Error: {e}")
            QMessageBox.warning(self, "Figure Error", str(e))

    def _save_figure(self):
        """Save the selected figure to a file."""
        ft = self._type_combo.currentData()
        if not ft:
            return

        runner = self._get_runner()
        if not runner or not runner._dataset:
            QMessageBox.information(self, "No Data",
                                     "Run a comparison first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure",
            os.path.join(self._last_output_dir or "", f"{ft}.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return

        try:
            from ...research.visualization import generate_figure
            generate_figure(
                ft, path,
                dataset=runner._dataset,
                metrics=runner._metrics,
                config=runner._config.visualization,
            )
            self._status.setText(f"Saved: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _generate_all(self):
        """Generate all figure types to the output directory."""
        if not self._last_output_dir:
            self._set_output_dir()
        if not self._last_output_dir:
            return

        runner = self._get_runner()
        if not runner or not runner._dataset:
            QMessageBox.information(self, "No Data",
                                     "Run a comparison first.")
            return

        from ...research.visualization import generate_figure

        fig_dir = os.path.join(self._last_output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        types = list(FIGURE_TYPES.keys())
        self._progress.setVisible(True)
        self._progress.setRange(0, len(types))
        generated = 0

        for i, ft in enumerate(types):
            self._progress.setValue(i)
            try:
                path = os.path.join(fig_dir, f"{ft}.png")
                generate_figure(
                    ft, path,
                    dataset=runner._dataset,
                    metrics=runner._metrics,
                    config=runner._config.visualization,
                )
                generated += 1
            except Exception as e:
                print(f"[Research] {ft} failed: {e}")

        self._progress.setValue(len(types))
        self._progress.setVisible(False)
        self._status.setText(
            f"Generated {generated}/{len(types)} figures in {fig_dir}")

    def _set_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._last_output_dir = d

    # Called by strip_window.update_research_results()
    def on_report_complete(self, result):
        files = result.get("files", {})
        n_figs = sum(1 for k in files if k.startswith("figure_"))
        self._status.setText(
            f"Report generated: {n_figs} figures available")
