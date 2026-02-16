"""
HV Forward Modeling page with integrated Profile Editor.
"""

import tempfile
from pathlib import Path

import numpy as np
import matplotlib.gridspec as gridspec
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QLineEdit, QTextEdit, QSplitter,
    QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget, QListWidgetItem,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal, QThread

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.layer_table_widget import LayerTableWidget
from ..widgets.profile_preview_widget import ProfilePreviewWidget
from ..dialogs.multi_profile_dialog import (
    MultiProfilePickerDialog, FigureSettings, ProfileResult,
    get_palette_colors, _PALETTE_NAMES, _MARKER_COLORS, _MARKER_SHAPES,
    _darken_color,
)
from ..dialogs.output_viewer_dialog import OutputViewerDialog, load_output_folder
from ...core.soil_profile import SoilProfile, Layer
from ...core.velocity_utils import VelocityConverter
from ...core.hv_forward import compute_hv_curve


class ForwardWorker(QThread):
    """Worker thread for HV forward modeling."""
    finished_signal = Signal(tuple)
    error = Signal(str)

    def __init__(self, model_path, config):
        super().__init__()
        self.model_path = model_path
        self.config = config

    def run(self):
        try:
            freqs, amps = compute_hv_curve(self.model_path, self.config)
            self.finished_signal.emit((freqs, amps))
        except Exception as e:
            self.error.emit(str(e))


class ForwardModelingPage(QWidget):
    """
    HV Forward Modeling with integrated Profile Editor.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ForwardModelingPage")
        self._current_profile = None
        self._active_profile = None  # Profile used for current computation
        self._temp_model_path = None
        self._last_freqs = None
        self._last_amps = None
        self._selected_peak = None  # (freq, amp, idx) primary f0 peak
        self._secondary_peaks = []  # list of (freq, amp, idx) secondary peaks
        self._pick_mode = True
        self._pick_secondary = False
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("HV Forward Modeling")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(header)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        input_tabs = QTabWidget()
        input_tabs.addTab(self._create_file_tab(), "From File")
        input_tabs.addTab(self._create_dinver_tab(), "From Dinver")
        input_tabs.addTab(self._create_editor_tab(), "Profile Editor")
        input_tabs.addTab(self._create_multi_tab(), "Multiple Profiles")
        left_layout.addWidget(input_tabs)
        self.input_tabs = input_tabs

        config_group = QGroupBox("Frequency Parameters")
        config_layout = QHBoxLayout(config_group)

        config_layout.addWidget(QLabel("Min:"))
        self.fmin_spin = QDoubleSpinBox()
        self.fmin_spin.setRange(0.01, 10.0)
        self.fmin_spin.setValue(0.5)
        self.fmin_spin.setSingleStep(0.1)
        self.fmin_spin.setSuffix(" Hz")
        config_layout.addWidget(self.fmin_spin)

        config_layout.addWidget(QLabel("Max:"))
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(1.0, 100.0)
        self.fmax_spin.setValue(20.0)
        self.fmax_spin.setSingleStep(1.0)
        self.fmax_spin.setSuffix(" Hz")
        config_layout.addWidget(self.fmax_spin)

        config_layout.addWidget(QLabel("Points:"))
        self.nf_spin = QSpinBox()
        self.nf_spin.setRange(50, 2000)
        self.nf_spin.setValue(500)
        config_layout.addWidget(self.nf_spin)

        config_layout.addStretch()
        left_layout.addWidget(config_group)

        # Output directory
        outdir_group = QGroupBox("Output Directory")
        outdir_layout = QHBoxLayout(outdir_group)
        self.outdir_edit = QLineEdit()
        self.outdir_edit.setPlaceholderText("Select output directory for figures and data...")
        outdir_layout.addWidget(self.outdir_edit)
        btn_outdir = QPushButton("Browse...")
        btn_outdir.clicked.connect(self._browse_output_dir)
        outdir_layout.addWidget(btn_outdir)
        left_layout.addWidget(outdir_group)

        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Compute HV Curve")
        self.btn_run.setStyleSheet(
            "background-color: #0078d4; color: white; padding: 8px 16px; font-size: 13px;"
        )
        self.btn_run.clicked.connect(self._run_forward)
        btn_layout.addWidget(self.btn_run)

        self.btn_save = QPushButton("Save Results...")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_results)
        btn_layout.addWidget(self.btn_save)

        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(120)
        results_layout.addWidget(self.results_text)
        left_layout.addWidget(results_group)

        main_splitter.addWidget(left_panel)

        # Right panel: plot area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.hv_plot = MatplotlibWidget(figsize=(14, 5))
        self.hv_plot.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        right_layout.addWidget(self.hv_plot)

        plot_options = QHBoxLayout()
        self.log_x_check = QCheckBox("Log X")
        self.log_x_check.setChecked(True)
        self.log_x_check.stateChanged.connect(self._update_plot)
        plot_options.addWidget(self.log_x_check)

        self.log_y_check = QCheckBox("Log Y")
        self.log_y_check.stateChanged.connect(self._update_plot)
        plot_options.addWidget(self.log_y_check)

        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self._update_plot)
        plot_options.addWidget(self.grid_check)

        self.show_vs_check = QCheckBox("Show Vs Profile")
        self.show_vs_check.setChecked(True)
        self.show_vs_check.stateChanged.connect(self._update_plot)
        plot_options.addWidget(self.show_vs_check)

        self.btn_pick = QPushButton("Select f0")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setChecked(True)
        self.btn_pick.setStyleSheet(
            "QPushButton:checked { background-color: #107c10; color: white; font-weight: bold; padding: 6px 12px; }"
            "QPushButton { padding: 6px 12px; background-color: #666; color: white; }"
        )
        self.btn_pick.clicked.connect(self._toggle_pick_f0)
        plot_options.addWidget(self.btn_pick)

        self.btn_pick_secondary = QPushButton("Select Secondary Peak")
        self.btn_pick_secondary.setCheckable(True)
        self.btn_pick_secondary.setChecked(False)
        self.btn_pick_secondary.setStyleSheet(
            "QPushButton:checked { background-color: #d4700a; color: white; font-weight: bold; padding: 6px 12px; }"
            "QPushButton { padding: 6px 12px; background-color: #666; color: white; }"
        )
        self.btn_pick_secondary.clicked.connect(self._toggle_pick_secondary)
        plot_options.addWidget(self.btn_pick_secondary)

        self.btn_clear_secondary = QPushButton("Clear Secondary")
        self.btn_clear_secondary.setStyleSheet("padding: 6px 12px;")
        self.btn_clear_secondary.clicked.connect(self._clear_secondary_peaks)
        plot_options.addWidget(self.btn_clear_secondary)

        # Selection label
        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("font-size: 11px; padding: 4px;")
        plot_options.addWidget(self.selection_label)

        plot_options.addStretch()
        right_layout.addLayout(plot_options)

        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([450, 550])

        layout.addWidget(main_splitter)
        self._draw_empty_plot()

    def _create_file_tab(self):
        """Create the From File tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Model File:"))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Select HVf format model file (.txt)")
        file_row.addWidget(self.model_edit)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_model)
        file_row.addWidget(btn_browse)
        layout.addLayout(file_row)

        self.file_preview = ProfilePreviewWidget()
        layout.addWidget(self.file_preview)

        return widget

    def _create_dinver_tab(self):
        """Create the From Dinver tab for importing Dinver output files."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_label = QLabel(
            "Import velocity model from Dinver output files.\n"
            "Requires Vs file; Vp and density files are optional."
        )
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)

        vs_row = QHBoxLayout()
        vs_row.addWidget(QLabel("Vs File:"))
        self.dinver_vs_edit = QLineEdit()
        self.dinver_vs_edit.setPlaceholderText("Required: *_vs.txt")
        vs_row.addWidget(self.dinver_vs_edit)
        btn_vs = QPushButton("...")
        btn_vs.setMaximumWidth(30)
        btn_vs.clicked.connect(lambda: self._browse_dinver_file("vs"))
        vs_row.addWidget(btn_vs)
        layout.addLayout(vs_row)

        vp_row = QHBoxLayout()
        vp_row.addWidget(QLabel("Vp File:"))
        self.dinver_vp_edit = QLineEdit()
        self.dinver_vp_edit.setPlaceholderText("Optional: *_vp.txt")
        vp_row.addWidget(self.dinver_vp_edit)
        btn_vp = QPushButton("...")
        btn_vp.setMaximumWidth(30)
        btn_vp.clicked.connect(lambda: self._browse_dinver_file("vp"))
        vp_row.addWidget(btn_vp)
        layout.addLayout(vp_row)

        rho_row = QHBoxLayout()
        rho_row.addWidget(QLabel("Density File:"))
        self.dinver_rho_edit = QLineEdit()
        self.dinver_rho_edit.setPlaceholderText("Optional: *_rho.txt")
        rho_row.addWidget(self.dinver_rho_edit)
        btn_rho = QPushButton("...")
        btn_rho.setMaximumWidth(30)
        btn_rho.clicked.connect(lambda: self._browse_dinver_file("rho"))
        rho_row.addWidget(btn_rho)
        layout.addLayout(rho_row)

        btn_load = QPushButton("Load Dinver Profile")
        btn_load.clicked.connect(self._load_dinver_profile)
        layout.addWidget(btn_load)

        self.dinver_preview = ProfilePreviewWidget()
        layout.addWidget(self.dinver_preview)

        self.dinver_status = QLabel("")
        layout.addWidget(self.dinver_status)

        return widget

    def _browse_dinver_file(self, file_type: str):
        """Browse for Dinver file."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {file_type.upper()} File", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            if file_type == "vs":
                self.dinver_vs_edit.setText(path)
                base = path.replace("_vs.txt", "").replace("_Vs.txt", "")
                for suffix, edit in [("_vp.txt", self.dinver_vp_edit), ("_rho.txt", self.dinver_rho_edit)]:
                    candidate = base + suffix
                    if Path(candidate).exists() and not edit.text():
                        edit.setText(candidate)
            elif file_type == "vp":
                self.dinver_vp_edit.setText(path)
            elif file_type == "rho":
                self.dinver_rho_edit.setText(path)

    def _load_dinver_profile(self):
        """Load profile from Dinver files."""
        vs_file = self.dinver_vs_edit.text().strip()
        if not vs_file:
            QMessageBox.warning(self, "Error", "Please select a Vs file.")
            return

        vp_file = self.dinver_vp_edit.text().strip() or None
        rho_file = self.dinver_rho_edit.text().strip() or None

        try:
            profile = SoilProfile.from_dinver_files(vs_file, vp_file, rho_file)
            self._dinver_profile = profile
            self.dinver_preview.set_profile(profile)
            self.dinver_status.setText(
                f"Loaded: {len(profile.layers)} layers, "
                f"depth: {profile.get_total_thickness():.1f} m"
            )
            self.dinver_status.setStyleSheet("color: green;")
        except Exception as e:
            self._dinver_profile = None
            self.dinver_status.setText(f"Error: {e}")
            self.dinver_status.setStyleSheet("color: red;")

    def _create_editor_tab(self):
        """Create the Profile Editor tab with full functionality."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        editor_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        file_ops = QHBoxLayout()
        btn_new = QPushButton("New")
        btn_new.clicked.connect(self._new_profile)
        file_ops.addWidget(btn_new)

        btn_open = QPushButton("Open...")
        btn_open.clicked.connect(self._open_profile)
        file_ops.addWidget(btn_open)

        btn_save_profile = QPushButton("Save...")
        btn_save_profile.clicked.connect(self._save_profile)
        file_ops.addWidget(btn_save_profile)

        file_ops.addStretch()
        left_layout.addLayout(file_ops)

        self.layer_table = LayerTableWidget()
        self.layer_table.profile_changed.connect(self._on_profile_changed)
        left_layout.addWidget(self.layer_table)

        ref_group = QGroupBox("Poisson's Ratio Reference")
        ref_layout = QVBoxLayout(ref_group)
        self.ref_table = QTableWidget()
        self.ref_table.setColumnCount(4)
        self.ref_table.setHorizontalHeaderLabels(["Material", "Min", "Max", "Typical"])
        self.ref_table.horizontalHeader().setStretchLastSection(True)
        self.ref_table.setMaximumHeight(150)
        self._populate_reference_table()
        ref_layout.addWidget(self.ref_table)
        left_layout.addWidget(ref_group)

        editor_splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.profile_preview = ProfilePreviewWidget()
        right_layout.addWidget(self.profile_preview)

        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_group)
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(80)
        validation_layout.addWidget(self.validation_text)
        right_layout.addWidget(validation_group)

        editor_splitter.addWidget(right_widget)
        editor_splitter.setSizes([400, 250])

        layout.addWidget(editor_splitter)

        self._create_default_profile()

        return widget

    def _create_multi_tab(self):
        """Create the Multiple Profiles tab for batch Excel processing."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel(
            "Load multiple Excel (.xlsx) profile files and process them sequentially.\n"
            "Each file: Thickness | Vs | Vp | Density (last row = halfspace)."
        )
        info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info)

        # File list
        self.multi_file_list = QListWidget()
        self.multi_file_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.multi_file_list)

        file_btns = QHBoxLayout()
        btn_add = QPushButton("Add Files…")
        btn_add.clicked.connect(self._multi_add_files)
        file_btns.addWidget(btn_add)

        btn_add_folder = QPushButton("Add Folder…")
        btn_add_folder.clicked.connect(self._multi_add_folder)
        file_btns.addWidget(btn_add_folder)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._multi_remove_selected)
        file_btns.addWidget(btn_remove)

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._multi_clear_all)
        file_btns.addWidget(btn_clear)

        file_btns.addStretch()
        layout.addLayout(file_btns)

        # Figure settings
        fig_group = QGroupBox("Figure Settings")
        fig_lay = QVBoxLayout(fig_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("DPI:"))
        self.multi_dpi_spin = QSpinBox()
        self.multi_dpi_spin.setRange(72, 600)
        self.multi_dpi_spin.setValue(300)
        row1.addWidget(self.multi_dpi_spin)

        row1.addWidget(QLabel("Width:"))
        self.multi_fig_w = QDoubleSpinBox()
        self.multi_fig_w.setRange(4.0, 24.0)
        self.multi_fig_w.setValue(10.0)
        self.multi_fig_w.setSuffix(" in")
        row1.addWidget(self.multi_fig_w)

        row1.addWidget(QLabel("Height:"))
        self.multi_fig_h = QDoubleSpinBox()
        self.multi_fig_h.setRange(3.0, 16.0)
        self.multi_fig_h.setValue(5.0)
        self.multi_fig_h.setSuffix(" in")
        row1.addWidget(self.multi_fig_h)

        row1.addWidget(QLabel("Font:"))
        self.multi_font_spin = QSpinBox()
        self.multi_font_spin.setRange(6, 24)
        self.multi_font_spin.setValue(12)
        row1.addWidget(self.multi_font_spin)

        row1.addStretch()
        fig_lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.multi_logx = QCheckBox("Log X")
        self.multi_logx.setChecked(True)
        row2.addWidget(self.multi_logx)

        self.multi_logy = QCheckBox("Log Y")
        row2.addWidget(self.multi_logy)

        self.multi_grid = QCheckBox("Grid")
        self.multi_grid.setChecked(True)
        row2.addWidget(self.multi_grid)

        self.multi_png = QCheckBox("PNG")
        self.multi_png.setChecked(True)
        row2.addWidget(self.multi_png)

        self.multi_pdf = QCheckBox("PDF")
        self.multi_pdf.setChecked(True)
        row2.addWidget(self.multi_pdf)

        self.multi_show_vs = QCheckBox("Show Vs Profile")
        row2.addWidget(self.multi_show_vs)

        row2.addStretch()
        fig_lay.addLayout(row2)

        # Combined plot settings
        comb_group = QGroupBox("Combined Plot Settings")
        comb_lay = QVBoxLayout(comb_group)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Color Palette:"))
        self.multi_palette = QComboBox()
        self.multi_palette.addItems(_PALETTE_NAMES)
        row3.addWidget(self.multi_palette)

        row3.addWidget(QLabel("Line α:"))
        self.multi_alpha = QDoubleSpinBox()
        self.multi_alpha.setRange(0.1, 1.0)
        self.multi_alpha.setValue(0.45)
        self.multi_alpha.setSingleStep(0.05)
        row3.addWidget(self.multi_alpha)

        row3.addWidget(QLabel("Line W:"))
        self.multi_line_w = QDoubleSpinBox()
        self.multi_line_w.setRange(0.3, 5.0)
        self.multi_line_w.setValue(1.0)
        self.multi_line_w.setSingleStep(0.1)
        row3.addWidget(self.multi_line_w)

        row3.addWidget(QLabel("Median W:"))
        self.multi_median_w = QDoubleSpinBox()
        self.multi_median_w.setRange(1.0, 8.0)
        self.multi_median_w.setValue(3.0)
        self.multi_median_w.setSingleStep(0.5)
        row3.addWidget(self.multi_median_w)

        row3.addStretch()
        comb_lay.addLayout(row3)

        row4 = QHBoxLayout()
        self.multi_show_median = QCheckBox("Show Median Curve")
        self.multi_show_median.setChecked(True)
        row4.addWidget(self.multi_show_median)

        self.multi_show_sec = QCheckBox("Show Secondary Peaks")
        self.multi_show_sec.setChecked(True)
        row4.addWidget(self.multi_show_sec)

        row4.addStretch()
        comb_lay.addLayout(row4)

        fig_lay.addWidget(comb_group)

        layout.addWidget(fig_group)

        # Action buttons row
        btn_row = QHBoxLayout()

        self.btn_run_multi = QPushButton("Run All Profiles")
        self.btn_run_multi.setStyleSheet(
            "background-color: #0078d4; color: white; padding: 8px 16px; font-size: 13px;"
        )
        self.btn_run_multi.clicked.connect(self._run_multi_profiles)
        btn_row.addWidget(self.btn_run_multi)

        self.btn_load_output = QPushButton("Load Results Folder")
        self.btn_load_output.setStyleSheet(
            "background-color: #107c10; color: white; padding: 8px 16px; font-size: 13px;"
        )
        self.btn_load_output.setToolTip(
            "Open a previously saved output folder to view combined curves"
        )
        self.btn_load_output.clicked.connect(self._load_multi_output)
        btn_row.addWidget(self.btn_load_output)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.multi_status = QLabel("")
        layout.addWidget(self.multi_status)

        return widget

    def _load_multi_output(self):
        """Open a saved output folder and display combined curves in a viewer."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder with Profile Subfolders"
        )
        if not folder:
            return

        folder_path = Path(folder)
        results, median_result = load_output_folder(folder_path)

        if not results:
            QMessageBox.warning(
                self, "No Data Found",
                f"No profile subfolders with hv_curve.csv found in:\n{folder}"
            )
            return

        fig_settings = self._get_fig_settings()
        self.multi_status.setText(
            f"Loaded {len(results)} profiles from {folder_path.name}"
        )
        self.multi_status.setStyleSheet("color: green; font-weight: bold;")

        dialog = OutputViewerDialog(
            results, median_result, fig_settings,
            source_folder=folder, parent=self,
        )
        dialog.exec()

    def _multi_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Excel Profile Files", "",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        for p in paths:
            if not any(
                self.multi_file_list.item(i).data(Qt.UserRole) == p
                for i in range(self.multi_file_list.count())
            ):
                item = QListWidgetItem(Path(p).name)
                item.setData(Qt.UserRole, p)
                self.multi_file_list.addItem(item)

    def _multi_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with .xlsx Files")
        if folder:
            for p in sorted(Path(folder).glob("*.xlsx")):
                full = str(p)
                if not any(
                    self.multi_file_list.item(i).data(Qt.UserRole) == full
                    for i in range(self.multi_file_list.count())
                ):
                    item = QListWidgetItem(p.name)
                    item.setData(Qt.UserRole, full)
                    self.multi_file_list.addItem(item)

    def _multi_remove_selected(self):
        for item in self.multi_file_list.selectedItems():
            self.multi_file_list.takeItem(self.multi_file_list.row(item))

    def _multi_clear_all(self):
        self.multi_file_list.clear()

    def _get_fig_settings(self) -> FigureSettings:
        return FigureSettings(
            dpi=self.multi_dpi_spin.value(),
            width=self.multi_fig_w.value(),
            height=self.multi_fig_h.value(),
            font_size=self.multi_font_spin.value(),
            title_size=self.multi_font_spin.value() + 1,
            legend_size=self.multi_font_spin.value() - 2,
            log_x=self.multi_logx.isChecked(),
            log_y=self.multi_logy.isChecked(),
            grid=self.multi_grid.isChecked(),
            save_png=self.multi_png.isChecked(),
            save_pdf=self.multi_pdf.isChecked(),
            show_vs=self.multi_show_vs.isChecked(),
            color_palette=self.multi_palette.currentText(),
            individual_alpha=self.multi_alpha.value(),
            individual_linewidth=self.multi_line_w.value(),
            median_linewidth=self.multi_median_w.value(),
            show_median=self.multi_show_median.isChecked(),
            show_secondary_peaks=self.multi_show_sec.isChecked(),
        )

    def _run_multi_profiles(self):
        """Load all Excel files, open the sequential picker dialog, and save results."""
        n = self.multi_file_list.count()
        if n == 0:
            QMessageBox.warning(self, "No Files", "Please add at least one Excel profile file.")
            return

        outdir = self.outdir_edit.text().strip()
        if not outdir:
            QMessageBox.warning(self, "No Output", "Please set an output directory first.")
            return

        # Parse all Excel files
        profiles = []
        errors = []
        for i in range(n):
            fpath = self.multi_file_list.item(i).data(Qt.UserRole)
            try:
                prof = SoilProfile.from_excel_file(fpath)
                profiles.append((Path(fpath).stem, prof))
            except Exception as e:
                errors.append(f"{Path(fpath).name}: {e}")

        if errors:
            QMessageBox.warning(
                self, "Parse Errors",
                f"Could not load {len(errors)} file(s):\n" + "\n".join(errors[:10])
            )
        if not profiles:
            return

        freq_config = {
            "fmin": self.fmin_spin.value(),
            "fmax": self.fmax_spin.value(),
            "nf": self.nf_spin.value(),
        }
        fig_settings = self._get_fig_settings()

        dialog = MultiProfilePickerDialog(profiles, freq_config, fig_settings, self)

        def _on_report(results, median):
            s = dialog._fig_settings
            self._save_multi_results(results, outdir, s, median)
            # Save the dialog's current plot view alongside report
            out = Path(outdir)
            out.mkdir(parents=True, exist_ok=True)
            for ext in (["png"] if s.save_png else []) + (["pdf"] if s.save_pdf else []):
                dialog.save_current_plot(
                    str(out / f"median_step_view.{ext}"), dpi=s.dpi
                )
            self.multi_status.setText(
                f"Report saved to {outdir} (dialog still open)"
            )
            self.multi_status.setStyleSheet("color: green; font-weight: bold;")

        dialog.report_requested.connect(_on_report)

        if dialog.exec():
            results = dialog.get_results()
            median_result = dialog.get_median_result()
            self._save_multi_results(results, outdir, dialog._fig_settings, median_result)

    def _save_multi_results(self, results: list, outdir: str, s: FigureSettings,
                            median_result=None):
        """Save per-profile folders + combined overlay plot."""
        import matplotlib.pyplot as plt

        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        extensions = []
        if s.save_png:
            extensions.append("png")
        if s.save_pdf:
            extensions.append("pdf")
        if not extensions:
            extensions = ["png"]

        saved_count = 0

        for r in results:
            if r.freqs is None:
                continue

            folder = out / r.name
            folder.mkdir(parents=True, exist_ok=True)

            # CSV
            csv_path = folder / "hv_curve.csv"
            with open(csv_path, "w") as f:
                f.write("frequency,amplitude\n")
                for freq, amp in zip(r.freqs, r.amps):
                    f.write(f"{freq},{amp}\n")

            # Peak info
            peak_path = folder / "peak_info.txt"
            with open(peak_path, "w") as f:
                if r.f0 is not None:
                    f.write(f"f0_Frequency_Hz,{r.f0[0]:.6f}\n")
                    f.write(f"f0_Amplitude,{r.f0[1]:.6f}\n")
                    f.write(f"f0_Index,{r.f0[2]}\n")
                for i, (sf, sa, si) in enumerate(r.secondary_peaks):
                    f.write(f"Secondary_{i+1}_Frequency_Hz,{sf:.6f}\n")
                    f.write(f"Secondary_{i+1}_Amplitude,{sa:.6f}\n")
                    f.write(f"Secondary_{i+1}_Index,{si}\n")

            # HV-only figure
            fig = plt.figure(figsize=(s.width, s.height))
            ax = fig.add_subplot(111)
            MultiProfilePickerDialog._draw_hv(ax, r, s)
            fig.tight_layout()
            for ext in extensions:
                fig.savefig(folder / f"hv_forward_curve.{ext}", dpi=s.dpi, bbox_inches="tight")
            plt.close(fig)

            # HV + Vs figure (optional)
            if s.show_vs and r.profile is not None:
                fig2 = plt.figure(figsize=(s.width * 1.4, s.height))
                gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.25)
                ax_hv = fig2.add_subplot(gs[0])
                ax_vs = fig2.add_subplot(gs[1])
                MultiProfilePickerDialog._draw_hv(ax_hv, r, s)
                MultiProfilePickerDialog._draw_vs(ax_vs, r.profile, s)
                fig2.tight_layout()
                for ext in extensions:
                    fig2.savefig(folder / f"hv_forward_with_vs.{ext}", dpi=s.dpi, bbox_inches="tight")
                plt.close(fig2)

            saved_count += 1

        # Combined overlay plot
        computed = [r for r in results if r.freqs is not None]
        if computed:
            fig_c = plt.figure(figsize=(s.width, s.height))
            ax_c = fig_c.add_subplot(111)

            n_prof = len(computed)
            colors = get_palette_colors(s.color_palette, n_prof)

            # Individual profiles — lighter, thinner
            for i, r in enumerate(computed):
                c = colors[i]
                ax_c.plot(
                    r.freqs, r.amps,
                    linewidth=s.individual_linewidth,
                    color=c, alpha=s.individual_alpha,
                    label=r.name,
                )
                # f0 peak
                if r.f0 is not None:
                    ax_c.scatter(
                        r.f0[0], r.f0[1], color=c, s=100, marker="*",
                        edgecolors="black", linewidth=0.6, zorder=5,
                        alpha=min(s.individual_alpha + 0.3, 1.0),
                    )
                # Secondary peaks
                if s.show_secondary_peaks:
                    for sec_f, sec_a, _ in r.secondary_peaks:
                        ax_c.scatter(
                            sec_f, sec_a, color=c, s=70, marker="d",
                            edgecolors="black", linewidth=0.5, zorder=4,
                            alpha=min(s.individual_alpha + 0.2, 1.0),
                        )

            # Median HV curve — use dialog result if available, else compute
            mr = median_result
            med_freqs = None
            med_amps = None

            if mr is not None and mr.freqs is not None:
                # User-selected median from dialog
                med_freqs = mr.freqs
                med_amps = mr.amps
            elif s.show_median and len(computed) >= 2:
                # Fallback: compute median automatically
                ref_freqs = computed[0].freqs
                all_a = np.column_stack([
                    np.interp(ref_freqs, r.freqs, r.amps) for r in computed
                ])
                med_freqs = ref_freqs
                med_amps = np.median(all_a, axis=1)
                # Auto-detect peak
                peak_idx = int(np.argmax(med_amps))
                mr = ProfileResult(
                    name="Median HV", profile=computed[0].profile,
                    freqs=med_freqs, amps=med_amps, computed=True,
                    f0=(float(med_freqs[peak_idx]),
                        float(med_amps[peak_idx]), peak_idx),
                )

            f0_c = _MARKER_COLORS.get(s.f0_marker_color, "#d62728")
            f0_m = _MARKER_SHAPES.get(s.f0_marker_shape, "*")
            sec_c = _MARKER_COLORS.get(s.secondary_marker_color, "#2ca02c")
            sec_m = _MARKER_SHAPES.get(s.secondary_marker_shape, "*")

            if med_freqs is not None and mr is not None:
                ax_c.plot(
                    med_freqs, med_amps, color="black",
                    linewidth=s.median_linewidth, linestyle="-",
                    label="Median HV", zorder=10,
                )
                if mr.f0 is not None:
                    ax_c.scatter(
                        mr.f0[0], mr.f0[1], color=f0_c,
                        s=s.f0_marker_size, marker=f0_m,
                        edgecolors=_darken_color(f0_c), linewidth=1.5,
                        zorder=11,
                        label=f"Median f0 = {mr.f0[0]:.2f} Hz",
                    )
                if s.show_secondary_peaks:
                    for sec_f, sec_a, _ in mr.secondary_peaks:
                        ax_c.scatter(
                            sec_f, sec_a, color=sec_c,
                            s=s.secondary_marker_size, marker=sec_m,
                            edgecolors=_darken_color(sec_c), linewidth=1.2,
                            zorder=10,
                            label=f"Median Sec. ({sec_f:.2f} Hz, A={sec_a:.2f})",
                        )

            ax_c.set_xlabel("Frequency (Hz)", fontsize=s.font_size)
            ax_c.set_ylabel("H/V Amplitude Ratio", fontsize=s.font_size)
            ax_c.set_title("Combined HV Curves — All Profiles",
                           fontsize=s.title_size, fontweight="bold")
            if s.log_x:
                ax_c.set_xscale("log")
            if s.log_y:
                ax_c.set_yscale("log")
            if s.grid:
                ax_c.grid(True, alpha=0.3, which="both")
            ax_c.legend(loc="upper right", fontsize=max(s.legend_size - 1, 6), ncol=2)
            fig_c.tight_layout()
            for ext in extensions:
                fig_c.savefig(out / f"combined_hv_curves.{ext}",
                              dpi=s.dpi, bbox_inches="tight")
            plt.close(fig_c)

            # Combined summary CSV
            summary_path = out / "combined_summary.csv"
            with open(summary_path, "w") as f:
                f.write("Profile,f0_Hz,f0_Amplitude")
                if s.show_secondary_peaks:
                    f.write(",Secondary_Peaks")
                f.write("\n")
                for r in computed:
                    f0_f = f"{r.f0[0]:.6f}" if r.f0 else ""
                    f0_a = f"{r.f0[1]:.6f}" if r.f0 else ""
                    row = f"{r.name},{f0_f},{f0_a}"
                    if s.show_secondary_peaks:
                        sec_str = "; ".join(
                            f"{sf:.3f} Hz" for sf, _, _ in r.secondary_peaks
                        ) if r.secondary_peaks else ""
                        row += f",{sec_str}"
                    f.write(row + "\n")
                # Median row
                if mr is not None and mr.f0 is not None:
                    m_f0 = f"{mr.f0[0]:.6f}"
                    m_a0 = f"{mr.f0[1]:.6f}"
                    row = f"Median,{m_f0},{m_a0}"
                    if s.show_secondary_peaks:
                        sec_str = "; ".join(
                            f"{sf:.3f} Hz" for sf, _, _ in mr.secondary_peaks
                        ) if mr.secondary_peaks else ""
                        row += f",{sec_str}"
                    f.write(row + "\n")

            # Save median curve CSV
            if med_freqs is not None and med_amps is not None:
                med_path = out / "median_hv_curve.csv"
                with open(med_path, "w") as f:
                    f.write("frequency,median_amplitude\n")
                    for freq, amp in zip(med_freqs, med_amps):
                        f.write(f"{freq},{amp}\n")
                # Save median peak info
                if mr is not None:
                    med_peak_path = out / "median_peak_info.txt"
                    with open(med_peak_path, "w") as f:
                        if mr.f0 is not None:
                            f.write(f"Median_f0_Frequency_Hz,{mr.f0[0]:.6f}\n")
                            f.write(f"Median_f0_Amplitude,{mr.f0[1]:.6f}\n")
                        for i, (sf, sa, si) in enumerate(mr.secondary_peaks):
                            f.write(f"Median_Secondary_{i+1}_Frequency_Hz,{sf:.6f}\n")
                            f.write(f"Median_Secondary_{i+1}_Amplitude,{sa:.6f}\n")

        self.multi_status.setText(
            f"Done! Saved {saved_count} profiles to {outdir}"
        )
        self.multi_status.setStyleSheet("color: green; font-weight: bold;")
        self.results_text.setText(
            f"Multiple Profiles: {saved_count} processed, saved to:\n{outdir}"
        )

    def _populate_reference_table(self):
        """Populate the reference table with typical nu values."""
        data = VelocityConverter.get_typical_values_table()
        self.ref_table.setRowCount(len(data))
        for row, (material, nu_min, nu_max, nu_typical) in enumerate(data):
            self.ref_table.setItem(row, 0, QTableWidgetItem(material))
            self.ref_table.setItem(row, 1, QTableWidgetItem(f"{nu_min:.2f}"))
            self.ref_table.setItem(row, 2, QTableWidgetItem(f"{nu_max:.2f}"))
            self.ref_table.setItem(row, 3, QTableWidgetItem(f"{nu_typical:.2f}"))
        self.ref_table.resizeColumnsToContents()

    def _create_default_profile(self):
        """Create a default profile."""
        profile = SoilProfile(name="New Profile")
        profile.add_layer(Layer(thickness=5.0, vs=150, density=1700))
        profile.add_layer(Layer(thickness=10.0, vs=300, density=1900))
        profile.add_layer(Layer(thickness=0, vs=600, density=2100, is_halfspace=True))
        self.layer_table.set_profile(profile)
        self.profile_preview.set_profile(profile)
        self._current_profile = profile
        self._validate_profile()

    def _new_profile(self):
        """Create a new empty profile."""
        self._create_default_profile()

    def _open_profile(self):
        """Open a profile file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Profile", "",
            "All Supported (*.txt *.csv);;Text Files (*.txt);;CSV Files (*.csv)"
        )
        if path:
            try:
                if path.lower().endswith('.csv'):
                    profile = SoilProfile.from_csv_file(path)
                else:
                    profile = SoilProfile.from_txt_file(path)
                self.layer_table.set_profile(profile)
                self.profile_preview.set_profile(profile)
                self._current_profile = profile
                self._validate_profile()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def _save_profile(self):
        """Save profile to file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", "", "HVf Format (*.txt);;CSV Format (*.csv)"
        )
        if path:
            try:
                profile = self.layer_table.get_profile()
                if path.lower().endswith('.csv'):
                    profile.save_csv(path)
                else:
                    profile.save_hvf(path)
                QMessageBox.information(self, "Saved", f"Profile saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _on_profile_changed(self):
        """Handle profile changes from editor."""
        self._current_profile = self.layer_table.get_profile()
        self.profile_preview.set_profile(self._current_profile)
        self._validate_profile()

    def _validate_profile(self):
        """Validate and show status."""
        profile = self.layer_table.get_profile()
        is_valid, errors = profile.validate()
        if is_valid:
            self.validation_text.setStyleSheet("color: green;")
            self.validation_text.setText(
                f"Valid profile\n"
                f"Layers: {len(profile.layers)}, "
                f"Depth: {profile.get_total_thickness():.1f} m"
            )
        else:
            self.validation_text.setStyleSheet("color: red;")
            self.validation_text.setText("Errors:\n" + "\n".join(errors))

    def _browse_model(self):
        """Browse for model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            self.model_edit.setText(path)
            try:
                profile = SoilProfile.from_txt_file(path)
                self.file_preview.set_profile(profile)
            except Exception as e:
                self.results_text.setText(f"Error loading preview: {e}")

    def _browse_output_dir(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.outdir_edit.setText(path)

    def _toggle_pick_f0(self):
        """Toggle f0 selection mode (mutually exclusive with secondary)."""
        if self.btn_pick.isChecked():
            self._pick_mode = True
            self._pick_secondary = False
            self.btn_pick_secondary.setChecked(False)
        else:
            self._pick_mode = False

    def _toggle_pick_secondary(self):
        """Toggle secondary peak selection mode (mutually exclusive with f0)."""
        if self.btn_pick_secondary.isChecked():
            self._pick_secondary = True
            self._pick_mode = False
            self.btn_pick.setChecked(False)
        else:
            self._pick_secondary = False

    def _clear_secondary_peaks(self):
        """Clear all secondary peaks."""
        self._secondary_peaks.clear()
        self._update_plot()
        self._update_selection_label()

    def _update_selection_label(self):
        """Update the selection info label."""
        parts = []
        if self._selected_peak is not None:
            f, a, _ = self._selected_peak
            parts.append(f"f0 = {f:.3f} Hz ({a:.2f})")
        for i, (f, a, _) in enumerate(self._secondary_peaks):
            parts.append(f"Sec.{i+1} = {f:.3f} Hz ({a:.2f})")
        if parts:
            self.selection_label.setText("  |  ".join(parts))
            self.selection_label.setStyleSheet(
                "font-size: 11px; padding: 4px; color: #107c10; font-weight: bold;"
            )
        else:
            self.selection_label.setText("")

    def _on_canvas_click(self, event):
        """Handle click on the matplotlib canvas to select peak."""
        if (not self._pick_mode and not self._pick_secondary) or self._last_freqs is None:
            return
        if event.inaxes is None or event.xdata is None:
            return
        if event.button != 1:
            return

        # Check toolbar mode
        toolbar_mode = getattr(self.hv_plot.toolbar, 'mode', '')
        if toolbar_mode:
            return

        freqs = np.array(self._last_freqs)
        amps = np.array(self._last_amps)
        click_freq = event.xdata

        if click_freq < freqs.min() or click_freq > freqs.max():
            return

        # Find nearest point using log distance (since x-axis is log)
        log_freqs = np.log10(freqs)
        log_click = np.log10(click_freq)
        idx = int(np.argmin(np.abs(log_freqs - log_click)))

        picked = (float(freqs[idx]), float(amps[idx]), idx)

        if self._pick_mode:
            self._selected_peak = picked
        elif self._pick_secondary:
            self._secondary_peaks.append(picked)

        self._update_selection_label()
        self._update_plot()

    def _get_active_profile(self) -> SoilProfile:
        """Get the profile used for the current computation."""
        tab_index = self.input_tabs.currentIndex()
        if tab_index == 0:  # From File
            model_path = self.model_edit.text().strip()
            if model_path and Path(model_path).exists():
                return SoilProfile.from_txt_file(model_path)
        elif tab_index == 1:  # From Dinver
            if hasattr(self, '_dinver_profile') and self._dinver_profile is not None:
                return self._dinver_profile
        else:  # Profile Editor
            return self.layer_table.get_profile()
        return None

    def _run_forward(self):
        """Run forward modeling."""
        tab_index = self.input_tabs.currentIndex()
        
        if tab_index == 0:  # From File
            model_path = self.model_edit.text().strip()
            if not model_path or not Path(model_path).exists():
                QMessageBox.warning(self, "Error", "Please select a valid model file.")
                return
            try:
                self._active_profile = SoilProfile.from_txt_file(model_path)
            except Exception:
                self._active_profile = None
        elif tab_index == 1:  # From Dinver
            if not hasattr(self, '_dinver_profile') or self._dinver_profile is None:
                QMessageBox.warning(self, "Error", "Please load a Dinver profile first.")
                return
            profile = self._dinver_profile
            is_valid, errors = profile.validate()
            if not is_valid:
                QMessageBox.warning(self, "Error", "Invalid profile:\n" + "\n".join(errors))
                return
            self._active_profile = profile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(profile.to_hvf_format())
                model_path = f.name
            self._temp_model_path = model_path
        else:  # Profile Editor
            profile = self.layer_table.get_profile()
            is_valid, errors = profile.validate()
            if not is_valid:
                QMessageBox.warning(self, "Error", "Invalid profile:\n" + "\n".join(errors))
                return
            self._active_profile = profile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(profile.to_hvf_format())
                model_path = f.name
            self._temp_model_path = model_path

        config = {
            "fmin": self.fmin_spin.value(),
            "fmax": self.fmax_spin.value(),
            "nf": self.nf_spin.value(),
        }

        self._selected_peak = None
        self._secondary_peaks.clear()
        self.selection_label.setText("")
        self.btn_run.setEnabled(False)
        self.results_text.setText("Computing HV curve...")

        self.worker = ForwardWorker(model_path, config)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, result):
        freqs, amps = result
        self._last_freqs = freqs
        self._last_amps = amps

        freqs_arr = np.array(freqs)
        amps_arr = np.array(amps)
        peak_idx = int(np.argmax(amps_arr))
        peak_freq = freqs_arr[peak_idx]
        peak_amp = amps_arr[peak_idx]

        self._selected_peak = (float(peak_freq), float(peak_amp), peak_idx)

        self.results_text.setText(
            f"f0 = {peak_freq:.3f} Hz, Amplitude: {peak_amp:.2f}\n"
            f"Range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz\n"
            f"Points: {len(freqs)}"
        )
        self._update_selection_label()

        self._update_plot()
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)

        if self._temp_model_path:
            try:
                Path(self._temp_model_path).unlink()
            except Exception:
                pass
            self._temp_model_path = None

    def _on_error(self, error):
        self.results_text.setText(f"Error: {error}")
        self.btn_run.setEnabled(True)

        if self._temp_model_path:
            try:
                Path(self._temp_model_path).unlink()
            except Exception:
                pass
            self._temp_model_path = None

    def _update_plot(self):
        """Update the HV curve plot with optional Vs profile panel."""
        if self._last_freqs is None or self._last_amps is None:
            return

        freqs = np.array(self._last_freqs)
        amps = np.array(self._last_amps)

        fig = self.hv_plot.get_figure()
        fig.clear()

        show_vs = self.show_vs_check.isChecked() and self._active_profile is not None

        if show_vs:
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.25)
            ax_hv = fig.add_subplot(gs[0])
            ax_vs = fig.add_subplot(gs[1])
        else:
            ax_hv = fig.add_subplot(111)
            ax_vs = None

        self._draw_hv_curve(ax_hv, freqs, amps)

        if ax_vs is not None and self._active_profile is not None:
            self._draw_vs_profile(ax_vs, self._active_profile)

        try:
            fig.tight_layout()
        except Exception:
            pass
        self.hv_plot.refresh()

    def _draw_hv_curve(self, ax, freqs, amps):
        """Draw HV curve with f0 and secondary peaks on the given axes."""
        ax.plot(freqs, amps, 'b-', linewidth=2, label='H/V Ratio')

        # Primary peak (f0) — red star
        if self._selected_peak is not None:
            sel_f, sel_a, _ = self._selected_peak
            ax.scatter(sel_f, sel_a, color='red', s=200, marker='*',
                      edgecolors='darkred', linewidth=1.5,
                      label=f'f0 = {sel_f:.2f} Hz', zorder=5)
            ax.axvline(x=sel_f, color='red', linestyle='--', alpha=0.4)

        # Secondary peaks — black stars
        for i, (sec_f, sec_a, _) in enumerate(self._secondary_peaks):
            label = f'Secondary Peak ({sec_f:.2f} Hz)'
            ax.scatter(sec_f, sec_a, color='black', s=200, marker='*',
                      edgecolors='black', linewidth=1.5,
                      label=label, zorder=4)
            ax.axvline(x=sec_f, color='black', linestyle=':', alpha=0.4)

        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('H/V Amplitude Ratio', fontsize=12)
        ax.set_title('HV Spectral Ratio - Forward Model', fontsize=13, fontweight='bold')

        if self.log_x_check.isChecked():
            ax.set_xscale('log')
        if self.log_y_check.isChecked():
            ax.set_yscale('log')
        if self.grid_check.isChecked():
            ax.grid(True, alpha=0.3, which='both')

        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(freqs[0], freqs[-1])

    def _draw_vs_profile(self, ax, profile: SoilProfile):
        """Draw compact Vs profile on the given axes."""
        depths = [0.0]
        vs_vals = []
        for layer in profile.layers:
            vs_vals.append(layer.vs)
            if layer.is_halfspace:
                depths.append(depths[-1] + max(100, depths[-1] * 0.3))
            else:
                depths.append(depths[-1] + layer.thickness)

        # Build step-function arrays
        plot_d, plot_v = [], []
        for i in range(len(vs_vals)):
            plot_d.extend([depths[i], depths[i + 1]])
            plot_v.extend([vs_vals[i], vs_vals[i]])

        ax.fill_betweenx(plot_d, 0, plot_v, alpha=0.3, color='teal')
        ax.step(plot_v + [plot_v[-1]], [0] + plot_d, where='pre',
               color='teal', linewidth=1.5, linestyle='-')

        # Depth annotation at deepest finite interface
        finite_layers = [l for l in profile.layers if not l.is_halfspace]
        if finite_layers:
            total_depth = sum(l.thickness for l in finite_layers)
            ax.axhline(y=total_depth, color='red', linestyle='-', alpha=0.6, linewidth=1.5)
            ax.text(0.95, 0.02, f'{total_depth:.0f}m',
                   transform=ax.transAxes, fontsize=8, fontweight='bold',
                   color='red', ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_xlabel('Vs', fontsize=9)
        ax.set_ylabel('Depth (m)', fontsize=9)
        n_layers = len([l for l in profile.layers if not l.is_halfspace])
        ax.set_title(f'{n_layers}L', fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        if plot_v:
            ax.set_xlim(0, max(plot_v) * 1.1)

    def _draw_empty_plot(self):
        """Draw empty placeholder plot."""
        fig = self.hv_plot.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, "Click 'Compute HV Curve' to generate results",
            ha='center', va='center', fontsize=12, color='gray'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.hv_plot.refresh()

    def _save_results(self):
        """Save HV curve data, figures, and peak info to output directory."""
        if self._last_freqs is None:
            return

        outdir = self.outdir_edit.text().strip()
        if not outdir:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "hv_curve.csv", "CSV Files (*.csv)"
            )
            if not path:
                return
            outdir = str(Path(path).parent)
            csv_path = Path(path)
        else:
            out_path = Path(outdir)
            out_path.mkdir(parents=True, exist_ok=True)
            csv_path = out_path / 'hv_curve.csv'

        freqs = np.array(self._last_freqs)
        amps = np.array(self._last_amps)

        # Save CSV data
        with open(csv_path, 'w') as f:
            f.write("frequency,amplitude\n")
            for freq, amp in zip(freqs, amps):
                f.write(f"{freq},{amp}\n")

        saved_files = [str(csv_path)]

        # Save peak info (f0 + secondary)
        peak_path = Path(outdir) / 'peak_info.txt'
        with open(peak_path, 'w') as f:
            if self._selected_peak is not None:
                sel_f, sel_a, sel_idx = self._selected_peak
                f.write(f"f0_Frequency_Hz,{sel_f:.6f}\n")
                f.write(f"f0_Amplitude,{sel_a:.6f}\n")
                f.write(f"f0_Index,{sel_idx}\n")
            for i, (sec_f, sec_a, sec_idx) in enumerate(self._secondary_peaks):
                f.write(f"Secondary_{i+1}_Frequency_Hz,{sec_f:.6f}\n")
                f.write(f"Secondary_{i+1}_Amplitude,{sec_a:.6f}\n")
                f.write(f"Secondary_{i+1}_Index,{sec_idx}\n")
        saved_files.append(str(peak_path))

        import matplotlib.pyplot as plt
        profile = self._active_profile

        # Figure 1: HV curve + Vs profile side-by-side
        if profile is not None:
            fig_combined = plt.figure(figsize=(14, 5))
            gs_c = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.25)
            ax_hv_c = fig_combined.add_subplot(gs_c[0])
            ax_vs_c = fig_combined.add_subplot(gs_c[1])
            self._draw_hv_curve(ax_hv_c, freqs, amps)
            self._draw_vs_profile(ax_vs_c, profile)
            fig_combined.tight_layout()
            for ext in ['png', 'pdf']:
                fig_path = Path(outdir) / f'hv_forward_with_vs.{ext}'
                fig_combined.savefig(fig_path, dpi=300, bbox_inches='tight')
                saved_files.append(str(fig_path))
            plt.close(fig_combined)

        # Figure 2: HV curve only (no Vs profile)
        fig_hv = plt.figure(figsize=(10, 5))
        ax_hv_only = fig_hv.add_subplot(111)
        self._draw_hv_curve(ax_hv_only, freqs, amps)
        fig_hv.tight_layout()
        for ext in ['png', 'pdf']:
            fig_path = Path(outdir) / f'hv_forward_curve.{ext}'
            fig_hv.savefig(fig_path, dpi=300, bbox_inches='tight')
            saved_files.append(str(fig_path))
        plt.close(fig_hv)

        self.results_text.append(f"\nSaved {len(saved_files)} files to: {outdir}")
        for f in saved_files:
            self.results_text.append(f"  - {Path(f).name}")
