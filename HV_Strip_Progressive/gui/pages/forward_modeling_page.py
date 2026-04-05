"""HV Forward Modeling Page.

Faithfully replicates the original PySide6 ForwardModelingPage structure:
  Left panel  — Input tabs (File / Dinver / Editor / Multiple) + config + controls + results
  Right panel — MatplotlibWidget with peak picking + plot options
"""
import os
import tempfile
import numpy as np
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QGroupBox, QFormLayout, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QTextEdit,
    QFileDialog, QMessageBox, QScrollArea,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.profile_preview_widget import ProfilePreviewWidget
from ..widgets.layer_table_widget import LayerTableWidget
from ..workers.forward_worker import ForwardWorker
from ..widgets.style_constants import (
    OUTER_MARGINS, SECONDARY_LABEL, MONOSPACE_PREVIEW,
    BUTTON_PRIMARY, GEAR_BUTTON, EMOJI,
)

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]
ENGINE_DESCRIPTIONS = {
    "diffuse_field": "Full diffuse wavefield H/V ratio (HVf.exe required)",
    "sh_wave": "SH-wave transfer function (pure Python, no external tools)",
    "ellipticity": "Rayleigh wave ellipticity (gpell.exe required)",
}


class ForwardModelingPage(QWidget):
    """Interactive HV forward modeling with profile editing and peak selection."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._main_window = main_window

        # State
        self._active_profile = None
        self._temp_model_path = None
        self._last_freqs = None
        self._last_amps = None
        self._selected_peak = None        # (freq, amp, idx)
        self._secondary_peaks = []        # [(freq, amp, idx), ...]
        self._pick_mode = False
        self._pick_secondary = False
        self._worker = None
        self._engine_settings_config = {}

        self._build_ui()

    # ═══════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*OUTER_MARGINS)

        # Header
        hdr = QLabel(f"<b>{EMOJI['forward']} HV Forward Modeling</b>")
        hdr.setStyleSheet("font-size: 14px; padding: 4px;")
        layout.addWidget(hdr)
        desc = QLabel("Compute theoretical H/V curves for velocity profiles using different engines")
        desc.setStyleSheet(SECONDARY_LABEL)
        layout.addWidget(desc)

        # Splitter: left (input + config) | right (plot)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # ── LEFT PANEL ──────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_w = QWidget()
        left_layout = QVBoxLayout(left_w)

        # Input Tabs (simplified: Load Model | Profile Editor | Multiple)
        self._input_tabs = QTabWidget()
        self._input_tabs.addTab(self._build_load_tab(), "Load Model")
        self._input_tabs.addTab(self._build_editor_tab(), "Profile Editor")
        self._input_tabs.addTab(self._build_multi_tab(), "Multiple Profiles")
        left_layout.addWidget(self._input_tabs)

        # Frequency Parameters
        left_layout.addWidget(self._build_freq_group())

        # Output Directory
        left_layout.addWidget(self._build_output_group())

        # Control Buttons
        left_layout.addWidget(self._build_control_group())

        # Results
        left_layout.addWidget(self._build_results_group())

        left_layout.addStretch()
        left_scroll.setWidget(left_w)
        splitter.addWidget(left_scroll)

        # ── RIGHT PANEL ─────────────────────────────────────────
        right_w = QWidget()
        right_layout = QVBoxLayout(right_w)

        self._plot = MatplotlibWidget(figsize=(14, 5))
        self._plot.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        right_layout.addWidget(self._plot)

        # Plot options
        right_layout.addLayout(self._build_plot_options())

        # Peak selection
        right_layout.addLayout(self._build_peak_selection())

        splitter.addWidget(right_w)
        splitter.setSizes([450, 550])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Tab 0: Load Model (unified) ────────────────────────────────
    def _build_load_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Load a velocity model from any supported format "
                                "(HVf .txt, Dinver .txt, CSV, Excel):"))

        # Browse row
        row = QHBoxLayout()
        row.addWidget(QLabel("Model File:"))
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Browse for any model file (.txt, .csv, .xlsx)")
        row.addWidget(self._file_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_model_file)
        row.addWidget(btn)
        layout.addLayout(row)

        # Advanced loader button (opens ProfileLoaderDialog for Dinver 3-file import)
        adv_row = QHBoxLayout()
        adv_btn = QPushButton("Advanced Import (Dinver 3-file)...")
        adv_btn.setToolTip("Open advanced loader for Dinver multi-file import")
        adv_btn.clicked.connect(self._open_profile_loader)
        adv_row.addWidget(adv_btn)
        adv_row.addStretch()
        layout.addLayout(adv_row)

        self._file_status = QLabel("")
        layout.addWidget(self._file_status)

        # "Send to Editor" button — lets user fine-tune auto-computed Vp/density
        edit_row = QHBoxLayout()
        self._send_to_editor_btn = QPushButton("📝 Send to Profile Editor (edit Vp/density)")
        self._send_to_editor_btn.setToolTip(
            "Load the profile into the editor tab where you can manually adjust\n"
            "auto-computed Vp, density, and Poisson's ratio values")
        self._send_to_editor_btn.setEnabled(False)
        self._send_to_editor_btn.clicked.connect(self._send_to_editor)
        edit_row.addWidget(self._send_to_editor_btn)
        edit_row.addStretch()
        layout.addLayout(edit_row)

        self._file_preview = ProfilePreviewWidget()
        layout.addWidget(self._file_preview)
        layout.addStretch()
        return w

    # ── Tab 2: Profile Editor ───────────────────────────────────
    def _build_editor_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        # File operations
        file_row = QHBoxLayout()
        for label, slot in [("New", self._editor_new), ("Open", self._editor_open), ("Save", self._editor_save)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            file_row.addWidget(b)
        file_row.addStretch()
        layout.addLayout(file_row)

        # Layer table + preview in splitter
        editor_split = QSplitter(Qt.Horizontal)
        self._layer_table = LayerTableWidget()
        self._layer_table.profile_changed.connect(self._on_editor_profile_changed)
        editor_split.addWidget(self._layer_table)

        right_col = QWidget()
        rc_layout = QVBoxLayout(right_col)
        self._editor_preview = ProfilePreviewWidget()
        rc_layout.addWidget(QLabel("Vs Profile Preview:"))
        rc_layout.addWidget(self._editor_preview)

        # Validation
        self._validation_text = QTextEdit()
        self._validation_text.setReadOnly(True)
        self._validation_text.setMaximumHeight(80)
        rc_layout.addWidget(QLabel("Validation:"))
        rc_layout.addWidget(self._validation_text)
        editor_split.addWidget(right_col)
        editor_split.setSizes([400, 200])

        layout.addWidget(editor_split)
        return w

    # ── Tab 3: Multiple Profiles ────────────────────────────────
    def _build_multi_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Load and compare multiple velocity profiles."))

        btn_row = QHBoxLayout()
        self._multi_add_btn = QPushButton("Add Files...")
        self._multi_add_btn.clicked.connect(self._multi_add_files)
        self._multi_add_dir_btn = QPushButton("Add Directory...")
        self._multi_add_dir_btn.clicked.connect(self._multi_add_dir)
        self._multi_clear_btn = QPushButton("Clear All")
        self._multi_clear_btn.clicked.connect(self._multi_clear)
        btn_row.addWidget(self._multi_add_btn)
        btn_row.addWidget(self._multi_add_dir_btn)
        btn_row.addWidget(self._multi_clear_btn)
        layout.addLayout(btn_row)

        from PyQt5.QtWidgets import QListWidget
        self._multi_list = QListWidget()
        self._multi_list.setMinimumHeight(120)
        layout.addWidget(self._multi_list)

        self._multi_run_btn = QPushButton("Run All Profiles")
        self._multi_run_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self._multi_run_btn.clicked.connect(self._multi_run)
        layout.addWidget(self._multi_run_btn)

        self._multi_load_btn = QPushButton("Load Results Folder...")
        self._multi_load_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 6px;")
        self._multi_load_btn.clicked.connect(self._multi_load_results)
        layout.addWidget(self._multi_load_btn)

        layout.addStretch()
        return w

    # ── Frequency Parameters ────────────────────────────────────
    def _build_freq_group(self):
        grp = QGroupBox(f"{EMOJI['frequency']} Frequency Parameters")
        form = QFormLayout(grp)

        self._fmin = QDoubleSpinBox(); self._fmin.setRange(0.01, 10); self._fmin.setValue(0.2); self._fmin.setDecimals(2)
        self._fmax = QDoubleSpinBox(); self._fmax.setRange(1, 100); self._fmax.setValue(20.0); self._fmax.setDecimals(1)
        self._nf = QSpinBox(); self._nf.setRange(10, 2000); self._nf.setValue(71)
        form.addRow("Freq Min (Hz):", self._fmin)
        form.addRow("Freq Max (Hz):", self._fmax)
        form.addRow("Points:", self._nf)

        engine_row = QHBoxLayout()
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(ENGINES)
        self._engine_combo.currentTextChanged.connect(self._on_fwd_engine_changed)
        engine_row.addWidget(self._engine_combo)
        self._gear_btn = QPushButton("⚙")
        self._gear_btn.setFixedWidth(30)
        self._gear_btn.setStyleSheet(GEAR_BUTTON)
        self._gear_btn.setToolTip("Engine Settings")
        self._gear_btn.clicked.connect(self._open_engine_settings)
        engine_row.addWidget(self._gear_btn)
        form.addRow("Engine:", engine_row)
        self._fwd_engine_desc = QLabel(ENGINE_DESCRIPTIONS.get("diffuse_field", ""))
        self._fwd_engine_desc.setStyleSheet(SECONDARY_LABEL)
        self._fwd_engine_desc.setWordWrap(True)
        form.addRow("", self._fwd_engine_desc)

        return grp

    # ── Output Directory ────────────────────────────────────────
    def _build_output_group(self):
        grp = QGroupBox(f"{EMOJI['folder']} Output Directory")
        layout = QHBoxLayout(grp)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select output directory for results")
        layout.addWidget(self._output_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_output)
        layout.addWidget(btn)
        return grp

    # ── Control Buttons ─────────────────────────────────────────
    def _build_control_group(self):
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        self._btn_compute = QPushButton(f"{EMOJI['run']} Compute HV Curve")
        self._btn_compute.setStyleSheet(BUTTON_PRIMARY)
        self._btn_compute.clicked.connect(self._run_forward)
        layout.addWidget(self._btn_compute)

        self._btn_save = QPushButton(f"{EMOJI['save']} Save Results...")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_results)
        layout.addWidget(self._btn_save)
        return w

    # ── Results ─────────────────────────────────────────────────
    def _build_results_group(self):
        grp = QGroupBox(f"{EMOJI['info']} Results")
        layout = QVBoxLayout(grp)
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(120)
        self._results_text.setStyleSheet(MONOSPACE_PREVIEW)
        self._results_text.setPlaceholderText("Run forward model to see results here...")
        layout.addWidget(self._results_text)
        return grp

    # ── Plot Options ────────────────────────────────────────────
    def _build_plot_options(self):
        row = QHBoxLayout()
        self._log_x = QCheckBox("Log X"); self._log_x.setChecked(True)
        self._log_y = QCheckBox("Log Y")
        self._grid = QCheckBox("Grid"); self._grid.setChecked(True)
        self._show_vs = QCheckBox("Show Vs Profile"); self._show_vs.setChecked(True)

        self._hs_pct = QDoubleSpinBox()
        self._hs_pct.setRange(10, 100); self._hs_pct.setValue(25); self._hs_pct.setSuffix(" %")
        self._hs_pct.setToolTip("Half-space display depth as % of total finite depth")

        for w in [self._log_x, self._log_y, self._grid, self._show_vs]:
            w.stateChanged.connect(lambda _: self._update_plot())
            row.addWidget(w)
        row.addWidget(QLabel("HS:"))
        row.addWidget(self._hs_pct)
        self._hs_pct.valueChanged.connect(lambda _: self._update_plot())
        row.addStretch()
        return row

    # ── Peak Selection ──────────────────────────────────────────
    def _build_peak_selection(self):
        row = QHBoxLayout()

        self._btn_pick_f0 = QPushButton("Select f0")
        self._btn_pick_f0.setCheckable(True)
        self._btn_pick_f0.setStyleSheet("QPushButton:checked { background-color: #4CAF50; color: white; }")
        self._btn_pick_f0.toggled.connect(self._on_pick_f0_toggled)
        row.addWidget(self._btn_pick_f0)

        self._btn_pick_sec = QPushButton("Select Secondary Peak")
        self._btn_pick_sec.setCheckable(True)
        self._btn_pick_sec.setStyleSheet("QPushButton:checked { background-color: #FF9800; color: white; }")
        self._btn_pick_sec.toggled.connect(self._on_pick_sec_toggled)
        row.addWidget(self._btn_pick_sec)

        self._btn_clear_sec = QPushButton("Clear Secondary")
        self._btn_clear_sec.clicked.connect(self._clear_secondary)
        row.addWidget(self._btn_clear_sec)

        self._selection_label = QLabel("")
        self._selection_label.setStyleSheet("font-size: 11px; color: #333;")
        row.addWidget(self._selection_label, 1)

        return row

    # ═══════════════════════════════════════════════════════════
    #  PROFILE LOADING
    # ═══════════════════════════════════════════════════════════
    def _get_active_profile(self):
        """Return the profile from the current input tab and a temp model path."""
        from ...core.soil_profile import SoilProfile
        tab_idx = self._input_tabs.currentIndex()

        if tab_idx == 0:  # Load Model
            path = self._file_edit.text().strip()
            if not path or not os.path.isfile(path):
                raise ValueError("Please select a valid model file.")
            profile = SoilProfile.from_auto(path)
            return profile, path

        elif tab_idx == 1:  # Profile Editor
            profile = self._layer_table.get_profile()
            valid, msgs = profile.validate()
            if not valid:
                raise ValueError(f"Profile validation failed:\n" + "\n".join(msgs))
            tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
            tmp.write(profile.to_hvf_format())
            tmp.close()
            self._temp_model_path = tmp.name
            return profile, tmp.name

        elif tab_idx == 2:  # Multiple
            raise ValueError("Use 'Run All Profiles' for multiple profiles.")

        raise ValueError("Unknown input tab.")

    def _browse_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "All Supported (*.txt *.csv *.xlsx);;Text (*.txt);;CSV (*.csv);;Excel (*.xlsx);;All (*)")
        if path:
            self._file_edit.setText(path)
            try:
                from ...core.soil_profile import SoilProfile
                prof = SoilProfile.from_auto(path)
                self._file_preview.set_profile(prof)
                self._active_profile = prof
                n = len([L for L in prof.layers if not L.is_halfspace])
                # Check if Vp/density were auto-computed
                has_missing = any(L.vp is None for L in prof.layers)
                note = ""
                if has_missing:
                    note = " — <i>Vp/density auto-computed from Vs</i>"
                self._file_status.setText(
                    f"<span style='color:green;'>✓ Loaded: {n} layers{note}</span>")
                self._send_to_editor_btn.setEnabled(True)
            except Exception as e:
                self._file_preview._draw_empty()
                self._file_status.setText(
                    f"<span style='color:red;'>✗ {e}</span>")
                self._send_to_editor_btn.setEnabled(False)

    def _open_profile_loader(self):
        """Open the advanced ProfileLoaderDialog (Dinver 3-file, etc.)."""
        from ..dialogs.profile_loader_dialog import ProfileLoaderDialog
        dlg = ProfileLoaderDialog(self)
        if dlg.exec_() == ProfileLoaderDialog.Accepted:
            data = dlg.get_data()
            if data and "path" in data:
                self._file_edit.setText(data["path"])
                try:
                    from ...core.soil_profile import SoilProfile
                    prof = SoilProfile.from_auto(data["path"])
                    self._file_preview.set_profile(prof)
                    self._active_profile = prof
                    n = len([L for L in prof.layers if not L.is_halfspace])
                    self._file_status.setText(
                        f"<span style='color:green;'>✓ Loaded via {data.get('source', 'dialog')}: {n} layers</span>")
                    self._send_to_editor_btn.setEnabled(True)
                except Exception as e:
                    self._file_status.setText(
                        f"<span style='color:red;'>✗ {e}</span>")
                    self._send_to_editor_btn.setEnabled(False)

    def _send_to_editor(self):
        """Send the loaded profile to the Profile Editor tab for manual editing."""
        if self._active_profile is None:
            return
        self._layer_table.set_profile(self._active_profile)
        self._on_editor_profile_changed()
        # Switch to editor tab (index 1)
        self._input_tabs.setCurrentIndex(1)

    # ── Editor operations ───────────────────────────────────────
    def _editor_new(self):
        self._layer_table._new_default()
        self._on_editor_profile_changed()

    def _editor_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Profile", "",
            "All Supported (*.txt *.csv *.xlsx);;Text (*.txt);;CSV (*.csv);;Excel (*.xlsx);;All (*)")
        if path:
            try:
                from ...core.soil_profile import SoilProfile
                prof = SoilProfile.from_auto(path)
                self._layer_table.set_profile(prof)
                self._on_editor_profile_changed()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load: {e}")

    def _editor_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", "", "HVf Files (*.txt);;CSV (*.csv)")
        if path:
            try:
                prof = self._layer_table.get_profile()
                if path.endswith(".csv"):
                    prof.save_csv(path)
                else:
                    prof.save_hvf(path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def _on_editor_profile_changed(self):
        try:
            prof = self._layer_table.get_profile()
            self._editor_preview.set_profile(prof)
            valid, msgs = prof.validate()
            if valid:
                self._validation_text.setStyleSheet("color: green;")
                self._validation_text.setText("✓ Profile is valid")
            else:
                self._validation_text.setStyleSheet("color: red;")
                self._validation_text.setText("✗ " + "\n".join(msgs))
        except Exception as e:
            self._validation_text.setStyleSheet("color: red;")
            self._validation_text.setText(f"✗ Error: {e}")

    # ── Multiple profiles ───────────────────────────────────────
    def _multi_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Profile Files", "",
            "All Supported (*.txt *.csv *.xlsx);;Text (*.txt);;CSV (*.csv);;Excel (*.xlsx);;All (*)")
        for p in paths:
            self._multi_list.addItem(p)

    def _multi_add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            for ext in ("*.txt", "*.csv", "*.xlsx"):
                for f in sorted(Path(d).glob(ext)):
                    self._multi_list.addItem(str(f))

    def _multi_clear(self):
        self._multi_list.clear()

    def _multi_run(self):
        n = self._multi_list.count()
        if n == 0:
            QMessageBox.warning(self, "Error", "No profiles loaded.")
            return
        paths = [self._multi_list.item(i).text() for i in range(n)]
        try:
            from ..dialogs.multi_profile_dialog import MultiProfileDialog
            dlg = MultiProfileDialog(
                paths, self._engine_combo.currentText(),
                self._build_engine_config(), parent=self)
            dlg.exec_()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Multi-profile failed: {e}")

    def _multi_load_results(self):
        d = QFileDialog.getExistingDirectory(self, "Select Results Folder")
        if d:
            try:
                from ..dialogs.output_viewer_dialog import OutputViewerDialog
                dlg = OutputViewerDialog(d, parent=self)
                dlg.exec_()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load results: {e}")

    # ═══════════════════════════════════════════════════════════
    #  FORWARD COMPUTATION
    # ═══════════════════════════════════════════════════════════
    def _build_engine_config(self):
        """Build engine-specific config dict from current UI state."""
        engine_name = self._engine_combo.currentText()
        cfg = dict(self._engine_settings_config.get(engine_name, {}))
        cfg["fmin"] = self._fmin.value()
        cfg["fmax"] = self._fmax.value()
        cfg["nf"] = self._nf.value()
        return cfg

    def _run_forward(self):
        try:
            profile, model_path = self._get_active_profile()
            self._active_profile = profile
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
            return

        engine_name = self._engine_combo.currentText()
        cfg = self._build_engine_config()

        self._btn_compute.setEnabled(False)
        self._btn_compute.setText("Computing...")
        self._results_text.setText("Running forward model...")

        self._worker = ForwardWorker(model_path, cfg, engine_name, parent=self)
        self._worker.finished_signal.connect(self._on_forward_finished)
        self._worker.error.connect(self._on_forward_error)
        self._worker.start()

    def _on_forward_finished(self, result):
        freqs, amps = result
        self._last_freqs = np.array(freqs)
        self._last_amps = np.array(amps)

        # Auto-detect f0 (global maximum)
        idx = np.argmax(self._last_amps)
        self._selected_peak = (self._last_freqs[idx], self._last_amps[idx], idx)
        self._secondary_peaks = []

        # Compute Vs30
        vs30_str = ""
        if self._active_profile:
            try:
                from ...core.vs_average import compute_vs30
                vs30 = compute_vs30(self._active_profile)
                vs30_str = f"\nVs30 = {vs30:.1f} m/s"
            except Exception:
                pass

        f0 = self._selected_peak[0]
        a0 = self._selected_peak[1]
        self._results_text.setText(
            f"f0 = {f0:.4f} Hz  (amplitude = {a0:.3f})\n"
            f"Frequency range: {self._last_freqs[0]:.3f} – {self._last_freqs[-1]:.3f} Hz\n"
            f"Points: {len(self._last_freqs)}{vs30_str}"
        )

        self._btn_compute.setEnabled(True)
        self._btn_compute.setText("Compute HV Curve")
        self._btn_save.setEnabled(True)
        self._update_plot()
        self._update_selection_label()

        if self._main_window:
            self._main_window.set_status(f"Forward model computed — f0 = {f0:.3f} Hz")

    def _on_forward_error(self, msg):
        self._btn_compute.setEnabled(True)
        self._btn_compute.setText("Compute HV Curve")
        self._results_text.setText(f"<span style='color:red;'>Error: {msg}</span>")
        QMessageBox.critical(self, "Computation Error", msg)

    # ═══════════════════════════════════════════════════════════
    #  PLOT
    # ═══════════════════════════════════════════════════════════
    def _update_plot(self):
        if self._last_freqs is None or self._last_amps is None:
            return

        self._plot.figure.clear()
        show_vs = self._show_vs.isChecked() and self._active_profile is not None

        if show_vs:
            import matplotlib.gridspec as gs
            spec = gs.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05, figure=self._plot.figure)
            ax_hv = self._plot.figure.add_subplot(spec[0])
            ax_vs = self._plot.figure.add_subplot(spec[1])
        else:
            ax_hv = self._plot.figure.add_subplot(111)
            ax_vs = None

        # HV Curve
        ax_hv.plot(self._last_freqs, self._last_amps, color="royalblue", linewidth=1.5, label="H/V Ratio")

        # f0 marker
        if self._selected_peak:
            f, a, _ = self._selected_peak
            ax_hv.plot(f, a, marker="*", color="red", markersize=14, zorder=5)
            ax_hv.axvline(f, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_hv.annotate(f"f0={f:.3f} Hz", xy=(f, a), xytext=(10, 10),
                           textcoords="offset points", fontsize=8, color="red",
                           arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        # Secondary peaks
        for sf, sa, _ in self._secondary_peaks:
            ax_hv.plot(sf, sa, marker="*", color="black", markersize=10, zorder=5)
            ax_hv.axvline(sf, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

        if self._log_x.isChecked():
            ax_hv.set_xscale("log")
        if self._log_y.isChecked():
            ax_hv.set_yscale("log")
        ax_hv.set_xlabel("Frequency (Hz)")
        ax_hv.set_ylabel("H/V Ratio")
        ax_hv.set_title("HV Forward Model")
        if self._grid.isChecked():
            ax_hv.grid(True, alpha=0.3, which="both")
        ax_hv.legend(fontsize=8)

        # Vs Profile
        if show_vs and ax_vs is not None:
            self._draw_vs_profile(ax_vs)

        self._plot.refresh()

    def _draw_vs_profile(self, ax):
        profile = self._active_profile
        if profile is None:
            return

        depths, vs_vals = [], []
        z = 0.0
        finite = [L for L in profile.layers if not L.is_halfspace]
        hs_list = [L for L in profile.layers if L.is_halfspace]

        for L in finite:
            depths.append(z); vs_vals.append(L.vs)
            z += L.thickness
            depths.append(z); vs_vals.append(L.vs)

        total_d = z
        if hs_list:
            hs_depth = total_d * (self._hs_pct.value() / 100.0)
            depths.append(z); vs_vals.append(hs_list[0].vs)
            z += hs_depth
            depths.append(z); vs_vals.append(hs_list[0].vs)

        ax.plot(vs_vals, depths, color="teal", linewidth=1.5)
        ax.axhline(total_d, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=7)
        ax.set_title(f"{len(finite)}L", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.yaxis.tick_right()
        ax.grid(True, alpha=0.2)

    # ═══════════════════════════════════════════════════════════
    #  PEAK PICKING
    # ═══════════════════════════════════════════════════════════
    def _on_pick_f0_toggled(self, checked):
        self._pick_mode = checked
        if checked:
            self._btn_pick_sec.setChecked(False)

    def _on_pick_sec_toggled(self, checked):
        self._pick_secondary = checked
        if checked:
            self._btn_pick_f0.setChecked(False)

    def _clear_secondary(self):
        self._secondary_peaks = []
        self._update_plot()
        self._update_selection_label()

    def _on_canvas_click(self, event):
        if event.inaxes is None or self._last_freqs is None:
            return
        if not (self._pick_mode or self._pick_secondary):
            return
        # Disable pick when toolbar is active (pan/zoom)
        if self._plot.toolbar.mode:
            return

        click_x, click_y = event.xdata, event.ydata

        # Find nearest point using log-distance
        log_freqs = np.log10(self._last_freqs + 1e-20)
        log_amps = np.log10(self._last_amps + 1e-20)
        log_cx = np.log10(click_x + 1e-20)
        log_cy = np.log10(click_y + 1e-20)

        x_range = log_freqs.max() - log_freqs.min()
        y_range = log_amps.max() - log_amps.min()
        if x_range == 0: x_range = 1
        if y_range == 0: y_range = 1

        dist = ((log_freqs - log_cx) / x_range) ** 2 + ((log_amps - log_cy) / y_range) ** 2
        idx = int(np.argmin(dist))
        freq = float(self._last_freqs[idx])
        amp = float(self._last_amps[idx])

        if self._pick_mode:
            self._selected_peak = (freq, amp, idx)
            self._btn_pick_f0.setChecked(False)
        elif self._pick_secondary:
            self._secondary_peaks.append((freq, amp, idx))

        self._update_plot()
        self._update_selection_label()

    def _update_selection_label(self):
        parts = []
        if self._selected_peak:
            f, a, _ = self._selected_peak
            parts.append(f"f0 = {f:.3f} Hz ({a:.2f})")
        for i, (sf, sa, _) in enumerate(self._secondary_peaks):
            parts.append(f"Sec.{i+1} = {sf:.3f} Hz ({sa:.2f})")
        self._selection_label.setText("  |  ".join(parts))

    # ═══════════════════════════════════════════════════════════
    #  SAVE RESULTS
    # ═══════════════════════════════════════════════════════════
    def _save_results(self):
        if self._last_freqs is None:
            return
        out_dir = self._output_edit.text().strip()
        if not out_dir:
            out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if not out_dir:
                return
            self._output_edit.setText(out_dir)

        os.makedirs(out_dir, exist_ok=True)
        try:
            # CSV
            csv_path = os.path.join(out_dir, "hv_forward_curve.csv")
            np.savetxt(csv_path, np.column_stack([self._last_freqs, self._last_amps]),
                       delimiter=",", header="frequency,amplitude", comments="")

            # Peak info
            info_path = os.path.join(out_dir, "peak_info.txt")
            with open(info_path, "w") as f:
                if self._selected_peak:
                    freq, amp, _ = self._selected_peak
                    f.write(f"f0 = {freq:.6f} Hz\n")
                    f.write(f"Amplitude = {amp:.6f}\n")
                for i, (sf, sa, _) in enumerate(self._secondary_peaks):
                    f.write(f"Secondary_{i+1} = {sf:.6f} Hz (amp={sa:.6f})\n")

            # Figures
            for ext in ["png", "pdf"]:
                fig_path = os.path.join(out_dir, f"hv_forward_curve.{ext}")
                self._plot.figure.savefig(fig_path, dpi=150, bbox_inches="tight")

            self._results_text.append(f"\n✓ Results saved to: {out_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    # ── Engine Settings Dialog ──────────────────────────────────
    def _open_engine_settings(self):
        from ..dialogs.engine_settings_dialog import EngineSettingsDialog
        dlg = EngineSettingsDialog(self._engine_settings_config, parent=self)
        if dlg.exec_() == EngineSettingsDialog.Accepted:
            self._engine_settings_config = dlg.get_config()
            if self._main_window:
                self._main_window.update_config({"engine_settings": self._engine_settings_config})

    def _on_fwd_engine_changed(self, engine_name):
        desc = ENGINE_DESCRIPTIONS.get(engine_name, "")
        if hasattr(self, '_fwd_engine_desc'):
            self._fwd_engine_desc.setText(desc)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_edit.setText(d)

    # ── Config propagation ──────────────────────────────────────
    def apply_config(self, cfg):
        """Apply settings from main window config."""
        hv = cfg.get("hv_forward", {})
        if "fmin" in hv: self._fmin.setValue(hv["fmin"])
        if "fmax" in hv: self._fmax.setValue(hv["fmax"])
        if "nf" in hv: self._nf.setValue(hv["nf"])

        engine = cfg.get("engine", {})
        engine_name = engine.get("name", "diffuse_field") if isinstance(engine, dict) else str(engine)
        idx = self._engine_combo.findText(engine_name)
        if idx >= 0:
            self._engine_combo.setCurrentIndex(idx)

        es = cfg.get("engine_settings", {})
        if es:
            self._engine_settings_config = dict(es)
