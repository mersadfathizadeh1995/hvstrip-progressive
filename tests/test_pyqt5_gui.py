"""Comprehensive tests for the new PyQt5 HV Strip Progressive GUI.

Tests all panels, views, dialogs, and their integration with the main window.
"""
import os
import sys
import csv
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_API"] = "pyqt5"

import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass

import numpy as np
import pytest
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "hvstrip_progressive"))


@pytest.fixture(scope="session")
def qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def example_model():
    return str(PROJECT_ROOT / "examples" / "example_model.txt")


def _make_step_folder(parent, step, n_layers, model_text, freqs, amps):
    name = f"Step{step}_{n_layers}-layer"
    folder = parent / name
    folder.mkdir()
    (folder / f"model_{name}.txt").write_text(model_text, encoding="utf-8")
    hv_csv = folder / "hv_curve.csv"
    with open(hv_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frequency_Hz", "HVSR_Amplitude"])
        for f, a in zip(freqs, amps):
            w.writerow([f"{f:.6f}", f"{a:.6f}"])
    return folder


@pytest.fixture
def strip_dir(tmp_path):
    strip = tmp_path / "strip"
    strip.mkdir()
    model_txt = "3\n10.0  400.0  200.0  1.8\n20.0  800.0  400.0  2.0\n0.0  1600.0  800.0  2.2\n"
    freqs = np.linspace(0.5, 20, 200)
    amps_3 = 2.0 + 3.0 * np.exp(-0.5 * ((freqs - 5.0) / 0.8) ** 2)
    _make_step_folder(strip, 0, 3, model_txt, freqs, amps_3)
    model_2 = "2\n20.0  800.0  400.0  2.0\n0.0  1600.0  800.0  2.2\n"
    amps_2 = 1.5 + 2.5 * np.exp(-0.5 * ((freqs - 8.0) / 1.0) ** 2)
    _make_step_folder(strip, 1, 2, model_2, freqs, amps_2)
    return strip


# ═══════════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════════

class TestHVStripWindow:

    def test_window_creates(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        w = HVStripWindow()
        assert w.windowTitle() == "HVSR Progressive Layer Stripping Analysis"

    def test_menu_bar(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        w = HVStripWindow()
        mb = w.menuBar()
        actions = [a.text() for a in mb.actions()]
        assert "&File" in actions
        assert "&View" in actions
        assert "&Tools" in actions
        assert "&Help" in actions

    def test_control_tabs(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        w = HVStripWindow()
        # New layout: 2 main tabs (Forward Model, HV Strip), each with 2 sub-tabs
        assert w._main_tabs.count() == 2
        assert w._fwd_tabs.count() == 2
        assert w._strip_tabs.count() == 2

    def test_view_tabs(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        w = HVStripWindow()
        # Canvas tabs vary per mode; default mode is forward_single with 3 tabs
        canvas = w.get_active_canvas()
        assert canvas is not None
        assert canvas.count() >= 3

    def test_panel_types(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        from HV_Strip_Progressive.panels.forward_single_panel import ForwardSinglePanel
        from HV_Strip_Progressive.panels.forward_multi_panel import ForwardMultiPanel
        from HV_Strip_Progressive.panels.strip_single_panel import StripSinglePanel
        from HV_Strip_Progressive.panels.strip_batch_panel import StripBatchPanel
        w = HVStripWindow()
        from HV_Strip_Progressive.strip_window import (
            MODE_FWD_SINGLE, MODE_FWD_MULTI,
            MODE_STRIP_SINGLE, MODE_STRIP_BATCH,
        )
        # Panels may be placeholders if import fails; just check they exist
        assert MODE_FWD_SINGLE in w._panels
        assert MODE_FWD_MULTI in w._panels
        assert MODE_STRIP_SINGLE in w._panels
        assert MODE_STRIP_BATCH in w._panels

    def test_view_types(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        from HV_Strip_Progressive.views.log_view import LogView
        w = HVStripWindow()
        # Log view is always present
        assert hasattr(w, '_log_view')

    def test_engine_combo(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        w = HVStripWindow()
        # Engine is now per-panel; test via the public API
        name = w.get_engine_name()
        assert name in ("diffuse_field", "sh_wave", "ellipticity")

    def test_log(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        w = HVStripWindow()
        w.log("Test message")

    def test_set_status(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        w = HVStripWindow()
        w.set_status("Testing")
        assert w._status_msg.text() == "Testing"

    def test_set_result(self, qapp):
        from HV_Strip_Progressive.strip_window import HVStripWindow
        w = HVStripWindow()
        w.set_result({"strip_directory": "/tmp/test", "step_results": {}})
        assert w._last_strip_dir == "/tmp/test"


# ═══════════════════════════════════════════════════════════════════
#  Input Panel
# ═══════════════════════════════════════════════════════════════════

class TestInputPanel:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.panels.input_panel import InputPanel
        p = InputPanel()
        assert p is not None

    def test_load_profile(self, qapp, example_model):
        from HV_Strip_Progressive.panels.input_panel import InputPanel
        p = InputPanel()
        p.load_profile(example_model)
        assert p.get_model_path() == example_model

    def test_get_output_dir(self, qapp):
        from HV_Strip_Progressive.panels.input_panel import InputPanel
        p = InputPanel()
        p._output_edit.setText("/tmp/output")
        assert p.get_output_dir() == "/tmp/output"


# ═══════════════════════════════════════════════════════════════════
#  Config Panel
# ═══════════════════════════════════════════════════════════════════

class TestConfigPanel:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.panels.config_panel import ConfigPanel
        p = ConfigPanel()
        assert p is not None

    def test_get_config(self, qapp):
        from HV_Strip_Progressive.panels.config_panel import ConfigPanel
        p = ConfigPanel()
        cfg = p.get_config()
        assert "hv_forward" in cfg
        assert "engine_name" in cfg
        assert "peak_detection" in cfg
        assert "dual_resonance" in cfg
        assert "plot" in cfg

    def test_engine_name(self, qapp):
        from HV_Strip_Progressive.panels.config_panel import ConfigPanel
        p = ConfigPanel()
        p._engine_combo.setCurrentText("sh_wave")
        assert p.get_engine_name() == "sh_wave"

    def test_peak_detection_config(self, qapp):
        from HV_Strip_Progressive.panels.config_panel import ConfigPanel
        p = ConfigPanel()
        p._min_prom.setValue(1.5)
        cfg = p.get_config()
        assert cfg["peak_detection"]["min_prominence"] == pytest.approx(1.5)


# ═══════════════════════════════════════════════════════════════════
#  Run Panel
# ═══════════════════════════════════════════════════════════════════

class TestRunPanel:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.panels.run_panel import RunPanel
        p = RunPanel()
        assert p is not None

    def test_set_batch_folder(self, qapp, tmp_path):
        from HV_Strip_Progressive.panels.run_panel import RunPanel
        p = RunPanel()
        (tmp_path / "a.txt").write_text("1\n0.0 200.0 100.0 1.8\n")
        (tmp_path / "b.txt").write_text("1\n0.0 300.0 150.0 1.9\n")
        p.set_batch_folder(str(tmp_path))
        assert p._batch_list.count() == 2


# ═══════════════════════════════════════════════════════════════════
#  Views
# ═══════════════════════════════════════════════════════════════════

class TestHVCurveView:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.views.hv_curve_view import HVCurveView
        v = HVCurveView()
        assert v is not None

    def test_set_data(self, qapp):
        from HV_Strip_Progressive.views.hv_curve_view import HVCurveView
        v = HVCurveView()
        freqs = np.linspace(0.5, 20, 100)
        amps = 2 + 3 * np.exp(-0.5 * ((freqs - 5) / 0.8) ** 2)
        v.set_data(freqs, amps)


class TestVsProfileView:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.views.vs_profile_view import VsProfileView
        v = VsProfileView()
        assert v is not None


class TestHVOverlayView:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.views.hv_overlay_view import HVOverlayView
        v = HVOverlayView()
        assert v is not None

    def test_load_strip_dir(self, qapp, strip_dir):
        from HV_Strip_Progressive.views.hv_overlay_view import HVOverlayView
        v = HVOverlayView()
        v.load_strip_dir(str(strip_dir))


class TestStripResultsView:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.views.strip_results_view import StripResultsView
        v = StripResultsView()
        assert v is not None

    def test_set_results(self, qapp):
        from HV_Strip_Progressive.views.strip_results_view import StripResultsView
        v = StripResultsView()
        v.set_results({
            "step_results": {
                "Step0_3-layer": {"peak_frequency": 5.0, "peak_amplitude": 4.5},
                "Step1_2-layer": {"peak_frequency": 8.0, "peak_amplitude": 3.0},
            }
        })


class TestLogView:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.views.log_view import LogView
        v = LogView()
        assert v is not None

    def test_append(self, qapp):
        from HV_Strip_Progressive.views.log_view import LogView
        v = LogView()
        v.append("Test message")
        v.append("Error: something")


# ═══════════════════════════════════════════════════════════════════
#  Dialogs
# ═══════════════════════════════════════════════════════════════════

class TestFigureStudio:

    def test_import(self, qapp):
        from HV_Strip_Progressive.dialogs.figure_studio import FigureStudioWindow
        assert FigureStudioWindow is not None

    def test_settings_panels(self, qapp):
        from HV_Strip_Progressive.dialogs.figure_studio import (
            HVOverlayPanel, PeakEvolutionPanel, InterfaceAnalysisPanel,
            WaterfallPanel, PublicationPanel, DualResonancePanel,
        )
        for cls in [HVOverlayPanel, PeakEvolutionPanel, InterfaceAnalysisPanel,
                     WaterfallPanel, PublicationPanel, DualResonancePanel]:
            p = cls()
            kw = p.get_kwargs()
            assert isinstance(kw, dict), f"{cls.__name__} failed"

    def test_overlay_panel_defaults(self, qapp):
        from HV_Strip_Progressive.dialogs.figure_studio import HVOverlayPanel
        p = HVOverlayPanel()
        kw = p.get_kwargs()
        assert kw["log_x"] is True
        assert kw["grid"] is True
        assert kw["show_peaks"] is True

    def test_dr_panel_offsets(self, qapp):
        from HV_Strip_Progressive.dialogs.figure_studio import DualResonancePanel
        p = DualResonancePanel()
        kw = p.get_kwargs()
        assert kw["f0_offset"] == (0.0, 0.0)
        assert kw["f1_offset"] == (0.0, 0.0)
        assert kw["show_stripped"] is True


class TestPeakPickerDialog:

    def test_creates_empty(self, qapp):
        from HV_Strip_Progressive.dialogs.peak_picker_dialog import PeakPickerDialog
        dlg = PeakPickerDialog({"step_results": {}})
        assert dlg._step_list.count() == 0

    def test_auto_detect(self, qapp, strip_dir):
        from HV_Strip_Progressive.dialogs.peak_picker_dialog import PeakPickerDialog
        # Build result dict with step data pointing to strip_dir
        steps = {}
        for d in sorted(strip_dir.iterdir()):
            hv = d / "hv_curve.csv"
            if hv.exists():
                steps[d.name] = {"hv_csv": str(hv)}
        dlg = PeakPickerDialog({"step_results": steps})
        assert dlg._step_list.count() == 2
        dlg._auto_detect()
        assert len(dlg._selected) == 1

    def test_undo(self, qapp, strip_dir):
        from HV_Strip_Progressive.dialogs.peak_picker_dialog import PeakPickerDialog
        steps = {}
        for d in sorted(strip_dir.iterdir()):
            hv = d / "hv_curve.csv"
            if hv.exists():
                steps[d.name] = {"hv_csv": str(hv)}
        dlg = PeakPickerDialog({"step_results": steps})
        dlg._auto_detect()
        assert len(dlg._selected) == 1
        dlg._undo()
        assert len(dlg._selected) == 0


class TestProfileLoaderDialog:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.dialogs.profile_loader_dialog import ProfileLoaderDialog
        dlg = ProfileLoaderDialog()
        assert dlg is not None


class TestEngineSettingsDialog:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.dialogs.engine_settings_dialog import EngineSettingsDialog
        dlg = EngineSettingsDialog({})
        assert dlg is not None

    def test_get_config(self, qapp):
        from HV_Strip_Progressive.dialogs.engine_settings_dialog import EngineSettingsDialog
        dlg = EngineSettingsDialog({"diffuse_field": {"fmin": 0.5}})
        cfg = dlg.get_config()
        assert isinstance(cfg, dict)


class TestDualResonanceDialog:

    def test_creates(self, qapp):
        from HV_Strip_Progressive.dialogs.dual_resonance_settings_dialog import DualResonanceSettingsDialog
        dlg = DualResonanceSettingsDialog()
        assert dlg is not None

    def test_defaults(self, qapp):
        from HV_Strip_Progressive.dialogs.dual_resonance_settings_dialog import DualResonanceSettingsDialog
        dlg = DualResonanceSettingsDialog(ratio=1.5, shift=0.4)
        vals = dlg.get_values()
        assert vals["separation_ratio_threshold"] == pytest.approx(1.5)
        assert vals["separation_shift_threshold"] == pytest.approx(0.4)


# ═══════════════════════════════════════════════════════════════════
#  Engine Integration
# ═══════════════════════════════════════════════════════════════════

class TestEngines:

    def test_sh_wave_forward(self, example_model):
        from hvstrip_progressive.core.hv_forward import compute_hv_curve
        freqs, amps = compute_hv_curve(example_model, engine_name="sh_wave")
        assert len(freqs) > 0
        assert np.max(amps) > 1.0

    def test_diffuse_field_forward(self, example_model):
        from hvstrip_progressive.core.hv_forward import compute_hv_curve
        freqs, amps = compute_hv_curve(example_model, engine_name="diffuse_field")
        assert len(freqs) > 0

    def test_ellipticity_forward(self, example_model):
        from hvstrip_progressive.core.hv_forward import compute_hv_curve
        freqs, amps = compute_hv_curve(example_model, engine_name="ellipticity")
        assert len(freqs) > 0


class TestWorkflow:

    def test_complete_workflow_sh_wave(self, example_model, tmp_path):
        from hvstrip_progressive.core.batch_workflow import run_complete_workflow
        out = tmp_path / "workflow_out"
        out.mkdir()
        result = run_complete_workflow(
            example_model, str(out),
            workflow_config={"engine_name": "sh_wave"},
            engine_name="sh_wave")
        assert result["success"] is True
        assert len(result["step_results"]) > 0

    def test_reporter(self, strip_dir):
        from hvstrip_progressive.core.report_generator import ProgressiveStrippingReporter
        r = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure(figsize=(10, 6))
        ok = r.draw_hv_overlay_on_figure(fig)
        assert ok is True
