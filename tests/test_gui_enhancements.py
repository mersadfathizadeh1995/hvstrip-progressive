"""
Tests for GUI UX enhancement phases A–L.

Covers:
- Engine dropdown propagation
- DualResonanceSettingsDialog
- BatchSettingsDialog
- FigureWizardDialog (headless, per-figure panels)
- Reporter draw-on-figure methods (extended kwargs)
- compute_halfspace_display_depth utility
- Vs30/VsAvg core module
- Per-figure settings panels
"""

import os
import sys
import csv
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


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
    """Build a minimal two-step strip directory."""
    strip = tmp_path / "strip"
    strip.mkdir()
    model_txt = (
        "3\n"
        "10.0  400.0  200.0  1.8\n"
        "20.0  800.0  400.0  2.0\n"
        "0.0  1600.0  800.0  2.2\n"
    )
    freqs = np.linspace(0.5, 20, 200)
    amps_3 = 2.0 + 3.0 * np.exp(-0.5 * ((freqs - 5.0) / 0.8) ** 2)
    _make_step_folder(strip, 0, 3, model_txt, freqs, amps_3)

    model_2 = "2\n20.0  800.0  400.0  2.0\n0.0  1600.0  800.0  2.2\n"
    amps_2 = 1.5 + 2.5 * np.exp(-0.5 * ((freqs - 8.0) / 1.0) ** 2)
    _make_step_folder(strip, 1, 2, model_2, freqs, amps_2)
    return strip


# ---------------------------------------------------------------------------
# Phase A: Engine dropdown
# ---------------------------------------------------------------------------

class TestEngineDropdown:

    def test_forward_worker_accepts_engine(self):
        from hvstrip_progressive.gui.pages.forward_modeling_page import ForwardWorker
        w = ForwardWorker("dummy.txt", {"fmin": 0.5}, engine_name="diffuse_field")
        assert w.engine_name == "diffuse_field"

    def test_compute_worker_accepts_engine(self):
        from hvstrip_progressive.gui.dialogs.multi_profile_dialog import _ComputeWorker
        w = _ComputeWorker(0, "dummy.txt", {}, engine_name="diffuse_field")
        assert w._engine_name == "diffuse_field"


# ---------------------------------------------------------------------------
# Phase B: Dual-Resonance Settings Dialog
# ---------------------------------------------------------------------------

class TestDualResonanceSettingsDialog:

    def test_dialog_defaults(self, qapp):
        from hvstrip_progressive.gui.dialogs.dual_resonance_settings_dialog import (
            DualResonanceSettingsDialog,
        )
        dlg = DualResonanceSettingsDialog(ratio=1.5, shift=0.4)
        vals = dlg.get_values()
        assert vals["separation_ratio_threshold"] == pytest.approx(1.5)
        assert vals["separation_shift_threshold"] == pytest.approx(0.4)

    def test_dialog_default_values(self, qapp):
        from hvstrip_progressive.gui.dialogs.dual_resonance_settings_dialog import (
            DualResonanceSettingsDialog,
        )
        dlg = DualResonanceSettingsDialog()
        vals = dlg.get_values()
        assert vals["separation_ratio_threshold"] == pytest.approx(1.2)
        assert vals["separation_shift_threshold"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Phase C: Batch Settings Dialog
# ---------------------------------------------------------------------------

class TestBatchSettingsDialog:

    def test_dialog_default_config(self, qapp):
        from hvstrip_progressive.gui.dialogs.batch_settings_dialog import (
            BatchSettingsDialog,
        )
        dlg = BatchSettingsDialog()
        cfg = dlg.get_config()
        assert "hv_forward" in cfg
        assert "engine_name" in cfg
        assert "generate_report" in cfg
        assert "dual_resonance" in cfg
        assert "figure" in cfg
        assert "peak_detection" in cfg

    def test_dialog_preserves_custom_defaults(self, qapp):
        from hvstrip_progressive.gui.dialogs.batch_settings_dialog import (
            BatchSettingsDialog,
        )
        defaults = {
            "fmin": 1.0,
            "fmax": 15.0,
            "nf": 300,
            "engine": "diffuse_field",
            "generate_report": False,
            "dual_resonance": {
                "enable": True,
                "separation_ratio_threshold": 2.0,
                "separation_shift_threshold": 0.5,
            },
        }
        dlg = BatchSettingsDialog(defaults=defaults)
        cfg = dlg.get_config()
        assert cfg["hv_forward"]["fmin"] == pytest.approx(1.0)
        assert cfg["hv_forward"]["fmax"] == pytest.approx(15.0)
        assert cfg["hv_forward"]["nf"] == 300
        assert cfg["generate_report"] is False
        assert cfg["dual_resonance"]["enable"] is True
        assert cfg["dual_resonance"]["separation_ratio_threshold"] == pytest.approx(2.0)

    def test_figure_defaults_present(self, qapp):
        from hvstrip_progressive.gui.dialogs.batch_settings_dialog import (
            BatchSettingsDialog,
        )
        dlg = BatchSettingsDialog()
        cfg = dlg.get_config()
        fig = cfg["figure"]
        assert fig["dpi"] == 300
        assert fig["save_png"] is True
        assert "palette" in fig


# ---------------------------------------------------------------------------
# Phase D: Reporter draw-on-figure methods
# ---------------------------------------------------------------------------

class TestReporterDrawOnFigure:

    def test_draw_hv_overlay(self, strip_dir):
        from hvstrip_progressive.core.report_generator import (
            ProgressiveStrippingReporter,
        )
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure(figsize=(10, 6))
        ok = reporter.draw_hv_overlay_on_figure(fig, cmap="viridis", linewidth=1.5)
        assert ok is True
        assert len(fig.axes) >= 1

    def test_draw_peak_evolution(self, strip_dir):
        from hvstrip_progressive.core.report_generator import (
            ProgressiveStrippingReporter,
        )
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure(figsize=(10, 8))
        ok = reporter.draw_peak_evolution_on_figure(fig)
        assert ok is True
        assert len(fig.axes) == 3

    def test_draw_interface_analysis(self, strip_dir):
        from hvstrip_progressive.core.report_generator import (
            ProgressiveStrippingReporter,
        )
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure(figsize=(12, 5))
        ok = reporter.draw_interface_analysis_on_figure(fig)
        assert ok is True
        assert len(fig.axes) == 2

    def test_draw_waterfall(self, strip_dir):
        from hvstrip_progressive.core.report_generator import (
            ProgressiveStrippingReporter,
        )
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure(figsize=(12, 8))
        ok = reporter.draw_waterfall_on_figure(fig, offset_factor=2.0)
        assert ok is True

    def test_draw_publication(self, strip_dir):
        from hvstrip_progressive.core.report_generator import (
            ProgressiveStrippingReporter,
        )
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure(figsize=(10, 8))
        ok = reporter.draw_publication_on_figure(fig)
        assert ok is True
        assert len(fig.axes) >= 4

    def test_draw_overlay_kwargs(self, strip_dir):
        from hvstrip_progressive.core.report_generator import (
            ProgressiveStrippingReporter,
        )
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure()
        ok = reporter.draw_hv_overlay_on_figure(
            fig, log_x=False, grid=False, alpha=0.5, font_size=8,
        )
        assert ok is True


# ---------------------------------------------------------------------------
# Phase D: Figure Wizard Dialog (headless, no strip_dir available → graceful)
# ---------------------------------------------------------------------------

class TestFigureWizardDialog:

    def test_wizard_creates(self, qapp, strip_dir, tmp_path):
        from hvstrip_progressive.gui.dialogs.figure_wizard_dialog import (
            FigureWizardDialog,
        )
        output = tmp_path / "wizard_out"
        output.mkdir()
        wizard = FigureWizardDialog(
            strip_dir=str(strip_dir),
            output_dir=str(output),
            has_dual_resonance=False,
        )
        assert wizard.fig_list.count() >= 4  # overlay, peak, interface, waterfall, publication
        assert wizard._reporter is not None

    def test_wizard_with_dr(self, qapp, strip_dir, tmp_path):
        from hvstrip_progressive.gui.dialogs.figure_wizard_dialog import (
            FigureWizardDialog,
        )
        output = tmp_path / "wizard_out2"
        output.mkdir()
        wizard = FigureWizardDialog(
            strip_dir=str(strip_dir),
            output_dir=str(output),
            has_dual_resonance=True,
        )
        # Should have 6 entries (5 base + dual_resonance)
        assert wizard.fig_list.count() == 6

    def test_wizard_draws_without_error(self, qapp, strip_dir, tmp_path):
        from hvstrip_progressive.gui.dialogs.figure_wizard_dialog import (
            FigureWizardDialog,
        )
        output = tmp_path / "wizard_out3"
        output.mkdir()
        wizard = FigureWizardDialog(
            strip_dir=str(strip_dir),
            output_dir=str(output),
        )
        # Draw each figure type
        for row in range(wizard.fig_list.count()):
            wizard.fig_list.setCurrentRow(row)
            # Should not raise

    def test_wizard_has_stacked_panels(self, qapp, strip_dir, tmp_path):
        from hvstrip_progressive.gui.dialogs.figure_wizard_dialog import (
            FigureWizardDialog,
        )
        output = tmp_path / "wizard_stack"
        output.mkdir()
        wizard = FigureWizardDialog(
            strip_dir=str(strip_dir),
            output_dir=str(output),
        )
        assert wizard.settings_stack.count() == wizard.fig_list.count()
        # Each panel should have get_kwargs
        for key, panel in wizard._panels.items():
            if hasattr(panel, "get_kwargs"):
                kw = panel.get_kwargs()
                assert isinstance(kw, dict)

    def test_wizard_panel_switching(self, qapp, strip_dir, tmp_path):
        from hvstrip_progressive.gui.dialogs.figure_wizard_dialog import (
            FigureWizardDialog,
        )
        output = tmp_path / "wizard_switch"
        output.mkdir()
        wizard = FigureWizardDialog(
            strip_dir=str(strip_dir),
            output_dir=str(output),
        )
        for row in range(wizard.fig_list.count()):
            wizard.fig_list.setCurrentRow(row)
            assert wizard.settings_stack.currentIndex() == row


# ---------------------------------------------------------------------------
# Phase G: compute_halfspace_display_depth
# ---------------------------------------------------------------------------

class TestHalfspaceDisplayDepth:

    def test_proportional_40m(self):
        from hvstrip_progressive.core.soil_profile import compute_halfspace_display_depth
        result = compute_halfspace_display_depth(40.0)
        assert result == pytest.approx(10.0)  # 40 * 0.25

    def test_min_extension(self):
        from hvstrip_progressive.core.soil_profile import compute_halfspace_display_depth
        result = compute_halfspace_display_depth(10.0)
        assert result == pytest.approx(5.0)  # min_extension kicks in (10*0.25=2.5 < 5)

    def test_max_cap(self):
        from hvstrip_progressive.core.soil_profile import compute_halfspace_display_depth
        result = compute_halfspace_display_depth(200.0)
        assert result == pytest.approx(50.0)  # 200*0.25=50, cap=200*0.5=100 → 50

    def test_custom_ratio(self):
        from hvstrip_progressive.core.soil_profile import compute_halfspace_display_depth
        result = compute_halfspace_display_depth(40.0, hs_ratio=0.5)
        assert result == pytest.approx(20.0)  # 40 * 0.5, cap=20

    def test_custom_max_extension(self):
        from hvstrip_progressive.core.soil_profile import compute_halfspace_display_depth
        result = compute_halfspace_display_depth(100.0, max_extension=15.0)
        assert result == pytest.approx(15.0)  # capped

    def test_zero_depth(self):
        from hvstrip_progressive.core.soil_profile import compute_halfspace_display_depth
        result = compute_halfspace_display_depth(0.0)
        assert result >= 20.0  # fallback


# ---------------------------------------------------------------------------
# Phase K: Vs30 / VsAvg
# ---------------------------------------------------------------------------

class TestVsAverage:

    def test_vs30_simple_layers(self):
        from hvstrip_progressive.core.vs_average import compute_vs_average
        layers = [(10.0, 200.0), (20.0, 400.0), (0.0, 800.0)]
        res = compute_vs_average(layers, target_depth=30.0)
        assert res.vs_avg > 0
        assert res.target_depth == 30.0
        assert res.actual_depth == 30.0
        assert res.extrapolated is False

    def test_vs30_needs_halfspace(self):
        from hvstrip_progressive.core.vs_average import compute_vs_average
        layers = [(10.0, 200.0), (0.0, 400.0)]
        res = compute_vs_average(layers, target_depth=30.0)
        assert res.vs_avg > 0
        assert res.extrapolated is True
        assert len(res.layer_contributions) == 2

    def test_vs30_no_halfspace_short(self):
        from hvstrip_progressive.core.vs_average import compute_vs_average
        layers = [(10.0, 200.0)]
        res = compute_vs_average(layers, target_depth=30.0, use_halfspace=False)
        # Only 10m of 30m available
        assert res.vs_avg > 0
        assert res.extrapolated is False

    def test_vs_weighted(self):
        from hvstrip_progressive.core.vs_average import compute_vs_weighted
        layers = [(10.0, 200.0), (20.0, 400.0), (0.0, 800.0)]
        result = compute_vs_weighted(layers)
        expected = (10 * 200 + 20 * 400) / 30.0
        assert result == pytest.approx(expected, rel=0.01)

    def test_from_model_file(self, tmp_path):
        from hvstrip_progressive.core.vs_average import vs_average_from_model_file
        model = tmp_path / "model.txt"
        model.write_text(
            "3\n10.0  400.0  200.0  1.8\n20.0  800.0  400.0  2.0\n0.0  1600.0  800.0  2.2\n"
        )
        res = vs_average_from_model_file(str(model), target_depth=30.0)
        assert res.vs_avg > 0
        assert res.actual_depth == 30.0

    def test_from_profile(self):
        from hvstrip_progressive.core.soil_profile import SoilProfile, Layer
        from hvstrip_progressive.core.vs_average import vs_average_from_profile
        p = SoilProfile()
        p.add_layer(Layer(thickness=15.0, vs=200.0))
        p.add_layer(Layer(thickness=15.0, vs=400.0))
        p.add_layer(Layer(thickness=0.0, vs=800.0, is_halfspace=True))
        res = vs_average_from_profile(p, target_depth=30.0)
        assert res.vs_avg > 0
        assert res.extrapolated is False

    def test_custom_depth(self):
        from hvstrip_progressive.core.vs_average import compute_vs_average
        layers = [(5.0, 150.0), (5.0, 300.0), (0.0, 600.0)]
        res = compute_vs_average(layers, target_depth=10.0)
        expected = 10.0 / (5.0 / 150.0 + 5.0 / 300.0)
        assert res.vs_avg == pytest.approx(expected, rel=0.01)
        assert res.extrapolated is False


# ---------------------------------------------------------------------------
# Phase H: Per-figure settings panels
# ---------------------------------------------------------------------------

class TestFigureSettingsPanels:

    def test_all_panels_instantiate(self, qapp):
        from hvstrip_progressive.gui.dialogs.figure_settings_panels import PANEL_REGISTRY
        for key, cls in PANEL_REGISTRY.items():
            panel = cls()
            kw = panel.get_kwargs()
            assert isinstance(kw, dict), f"{key} panel get_kwargs failed"

    def test_hv_overlay_defaults(self, qapp):
        from hvstrip_progressive.gui.dialogs.figure_settings_panels import HVOverlaySettingsPanel
        p = HVOverlaySettingsPanel()
        kw = p.get_kwargs()
        assert kw["log_x"] is True
        assert kw["grid"] is True
        assert kw["show_peaks"] is True
        assert "cmap" in kw
        assert "linewidth" in kw

    def test_waterfall_defaults(self, qapp):
        from hvstrip_progressive.gui.dialogs.figure_settings_panels import WaterfallSettingsPanel
        p = WaterfallSettingsPanel()
        kw = p.get_kwargs()
        assert "offset_factor" in kw
        assert kw["normalize"] is False

    def test_dual_resonance_annotation_offsets(self, qapp):
        from hvstrip_progressive.gui.dialogs.figure_settings_panels import DualResonanceSettingsPanel
        p = DualResonanceSettingsPanel()
        kw = p.get_kwargs()
        assert kw["f0_offset"] == (0.0, 0.0)
        assert kw["f1_offset"] == (0.0, 0.0)
        assert kw["show_stripped"] is True
        assert "hs_ratio" in kw

    def test_publication_table_font(self, qapp):
        from hvstrip_progressive.gui.dialogs.figure_settings_panels import PublicationSettingsPanel
        p = PublicationSettingsPanel()
        kw = p.get_kwargs()
        assert "table_font" in kw


# ---------------------------------------------------------------------------
# Phase I: Extended draw kwargs
# ---------------------------------------------------------------------------

class TestExtendedDrawKwargs:

    def test_overlay_show_peaks_false(self, strip_dir):
        from hvstrip_progressive.core.report_generator import ProgressiveStrippingReporter
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure()
        ok = reporter.draw_hv_overlay_on_figure(fig, show_peaks=False, marker_size=12)
        assert ok is True

    def test_peak_evolution_no_fill(self, strip_dir):
        from hvstrip_progressive.core.report_generator import ProgressiveStrippingReporter
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure()
        ok = reporter.draw_peak_evolution_on_figure(fig, show_fill=False, marker_size=10)
        assert ok is True

    def test_interface_annot_font(self, strip_dir):
        from hvstrip_progressive.core.report_generator import ProgressiveStrippingReporter
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure()
        ok = reporter.draw_interface_analysis_on_figure(fig, annot_font=8, marker_size=6)
        assert ok is True

    def test_publication_table_font(self, strip_dir):
        from hvstrip_progressive.core.report_generator import ProgressiveStrippingReporter
        reporter = ProgressiveStrippingReporter(str(strip_dir))
        fig = Figure()
        ok = reporter.draw_publication_on_figure(fig, table_font=7)
        assert ok is True
