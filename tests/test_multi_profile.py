"""
Headless test for the Multiple Profiles workflow.

Loads 10 Excel profiles, runs forward modeling (0.1–10 Hz, 500 pts),
auto-detects peaks, and saves results to an output directory.
"""

import os
import sys
import shutil
from pathlib import Path

# Offscreen rendering — must be set before any Qt import
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest

PROFILES_DIR = Path(
    r"D:\Runs\Readfield\Proccessed\Dinver_3\Redfield_extra\Final_table\profiles_Input"
)
OUTPUT_DIR = PROFILES_DIR / "output"

# Project root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture(autouse=True)
def clean_output():
    """Clean output directory before each test."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield


class TestExcelParser:
    """Test SoilProfile.from_excel_file with real data."""

    def test_parse_all_profiles(self):
        from hvstrip_progressive.core.soil_profile import SoilProfile

        xlsx_files = sorted(PROFILES_DIR.glob("*.xlsx"))
        assert len(xlsx_files) == 10, f"Expected 10 xlsx files, got {len(xlsx_files)}"

        for f in xlsx_files:
            profile = SoilProfile.from_excel_file(str(f))
            assert len(profile.layers) > 0, f"{f.name}: no layers"
            assert profile.layers[-1].is_halfspace, f"{f.name}: last layer not halfspace"
            for layer in profile.layers:
                assert layer.vs > 0, f"{f.name}: Vs must be > 0"

    def test_halfspace_detection(self):
        from hvstrip_progressive.core.soil_profile import SoilProfile

        profile = SoilProfile.from_excel_file(str(PROFILES_DIR / "Profile_1.xlsx"))
        hs = profile.layers[-1]
        assert hs.is_halfspace
        assert hs.thickness == 0.0
        assert hs.vs > 0


class TestHVComputation:
    """Test HV forward computation from Excel profiles."""

    def test_compute_single_profile(self):
        import tempfile
        from hvstrip_progressive.core.soil_profile import SoilProfile
        from hvstrip_progressive.core.hv_forward import compute_hv_curve

        profile = SoilProfile.from_excel_file(str(PROFILES_DIR / "Profile_1.xlsx"))

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="test_fwd_"
        )
        tmp.write(profile.to_hvf_format())
        tmp.close()

        try:
            freqs, amps = compute_hv_curve(
                tmp.name, {"fmin": 0.1, "fmax": 10.0, "nf": 500}
            )
            freqs = np.asarray(freqs)
            amps = np.asarray(amps)

            assert len(freqs) == 500
            assert freqs[0] >= 0.09  # ~0.1 Hz
            assert freqs[-1] <= 10.1  # ~10 Hz
            assert np.all(amps > 0), "Amplitudes must be positive"

            peak_idx = np.argmax(amps)
            print(f"Profile_1 peak: {freqs[peak_idx]:.3f} Hz, amp={amps[peak_idx]:.2f}")
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_compute_all_profiles(self):
        """Run all 10 profiles and verify each produces valid HV curves."""
        import tempfile
        from hvstrip_progressive.core.soil_profile import SoilProfile
        from hvstrip_progressive.core.hv_forward import compute_hv_curve

        config = {"fmin": 0.1, "fmax": 10.0, "nf": 500}
        xlsx_files = sorted(PROFILES_DIR.glob("*.xlsx"))

        for f in xlsx_files:
            profile = SoilProfile.from_excel_file(str(f))
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, prefix="test_fwd_"
            )
            tmp.write(profile.to_hvf_format())
            tmp.close()

            try:
                freqs, amps = compute_hv_curve(tmp.name, config)
                freqs = np.asarray(freqs)
                amps = np.asarray(amps)
                assert len(freqs) == 500, f"{f.name}: wrong freq count"
                assert np.all(np.isfinite(amps)), f"{f.name}: non-finite amplitudes"
                peak_idx = np.argmax(amps)
                print(f"  {f.stem}: peak={freqs[peak_idx]:.3f} Hz, A={amps[peak_idx]:.2f}")
            finally:
                Path(tmp.name).unlink(missing_ok=True)


class TestMultiProfileDialog:
    """Test the MultiProfilePickerDialog in headless mode."""

    def test_dialog_creates(self, qapp):
        from hvstrip_progressive.core.soil_profile import SoilProfile
        from hvstrip_progressive.gui.dialogs.multi_profile_dialog import (
            MultiProfilePickerDialog, FigureSettings,
        )

        xlsx_files = sorted(PROFILES_DIR.glob("*.xlsx"))[:2]  # just 2 for speed
        profiles = []
        for f in xlsx_files:
            prof = SoilProfile.from_excel_file(str(f))
            profiles.append((f.stem, prof))

        freq_config = {"fmin": 0.1, "fmax": 10.0, "nf": 500}
        fig_settings = FigureSettings()

        dialog = MultiProfilePickerDialog(profiles, freq_config, fig_settings)

        # Wait for first computation to finish
        qapp.processEvents()
        import time
        time.sleep(3)
        qapp.processEvents()

        # Check first profile computed
        r = dialog._results[0]
        assert r.computed, "First profile should be computed"
        assert r.f0 is not None, "f0 should be auto-detected"
        assert r.freqs is not None

        print(f"  {r.name}: f0={r.f0[0]:.3f} Hz")

        # Auto-detect all remaining
        dialog._auto_all_remaining()
        qapp.processEvents()

        for r in dialog._results:
            assert r.computed, f"{r.name}: not computed"
            assert r.f0 is not None, f"{r.name}: no f0"
            print(f"  {r.name}: f0={r.f0[0]:.3f} Hz")

        # Close cleanly
        dialog._wait_for_workers()
        dialog.close()

    def test_median_step(self, qapp):
        """Test the median step: navigate to it, pick peaks, verify results."""
        from hvstrip_progressive.core.soil_profile import SoilProfile
        from hvstrip_progressive.gui.dialogs.multi_profile_dialog import (
            MultiProfilePickerDialog, FigureSettings,
        )

        xlsx_files = sorted(PROFILES_DIR.glob("*.xlsx"))[:3]  # 3 for speed
        profiles = []
        for f in xlsx_files:
            prof = SoilProfile.from_excel_file(str(f))
            profiles.append((f.stem, prof))

        freq_config = {"fmin": 0.1, "fmax": 10.0, "nf": 500}
        fig_settings = FigureSettings()

        dialog = MultiProfilePickerDialog(profiles, freq_config, fig_settings)

        qapp.processEvents()
        import time
        time.sleep(3)
        qapp.processEvents()

        # Auto-detect all individual profiles
        dialog._auto_all_remaining()
        qapp.processEvents()

        # Navigate to median step (index = n_profiles)
        assert dialog._median_idx == 3
        dialog._navigate_to(dialog._median_idx)
        qapp.processEvents()

        assert dialog._is_median_step()
        assert dialog._median_result is not None, "Median should be computed"
        assert dialog._median_result.freqs is not None
        assert dialog._median_result.f0 is not None, "Median f0 auto-detected"
        print(f"  Median f0 (auto): {dialog._median_result.f0[0]:.3f} Hz")

        # Manually override median f0
        old_f0 = dialog._median_result.f0
        dialog._median_result.f0 = (0.35, 4.5, 42)
        assert dialog._median_result.f0 != old_f0
        print(f"  Median f0 (manual): {dialog._median_result.f0[0]:.3f} Hz")

        # Add a secondary peak on median
        dialog._median_result.secondary_peaks.append((1.5, 2.0, 100))
        assert len(dialog._median_result.secondary_peaks) == 1

        # Verify get_median_result works
        mr = dialog.get_median_result()
        assert mr is not None
        assert mr.f0 == (0.35, 4.5, 42)

        # Test skip median
        dialog._median_skipped = True
        assert dialog.get_median_result() is None
        dialog._median_skipped = False  # reset

        # Test settings controls exist and update fig_settings
        dialog.palette_combo.setCurrentText("Bold")
        dialog.alpha_spin.setValue(0.6)
        dialog.ilw_spin.setValue(1.5)
        dialog.mlw_spin.setValue(4.0)
        assert dialog._fig_settings.color_palette == "Bold"
        assert dialog._fig_settings.individual_alpha == 0.6
        assert dialog._fig_settings.individual_linewidth == 1.5
        assert dialog._fig_settings.median_linewidth == 4.0

        # Test generate report signal
        received = []
        dialog.report_requested.connect(lambda r, m: received.append((r, m)))
        dialog.chk_include_median.setChecked(True)
        dialog._generate_report()
        assert len(received) == 1
        assert received[0][1] is not None, "Median should be included"

        # Generate report without median
        dialog.chk_include_median.setChecked(False)
        dialog._generate_report()
        assert len(received) == 2
        assert received[1][1] is None, "Median should be excluded"

        dialog._wait_for_workers()
        dialog.close()

    def test_full_workflow_with_save(self, qapp):
        """Full end-to-end: load all 10, auto-detect + median step, save."""
        from hvstrip_progressive.core.soil_profile import SoilProfile
        from hvstrip_progressive.gui.dialogs.multi_profile_dialog import (
            MultiProfilePickerDialog, FigureSettings,
        )
        from hvstrip_progressive.gui.pages.forward_modeling_page import ForwardModelingPage

        xlsx_files = sorted(PROFILES_DIR.glob("*.xlsx"))
        profiles = []
        for f in xlsx_files:
            prof = SoilProfile.from_excel_file(str(f))
            profiles.append((f.stem, prof))

        freq_config = {"fmin": 0.1, "fmax": 10.0, "nf": 500}
        fig_settings = FigureSettings(
            dpi=150, width=12, height=5, font_size=12,
            log_x=True, grid=True, save_png=True, save_pdf=False,
            color_palette="tab10",
            individual_alpha=0.45,
            individual_linewidth=1.0,
            median_linewidth=3.0,
            show_median=True,
            show_secondary_peaks=True,
        )

        dialog = MultiProfilePickerDialog(profiles, freq_config, fig_settings)

        # Wait for first computation
        qapp.processEvents()
        import time
        time.sleep(3)
        qapp.processEvents()

        # Auto-detect all individual profiles
        dialog._auto_all_remaining()
        qapp.processEvents()

        # Navigate to median step and auto-detect
        dialog._navigate_to(dialog._median_idx)
        qapp.processEvents()

        results = dialog.get_results()
        median_result = dialog.get_median_result()
        assert len(results) == 10
        assert median_result is not None, "Median result should exist"
        assert median_result.f0 is not None, "Median f0 should be set"
        print(f"  Median f0: {median_result.f0[0]:.3f} Hz")

        # Use the save method from MultiProfileTab
        from hvstrip_progressive.gui.pages.multi_profile_tab import MultiProfileTab
        tab = MultiProfileTab(
            get_freq_config=lambda: {"fmin": 0.2, "fmax": 20.0, "nf": 500},
            get_output_dir=lambda: str(OUTPUT_DIR),
        )
        qapp.processEvents()

        outdir = str(OUTPUT_DIR)
        tab._save_results(results, outdir, fig_settings, median_result)

        # Verify per-profile output
        for r in results:
            folder = OUTPUT_DIR / r.name
            assert (folder / "hv_curve.csv").exists(), f"{r.name}: no CSV"
            assert (folder / "peak_info.txt").exists(), f"{r.name}: no peak_info"
            assert (folder / "hv_forward_curve.png").exists(), f"{r.name}: no figure"

        # Verify combined output
        assert (OUTPUT_DIR / "combined_hv_curves.png").exists(), "No combined plot"
        assert (OUTPUT_DIR / "combined_summary.csv").exists(), "No combined CSV"
        assert (OUTPUT_DIR / "median_hv_curve.csv").exists(), "No median CSV"
        assert (OUTPUT_DIR / "median_peak_info.txt").exists(), "No median peak info"

        # Verify median row in summary CSV
        summary = (OUTPUT_DIR / "combined_summary.csv").read_text()
        assert "Median," in summary, "Summary CSV should contain Median row"

        # Print summary
        print(f"\nAll 10 profiles saved to: {OUTPUT_DIR}")
        for r in results:
            print(f"  {r.name}: f0={r.f0[0]:.3f} Hz, A={r.f0[1]:.2f}")
        print(f"  Median: f0={median_result.f0[0]:.3f} Hz")

        # Verify median CSV has data
        med_lines = (OUTPUT_DIR / "median_hv_curve.csv").read_text().strip().split("\n")
        assert len(med_lines) == 501, f"Median CSV: expected 501 lines, got {len(med_lines)}"

        dialog._wait_for_workers()
        dialog.close()
        tab.close()

    def test_load_output_folder(self, qapp):
        """Load an existing output folder and verify profiles + median are parsed."""
        from hvstrip_progressive.gui.dialogs.output_viewer_dialog import (
            load_output_folder, OutputViewerDialog,
        )
        from hvstrip_progressive.gui.dialogs.multi_profile_dialog import FigureSettings

        output_dir = Path(
            r"D:\Research\Narm_Afzar\hvstrip-progressive"
            r"\hvstrip_progressive\Example\profiles_Input\output8"
        )
        if not output_dir.exists():
            pytest.skip(f"Output folder not found: {output_dir}")

        results, median = load_output_folder(output_dir)
        assert len(results) == 10, f"Expected 10 profiles, got {len(results)}"
        for r in results:
            assert r.freqs is not None
            assert r.amps is not None
            assert r.f0 is not None, f"{r.name} missing f0"
            print(f"  {r.name}: f0={r.f0[0]:.3f} Hz, secs={len(r.secondary_peaks)}")

        assert median is not None, "Median should be loaded"
        assert median.freqs is not None
        assert median.f0 is not None, "Median f0 missing"
        print(f"  Median: f0={median.f0[0]:.3f} Hz, secs={len(median.secondary_peaks)}")

        # Open viewer dialog
        s = FigureSettings()
        viewer = OutputViewerDialog(results, median, s, str(output_dir))
        qapp.processEvents()

        # Toggle median off via tree
        from PySide6.QtCore import Qt
        viewer._tree_median.setCheckState(0, Qt.Unchecked)
        qapp.processEvents()
        viewer._tree_median.setCheckState(0, Qt.Checked)
        qapp.processEvents()

        # Toggle individual peaks off
        viewer._tree_prof_f0.setCheckState(0, Qt.Unchecked)
        qapp.processEvents()
        viewer._tree_prof_sec.setCheckState(0, Qt.Unchecked)
        qapp.processEvents()

        # Turn them back on
        viewer._tree_prof_f0.setCheckState(0, Qt.Checked)
        viewer._tree_prof_sec.setCheckState(0, Qt.Checked)
        qapp.processEvents()

        # Toggle median peaks only
        viewer._tree_med_peaks.setCheckState(0, Qt.Unchecked)
        qapp.processEvents()

        # Change palette and peak alpha
        viewer.palette_combo.setCurrentText("Blues")
        viewer.peak_alpha_spin.setValue(0.5)
        viewer.alpha_spin.setValue(0.8)
        qapp.processEvents()

        viewer.close()
