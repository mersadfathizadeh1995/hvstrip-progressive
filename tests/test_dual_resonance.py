"""Tests for the dual_resonance module."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hvstrip_progressive.core.dual_resonance import (
    BatchDualResonanceStats,
    DualResonanceResult,
    compute_batch_statistics,
    extract_dual_resonance,
    save_results_csv,
    save_statistics_json,
    theoretical_frequency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step_folder(parent: Path, step: int, n_layers: int,
                      model_text: str, freqs, amps):
    """Create a Step folder with model file and HV CSV."""
    name = f"Step{step}_{n_layers}-layer"
    folder = parent / name
    folder.mkdir()
    model_path = folder / f"model_{name}.txt"
    model_path.write_text(model_text, encoding="utf-8")
    hv_csv = folder / "hv_curve.csv"
    with open(hv_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frequency_Hz", "HVSR_Amplitude"])
        for f, a in zip(freqs, amps):
            w.writerow([f"{f:.6f}", f"{a:.6f}"])
    return folder


def _build_strip_dir(tmp_path):
    """Build a minimal two-step strip directory for testing."""
    strip = tmp_path / "strip"
    strip.mkdir()

    # 3-layer model: 10m@200, 20m@400, halfspace@800
    model_txt = "3\n10.0  400.0  200.0  1.8\n20.0  800.0  400.0  2.0\n0.0  1600.0  800.0  2.2\n"

    freqs = np.linspace(0.5, 20, 100)

    # Step 0: peak near 2 Hz (deep resonance)
    amps0 = 1.0 + 3.0 * np.exp(-0.5 * ((freqs - 2.0) / 0.3) ** 2)
    _make_step_folder(strip, 0, 3, model_txt, freqs, amps0)

    # Step 1: stripped model, peak shifts to ~6 Hz
    model1 = "2\n10.0  400.0  200.0  1.8\n0.0  1600.0  800.0  2.2\n"
    amps1 = 1.0 + 4.0 * np.exp(-0.5 * ((freqs - 6.0) / 0.5) ** 2)
    _make_step_folder(strip, 1, 2, model1, freqs, amps1)

    return strip


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTheoreticalFrequency:
    def test_single_layer_over_halfspace(self):
        layers = [
            {"thickness": 30.0, "vs": 300.0},
            {"thickness": 0, "vs": 800.0},
        ]
        f0, f1 = theoretical_frequency(layers)
        assert f0 == pytest.approx(300.0 / (4 * 30.0), rel=0.01)
        assert f1 == pytest.approx(300.0 / (4 * 30.0), rel=0.01)

    def test_two_finite_layers(self):
        layers = [
            {"thickness": 10.0, "vs": 200.0},
            {"thickness": 20.0, "vs": 400.0},
            {"thickness": 0, "vs": 800.0},
        ]
        f0, f1 = theoretical_frequency(layers)
        assert f0 > 0
        assert f1 > 0
        # f1 (top layer only) should be higher than f0 (full depth)
        assert f1 > f0

    def test_empty_layers(self):
        f0, f1 = theoretical_frequency([])
        assert f0 == 0.0
        assert f1 == 0.0


class TestExtractDualResonance:
    def test_basic_extraction(self, tmp_path):
        strip = _build_strip_dir(tmp_path)
        dr = extract_dual_resonance(str(strip))

        assert dr.success is True
        assert dr.n_layers == 3
        assert dr.f0 > 0
        assert dr.f1 > 0
        assert dr.f1 > dr.f0  # shallow resonance at higher freq
        assert dr.freq_ratio > 1.0
        assert len(dr.freq_per_step) == 2
        assert len(dr.amp_per_step) == 2

    def test_separation_success_flag(self, tmp_path):
        strip = _build_strip_dir(tmp_path)
        dr = extract_dual_resonance(str(strip))
        # With our synthetic data, f1/f0 ~ 3 and shift ~ 4 Hz → success
        assert dr.separation_success is True

    def test_missing_step_folders(self, tmp_path):
        empty = tmp_path / "empty_strip"
        empty.mkdir()
        dr = extract_dual_resonance(str(empty))
        assert dr.success is False
        assert "No step folders" in dr.error_message

    def test_custom_profile_name(self, tmp_path):
        strip = _build_strip_dir(tmp_path)
        dr = extract_dual_resonance(str(strip), profile_name="TestProfile")
        assert dr.profile_name == "TestProfile"


class TestBatchStatistics:
    def test_empty_results(self):
        stats = compute_batch_statistics([])
        assert stats.n_profiles == 0
        assert stats.success_rate == 0.0

    def test_all_failures(self):
        fails = [
            DualResonanceResult(profile_name="a", profile_path="a", success=False),
            DualResonanceResult(profile_name="b", profile_path="b", success=False),
        ]
        stats = compute_batch_statistics(fails)
        assert stats.n_profiles == 2
        assert stats.n_successful == 0

    def test_mixed_results(self, tmp_path):
        strip = _build_strip_dir(tmp_path)
        dr_ok = extract_dual_resonance(str(strip))
        dr_fail = DualResonanceResult(
            profile_name="bad", profile_path="bad", success=False
        )
        stats = compute_batch_statistics([dr_ok, dr_fail])
        assert stats.n_profiles == 2
        assert stats.n_successful == 1
        assert stats.success_rate == pytest.approx(50.0)
        assert stats.f0_mean > 0


class TestIO:
    def test_save_results_csv(self, tmp_path):
        dr = DualResonanceResult(
            profile_name="test", profile_path="test.txt",
            success=True, f0=2.0, f1=6.0, freq_ratio=3.0,
        )
        csv_path = tmp_path / "results.csv"
        save_results_csv([dr], str(csv_path))
        assert csv_path.exists()
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 2  # header + 1 row

    def test_save_statistics_json(self, tmp_path):
        stats = BatchDualResonanceStats(n_profiles=5, n_successful=4)
        json_path = tmp_path / "stats.json"
        save_statistics_json(stats, str(json_path))
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["n_profiles"] == 5
        assert data["n_successful"] == 4
