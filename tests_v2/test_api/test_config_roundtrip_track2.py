"""Spec 002 T02 — the Track-2 config additions round-trip the funnel.

New surface: ``AutoPeakConfig.strategy`` + per-peak ``ranges`` (armed
bands), ``PeakDetectionConfig.width``, ``MarkerStyleConfig`` and
``ResearchStudyConfig`` as top-level sections.
"""

from __future__ import annotations

import pytest

from HV_Strip_Progressive.api import (
    AUTO_PEAK_STRATEGIES,
    HVStripConfig,
)


def test_new_sections_survive_the_funnel():
    cfg = HVStripConfig()
    cfg.markers.f0_shape = "D"
    cfg.markers.f0_size = 20.0
    cfg.markers.show_annotations = False
    cfg.research.profiles_dir = "C:/models"
    cfg.research.engines = ["sh_wave"]
    cfg.research.n_per_scenario = 5
    cfg.auto_peak.strategy = "range_constrained"
    cfg.auto_peak.ranges = [
        {"fmin": 0.5, "fmax": 6.0, "use": True},
        {"fmin": 6.0, "fmax": 15.0, "use": False},
    ]
    cfg.peak_detection.width = 3.0

    back = HVStripConfig.from_dict(cfg.to_dict())
    assert back.markers.f0_shape == "D"
    assert back.markers.f0_size == pytest.approx(20.0)
    assert back.markers.show_annotations is False
    assert back.research.profiles_dir == "C:/models"
    assert back.research.engines == ["sh_wave"]
    assert back.research.n_per_scenario == 5
    assert back.auto_peak.strategy == "range_constrained"
    assert back.auto_peak.ranges == cfg.auto_peak.ranges
    assert back.peak_detection.width == pytest.approx(3.0)


def test_auto_peak_strategies_registry():
    assert set(AUTO_PEAK_STRATEGIES) == {
        "range_constrained", "preset", "advanced"}
    assert HVStripConfig().auto_peak.strategy in AUTO_PEAK_STRATEGIES


def test_effective_ranges_armed_bands_win():
    ap = HVStripConfig().auto_peak
    # Legacy fallback: the f0/f1/f2 triple capped by n_secondary.
    ap.n_secondary = 1
    assert ap.effective_ranges() == [(0.1, 50.0), (0.1, 50.0)]
    # Explicit bands win; disarmed bands are skipped.
    ap.ranges = [
        {"fmin": 1.0, "fmax": 5.0, "use": True},
        {"fmin": 6.0, "fmax": 9.0, "use": False},
        {"fmin": 10.0, "fmax": 20.0},          # "use" defaults to armed
    ]
    assert ap.effective_ranges() == [(1.0, 5.0), (10.0, 20.0)]


def test_width_only_enters_core_params_when_set():
    pd = HVStripConfig().peak_detection
    assert "width" not in pd.to_core_config()["find_peaks_params"], \
        "default (None) must be behavior-preserving"
    pd.width = 2.5
    assert pd.to_core_config()["find_peaks_params"]["width"] == \
        pytest.approx(2.5)
