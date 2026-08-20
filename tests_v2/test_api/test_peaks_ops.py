"""Spec 002 T03 — the facade's peaks surface (Qt-free).

The exact-store forward picks (legacy click semantics: exact frequency +
interpolated amplitude, NO snapping) and the per-step picks store
(secondaries + label positions + Vs context; merge semantics).
"""

from __future__ import annotations

import numpy as np
import pytest

from HV_Strip_Progressive.api import HVStripAnalysis
from HV_Strip_Progressive.api.forward_engine import (
    ForwardResult,
    set_exact_peaks,
)


def _curve_result() -> ForwardResult:
    freqs = np.linspace(1.0, 10.0, 10)          # 1, 2, … 10
    amps = np.linspace(2.0, 4.0, 10)
    return ForwardResult(profile_name="p", frequencies=freqs,
                        amplitudes=amps, success=True)


# ----------------------------------------------------------------------
#  set_exact_peaks — the click semantics
# ----------------------------------------------------------------------
def test_exact_peaks_keep_the_clicked_frequency():
    res = set_exact_peaks(_curve_result(), [
        {"frequency": 4.37, "amplitude": 3.21, "label": "f0"},
    ])
    p = res.peaks[0]
    assert p.frequency == pytest.approx(4.37)   # NOT snapped to 4.0
    assert p.amplitude == pytest.approx(3.21)   # the given (interp) value
    assert p.index == 3                         # nearest-bin REFERENCE only
    assert p.source == "manual"


def test_exact_peaks_interpolate_missing_amplitude():
    res = set_exact_peaks(_curve_result(), [{"frequency": 1.5}])
    # curve: amp(1)=2.0, amp(2)=2.2222… → amp(1.5) by interp
    assert res.peaks[0].amplitude == pytest.approx(
        np.interp(1.5, np.linspace(1, 10, 10), np.linspace(2, 4, 10)))


def test_exact_peaks_carry_label_positions():
    res = set_exact_peaks(_curve_result(), [
        {"frequency": 4.0, "amplitude": 3.0, "label": "f0",
         "label_pos": [5.2, 3.6]},
        {"frequency": 8.0, "amplitude": 2.0},
    ])
    assert res.peaks[0].label_pos == [5.2, 3.6]
    assert res.peaks[1].label_pos is None
    assert res.peaks[1].label == "sec1"          # default labelling


def test_facade_set_profile_peaks_roundtrip():
    a = HVStripAnalysis()
    a._forward_results["p"] = _curve_result()    # test seam: seed a result
    env = a.set_profile_peaks("p", [
        {"frequency": 4.37, "amplitude": 3.21, "label": "f0",
         "label_pos": [5.0, 3.5]},
    ])
    assert env["success"] is True
    assert env["peaks"][0]["frequency"] == pytest.approx(4.37)
    assert env["peaks"][0]["label_pos"] == [5.0, 3.5]
    got = a.get_peaks("p")["peaks"][0]
    assert got["frequency"] == pytest.approx(4.37)

    missing = a.set_profile_peaks("nope", [])
    assert missing["success"] is False


# ----------------------------------------------------------------------
#  The per-step picks store
# ----------------------------------------------------------------------
def test_step_picks_merge_semantics():
    a = HVStripAnalysis()
    env = a.set_step_peaks(
        "prof", "Step0",
        f0=(4.1, 3.3),                            # legacy tuple shape
        secondary=[{"frequency": 8.5, "amplitude": 1.9,
                    "label_pos": [9.0, 2.2]}],
        vs30=250.0)
    assert env["f0"]["frequency"] == pytest.approx(4.1)
    assert env["f0"]["source"] == "manual"
    assert env["secondary"][0]["label_pos"] == [9.0, 2.2]

    # None args leave stored values untouched; [] clears secondaries.
    env = a.set_step_peaks("prof", "Step0", vsavg=310.0)
    assert env["f0"] is not None and env["vs30"] == pytest.approx(250.0)
    env = a.set_step_peaks("prof", "Step0", secondary=[])
    assert env["secondary"] == [] and env["f0"] is not None

    got = a.get_step_peaks("prof", "Step0")
    assert got["vsavg"] == pytest.approx(310.0)

    # Envelope copies are detached from the store.
    got["secondary"].append("junk")
    assert a.get_step_peaks("prof", "Step0")["secondary"] == []


def test_step_picks_clear_and_overrides():
    a = HVStripAnalysis()
    a.set_step_peaks("prof", "Step0", f0=(4.1, 3.3))
    a.set_step_peaks("prof", "Step1", f0=(4.6, 3.1))
    ov = a.dual_resonance_overrides("prof")["overrides"]
    assert ov == {"Step0": (4.1, 3.3), "Step1": (4.6, 3.1)}

    a.clear_step_peaks("prof", "Step0")
    assert a.get_step_peaks("prof", "Step0")["f0"] is None
    assert a.dual_resonance_overrides("prof")["overrides"] == {
        "Step1": (4.6, 3.1)}

    a.clear_step_peaks("prof")
    assert a.picked_peaks("prof")["steps"] == {}


def test_persist_without_strip_result_fails_cleanly():
    a = HVStripAnalysis()
    assert a.persist_picked_peaks("prof")["success"] is False
    a.set_step_peaks("prof", "Step0", f0=(4.1, 3.3))
    env = a.persist_picked_peaks("prof")
    assert env["success"] is False and "strip" in env["error"].lower()
