"""Spec 002 T13/T14/T16 — the interactive HV figure's picking semantics.

The legacy-verbatim behaviours pinned at widget level: click = EXACT
frequency + interpolated amplitude; drag > 2 % of the span = band argmax;
f0 one-shot / secondary accumulating; right-click delete-nearest; undo /
clear; the toolbar guard; persisted draggable label positions; marker
visibility + style hooks.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 required for GUI tests")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


FREQS = np.linspace(1.0, 10.0, 10)               # 1, 2, … 10
AMPS = np.array([2.0, 2.2, 2.5, 3.0, 2.8, 2.4, 3.6, 2.9, 2.3, 2.1])
#                                 ^idx3=3.0            ^idx6=3.6 (band max)


@pytest.fixture()
def fig(_qapp):
    from HV_Strip_Progressive.gui.v2.canvas.hv_interactive_mpl import (
        HVInteractiveFigure,
    )

    w = HVInteractiveFigure(log_x=False)
    w.set_curve(FREQS, AMPS, title="Step0")
    return w


def _ev(w, x, button=1):
    return SimpleNamespace(inaxes=w.figure.axes[0], xdata=x,
                           ydata=2.5, button=button)


def _click(w, x):
    w._on_press(_ev(w, x))
    w._on_release(_ev(w, x))


def _drag(w, x0, x1):
    w._on_press(_ev(w, x0))
    w._on_motion(_ev(w, (x0 + x1) / 2))
    w._on_release(_ev(w, x1))


# ----------------------------------------------------------------------
def test_click_places_exact_frequency_with_interp_amplitude(fig):
    changed = []
    fig.peaks_changed.connect(lambda: changed.append(1))
    fig.arm("f0")
    _click(fig, 4.37)
    peaks = fig.get_peaks()
    assert peaks["f0"]["frequency"] == pytest.approx(4.37)   # NOT snapped
    assert peaks["f0"]["amplitude"] == pytest.approx(
        float(np.interp(4.37, FREQS, AMPS)))
    assert peaks["f0"]["source"] == "manual"
    assert fig.pick_mode == "", "f0 is one-shot — the arm must release"
    assert changed, "peaks_changed must fire on a pick"


def test_drag_snaps_to_band_argmax(fig):
    fig.arm("f0")
    _drag(fig, 5.0, 9.0)                          # span 4 ≫ 2 % of 9
    f0 = fig.get_peaks()["f0"]
    assert f0["frequency"] == pytest.approx(7.0)  # grid argmax in [5, 9]
    assert f0["amplitude"] == pytest.approx(3.6)


def test_secondaries_accumulate_and_arm_persists(fig):
    fig.arm("secondary")
    _click(fig, 3.1)
    _click(fig, 8.2)
    peaks = fig.get_peaks()
    assert [s["label"] for s in peaks["secondary"]] == ["sec1", "sec2"]
    assert fig.pick_mode == "secondary", "the secondary arm persists"


def test_right_click_deletes_nearest_peak(fig):
    fig.arm("f0")
    _click(fig, 4.0)
    fig.arm("secondary")
    _click(fig, 8.0)
    fig.arm("")
    fig._on_press(_ev(fig, 7.7, button=3))        # nearest = the secondary
    peaks = fig.get_peaks()
    assert peaks["secondary"] == []
    assert peaks["f0"] is not None
    fig._on_press(_ev(fig, 4.2, button=3))        # now the f0
    assert fig.get_peaks()["f0"] is None


def test_undo_and_clear(fig):
    fig.arm("f0")
    _click(fig, 4.0)
    fig.arm("secondary")
    _click(fig, 6.0)
    _click(fig, 8.0)
    fig.undo_secondary()
    assert len(fig.get_peaks()["secondary"]) == 1
    fig.clear_peaks()
    peaks = fig.get_peaks()
    assert peaks["f0"] is None and peaks["secondary"] == []


def test_toolbar_mode_suppresses_picking(fig, monkeypatch):
    monkeypatch.setattr(fig, "_toolbar_active", lambda: True)
    fig.arm("f0")
    _click(fig, 4.0)
    assert fig.get_peaks()["f0"] is None, \
        "picking must be OFF while pan/zoom is armed"


def test_label_positions_persist_and_harvest(fig):
    fig.set_peaks({"frequency": 4.0, "amplitude": 3.0, "label": "f0",
                   "label_pos": [5.5, 3.4]})
    peak, ann = fig._annotations[0]
    assert tuple(ann.xyann) == (5.5, 3.4), \
        "a stored label position must be re-applied on redraw"

    changed = []
    fig.peaks_changed.connect(lambda: changed.append(1))
    ann.xyann = (6.2, 2.9)                        # simulate the drag
    fig._harvest_label_positions()
    assert fig.get_peaks()["f0"]["label_pos"] == [6.2, 2.9]
    assert changed, "a label drag is a persistable change"


def test_marker_visibility_and_style(fig):
    fig.set_peaks({"frequency": 4.0, "amplitude": 3.0, "label": "f0"},
                  [{"frequency": 8.0, "amplitude": 2.3}])
    assert len(fig._annotations) == 2
    fig.set_markers_visible(False)                # the layer-tree toggle
    assert fig._annotations == []
    fig.set_markers_visible(True)
    fig.set_marker_style({"show_annotations": False})
    assert fig._annotations == [], "style can hide annotations alone"


def test_arm_is_mutually_exclusive(fig):
    fig.arm("f0")
    fig._btn_sec.setChecked(True)
    assert not fig._btn_f0.isChecked()
    assert fig.pick_mode == "secondary"
