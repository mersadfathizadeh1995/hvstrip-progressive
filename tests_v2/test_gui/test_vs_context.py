"""Spec 002 T15 — the Vs mini-panel + the ``vs_context`` facade op.

The legacy rules pinned: Vs30 = 30 m WITH half-space extrapolation;
VsAvg = to the bedrock interface WITHOUT it; bedrock selectable from the
combo AND by clicking the Vs plot.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 required for GUI tests")

from PySide6.QtWidgets import QApplication  # noqa: E402

_LAYERS = [
    {"thickness": 5.0, "vs": 200.0},
    {"thickness": 15.0, "vs": 400.0},
    {"thickness": 0.0, "vs": 800.0},             # half-space
]


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def _state(_qapp):
    from HV_Strip_Progressive.gui.v2.state.app_state import AppState

    state = AppState()
    yield state
    state.shutdown()


# ----------------------------------------------------------------------
#  The facade op (Qt-free semantics, checked here for locality)
# ----------------------------------------------------------------------
def test_vs_context_facade_numbers(_state):
    env = _state.vs_context(_LAYERS)
    assert env["success"] is True
    assert env["interfaces"] == [5.0, 20.0]
    # Vs30 WITH half-space: 30 / (5/200 + 15/400 + 10/800) = 400.
    assert env["vs30"] == pytest.approx(400.0)
    assert env["vs30_extrapolated"] is True      # 30 m > 20 m of layers
    # VsAvg defaults to the finite bottom (20 m), WITHOUT the half-space:
    # 20 / (5/200 + 15/400) = 320.
    assert env["vsavg"] == pytest.approx(320.0)

    env5 = _state.vs_context(_LAYERS, bedrock_depth=5.0)
    assert env5["vsavg"] == pytest.approx(200.0)  # only the first layer


# ----------------------------------------------------------------------
#  The panel
# ----------------------------------------------------------------------
def test_panel_combo_and_click_selection(_qapp, _state):
    from HV_Strip_Progressive.gui.v2.canvas.vs_context_mpl import (
        VsContextPanel,
    )

    panel = VsContextPanel(_state)
    emitted: list = []
    panel.context_changed.connect(emitted.append)
    panel.set_profile(_LAYERS)

    assert panel._combo.count() == 3              # auto + 2 interfaces
    assert panel.context()["bedrock_depth"] is None
    assert "Vs30 = 400.0" in panel._readout.text()

    panel._combo.setCurrentIndex(1)               # Interface 1: 5 m
    assert emitted and emitted[-1]["bedrock_depth"] == pytest.approx(5.0)
    assert emitted[-1]["vsavg"] == pytest.approx(200.0)

    # Click near 18 m depth → the nearest interface (20 m) becomes bedrock.
    ax = panel._fig.figure.axes[0]
    panel._on_click(SimpleNamespace(inaxes=ax, ydata=18.0, xdata=300.0))
    assert emitted[-1]["bedrock_depth"] == pytest.approx(20.0)
    assert emitted[-1]["vsavg"] == pytest.approx(320.0)
    assert panel._combo.currentIndex() == 2

    # A persisted bedrock restores through set_profile.
    panel.set_profile(_LAYERS, bedrock_depth=5.0)
    assert panel.context()["bedrock_depth"] == pytest.approx(5.0)
    assert panel._combo.currentIndex() == 1
