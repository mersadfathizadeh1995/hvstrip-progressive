"""Spec 002 T08/T09/T10 — the AppState rewire pins.

DC-5: no private facade reach-ins; FR-11 (SC-3): a config keystroke
refreshes ONLY the panels that display that section, and never reformats
a focused widget; FR-12: the one-time preload emits busy hints.
"""

from __future__ import annotations

import inspect
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 required for GUI tests")

from PySide6.QtWidgets import QApplication  # noqa: E402


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
#  DC-5 — the reach-in pin
# ----------------------------------------------------------------------
def test_appstate_has_no_private_facade_reach_ins():
    from HV_Strip_Progressive.gui.v2.state import app_state as mod

    src = inspect.getsource(mod)
    assert "_analysis._" not in src, \
        "AppState must use the facade's PUBLIC surface (spec 002 DC-5)"


# ----------------------------------------------------------------------
#  FR-11 / SC-3 — section-scoped refresh routing
# ----------------------------------------------------------------------
def _counting(cls, *args):
    class _C(cls):
        def __init__(self, *a, **k):
            self.n_refresh = 0
            super().__init__(*a, **k)

        def refresh(self):
            self.n_refresh += 1
            super().refresh()

    return _C(*args)


def test_config_keystroke_refreshes_only_owning_panels(_qapp, _state):
    from HV_Strip_Progressive.gui.v2.tools.forward.panel import (
        ForwardSinglePanel,
    )
    from HV_Strip_Progressive.gui.v2.tools.strip.panel import (
        StripModelPanel,
        StripReviewPanel,
        StripRunPanel,
    )

    forward = _counting(ForwardSinglePanel, _state)
    model = _counting(StripModelPanel, _state)
    run = _counting(StripRunPanel, _state, model)
    review = _counting(StripReviewPanel, _state)
    panels = {"forward": forward, "model": model,
              "run": run, "review": review}

    def deltas(action):
        before = {k: p.n_refresh for k, p in panels.items()}
        action()
        return {k: p.n_refresh - before[k] for k, p in panels.items()}

    # A strip-section keystroke refreshes ONLY the strip-run panel.
    d = deltas(lambda: _state.update_config("strip", generate_report=True))
    assert d == {"forward": 0, "model": 0, "run": 1, "review": 0}, d

    # A frequency keystroke refreshes the two panels showing frequency.
    d = deltas(lambda: _state.update_config("frequency", fmin=0.7))
    assert d == {"forward": 1, "model": 0, "run": 1, "review": 0}, d

    # Output touches the model panel (and run's gating note).
    d = deltas(lambda: _state.update_config("output", output_dir="x"))
    assert d == {"forward": 0, "model": 1, "run": 1, "review": 0}, d

    # The wildcard (payload load) refreshes everyone.
    d = deltas(lambda: _state.apply_config_payload(
        _state.config_payload()))
    assert d == {"forward": 1, "model": 1, "run": 1, "review": 1}, d


def test_set_unfocused_never_touches_a_focused_widget(_qapp):
    from PySide6.QtWidgets import QDoubleSpinBox

    from HV_Strip_Progressive.gui.v2.stages.base import set_unfocused

    spin = QDoubleSpinBox()
    spin.setDecimals(3)
    spin.setValue(1.0)
    set_unfocused(spin, 2.5)
    assert spin.value() == pytest.approx(2.5)

    spin.hasFocus = lambda: True                 # simulate mid-edit
    set_unfocused(spin, 9.9)
    assert spin.value() == pytest.approx(2.5), \
        "a refresh must never reformat the field being typed in"


# ----------------------------------------------------------------------
#  FR-12 — busy hints around the one-time preload
# ----------------------------------------------------------------------
def test_busy_hint_wraps_first_preload(_qapp, monkeypatch):
    import HV_Strip_Progressive.api as api_pkg
    from HV_Strip_Progressive.gui.v2.state.app_state import AppState

    state = AppState()
    try:
        monkeypatch.setattr(AppState, "_compute_imported", False)
        monkeypatch.setattr(api_pkg, "preload_heavy_modules", lambda: None)
        notes: list = []
        state.busy_hint.connect(notes.append)
        state._ensure_compute_imports()
        assert notes == ["Preparing compute libraries (one-time)…", ""]
        state._ensure_compute_imports()          # idempotent → no new hints
        assert len(notes) == 2
    finally:
        state.shutdown()
