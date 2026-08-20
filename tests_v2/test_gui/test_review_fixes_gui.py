"""Regression pins for the 2026-08-15 review fixes — gui/v2 layer.

Offscreen PySide6 lane; manual QApplication fixture (no qtbot).
Covers: the OpWorker failure-envelope guard (a raising op must never
wedge the queue), the LayerTable stale-row cell-widget handlers, the
stale cached profile names in the Forward/Strip panels, AppState's
pre-submit ``None``-profile resolution, and the research engines card
healing its "(unavailable)" state.
"""

from __future__ import annotations

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


def _pump(app, predicate, timeout_s: float = 30.0) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


# ----------------------------------------------------------------------
#  OpWorker — a raising op must fail the op, never wedge the queue
# ----------------------------------------------------------------------
def test_raising_op_fails_cleanly_and_queue_recovers(_qapp):
    from HV_Strip_Progressive.gui.v2.workers.op_worker import OpQueue

    queue = OpQueue()
    done: list = []
    queue.op_finished.connect(lambda name, env: done.append((name, env)))

    def _boom() -> dict:
        raise KeyError("Profile 'None' not found")

    queue.submit("run_forward", _boom)
    assert _pump(_qapp, lambda: len(done) == 1), "finished_env never fired"
    name, env = done[0]
    assert name == "run_forward"
    assert env["success"] is False
    assert "KeyError" in env["error"]
    assert not queue.is_busy, "queue stayed wedged after the raise"

    # The queue must still run the NEXT op normally.
    queue.submit("ok_op", lambda: {"success": True})
    assert _pump(_qapp, lambda: len(done) == 2)
    assert done[1][1]["success"] is True
    assert not queue.is_busy


def test_non_dict_return_becomes_failure_envelope(_qapp):
    from HV_Strip_Progressive.gui.v2.workers.op_worker import OpQueue

    queue = OpQueue()
    done: list = []
    queue.op_finished.connect(lambda _n, env: done.append(env))
    queue.submit("bad_op", lambda: 42)           # not an envelope dict
    assert _pump(_qapp, lambda: len(done) == 1)
    assert done[0]["success"] is False
    assert not queue.is_busy


# ----------------------------------------------------------------------
#  LayerTable — handlers must follow their row across removals/moves
# ----------------------------------------------------------------------
def _fill(vs, nu=None):
    nu_val = 0.3 if nu is None else float(nu)
    return {"nu": nu_val, "vp": vs * (1.0 + nu_val), "density": 1900.0,
            "soil_type": "test"}


def test_layer_table_mode_handler_follows_row_after_removal(_qapp):
    from HV_Strip_Progressive.gui.v2.widgets.house.layer_table import (
        COL_NU, COL_VP, COL_VPMODE, LayerTable)

    table = LayerTable(_fill)                    # default 3-layer model
    table.table.selectRow(0)
    table._remove_layer()                        # rows shift up, widgets too
    # Row 0 is now the old row 1 (vs=400).  Type a custom ν, then flip ITS
    # combo to "From Nu": the derived Vp must land in row 0 — with the old
    # creation-time row capture the handler fired on the stale index and
    # the user's action never touched this row.
    table.table.item(0, COL_NU).setText("0.450")
    row1_vp_before = table.table.item(1, COL_VP).text()
    table.table.cellWidget(0, COL_VPMODE).setCurrentIndex(1)   # From Nu
    assert table.table.item(0, COL_VP).text() == \
        f"{400.0 * 1.45:.1f}", "the edited row was not re-derived"
    assert table.table.item(1, COL_VP).text() == row1_vp_before


def test_layer_table_move_swaps_vp_mode_with_the_layer(_qapp):
    from HV_Strip_Progressive.gui.v2.widgets.house.layer_table import (
        COL_VPMODE, LayerTable)

    table = LayerTable(_fill)
    table.table.cellWidget(0, COL_VPMODE).setCurrentIndex(2)   # Manual Vp
    table.table.setCurrentCell(0, 0)
    table._move_down()
    assert table.table.cellWidget(1, COL_VPMODE).currentIndex() == 2, \
        "the Vp mode did not travel with its layer"
    assert table.table.cellWidget(0, COL_VPMODE).currentIndex() == 0


# ----------------------------------------------------------------------
#  Panels — stale cached profile names dropped on refresh
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def _app_state(_qapp):
    from HV_Strip_Progressive.gui.v2.state.app_state import AppState

    state = AppState()
    yield state
    state.shutdown()


def _add_profile(state, name):
    env = state.add_profile_from_layers(
        [{"thickness": 5.0, "vs": 200.0},
         {"thickness": 0.0, "vs": 600.0}], name=name)
    assert env.get("name") == name
    return env


def test_forward_panel_drops_stale_profile_name(_qapp, _app_state):
    from HV_Strip_Progressive.gui.v2.tools.forward.panel import (
        ForwardSinglePanel)

    _add_profile(_app_state, "keep_me")
    _add_profile(_app_state, "delete_me")
    panel = ForwardSinglePanel(_app_state)
    panel._profile_name = "delete_me"
    _app_state.remove_profile("delete_me")
    panel.refresh()
    assert panel._profile_name == "keep_me"

    _app_state.remove_profile("keep_me")
    panel.refresh()
    assert panel._profile_name is None
    assert not panel._run_btn.isEnabled(), \
        "Run stayed enabled with no profile — the wedge trigger"


def test_strip_model_panel_drops_stale_profile_name(_qapp, _app_state):
    from HV_Strip_Progressive.gui.v2.tools.strip.panel import StripModelPanel

    _add_profile(_app_state, "gone_soon")
    panel = StripModelPanel(_app_state)
    panel.refresh()
    assert panel.profile_name == "gone_soon"
    _app_state.remove_profile("gone_soon")
    panel.refresh()
    assert panel.profile_name is None


def test_resolve_op_profile_pre_submit(_qapp, _app_state):
    errors: list = []
    _app_state.error.connect(errors.append)

    assert _app_state._resolve_op_profile("explicit") == "explicit"

    _add_profile(_app_state, "first")
    _add_profile(_app_state, "second")
    # The api raises KeyError for None with 2+ profiles — AppState must
    # resolve BEFORE submit (to the first profile, matching the badges).
    assert _app_state._resolve_op_profile(None) == "first"

    _app_state.remove_profile("first")
    _app_state.remove_profile("second")
    assert _app_state._resolve_op_profile(None) is None
    assert errors and "No profiles loaded." in errors[-1]


# ----------------------------------------------------------------------
#  Research engines card — "(unavailable)" must heal on availability
# ----------------------------------------------------------------------
def test_research_engine_checkbox_restores(_qapp, _app_state):
    from HV_Strip_Progressive.gui.v2.tools.research.panel import (
        ComparisonPage)

    page = ComparisonPage(_app_state, owner=None)   # owner only used on Run
    down = {e: {"available": False} for e in page.engine_checks}
    up = {e: {"available": True} for e in page.engine_checks}

    _app_state.engines_report = lambda refresh=False: down  # type: ignore
    page.refresh()
    for chk in page.engine_checks.values():
        assert not chk.isChecked() and "(unavailable)" in chk.text()

    _app_state.engines_report = lambda refresh=False: up  # type: ignore
    page.refresh()
    for eng, chk in page.engine_checks.items():
        assert chk.isEnabled() and chk.isChecked()
        assert chk.text() == eng, "the '(unavailable)' label stuck"
