"""S4+S5 tool tests — ONE ``StripMainWindow`` shared by the Strip and
Research tool tests.

ONE window module, deliberately: every extra module-scoped window in the
process compounds Qt/native teardown state until a later worker-thread op
aborts on Windows (the multi-window composition crashed where each pairwise
subset passed).  The module also sorts AFTER ``test_scaffold`` so the
headless AppState tests never run behind a window's teardown.  Keep any new
window-based test IN THIS MODULE, on this window.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 required for GUI tests")

from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_MODEL = ROOT / "examples" / "different_files" / "example_model.txt"
GOLDEN = ROOT / "tests" / "golden" / "workflow_sh_wave.json"


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _pump(app, predicate, timeout_s: float = 120.0) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


@pytest.fixture(scope="module")
def _window(_qapp, tmp_path_factory):
    """THE window: one sh_wave strip run through the Strip panel, then
    shared read-only by every test here."""
    from HV_Strip_Progressive.gui.v2.main_window import StripMainWindow
    from HV_Strip_Progressive.gui.v2.state import StripTool

    win = StripMainWindow()
    win.show_tool(StripTool.STRIP)
    panel = win.panel(StripTool.STRIP)

    # Gating before any data.
    assert not panel._sub_bar._tabs[1].isEnabled()
    assert not panel._sub_bar._tabs[2].isEnabled()

    env = win.app_state.load_profile(str(EXAMPLE_MODEL))
    panel.model_panel.profile_name = env["name"]
    win.app_state.set_engine("sh_wave")
    out = tmp_path_factory.mktemp("strip_out")
    win.app_state.update_config("output", output_dir=str(out))

    finished = []
    win.app_state.op_finished.connect(
        lambda n, e: finished.append((n, e)) if n == "run_strip" else None)
    panel._set_index(1)
    panel.run_panel._on_run()
    assert _pump(_qapp, lambda: bool(finished)), "run_strip never finished"
    assert finished[0][1].get("success")

    yield win
    win.close()
    _qapp.processEvents()


# ======================================================================
#  Strip tool (S4)
# ======================================================================
def test_strip_tool_installed(_window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    assert type(_window.panel(StripTool.STRIP)).__name__ == "StripToolPanel"
    assert type(_window.canvas(StripTool.STRIP)).__name__ == "StripCanvas"


def test_wizard_gating_after_run(_window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    panel = _window.panel(StripTool.STRIP)
    assert panel._sub_bar._tabs[1].isEnabled()
    assert panel._sub_bar._tabs[2].isEnabled()   # Review opens with a result
    panel._set_index(2)
    assert panel.active_index == 2
    assert panel.review_panel._steps.rowCount() == 6


def test_steps_match_goldens(_window):
    steps = next(iter(_window.app_state.strip_results().values())).steps
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["steps"]
    assert len(steps) == 6
    for step in steps:
        g = golden[f"Step{step.step_number}_{step.n_layers}-layer"]
        assert step.peak_frequency == pytest.approx(
            g["peak_frequency"], abs=1e-6)
        assert step.peak_amplitude == pytest.approx(
            g["peak_amplitude"], abs=1e-6)


def test_canvas_steps_as_layers(_qapp, _window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    canvas = _window.canvas(StripTool.STRIP)
    assert sorted(canvas.overlay.item_keys()) == [
        f"step::{i}" for i in range(6)]
    assert canvas.step_combo.count() == 6
    assert canvas.summary.rowCount() == 6

    lm = _window.layer_model
    _window.show_tool(StripTool.STRIP)
    assert lm.find_node("step::0") is not None
    lm.set_display("step::0", color="#336699")
    lm.set_visible("step::0", False)
    _qapp.processEvents()
    assert lm.is_visible("step::0") is False
    lm.set_visible("step::0", True)


def test_step_view_and_figure_studio(_qapp, _window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    canvas = _window.canvas(StripTool.STRIP)
    canvas._on_step_selected(0)     # per-step HV ∥ Vs populates
    _qapp.processEvents()

    canvas.figure_studio._on_draw()
    assert len(canvas.figure_studio._figure.axes) > 0


# ======================================================================
#  Research tool (S5) — same window; payload/plumbing only (the study run
#  is exercised api-side in test_api/test_research_study.py + the smoke)
# ======================================================================
def test_research_tool_installed(_window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    panel = _window.panel(StripTool.RESEARCH)
    assert type(panel).__name__ == "ResearchToolPanel"
    assert panel._sub_bar.n_phases == 5
    assert type(_window.canvas(StripTool.RESEARCH)).__name__ \
        == "ResearchCanvas"


def test_research_phases_freely_navigable(_window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    panel = _window.panel(StripTool.RESEARCH)
    for i in range(5):
        panel._set_index(i)
        assert panel.active_index == i
    panel._set_index(0)


def test_research_study_payload_assembly(_window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    panel = _window.panel(StripTool.RESEARCH)
    panel.profiles_page.n_random.setValue(3)
    panel.profiles_page.seed.setValue(7)
    panel.profiles_page.suite_edit.setText("X:/suite")
    panel.profiles_page.soilgen_edit.setText("X:/soilgen")
    panel.comparison_page.fmin.setText("0.2")
    panel.report_page.out_edit.setText("X:/out")

    payload = panel.study_payload()
    assert payload["profiles"]["n_random"] == 3
    assert payload["profiles"]["seed"] == 7
    assert payload["profiles"]["soilgen_path"] == "X:/soilgen"
    assert payload["profiles_dir"] == "X:/suite"
    assert payload["engines"]["fmin"] == 0.2
    assert payload["output"]["output_dir"] == "X:/out"
    # Only available engines stay checked/enabled.
    report = _window.app_state.engines_report()
    for eng, chk in panel.comparison_page.engine_checks.items():
        if not report.get(eng, {}).get("available"):
            assert not chk.isEnabled() and not chk.isChecked()
    assert "sh_wave" in payload["engines"]["engines"]


def test_research_cancel_button_wired(_window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    panel = _window.panel(StripTool.RESEARCH)
    panel.report_page.cancel_btn.click()
    assert _window.app_state._research_cancelled is True
    _window.app_state._research_cancelled = False


# ======================================================================
#  Data Input stage + ProfilesPanel (Round 2 Track 1) — same window
# ======================================================================
def test_data_tool_installed_and_first(_window):
    from HV_Strip_Progressive.gui.v2.state import TOOL_ORDER, StripTool

    assert TOOL_ORDER[0] is StripTool.DATA
    assert type(_window.panel(StripTool.DATA)).__name__ == "DataToolPanel"
    assert type(_window.canvas(StripTool.DATA)).__name__ == "DataCanvas"


def test_profiles_panel_stage_aware_columns(_qapp, _window):
    from HV_Strip_Progressive.gui.v2.state import StripTool

    pp = _window.profiles_panel
    _window.show_tool(StripTool.DATA)
    _qapp.processEvents()
    assert [h for _k, h, _t in pp._badge_cols] == ["Fwd", "Str", "Res"]
    _window.show_tool(StripTool.STRIP)
    _qapp.processEvents()
    assert [h for _k, h, _t in pp._badge_cols] == ["Set", "Run"]
    _window.show_tool(StripTool.DATA)
    _qapp.processEvents()


def test_profiles_panel_badges_reflect_the_strip(_qapp, _window):
    """The module fixture stripped example_model — its Str badge is DONE."""
    from PySide6.QtCore import Qt

    from HV_Strip_Progressive.gui.v2.state import ProcessingStatus, StripTool

    _window.show_tool(StripTool.DATA)
    _qapp.processEvents()
    pp = _window.profiles_panel
    item = pp._find_item("example_model")
    assert item is not None
    assert item.data(2, Qt.UserRole) is ProcessingStatus.DONE   # Str col
    assert _window.app_state.profile_status(
        StripTool.STRIP, "example_model") is ProcessingStatus.DONE


def test_data_stage_load_focus_table_apply(_qapp, _window):
    """Load a SECOND profile through the Data panel; focus fills the mpl
    preview + table; Apply writes an edit back through the api."""
    from HV_Strip_Progressive.gui.v2.state import StripTool

    _window.show_tool(StripTool.DATA)
    panel = _window.panel(StripTool.DATA)
    canvas = _window.canvas(StripTool.DATA)

    row, _fmt = panel._single_rows["HVf File (.txt)"]
    row.edit.setText(str(ROOT / "examples" / "different_files" / "model.txt"))
    panel.name_edit.setText("second_model")
    panel._on_load()
    _qapp.processEvents()

    assert "second_model" in _window.app_state.profile_names()
    assert _window.app_state.focus == "second_model"
    assert len(canvas.vs_view.figure.axes) > 0
    n_rows = canvas.layer_table.table.rowCount()
    assert n_rows >= 2

    canvas.layer_table.table.item(0, 2).setText("222.0")   # Vs edit
    _qapp.processEvents()
    canvas._on_apply()
    _qapp.processEvents()
    layers = _window.app_state.profile_dict("second_model")["layers"]
    assert layers[0]["vs"] == pytest.approx(222.0)
    # ν was re-derived through the ONE api surface
    fill = _window.app_state.suggest_layer_fill(222.0)
    assert layers[0]["nu"] == pytest.approx(fill["nu"], abs=1e-6)
