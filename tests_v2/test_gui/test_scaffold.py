"""S1 scaffold tests — import hygiene · amber theme conformance · the
AppState spine (ops → coalesced frames → envelopes; two queues; the funnel).

Manual QApplication fixture; offscreen; PySide6 lane (see tests_v2/conftest).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 required for GUI tests")

from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_MODEL = ROOT / "examples" / "different_files" / "example_model.txt"


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
#  Import hygiene (DC-4)
# ----------------------------------------------------------------------
def test_v2_import_pulls_no_pyqt_and_no_legacy_gui(_qapp):
    import HV_Strip_Progressive.gui.v2.state  # noqa: F401
    import HV_Strip_Progressive.gui.v2.theme  # noqa: F401
    import HV_Strip_Progressive.gui.v2.widgets.house.tool_switcher  # noqa: F401
    import HV_Strip_Progressive.gui.v2.widgets.house.layer_tree  # noqa: F401
    import HV_Strip_Progressive.gui.v2.workbench.side_rail  # noqa: F401
    import HV_Strip_Progressive.gui.v2.stages.base  # noqa: F401

    assert not any(m.startswith("PyQt5") for m in sys.modules), \
        "v2 must never pull PyQt5"
    legacy = [m for m in sys.modules
              if m.startswith("HV_Strip_Progressive.gui.")
              and not m.startswith("HV_Strip_Progressive.gui.v2")
              and m != "HV_Strip_Progressive.gui"]
    assert not legacy, f"v2 imported legacy gui modules: {legacy}"
    import matplotlib

    assert matplotlib.get_backend().lower() != "qt5agg"


# ----------------------------------------------------------------------
#  Theme — amber, 3 modes, distinct from the family accents
# ----------------------------------------------------------------------
def test_amber_theme_conformance(_qapp):
    from PySide6.QtGui import QColor

    from HV_Strip_Progressive.gui.v2 import theme

    palettes = {"light": theme.LIGHT, "gray": theme.GRAY, "dark": theme.DARK}
    family = {"#C8102E", "#0078D4", "#3E7A45", "#6A4CAF", "#2E86AB"}
    for name, p in palettes.items():
        accent = QColor(p.accent)
        assert accent.isValid(), name
        assert p.accent.upper() not in family, f"{name} collides"
        # Amber reads warm: red > blue and green between them.
        r, g, b = accent.red(), accent.green(), accent.blue()
        assert r > b and r > 0.5 * 255 * 0.4 and g > b, \
            f"{name} accent {p.accent} does not read amber"
        assert p.primary == p.accent
    qss = theme.build_qss(theme.LIGHT)
    assert 'role="groupHeader"' in qss and 'role="subTab"' in qss
    assert "layerTree" in qss


# ----------------------------------------------------------------------
#  AppState — ops on the queue, coalesced frames, envelopes, status
# ----------------------------------------------------------------------
def _make_state():
    from HV_Strip_Progressive.gui.v2.state import AppState

    s = AppState()
    s.update_config("engine", name="sh_wave")
    s.load_profile(str(EXAMPLE_MODEL), name="example_model")
    return s


@pytest.fixture()
def _state():
    """A fresh AppState for NON-op tests, ALWAYS shut down.  Leaving an
    AppState's OpQueue QThreads alive lets later garbage collection destroy
    a live QThread mid-processEvents — a hard Windows abort."""
    s = _make_state()
    yield s
    s.shutdown()


@pytest.fixture(scope="module")
def _mstate():
    """ONE module-scoped AppState for every OP-RUNNING test — the product
    shape (one spine, many ops).  Many short-lived AppStates each running
    worker-thread strips in one process abort intermittently on Windows;
    one long-lived spine running repeated ops is verified stable."""
    s = _make_state()
    yield s
    s.shutdown()


def test_appstate_strip_op_streams_and_lands(_qapp, tmp_path, _mstate):
    from HV_Strip_Progressive.gui.v2.state import StripTool, ToolStatus

    s = _mstate
    frames, finished = [], []
    s.op_progress.connect(frames.append)
    s.op_finished.connect(lambda n, e: finished.append((n, e)))

    assert s.status_for(StripTool.STRIP) is ToolStatus.IDLE
    s.run_strip("example_model", output_dir=str(tmp_path / "out"))
    assert _pump(_qapp, lambda: bool(finished)), "strip op never finished"

    name, env = finished[0]
    assert name == "run_strip" and env.get("success")
    assert env["steps"], "envelope carries the per-step results"
    assert any(f.get("type") == "phase" for f in frames), \
        "coalesced phase frames reached the GUI thread"
    assert s.status_for(StripTool.STRIP) is ToolStatus.DONE
    assert s.strip_results(), "results accessible via AppState"


def test_two_queues_run_concurrently(_qapp, tmp_path, _mstate):
    """A research op must not block the main queue (per-tool lanes)."""
    s = _mstate
    order = []
    s.op_started.connect(lambda n: order.append(("start", n)))
    s.op_finished.connect(lambda n, e: order.append(("end", n)))

    # A research phase that fails fast (no study config) still exercises the
    # research lane; the forward op runs on the main lane simultaneously.
    s.run_research_phase("metrics")
    s.run_forward("example_model")
    assert _pump(_qapp, lambda: sum(1 for k, _ in order if k == "end") >= 2)
    started = [n for k, n in order if k == "start"]
    assert "run_research" in started and "run_forward" in started


def test_engines_report_and_status_badges(_qapp, _state):
    s = _state
    report = s.engines_report()
    assert report["sh_wave"]["available"] is True
    assert set(report) == {"sh_wave", "diffuse_field", "ellipticity"}


def test_config_funnel_via_appstate(_qapp, _state):
    import json

    s = _state
    legacy = json.loads(
        (ROOT / "tests" / "golden" / "legacy_gui_config.json")
        .read_text(encoding="utf-8"))
    s.apply_config_payload(legacy)
    assert s.config.frequency.nf == 71          # migrated legacy values
    payload = s.config_payload()
    assert payload["config_version"] == 2       # saves are always v2


def test_profile_state_model(_qapp, _state):
    """R2-T1.2 — checked set, focus, per-tool settings (no ops here)."""
    from HV_Strip_Progressive.gui.v2.state import ProcessingStatus, StripTool

    s = _state
    events = {"checked": [], "focus": [], "settings": []}
    s.checked_changed.connect(events["checked"].append)
    s.focus_changed.connect(events["focus"].append)
    s.profile_settings_changed.connect(events["settings"].append)

    assert s.checked_profiles() == []
    s.set_profile_checked("example_model", True)
    assert s.checked_profiles() == ["example_model"]
    s.set_checked([])
    assert s.checked_profiles() == []
    assert events["checked"] == [["example_model"], []]

    s.set_focus("example_model")
    assert s.focus == "example_model"
    assert events["focus"] == ["example_model"]

    assert s.profile_status(StripTool.STRIP, "example_model") \
        is ProcessingStatus.NOT_STARTED
    assert not s.has_settings(StripTool.STRIP, "example_model")
    s.assign_settings(StripTool.STRIP, ["example_model"], {"generate_report": False})
    assert s.has_settings(StripTool.STRIP, "example_model")
    assert s.profile_settings(StripTool.STRIP, "example_model") \
        == {"generate_report": False}
    assert events["settings"] == ["example_model"]

    fill = s.suggest_layer_fill(250.0)
    assert set(fill) == {"nu", "vp", "density", "soil_type"}


def test_layer_model_steps_as_layers(_qapp, tmp_path, _mstate):
    """Runs on the MODULE state: the strip test above already landed a
    result, so LayerModel derives the steps tree at construction — same
    contract, no second worker-thread strip in this process."""
    from HV_Strip_Progressive.gui.v2.state.layer_model import LayerModel
    from HV_Strip_Progressive.gui.v2.state import StripTool

    s = _mstate
    if not s.strip_results():           # standalone-run fallback
        finished = []
        s.op_finished.connect(lambda n, e: finished.append(n))
        s.run_strip("example_model", output_dir=str(tmp_path / "out"))
        assert _pump(_qapp, lambda: bool(finished))
    s.set_active_tool(StripTool.STRIP)
    lm = LayerModel(s)

    groups = lm.groups()
    steps_group = next(g for g in groups if g.key == "steps")
    assert len(steps_group.children) >= 2
    first = steps_group.children[0]
    assert first.key == "step::0" and "-layer" in first.label
    assert first.detail.endswith("Hz")
    # visibility + display round-trip
    lm.set_visible(first.key, False)
    assert lm.is_visible(first.key) is False
    lm.set_display(first.key, color="#123456")
    assert lm.display(first.key)["color"] == "#123456"


def test_profile_status_transitions_from_ops(_qapp, tmp_path, _mstate):
    """R2-T1.2 — a real strip flips the profile's Strip badge
    QUEUED→RUNNING→DONE; update_profile resets it.  Runs LAST in the
    module: it edits the shared profile."""
    from HV_Strip_Progressive.gui.v2.state import ProcessingStatus, StripTool

    s = _mstate
    seen = []
    s.profile_status_changed.connect(
        lambda tool, name: seen.append(
            (tool, name, s.profile_status(tool, name))))

    finished = []
    s.op_finished.connect(lambda n, e: finished.append(n))
    s.run_strip("example_model", output_dir=str(tmp_path / "st_out"))
    assert _pump(_qapp, lambda: bool(finished))

    strip_states = [st for t, n, st in seen
                    if t is StripTool.STRIP and n == "example_model"]
    assert strip_states[0] is ProcessingStatus.QUEUED
    assert ProcessingStatus.RUNNING in strip_states
    assert strip_states[-1] is ProcessingStatus.DONE
    assert s.profile_status(StripTool.STRIP, "example_model") \
        is ProcessingStatus.DONE

    # Editing the model invalidates the badge (results no longer describe it)
    env = s.update_profile("example_model", [
        {"thickness": 10.0, "vs": 200.0},
        {"thickness": 0.0, "vs": 700.0, "is_halfspace": True},
    ])
    assert env.get("success")
    assert s.profile_status(StripTool.STRIP, "example_model") \
        is ProcessingStatus.NOT_STARTED
