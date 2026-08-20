"""Spec 002 T06 — checked-set dispatch ops (Qt-free).

``compute_forward_for`` / ``run_stripping_for`` run EXACTLY the named
profiles, apply per-profile setting overrides to a config COPY, emit
per-item ``profile`` frames, and honor a cooperative cancel token
BETWEEN items.
"""

from __future__ import annotations

import threading

import pytest

from HV_Strip_Progressive.api import HVStripAnalysis

_LAYERS = [
    {"thickness": 5.0, "vs": 200.0},
    {"thickness": 15.0, "vs": 400.0},
    {"thickness": 0.0, "vs": 800.0},
]


@pytest.fixture()
def analysis() -> HVStripAnalysis:
    a = HVStripAnalysis()
    a.set_engine(name="sh_wave")                 # pure-python engine
    a.set_frequency(fmin=0.5, fmax=20.0, n_samples=128)
    a.set_strip(generate_report=False)
    for name in ("p1", "p2", "p3"):
        a.create_profile_from_layers(_LAYERS, name=name)
    return a


def test_forward_dispatch_runs_exactly_the_named_set(analysis):
    frames = []
    env = analysis.compute_forward_for(["p1", "p3"],
                                       progress_cb=frames.append)
    assert env["success"] is True and env["cancelled"] is False
    assert env["completed"] == ["p1", "p3"]
    assert set(analysis.forward_results()) == {"p1", "p3"}   # NOT p2
    prof = [f for f in frames if f.get("type") == "profile"]
    assert [(f["index"], f["total"], f["state"]) for f in prof] == [
        (1, 2, "started"), (1, 2, "finished"),
        (2, 2, "started"), (2, 2, "finished"),
    ]


def test_forward_dispatch_per_profile_overrides_leave_session_intact(
        analysis):
    env = analysis.compute_forward_for(
        ["p1", "p2"],
        settings_by_name={"p1": {"frequency": {"n_samples": 64}}})
    assert env["success"] is True
    n1 = len(analysis.forward_results()["p1"].frequencies)
    n2 = len(analysis.forward_results()["p2"].frequencies)
    assert n1 != n2, "the override must change p1's grid only"
    # The SESSION config is untouched (overrides applied to a copy).
    assert analysis.config.frequency.n_samples == 128


def test_forward_dispatch_missing_profile_fails_that_item(analysis):
    env = analysis.compute_forward_for(["p1", "ghost"])
    assert env["success"] is False
    assert env["completed"] == ["p1"]
    assert env["failed"] == [{"name": "ghost",
                              "error": "profile not loaded"}]


def test_forward_dispatch_cancel_between_items(analysis):
    ev = threading.Event()

    def cb(frame):
        if (frame.get("type") == "profile"
                and frame.get("state") == "finished"
                and frame.get("index") == 1):
            ev.set()                             # cancel after item 1

    env = analysis.compute_forward_for(["p1", "p2", "p3"],
                                       progress_cb=cb, cancel=ev)
    assert env["cancelled"] is True and env["success"] is False
    assert env["completed"] == ["p1"]
    assert set(analysis.forward_results()) == {"p1"}


def test_strip_dispatch_precancelled_runs_nothing(analysis, tmp_path):
    ev = threading.Event()
    ev.set()
    env = analysis.run_stripping_for(["p1"], output_dir=str(tmp_path),
                                     cancel=ev)
    assert env["cancelled"] is True and env["completed"] == []
    assert analysis.strip_results() == {}


def test_strip_dispatch_named_set(analysis, tmp_path):
    frames = []
    env = analysis.run_stripping_for(["p2"], output_dir=str(tmp_path),
                                     progress_cb=frames.append)
    assert env["success"] is True
    assert env["completed"] == ["p2"]
    assert set(analysis.strip_results()) == {"p2"}
    out = tmp_path / "p2"
    assert out.is_dir(), "each profile strips into <output>/<name>/"
    prof = [f for f in frames if f.get("type") == "profile"]
    assert prof[0]["name"] == "p2" and prof[-1]["state"] == "finished"
    assert analysis.strip_results()["p2"].steps, "steps parsed"
