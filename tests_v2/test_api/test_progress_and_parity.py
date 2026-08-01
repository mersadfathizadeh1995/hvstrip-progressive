"""Progress streaming + golden parity + the engine probe + the research pins.

The compute-freeze bar: with ``progress_cb=None`` the api path produces the
SAME numbers as the committed golden fixtures; with a callback it streams
ordered frames WITHOUT changing those numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from HV_Strip_Progressive.api import HVStripAnalysis, HVStripConfig
from HV_Strip_Progressive.api._progress import parse_line
from HV_Strip_Progressive.api.strip_engine import run_stripping

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_MODEL = ROOT / "examples" / "different_files" / "example_model.txt"
GOLDEN_SH = ROOT / "tests" / "golden" / "workflow_sh_wave.json"


def _sh_config() -> HVStripConfig:
    cfg = HVStripConfig()
    cfg.engine.name = "sh_wave"
    return cfg


# ----------------------------------------------------------------------
#  The stdout-line parser
# ----------------------------------------------------------------------
def test_parse_line_frames():
    assert parse_line("[1/3] Layer Stripping") == {
        "type": "phase", "index": 1, "total": 3, "label": "Layer Stripping"}
    assert parse_line("[4/8] Creating interface analysis...")["index"] == 4
    assert parse_line("[OK] done in 1.75s") == {
        "type": "log", "text": "[OK] done in 1.75s"}
    assert parse_line("random noise") is None
    assert parse_line("") is None


# ----------------------------------------------------------------------
#  Streaming: ordered frames, numbers unchanged
# ----------------------------------------------------------------------
def test_run_stripping_streams_frames_and_matches_golden(tmp_path):
    frames = []
    result = run_stripping(
        str(EXAMPLE_MODEL), output_dir=str(tmp_path / "out"),
        config=_sh_config(), generate_report=True,
        progress_cb=frames.append,
    )
    assert result.success, result.error

    # Frames: op start … numbered phases in order … op finish.
    assert frames[0] == {"type": "op", "op": "strip",
                         "profile": "example_model", "state": "started"}
    assert frames[-1]["state"] == "finished"
    phase_idx = [f["index"] for f in frames if f["type"] == "phase"
                 and f["total"] in (3, 4, 5)]
    assert phase_idx == sorted(phase_idx) and phase_idx[0] == 1
    assert any(f["type"] == "log" and f["text"].startswith("[OK]")
               for f in frames)

    # Numbers: each step's peak matches the committed golden fixture.
    golden = json.loads(GOLDEN_SH.read_text(encoding="utf-8"))
    by_name = {f"Step{s.step_number}_{s.n_layers}-layer": s
               for s in result.steps}
    for name, want in golden["steps"].items():
        step = by_name[name]
        assert step.peak_frequency == pytest.approx(want["peak_frequency"])
        assert step.peak_amplitude == pytest.approx(want["peak_amplitude"])
        assert len(step.frequencies) == want["n_frequencies"]


def test_progress_cb_none_is_byte_identical(tmp_path):
    """The legacy path: no callback → same numbers, nothing captured."""
    result = run_stripping(
        str(EXAMPLE_MODEL), output_dir=str(tmp_path / "out"),
        config=_sh_config(), generate_report=False,
    )
    assert result.success
    golden = json.loads(GOLDEN_SH.read_text(encoding="utf-8"))
    first = result.steps[0]
    want = golden["steps"][f"Step0_{first.n_layers}-layer"]
    assert first.peak_frequency == pytest.approx(want["peak_frequency"])
    assert float(np.sum(
        _read_curve(result, 0))) == pytest.approx(want["amp_sum"], rel=1e-9)


def _read_curve(result, i):
    step_dir = Path(result.strip_directory)
    name = f"Step{result.steps[i].step_number}_{result.steps[i].n_layers}-layer"
    data = np.genfromtxt(step_dir / name / "hv_curve.csv",
                         delimiter=",", skip_header=1)
    return data[:, 1]


# ----------------------------------------------------------------------
#  The facade end-to-end (headless session)
# ----------------------------------------------------------------------
def test_facade_forward_and_strip_envelopes(tmp_path):
    a = HVStripAnalysis()
    a.set_engine(name="sh_wave")
    a.load_profile_from_file(str(EXAMPLE_MODEL), name="example_model")

    fwd = a.compute_forward_single("example_model")
    assert isinstance(fwd, dict) and fwd.get("success")

    frames = []
    strip = a.run_stripping_for_profile(
        "example_model", output_dir=str(tmp_path / "strip"),
        progress_cb=frames.append,
    )
    assert isinstance(strip, dict) and strip.get("success")
    assert strip["steps"] and "peak_frequency" in strip["steps"][0]
    assert any(f["type"] == "phase" for f in frames)


# ----------------------------------------------------------------------
#  check_engines — existence probe only
# ----------------------------------------------------------------------
def test_check_engines_shape_and_sh_wave():
    report = HVStripAnalysis().check_engines()
    assert set(report) == {"sh_wave", "diffuse_field", "ellipticity"}
    for entry in report.values():
        assert set(entry) == {"available", "reason"}
        assert isinstance(entry["available"], bool)
    assert report["sh_wave"]["available"] is True
    # The vendored HVf.exe ships in-tree on this repo.
    assert report["diffuse_field"]["available"] is True


# ----------------------------------------------------------------------
#  research/ stays pinned to the api surface it imports
# ----------------------------------------------------------------------
def test_research_api_imports_pinned():
    from HV_Strip_Progressive.api.strip_engine import run_stripping as rs
    from HV_Strip_Progressive.api.config import HVStripConfig as C

    import inspect

    sig = inspect.signature(rs)
    for name in ("profile_or_path", "output_dir", "config",
                 "generate_report", "progress_cb"):
        assert name in sig.parameters
    assert C().engine.name == "diffuse_field"
    # The research modules import without error.
    import HV_Strip_Progressive.research.strip_comparison  # noqa: F401


def test_run_research_study_rejects_unknown_phase():
    env = HVStripAnalysis().run_research_study(phase="nope")
    assert env["success"] is False and "Unknown" in env["error"]
