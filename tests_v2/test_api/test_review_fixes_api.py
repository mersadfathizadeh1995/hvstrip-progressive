"""Regression pins for the 2026-08-15 review fixes — api layer.

Each test pins one CONFIRMED finding from the HV_Pro code review:
``HVStripConfig.from_dict`` misrouting partial v2 dicts to the legacy
migrator (silently dropping valid sections), and the stdout tee's
behaviour now that TWO OpQueues can print concurrently.
(The metrics single-category fix is pinned in test_research_study.py.)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from HV_Strip_Progressive.api.config import HVStripConfig, _apply_dict

FIXTURE = (Path(__file__).resolve().parent.parent.parent
           / "tests" / "golden" / "legacy_gui_config.json")


# ----------------------------------------------------------------------
#  HVStripConfig.from_dict routing
# ----------------------------------------------------------------------
def test_from_dict_partial_v2_applies_directly():
    # The review repro: no config_version and no marker key — this used to
    # route to the legacy migrator, which dropped BOTH sections (the run
    # then used the vendored HVf.exe and generated an unwanted report).
    cfg = HVStripConfig.from_dict({
        "engine": {"name": "diffuse_field", "exe_path": "C:/custom/HVf.exe"},
        "strip": {"generate_report": False},
    })
    assert cfg.engine.exe_path == "C:/custom/HVf.exe"
    assert cfg.strip.generate_report is False


def test_from_dict_overlap_only_key_is_v2():
    # "engine" exists in BOTH shapes; alone it must be treated as v2.
    cfg = HVStripConfig.from_dict({"engine": {"name": "sh_wave"}})
    assert cfg.engine.name == "sh_wave"


def test_from_dict_legacy_fixture_still_migrates():
    legacy = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cfg = HVStripConfig.from_dict(legacy)
    # hv_forward → frequency is a legacy-migrator mapping; the v2
    # _apply_dict path could never produce it.
    assert cfg.frequency.fmin == pytest.approx(0.2)
    assert cfg.frequency.nf == 71


def test_from_dict_legacy_only_key_routes_legacy():
    cfg = HVStripConfig.from_dict({
        "engine": {"name": "sh_wave"},
        "engine_settings": {"sh_wave": {"fmin": 0.5}},
    })
    assert cfg.engine.name == "sh_wave"
    assert cfg.frequency.fmin == pytest.approx(0.5)


def test_apply_dict_collects_unmapped_keys():
    cfg = HVStripConfig()
    unmapped: list = []
    _apply_dict(cfg, {"engine": {"name": "sh_wave", "bogus": 1},
                      "no_such_section": {}}, unmapped=unmapped)
    assert set(unmapped) == {"engine.bogus", "no_such_section"}
    assert cfg.engine.name == "sh_wave"          # valid keys still apply


# ----------------------------------------------------------------------
#  api._progress — the stdout tee under two queues
# ----------------------------------------------------------------------
def test_tee_parses_owner_thread_only():
    from HV_Strip_Progressive.api._progress import tee_progress

    frames: list = []
    with tee_progress(frames.append):
        print("[1/2] Owner phase")
        errs: list = []

        def _other_thread():
            try:
                # Another OpQueue's op printing concurrently — must pass
                # through unparsed, never cross-attributed to this op.
                print("[2/2] Intruder phase")
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

        t = threading.Thread(target=_other_thread)
        t.start()
        t.join()
        assert not errs
    labels = [f["label"] for f in frames if f.get("type") == "phase"]
    assert labels == ["Owner phase"]


def test_tee_second_concurrent_install_degrades():
    from HV_Strip_Progressive.api import _progress

    outer: list = []
    inner: list = []
    with _progress.tee_progress(outer.append):
        # Single-flight: the nested install must degrade to a no-op
        # instead of corrupting the sys.stdout swap nesting.
        with _progress.tee_progress(inner.append):
            print("[1/1] Only the outer tee parses this")
    assert [f["label"] for f in outer if f.get("type") == "phase"] == \
        ["Only the outer tee parses this"]
    assert inner == []
