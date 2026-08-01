"""The config funnel — v2 payloads, the legacy-GUI-dict migration, round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from HV_Strip_Progressive.api import (
    CONFIG_VERSION,
    HVStripConfig,
    load_config_payload,
)

FIXTURE = (Path(__file__).resolve().parent.parent.parent
           / "tests" / "golden" / "legacy_gui_config.json")


@pytest.fixture()
def legacy() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
def test_v2_round_trip():
    cfg = HVStripConfig()
    cfg.engine.name = "sh_wave"
    cfg.frequency.fmax = 33.0
    cfg.strip.generate_report = False
    d = cfg.to_dict()
    assert d["config_version"] == CONFIG_VERSION
    back = HVStripConfig.from_dict(d)
    assert back.engine.name == "sh_wave"
    assert back.frequency.fmax == pytest.approx(33.0)
    assert back.strip.generate_report is False


def test_legacy_dict_migrates_with_legacy_values_winning(legacy):
    cfg = HVStripConfig.from_legacy_gui_dict(legacy)
    # Engine + frequency from the active engine's settings.
    assert cfg.engine.name == "diffuse_field"
    assert cfg.frequency.fmin == pytest.approx(0.2)
    assert cfg.frequency.fmax == pytest.approx(20.0)
    assert cfg.frequency.nf == 71
    # Adaptive block mapped 1:1.
    assert cfg.adaptive.enable is True
    assert cfg.adaptive.fmax_limit == pytest.approx(60.0)
    # Renames: dual_resonance.enable → enabled; hv_postprocess.output →
    # output_files.
    assert cfg.dual_resonance.enabled is False
    assert cfg.postprocess.output_files.summary_filename == "step_summary.csv"
    # Non-active engine params survive onto the single EngineConfig.
    assert cfg.engine.Drock == pytest.approx(0.5)
    assert cfg.engine.gpell_path
    # Workflow flags.
    assert cfg.strip.generate_report is True
    assert cfg.strip.interactive_mode is False
    # Smoothing nested values (legacy window_length=9 beats the default 7).
    assert cfg.postprocess.hv_plot.smoothing.window_length == 9


def test_workflow_config_equivalence(legacy):
    """The MIGRATED dataclasses must hand core the same workflow knobs the
    legacy GUI dict carried — the real correctness bar."""
    cfg = HVStripConfig.from_legacy_gui_dict(legacy)
    wf = cfg.build_workflow_config()
    assert wf["engine_name"] == legacy["engine_name"]
    assert wf["hv_forward"]["fmin"] == legacy["hv_forward"]["fmin"]
    assert wf["hv_forward"]["fmax"] == legacy["hv_forward"]["fmax"]
    assert wf["hv_forward"]["nf"] == legacy["hv_forward"]["nf"]
    assert wf["hv_forward"]["nmr"] == legacy["hv_forward"]["nmr"]
    assert (wf["dual_resonance"]["enable"]
            == legacy["dual_resonance"]["enable"])
    assert wf["generate_report"] == legacy["generate_report"]
    assert wf["interactive_mode"] == legacy["interactive_mode"]


def test_funnel_detects_both_shapes(legacy):
    # Legacy shape → migrated.
    via_funnel = load_config_payload(legacy)
    assert via_funnel.frequency.nf == 71
    # v2 shape → applied directly.
    v2 = HVStripConfig()
    v2.frequency.nf = 99
    again = load_config_payload(v2.to_dict())
    assert again.frequency.nf == 99
    # Garbage → defaults.
    assert load_config_payload(None).frequency.nf == HVStripConfig().frequency.nf


def test_unknown_legacy_keys_are_logged_not_fatal(legacy, caplog):
    import logging

    legacy = dict(legacy)
    legacy["mystery_section"] = {"a": 1}
    with caplog.at_level(logging.INFO):
        cfg = HVStripConfig.from_legacy_gui_dict(legacy)
    assert cfg.engine.name == "diffuse_field"
    assert any("mystery_section" in r.message for r in caplog.records)
