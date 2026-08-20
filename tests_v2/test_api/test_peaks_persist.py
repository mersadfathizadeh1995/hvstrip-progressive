"""Spec 002 T04+T05 — the write-back chain + results-folder rehydration.

The heart of SC-1: the new api's persistence is compared BYTE-for-byte
against files the REAL legacy code produced (the committed
``fixtures/legacy_picked_run`` tree), and rehydration reads both legacy
trees (no sidecar) and v2 trees (sidecar, secondaries recovered).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from HV_Strip_Progressive.api import (
    HVStripAnalysis,
    SIDECAR_NAME,
    load_picks,
    persist_peaks,
    rehydrate_results_folder,
)
from HV_Strip_Progressive.api.persist_ops import (
    VS_RESULTS_NAME,
    create_minimal_summary,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / \
    "legacy_picked_run"
STRIP_SRC = FIXTURE / "run" / "strip"
PICKS = json.loads((FIXTURE / "picks.json").read_text(encoding="utf-8"))
PICKED_STEPS = list(PICKS["steps"])          # the 3 legacy-picked folders


@pytest.fixture()
def strip_copy(tmp_path) -> Path:
    dst = tmp_path / "strip"
    shutil.copytree(STRIP_SRC, dst)
    return dst


# ----------------------------------------------------------------------
#  persist_peaks vs the legacy-produced bytes
# ----------------------------------------------------------------------
def test_persist_matches_legacy_bytes(strip_copy):
    env = persist_peaks(str(strip_copy), PICKS["steps"])
    assert env["success"] is True
    assert env["unmatched_steps"] == []
    assert len(env["updated_summaries"]) == len(PICKED_STEPS)

    # The summary CSVs and vs_results.json must be BYTE-identical to what
    # the real legacy writers produced for the same picks.
    for step in PICKED_STEPS:
        ours = (strip_copy / step / "step_summary.csv").read_bytes()
        legacy = (STRIP_SRC / step / "step_summary.csv").read_bytes()
        assert ours == legacy, f"summary drift in {step}"
    assert (strip_copy / VS_RESULTS_NAME).read_bytes() == \
        (STRIP_SRC / VS_RESULTS_NAME).read_bytes()

    # And the NEW sidecar carries what legacy always dropped.
    sidecar = json.loads((strip_copy / SIDECAR_NAME).read_text())
    s0 = sidecar["steps"][PICKED_STEPS[0]]
    assert s0["f0"]["frequency"] == pytest.approx(4.10)
    assert len(s0["secondary"]) == 2             # secondaries PERSISTED
    assert s0["secondary"][0]["frequency"] == pytest.approx(8.50)


def test_create_minimal_summary_matches_legacy_format(tmp_path):
    path = tmp_path / "step_summary.csv"
    assert create_minimal_summary(path, "Step4_2-layer", 6.5, 2.25)
    assert path.read_bytes() == (
        b"Step,N_Finite_Layers,Peak_Frequency_Hz,Peak_Amplitude\r\n"
        b"Step4,2,6.500000,2.250000\r\n"
    )


# ----------------------------------------------------------------------
#  load_picks — legacy fallback and sidecar round-trip
# ----------------------------------------------------------------------
def test_load_picks_legacy_fallback_reads_summaries_and_vs():
    env = load_picks(str(STRIP_SRC))             # fixture has NO sidecar
    assert env["source"] == "legacy"
    s0 = env["steps"][PICKED_STEPS[0]]
    assert s0["f0"]["frequency"] == pytest.approx(4.10)
    assert s0["f0"]["source"] == "auto"          # provenance unknown
    assert s0["secondary"] == []                 # the legacy gap — nothing
    assert s0["vs30"] == pytest.approx(252.7)    # to recover
    assert s0["bedrock_depth"] == pytest.approx(44.0)


def test_load_picks_sidecar_recovers_secondaries(strip_copy):
    persist_peaks(str(strip_copy), PICKS["steps"])
    env = load_picks(str(strip_copy))
    assert env["source"] == "sidecar"
    s0 = env["steps"][PICKED_STEPS[0]]
    assert [s["frequency"] for s in s0["secondary"]] == \
        [pytest.approx(8.50), pytest.approx(12.30)]
    assert s0["f0"]["source"] == "manual"


# ----------------------------------------------------------------------
#  rehydrate_results_folder — no recompute
# ----------------------------------------------------------------------
def test_rehydrate_legacy_tree():
    env = rehydrate_results_folder(str(STRIP_SRC))
    assert env["success"] is True
    assert env["picks_source"] == "legacy"
    result = env["result"]
    assert [s.step_number for s in result.steps] == [0, 1, 2, 3, 4, 5]
    assert [s.n_layers for s in result.steps] == [6, 5, 4, 3, 2, 1]
    s0 = result.steps[0]
    assert len(s0.frequencies) > 0 and len(s0.frequencies) == \
        len(s0.amplitudes)
    assert s0.peak_frequency == pytest.approx(4.10)   # the persisted pick
    assert s0.model_path.endswith("model_Step0_6-layer.txt")


def test_rehydrate_bad_paths():
    assert rehydrate_results_folder("Z:/no/such/dir")["success"] is False
    assert "error" in rehydrate_results_folder(str(FIXTURE))  # no Step dirs


# ----------------------------------------------------------------------
#  The facade chain: re-open → pick → persist → re-open (FR-4 + FR-5)
# ----------------------------------------------------------------------
def test_facade_reopen_pick_persist_reopen(strip_copy):
    a = HVStripAnalysis()
    env = a.load_results_folder(str(strip_copy), profile_name="reopened")
    assert env["success"] is True and env["n_steps"] == 6
    assert env["picks_source"] == "legacy"

    # Step keys resolve to folder names through the strip result.
    env = a.set_step_peaks("reopened", "Step3", f0=(6.2, 2.5),
                           secondary=[(11.0, 1.4)], vs30=280.0)
    assert env["step"] == "Step3_3-layer"

    env = a.persist_picked_peaks("reopened", regenerate_report=False)
    assert env["success"] is True

    # Disk: the Step3 summary now carries the pick.
    text = (strip_copy / "Step3_3-layer" / "step_summary.csv").read_text()
    assert "6.200000" in text and "2.500000" in text
    # Memory: the StripResult mirrors it (the legacy in-memory update).
    step3 = a.strip_results()["reopened"].steps[3]
    assert step3.peak_frequency == pytest.approx(6.2)

    # A FRESH session re-opens with everything recovered from the sidecar.
    b = HVStripAnalysis()
    env = b.load_results_folder(str(strip_copy), profile_name="again")
    assert env["picks_source"] == "sidecar"
    got = b.get_step_peaks("again", "Step3")
    assert got["f0"]["frequency"] == pytest.approx(6.2)
    assert got["secondary"][0]["frequency"] == pytest.approx(11.0)
    assert got["vs30"] == pytest.approx(280.0)
    assert b.strip_results()["again"].steps[3].peak_frequency == \
        pytest.approx(6.2)
