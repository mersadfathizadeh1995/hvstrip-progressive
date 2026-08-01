"""R2-T1.1 — unified profile input through the api.

Every legacy GUI input mode must be loadable through the facade (the legacy
GUI called ``core.SoilProfile.*`` directly — these tests pin the api parity),
plus the ONE derivation surface for the layer table's auto-fill.

Qt-free; fixtures = the real ``examples/different_files/`` files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from HV_Strip_Progressive.api import HVStripAnalysis
from HV_Strip_Progressive.api.profile_io import (
    load_profile_dinver,
    load_profiles_from_directory,
    suggest_layer_fill,
)
from HV_Strip_Progressive.core.velocity_utils import VelocityConverter

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES = ROOT / "examples" / "different_files"
DINVER = EXAMPLES / "Dinver_style"


def test_hvf_file_via_facade():
    a = HVStripAnalysis()
    env = a.load_profile_from_file(str(EXAMPLES / "example_model.txt"))
    assert env["name"] == "example_model"
    assert env["summary"]["n_layers"] == 7
    assert env["summary"]["has_halfspace"] is True


def test_dinver_three_files_via_facade():
    """Vs step-polyline + Vp step-polyline + the HVf-shaped density file."""
    a = HVStripAnalysis()
    env = a.load_profile_dinver(
        vs_file=str(DINVER / "Vs_median_dinver.txt"),
        vp_file=str(DINVER / "Vp_median_dinver.txt"),
        rho_file=str(DINVER / "Density_median_model.txt"),
        name="dinver_combo",
    )
    assert env["name"] == "dinver_combo"
    layers = a.get_profile("dinver_combo")["layers"]
    assert len(layers) >= 2
    assert layers[-1]["is_halfspace"] or layers[-1]["thickness"] == 0
    # All three quantities came from FILES, not derivation defaults.
    assert all(ly["vs"] > 0 for ly in layers)
    assert all(ly["vp"] and ly["vp"] > ly["vs"] for ly in layers)
    assert all(ly["density"] and ly["density"] > 500 for ly in layers)


def test_dinver_vs_only_derives_missing():
    profile = load_profile_dinver(str(EXAMPLES / "Vs_median_dinver.txt"))
    assert profile.layers
    for layer in profile.layers:
        assert layer.vs > 0


def test_directory_load_collects_errors(tmp_path):
    a = HVStripAnalysis()
    env = a.load_profiles_from_directory(str(EXAMPLES))
    assert env["success"] is True
    assert "example_model" in env["loaded"]
    assert "model" in env["loaded"]
    # every loaded name is queryable
    for name in env["loaded"]:
        assert a.get_profile(name)["layers"]

    # a bad file is an ERROR entry, not an exception
    bad_dir = tmp_path / "mixed"
    bad_dir.mkdir()
    (bad_dir / "good.txt").write_text(
        (EXAMPLES / "example_model.txt").read_text(encoding="utf-8"),
        encoding="utf-8")
    (bad_dir / "junk.txt").write_text("not a profile at all", encoding="utf-8")
    profiles, errors = load_profiles_from_directory(str(bad_dir))
    assert [p.name for p in profiles] == ["good"]
    assert len(errors) == 1 and "junk.txt" in errors[0][0]


def test_suggest_layer_fill_matches_velocity_converter():
    for vs in (120.0, 250.0, 420.0, 900.0, 1600.0):
        fill = suggest_layer_fill(vs)
        nu = VelocityConverter.suggest_nu(vs)
        assert fill["nu"] == pytest.approx(nu)
        assert fill["vp"] == pytest.approx(
            VelocityConverter.vp_from_vs_nu(vs, nu))
        assert fill["density"] == pytest.approx(
            VelocityConverter.suggest_density(vs))
        assert fill["soil_type"]
    # facade passthrough
    assert HVStripAnalysis.suggest_layer_fill(250.0) == suggest_layer_fill(250.0)


def test_update_profile_replaces_and_invalidates():
    a = HVStripAnalysis()
    a.set_engine(name="sh_wave")
    a.load_profile_from_file(str(EXAMPLES / "example_model.txt"))
    env = a.compute_forward_single("example_model")
    assert env.get("success", True) and a._forward_results  # noqa: SLF001

    new_layers = [
        {"thickness": 12.0, "vs": 210.0},
        {"thickness": 0.0, "vs": 760.0, "is_halfspace": True},
    ]
    env = a.update_profile("example_model", new_layers)
    assert env["success"] is True
    assert env["summary"]["n_layers"] == 2
    assert "example_model" not in a._forward_results       # noqa: SLF001
    layers = a.get_profile("example_model")["layers"]
    assert layers[0]["vp"] and layers[0]["density"]         # auto-derived

    missing = a.update_profile("nope", new_layers)
    assert missing["success"] is False
