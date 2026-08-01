"""S5 research-api pins — the persistent runner, the ``profiles_dir`` load
path, inner-error propagation, and the metrics single-category recursion fix.

Qt-free (api-only); the runner is stubbed so no engines/SoilGen are needed.
"""

from __future__ import annotations

import pytest


class _StubRunner:
    """Records calls; instances are counted so tests can pin identity."""

    instances = 0

    def __init__(self):
        _StubRunner.instances += 1
        self.calls = []
        self.configured = {}
        self.metrics_result = {"phase": "metrics"}

    def configure(self, **kw):
        self.configured.update(kw)

    def set_progress_callback(self, cb):
        pass

    def generate_profiles(self):
        self.calls.append("generate_profiles")
        return {"phase": "profile_generation", "n_profiles": 3}

    def load_profiles(self, profiles_dir):
        self.calls.append(("load_profiles", profiles_dir))
        return {"phase": "profile_loading", "n_profiles": 2}

    def run_comparison(self):
        self.calls.append("run_comparison")
        return {"phase": "forward_comparison", "total_runs": 6,
                "successful_runs": 6}

    def compute_metrics(self):
        self.calls.append("compute_metrics")
        return self.metrics_result

    def run_field_validation(self):
        self.calls.append("run_field_validation")
        return {"phase": "field_validation", "n_sites": 0}

    def generate_report(self):
        self.calls.append("generate_report")
        return {"phase": "report", "n_files": 1, "files": {"a": "a.png"}}


@pytest.fixture()
def analysis(monkeypatch):
    import HV_Strip_Progressive.research.runner as runner_mod
    from HV_Strip_Progressive.api import HVStripAnalysis

    monkeypatch.setattr(runner_mod, "ComparisonStudyRunner", _StubRunner)
    _StubRunner.instances = 0
    return HVStripAnalysis()


def test_runner_persists_across_phase_calls(analysis):
    """Phases sequenced as separate calls must land on ONE runner —
    profiles → comparison state would otherwise be lost."""
    env1 = analysis.run_research_study(phase="profiles", reset=True)
    env2 = analysis.run_research_study(phase="comparison")
    assert env1["success"] and env2["success"]
    assert _StubRunner.instances == 1
    assert analysis._research_runner.calls == [
        "generate_profiles", "run_comparison"]


def test_reset_starts_a_fresh_runner(analysis):
    analysis.run_research_study(phase="profiles", reset=True)
    analysis.run_research_study(phase="profiles", reset=True)
    assert _StubRunner.instances == 2


def test_profiles_dir_loads_instead_of_generating(analysis):
    env = analysis.run_research_study(
        study_config={"profiles_dir": "X:/suite"},
        phase="profiles", reset=True)
    assert env["success"]
    assert env["results"]["phase"] == "profile_loading"
    runner = analysis._research_runner
    assert ("load_profiles", "X:/suite") in runner.calls
    assert "generate_profiles" not in runner.calls
    # profiles_dir is NOT a ComparisonStudyConfig field — never configured.
    assert "profiles_dir" not in runner.configured


def test_inner_phase_error_surfaces_as_failure(analysis):
    """Runner phases report failures as ``{"error": ...}`` dicts — the
    envelope must not mask them under ``success: True``."""
    analysis.run_research_study(phase="profiles", reset=True)
    analysis._research_runner.metrics_result = {"error": "boom"}
    env = analysis.run_research_study(phase="metrics")
    assert env["success"] is False
    assert env["error"] == "boom"

    analysis.run_research_study(phase="profiles", reset=True)
    analysis._research_runner.metrics_result = {"error": "boom"}
    full = analysis.run_research_study(phase="full")
    assert full["success"] is False and full["phase"] == "metrics"
    assert full["error"] == "boom"
    assert "profiles" in full["results"]        # partials still returned


def test_metrics_single_category_terminates():
    """compute_metrics recursed unconditionally per category and could
    never finish on a non-empty dataset (fixed: recurse only when the
    dataset spans >1 category)."""
    from HV_Strip_Progressive.research.config import MetricsConfig
    from HV_Strip_Progressive.research.forward_comparison import (
        ComparisonDataset,
        EngineResult,
        ProfileComparison,
    )
    from HV_Strip_Progressive.research.metrics import compute_metrics

    def comp(name, category):
        return ProfileComparison(
            profile_name=name, category=category,
            engine_results={"sh_wave": EngineResult(
                profile_name=name, engine_name="sh_wave", success=True,
                peaks=[{"frequency": 4.2, "amplitude": 3.0}])},
        )

    one_cat = ComparisonDataset(
        comparisons=[comp("a", "models"), comp("b", "models")],
        engine_names=["sh_wave"])
    m = compute_metrics(one_cat, MetricsConfig())      # must terminate
    assert m.engine_stats[0].n_successful == 2
    assert m.per_category == {}

    two_cats = ComparisonDataset(
        comparisons=[comp("a", "soft"), comp("b", "stiff")],
        engine_names=["sh_wave"])
    m2 = compute_metrics(two_cats, MetricsConfig())
    assert set(m2.per_category) == {"soft", "stiff"}
    for sub in m2.per_category.values():
        assert sub.per_category == {}                  # one level deep
